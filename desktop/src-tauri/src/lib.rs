mod engine;
mod jobs;
mod runtime;

use engine::{find_workspace_ready, EngineLocator};
use jobs::JobManager;
use runtime::{RuntimeManager, RuntimeProgressEvent, RuntimeStatus};
use serde::Serialize;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::RwLock;
use tauri::{AppHandle, Emitter, Manager, RunEvent, State};
use uuid::Uuid;

struct AppState {
    engine: RwLock<EngineLocator>,
    runtime: RuntimeManager,
    jobs: JobManager,
    setup_in_progress: AtomicBool,
}

#[derive(Serialize)]
struct StartJobResponse {
    job_id: String,
}

struct SetupFlagGuard<'a>(&'a AtomicBool);

impl Drop for SetupFlagGuard<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

#[tauri::command]
fn get_runtime_status(state: State<'_, AppState>) -> Result<RuntimeStatus, String> {
    let workspace = find_workspace_ready();
    let ws_refs = workspace.as_ref().map(|(r, p)| (r.as_path(), p.as_path()));
    Ok(state.runtime.inspect_status(ws_refs))
}

#[tauri::command]
async fn ensure_runtime_ready(
    app: AppHandle,
    state: State<'_, AppState>,
    force_reinstall: Option<bool>,
) -> Result<RuntimeStatus, String> {
    let force = force_reinstall.unwrap_or(false);
    let workspace = find_workspace_ready();
    let ws_refs = workspace.as_ref().map(|(r, p)| (r.as_path(), p.as_path()));
    let current = state.runtime.inspect_status(ws_refs);

    if !force && current.ready {
        return Ok(current);
    }
    if current.mode == "workspace" {
        return Ok(current);
    }

    if state
        .setup_in_progress
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return Err("Runtime-Einrichtung läuft bereits.".to_string());
    }
    let _guard = SetupFlagGuard(&state.setup_in_progress);

    let app_handle = app.clone();
    let next_status = state
        .runtime
        .bootstrap_managed_runtime(force, move |ev: RuntimeProgressEvent| {
            let _ = app_handle.emit("ruby://runtime-event", ev);
        })
        .await?;

    let refreshed = EngineLocator::discover_with_runtime(&state.runtime);
    if let Ok(mut guard) = state.engine.write() {
        *guard = refreshed;
    }

    Ok(next_status)
}

#[tauri::command]
async fn get_engine_info(state: State<'_, AppState>) -> Result<Value, String> {
    let engine = state
        .engine
        .read()
        .map_err(|_| "Engine state lock poisoned".to_string())?
        .clone();
    tauri::async_runtime::spawn_blocking(move || engine.run_sync_command("info", &json!({})))
        .await
        .map_err(|err| format!("Bridge task failed: {err}"))?
}

#[tauri::command]
async fn probe_video(
    state: State<'_, AppState>,
    path: String,
    max_short_edge: Option<u32>,
    model: Option<String>,
) -> Result<Value, String> {
    let engine = state
        .engine
        .read()
        .map_err(|_| "Engine state lock poisoned".to_string())?
        .clone();
    let payload = json!({
        "path": path,
        "max_short_edge": max_short_edge.unwrap_or(1080),
        "model": model,
        "generate_poster": true
    });
    tauri::async_runtime::spawn_blocking(move || engine.run_sync_command("probe", &payload))
        .await
        .map_err(|err| format!("Probe task failed: {err}"))?
}

#[tauri::command]
async fn analyze_video(state: State<'_, AppState>, path: String) -> Result<Value, String> {
    let engine = state
        .engine
        .read()
        .map_err(|_| "Engine state lock poisoned".to_string())?
        .clone();
    let payload = json!({ "path": path });
    tauri::async_runtime::spawn_blocking(move || engine.run_sync_command("analyze", &payload))
        .await
        .map_err(|err| format!("Analyze task failed: {err}"))?
}

#[tauri::command]
fn start_job(
    app: AppHandle,
    state: State<'_, AppState>,
    mode: String,
    payload: Value,
) -> Result<StartJobResponse, String> {
    if mode != "enhance" && mode != "benchmark" && mode != "preview" {
        return Err(format!("Unsupported job mode '{mode}'"));
    }

    let engine = state
        .engine
        .read()
        .map_err(|_| "Engine state lock poisoned".to_string())?
        .clone();

    let job_id = Uuid::new_v4().to_string();
    let mut job_payload = match payload {
        Value::Object(map) => map,
        _ => return Err("Job payload must be a JSON object".to_string()),
    };
    job_payload.insert("job_id".to_string(), Value::String(job_id.clone()));

    let mut child = engine.spawn_streaming_command(&mode, &Value::Object(job_payload))?;

    let pid = child.id();
    let cancelled = state.jobs.register(&job_id, pid);
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "Failed to attach stdout pipe".to_string())?;
    let stderr = child.stderr.take();

    let app_handle = app.clone();
    let job_id_thread = job_id.clone();

    std::thread::spawn(move || {
        let stderr_handle = stderr.map(|mut err_pipe| {
            std::thread::spawn(move || {
                let mut buf = String::new();
                let _ = err_pipe.read_to_string(&mut buf);
                buf
            })
        });

        let mut saw_terminal_event = false;
        let reader = BufReader::new(stdout);
        for line in reader.lines().map_while(Result::ok) {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Ok(val) = serde_json::from_str::<Value>(trimmed) {
                let ev_type = val.get("type").and_then(Value::as_str);
                if matches!(ev_type, Some("completed") | Some("error") | Some("cancelled")) {
                    saw_terminal_event = true;
                }
                let _ = app_handle.emit("ruby://job-event", val);
            }
        }

        let status = child.wait();
        let stderr_text = stderr_handle
            .and_then(|h| h.join().ok())
            .unwrap_or_default();

        if let Some(app_state) = app_handle.try_state::<AppState>() {
            app_state.jobs.remove(&job_id_thread);
        }

        if cancelled.load(Ordering::SeqCst) {
            if !saw_terminal_event {
                let _ = app_handle.emit(
                    "ruby://job-event",
                    json!({
                        "type": "cancelled",
                        "job_id": job_id_thread
                    }),
                );
            }
            return;
        }

        if !saw_terminal_event {
            let err_msg = match status {
                Ok(exit) if !exit.success() => {
                    let detail = stderr_text.trim();
                    if detail.is_empty() {
                        format!("Worker exited with status {exit}")
                    } else {
                        detail.to_string()
                    }
                }
                Err(err) => format!("Failed waiting for worker: {err}"),
                _ => "Worker exited unexpectedly without completion event".to_string(),
            };
            let _ = app_handle.emit(
                "ruby://job-event",
                json!({
                    "type": "error",
                    "job_id": job_id_thread,
                    "message": err_msg
                }),
            );
        }
    });

    Ok(StartJobResponse { job_id })
}

#[tauri::command]
fn cancel_job(app: AppHandle, state: State<'_, AppState>, job_id: String) -> bool {
    let found = state.jobs.cancel(&job_id);
    if found {
        let _ = app.emit(
            "ruby://job-event",
            json!({
                "type": "cancelled",
                "job_id": job_id
            }),
        );
    }
    found
}

#[tauri::command]
fn reveal_in_explorer(path: String) -> Result<(), String> {
    let target = Path::new(&path);
    #[cfg(target_os = "windows")]
    {
        if target.is_file() {
            Command::new("explorer")
                .arg(format!("/select,{}", target.display()))
                .spawn()
                .map_err(|err| format!("Failed to open Explorer: {err}"))?;
        } else {
            let dir = if target.is_dir() {
                target
            } else {
                target.parent().unwrap_or(target)
            };
            Command::new("explorer")
                .arg(dir)
                .spawn()
                .map_err(|err| format!("Failed to open Explorer: {err}"))?;
        }
    }
    #[cfg(target_os = "macos")]
    {
        if target.is_file() {
            Command::new("open")
                .args(["-R", &target.to_string_lossy()])
                .spawn()
                .map_err(|err| format!("Failed to reveal in Finder: {err}"))?;
        } else {
            Command::new("open")
                .arg(target)
                .spawn()
                .map_err(|err| format!("Failed to open directory: {err}"))?;
        }
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let dir = if target.is_dir() {
            target
        } else {
            target.parent().unwrap_or(target)
        };
        Command::new("xdg-open")
            .arg(dir)
            .spawn()
            .map_err(|err| format!("Failed to open file manager: {err}"))?;
    }
    Ok(())
}

pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .setup(|app| {
            let resource_dir = app.path().resource_dir().ok();
            let app_local_data_dir = app.path().app_local_data_dir().ok();
            let runtime = RuntimeManager::discover(resource_dir, app_local_data_dir);
            let engine = EngineLocator::discover_with_runtime(&runtime);
            app.manage(AppState {
                engine: RwLock::new(engine),
                runtime,
                jobs: JobManager::new(),
                setup_in_progress: AtomicBool::new(false),
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_runtime_status,
            ensure_runtime_ready,
            get_engine_info,
            probe_video,
            analyze_video,
            start_job,
            cancel_job,
            reveal_in_explorer
        ])
        .build(tauri::generate_context!())
        .expect("error while building ruby.ai Tauri application");

    app.run(|app_handle, event| {
        if matches!(event, RunEvent::Exit) {
            if let Some(state) = app_handle.try_state::<AppState>() {
                state.jobs.cancel_all();
            }
        }
    });
}
