use crate::engine::{find_binary_on_path, find_repo_root, normalize_windows_path};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use zip::ZipArchive;

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TorchIndexMap {
    pub cuda: Option<String>,
    pub cpu: Option<String>,
    pub mps: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RuntimeManifest {
    pub schema_version: u32,
    pub engine_version: String,
    pub python_version: String,
    pub base_dependencies: Vec<String>,
    pub torch_dependencies: Vec<String>,
    pub torch_indexes: TorchIndexMap,
    pub deps_hash: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RuntimeStateFile {
    pub deps_hash: String,
    pub accelerator: String,
    pub python_version: String,
    pub engine_version: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct RuntimeStatus {
    pub mode: String,
    pub ready: bool,
    pub needs_setup: bool,
    pub needs_sync: bool,
    pub accelerator: String,
    pub runtime_dir: String,
    pub engine_src_dir: Option<String>,
    pub python_bin: Option<String>,
    pub uv_installed: bool,
    pub ffmpeg_installed: bool,
    pub venv_installed: bool,
    pub installed_deps_hash: Option<String>,
    pub expected_deps_hash: Option<String>,
    pub reason: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RuntimeProgressEvent {
    pub stage: String,
    pub step: u32,
    pub total_steps: u32,
    pub percent: f64,
    pub message: String,
    pub detail: String,
}

#[derive(Clone, Copy, Debug)]
pub enum ArchiveKind {
    Zip,
}

#[derive(Clone, Copy, Debug)]
pub struct ToolArtifactSpec {
    pub id: &'static str,
    pub display_name: &'static str,
    pub url: &'static str,
    pub archive_kind: ArchiveKind,
    pub binaries: &'static [&'static str],
    pub sha256: Option<&'static str>,
}

pub fn platform_tool_artifacts() -> Vec<ToolArtifactSpec> {
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    {
        vec![
            ToolArtifactSpec {
                id: "uv",
                display_name: "uv Paketmanager",
                url: "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip",
                archive_kind: ArchiveKind::Zip,
                binaries: &["uv.exe"],
                sha256: None,
            },
            ToolArtifactSpec {
                id: "ffmpeg",
                display_name: "FFmpeg + FFprobe",
                url: "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip",
                archive_kind: ArchiveKind::Zip,
                binaries: &["ffmpeg.exe", "ffprobe.exe"],
                sha256: None,
            },
        ]
    }
    #[cfg(not(all(target_os = "windows", target_arch = "x86_64")))]
    {
        Vec::new()
    }
}

#[derive(Clone, Debug)]
pub struct RuntimeManager {
    runtime_root: PathBuf,
    bundled_engine_dir: Option<PathBuf>,
}

impl RuntimeManager {
    pub fn discover(resource_dir: Option<PathBuf>, app_local_data_dir: Option<PathBuf>) -> Self {
        let runtime_root = resolve_runtime_root(app_local_data_dir);
        let bundled_engine_dir = resolve_bundled_engine_dir(resource_dir);
        Self {
            runtime_root,
            bundled_engine_dir,
        }
    }

    pub fn from_paths(runtime_root: PathBuf, bundled_engine_dir: Option<PathBuf>) -> Self {
        Self {
            runtime_root: normalize_windows_path(runtime_root),
            bundled_engine_dir: bundled_engine_dir.map(normalize_windows_path),
        }
    }

    pub fn runtime_root(&self) -> &Path {
        &self.runtime_root
    }

    pub fn bin_dir(&self) -> PathBuf {
        self.runtime_root.join("bin")
    }

    pub fn venv_dir(&self) -> PathBuf {
        self.runtime_root.join("venv")
    }

    pub fn venv_python_path(&self) -> PathBuf {
        #[cfg(target_os = "windows")]
        {
            self.venv_dir().join("Scripts").join("python.exe")
        }
        #[cfg(not(target_os = "windows"))]
        {
            self.venv_dir().join("bin").join("python")
        }
    }

    pub fn python_install_dir(&self) -> PathBuf {
        self.runtime_root.join("python")
    }

    pub fn uv_cache_dir(&self) -> PathBuf {
        self.runtime_root.join("uv-cache")
    }

    pub fn tmp_dir(&self) -> PathBuf {
        self.runtime_root.join("tmp")
    }

    pub fn state_file_path(&self) -> PathBuf {
        self.runtime_root.join("runtime-state.json")
    }

    pub fn bundled_engine_dir(&self) -> Option<&Path> {
        self.bundled_engine_dir.as_deref()
    }

    pub fn bundled_engine_src_dir(&self) -> Option<PathBuf> {
        self.bundled_engine_dir
            .as_ref()
            .map(|dir| dir.join("src"))
            .filter(|src| src.join("ruby_ai").join("__init__.py").is_file())
    }

    pub fn detect_accelerator(&self) -> String {
        if let Ok(explicit) = std::env::var("RUBY_AI_ACCELERATOR") {
            let clean = explicit.trim().to_ascii_lowercase();
            if matches!(clean.as_str(), "cuda" | "cpu" | "mps") {
                return clean;
            }
        }

        #[cfg(target_os = "windows")]
        {
            let sys_root = std::env::var("SystemRoot").unwrap_or_else(|_| r"C:\Windows".to_string());
            let nvcuda = PathBuf::from(sys_root).join("System32").join("nvcuda.dll");
            if nvcuda.is_file() || find_binary_on_path("nvidia-smi", &[]).is_some() {
                return "cuda".to_string();
            }
            return "cpu".to_string();
        }

        #[cfg(target_os = "macos")]
        {
            return "mps".to_string();
        }

        #[cfg(all(unix, not(target_os = "macos")))]
        {
            if Path::new("/dev/nvidiactl").exists()
                || find_binary_on_path("nvidia-smi", &[]).is_some()
            {
                return "cuda".to_string();
            }
            return "cpu".to_string();
        }
    }

    pub fn load_manifest(&self) -> Result<RuntimeManifest, String> {
        let engine_dir = self.bundled_engine_dir.as_ref().ok_or_else(|| {
            "Gebündeltes ruby_ai-Engine-Verzeichnis (resources/engine) nicht gefunden.".to_string()
        })?;
        let manifest_path = engine_dir.join("runtime-manifest.json");
        let text = std::fs::read_to_string(&manifest_path).map_err(|err| {
            format!(
                "Konnte runtime-manifest.json nicht lesen ({}): {err}",
                manifest_path.display()
            )
        })?;
        serde_json::from_str::<RuntimeManifest>(&text).map_err(|err| {
            format!(
                "Ungültiges runtime-manifest.json ({}): {err}",
                manifest_path.display()
            )
        })
    }

    pub fn read_state(&self) -> Option<RuntimeStateFile> {
        let text = std::fs::read_to_string(self.state_file_path()).ok()?;
        serde_json::from_str::<RuntimeStateFile>(&text).ok()
    }

    fn resolved_uv_path(&self) -> Option<PathBuf> {
        let extra = [self.bin_dir()];
        find_binary_on_path("uv", &extra)
    }

    fn has_ffmpeg_pair(&self) -> bool {
        let extra = [self.bin_dir()];
        find_binary_on_path("ffmpeg", &extra).is_some()
            && find_binary_on_path("ffprobe", &extra).is_some()
    }

    pub fn inspect_status(&self, workspace_ready: Option<(&Path, &Path)>) -> RuntimeStatus {
        let accelerator = self.detect_accelerator();
        let force_managed = std::env::var("RUBY_AI_FORCE_MANAGED_RUNTIME")
            .map(|v| v == "1")
            .unwrap_or(false);

        if !force_managed {
            if let Some((repo_root, py_bin)) = workspace_ready {
                let ffmpeg_ok = self.has_ffmpeg_pair();
                return RuntimeStatus {
                    mode: "workspace".to_string(),
                    ready: true,
                    needs_setup: false,
                    needs_sync: false,
                    accelerator,
                    runtime_dir: repo_root.display().to_string(),
                    engine_src_dir: Some(repo_root.join("src").display().to_string()),
                    python_bin: Some(py_bin.display().to_string()),
                    uv_installed: self.resolved_uv_path().is_some(),
                    ffmpeg_installed: ffmpeg_ok,
                    venv_installed: true,
                    installed_deps_hash: None,
                    expected_deps_hash: None,
                    reason: if ffmpeg_ok {
                        None
                    } else {
                        Some("FFmpeg/FFprobe nicht im PATH gefunden.".to_string())
                    },
                };
            }
        }

        let bin_dir = self.bin_dir();
        let venv_py = self.venv_python_path();
        let uv_installed = self.resolved_uv_path().is_some();
        let ffmpeg_installed = self.has_ffmpeg_pair();
        let venv_installed = venv_py.is_file();
        let engine_src_dir = self.bundled_engine_src_dir();
        let state = self.read_state();
        let installed_deps_hash = state.as_ref().map(|s| s.deps_hash.clone());

        let manifest = self.load_manifest();
        let (expected_deps_hash, state_matches, manifest_err) = match &manifest {
            Ok(m) => {
                let matches = state
                    .as_ref()
                    .map(|s| s.deps_hash == m.deps_hash && s.accelerator == accelerator)
                    .unwrap_or(false);
                (Some(m.deps_hash.clone()), matches, None)
            }
            Err(err) => (None, false, Some(err.clone())),
        };

        let needs_setup = !ffmpeg_installed || !venv_installed;
        let ready =
            ffmpeg_installed && venv_installed && state_matches && engine_src_dir.is_some();
        let needs_sync = !ready && !needs_setup && engine_src_dir.is_some();

        let reason = if let Some(err) = manifest_err {
            Some(err)
        } else if engine_src_dir.is_none() {
            Some("Gebündelter Python-Engine-Quellcode fehlt.".to_string())
        } else if !ffmpeg_installed {
            Some("FFmpeg / FFprobe wird für die Erstinstallation benötigt.".to_string())
        } else if !venv_installed {
            Some("Isolierte Python- & KI-Umgebung wird eingerichtet.".to_string())
        } else if !state_matches {
            Some("KI-Abhängigkeiten werden auf die aktuelle App-Version synchronisiert.".to_string())
        } else {
            None
        };

        let _ = bin_dir;
        RuntimeStatus {
            mode: "managed".to_string(),
            ready,
            needs_setup,
            needs_sync,
            accelerator,
            runtime_dir: self.runtime_root.display().to_string(),
            engine_src_dir: engine_src_dir.map(|p| p.display().to_string()),
            python_bin: venv_installed.then(|| venv_py.display().to_string()),
            uv_installed,
            ffmpeg_installed,
            venv_installed,
            installed_deps_hash,
            expected_deps_hash,
            reason,
        }
    }

    pub async fn bootstrap_managed_runtime<F>(
        &self,
        force_reinstall: bool,
        emit: F,
    ) -> Result<RuntimeStatus, String>
    where
        F: Fn(RuntimeProgressEvent) + Send + Sync + Clone + 'static,
    {
        let manifest = self.load_manifest()?;
        let engine_src_dir = self.bundled_engine_src_dir().ok_or_else(|| {
            "Gebündelter Python-Engine-Quellcode (resources/engine/src/ruby_ai) fehlt.".to_string()
        })?;
        let accelerator = self.detect_accelerator();

        let bin_dir = self.bin_dir();
        let venv_dir = self.venv_dir();
        let venv_py = self.venv_python_path();
        let python_install_dir = self.python_install_dir();
        let uv_cache_dir = self.uv_cache_dir();
        let tmp_dir = self.tmp_dir();

        for dir in [
            &self.runtime_root,
            &bin_dir,
            &python_install_dir,
            &uv_cache_dir,
            &tmp_dir,
        ] {
            std::fs::create_dir_all(dir).map_err(|err| {
                format!("Konnte Verzeichnis {} nicht anlegen: {err}", dir.display())
            })?;
        }

        let total_steps = 5_u32;
        let artifacts = platform_tool_artifacts();
        if artifacts.is_empty() {
            return Err(
                "Automatischer Runtime-Download wird auf diesem Betriebssystem noch nicht unterstützt."
                    .to_string(),
            );
        }

        let client = reqwest::Client::builder()
            .user_agent("ruby-ai-desktop-setup/0.1")
            .build()
            .map_err(|err| format!("HTTP-Client-Fehler: {err}"))?;

        // Step 1/5: uv
        let uv_spec = artifacts
            .iter()
            .find(|a| a.id == "uv")
            .copied()
            .ok_or_else(|| "Keine uv-Artifact-Definition gefunden.".to_string())?;

        let mut uv_bin = if force_reinstall {
            None
        } else {
            self.resolved_uv_path()
        };

        if uv_bin.is_none() {
            emit(RuntimeProgressEvent {
                stage: "uv".to_string(),
                step: 1,
                total_steps,
                percent: 2.0,
                message: "Lade uv Paketmanager…".to_string(),
                detail: uv_spec.url.to_string(),
            });
            let zip_path = tmp_dir.join("uv.zip");
            download_file_with_progress(&client, uv_spec.url, &zip_path, uv_spec.sha256, {
                let emit = emit.clone();
                move |downloaded, total| {
                    let frac = progress_fraction(downloaded, total);
                    emit(RuntimeProgressEvent {
                        stage: "uv".to_string(),
                        step: 1,
                        total_steps,
                        percent: 2.0 + frac * 8.0,
                        message: "Lade uv Paketmanager…".to_string(),
                        detail: format_download_detail(downloaded, total),
                    });
                }
            })
            .await?;

            let bin_dir_clone = bin_dir.clone();
            let zip_clone = zip_path.clone();
            let extracted = tauri::async_runtime::spawn_blocking(move || {
                extract_binaries_from_zip(&zip_clone, &bin_dir_clone, uv_spec.binaries)
            })
            .await
            .map_err(|err| format!("Task-Fehler beim Entpacken von uv: {err}"))??;
            let _ = std::fs::remove_file(&zip_path);
            uv_bin = extracted.into_iter().next();
        } else {
            emit(RuntimeProgressEvent {
                stage: "uv".to_string(),
                step: 1,
                total_steps,
                percent: 10.0,
                message: "uv Paketmanager bereits vorhanden.".to_string(),
                detail: uv_bin
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_default(),
            });
        }
        let uv_exe = uv_bin.ok_or_else(|| "uv-Binary konnte nicht bereitgestellt werden.".to_string())?;

        // Step 2/5: FFmpeg + FFprobe
        let ffmpeg_spec = artifacts
            .iter()
            .find(|a| a.id == "ffmpeg")
            .copied()
            .ok_or_else(|| "Keine FFmpeg-Artifact-Definition gefunden.".to_string())?;

        if force_reinstall || !self.has_ffmpeg_pair() {
            emit(RuntimeProgressEvent {
                stage: "ffmpeg".to_string(),
                step: 2,
                total_steps,
                percent: 12.0,
                message: "Lade FFmpeg & FFprobe (mit NVENC & zscale)…".to_string(),
                detail: ffmpeg_spec.url.to_string(),
            });
            let zip_path = tmp_dir.join("ffmpeg.zip");
            download_file_with_progress(&client, ffmpeg_spec.url, &zip_path, ffmpeg_spec.sha256, {
                let emit = emit.clone();
                move |downloaded, total| {
                    let frac = progress_fraction(downloaded, total);
                    emit(RuntimeProgressEvent {
                        stage: "ffmpeg".to_string(),
                        step: 2,
                        total_steps,
                        percent: 12.0 + frac * 18.0,
                        message: "Lade FFmpeg & FFprobe…".to_string(),
                        detail: format_download_detail(downloaded, total),
                    });
                }
            })
            .await?;

            emit(RuntimeProgressEvent {
                stage: "ffmpeg".to_string(),
                step: 2,
                total_steps,
                percent: 31.0,
                message: "Entpacke FFmpeg & FFprobe…".to_string(),
                detail: bin_dir.display().to_string(),
            });

            let bin_dir_clone = bin_dir.clone();
            let zip_clone = zip_path.clone();
            tauri::async_runtime::spawn_blocking(move || {
                extract_binaries_from_zip(&zip_clone, &bin_dir_clone, ffmpeg_spec.binaries)
            })
            .await
            .map_err(|err| format!("Task-Fehler beim Entpacken von FFmpeg: {err}"))??;
            let _ = std::fs::remove_file(&zip_path);
        } else {
            emit(RuntimeProgressEvent {
                stage: "ffmpeg".to_string(),
                step: 2,
                total_steps,
                percent: 32.0,
                message: "FFmpeg & FFprobe bereits installiert.".to_string(),
                detail: bin_dir.display().to_string(),
            });
        }

        // Step 3/5: Python venv
        let prev_state = self.read_state();
        let venv_existed = venv_py.is_file();
        if force_reinstall || !venv_existed {
            emit(RuntimeProgressEvent {
                stage: "venv".to_string(),
                step: 3,
                total_steps,
                percent: 35.0,
                message: format!("Installiere Python {} & erstelle Umgebung…", manifest.python_version),
                detail: venv_dir.display().to_string(),
            });

            let args = vec![
                "venv".to_string(),
                "--python".to_string(),
                manifest.python_version.clone(),
                "--managed-python".to_string(),
                "--clear".to_string(),
                venv_dir.display().to_string(),
            ];
            run_uv_command(
                &uv_exe,
                &args,
                &uv_cache_dir,
                &python_install_dir,
                {
                    let emit = emit.clone();
                    move |line| {
                        emit(RuntimeProgressEvent {
                            stage: "venv".to_string(),
                            step: 3,
                            total_steps,
                            percent: 42.0,
                            message: "Erstelle isolierte Python-Umgebung…".to_string(),
                            detail: line,
                        });
                    }
                },
            )
            .await?;
        } else {
            emit(RuntimeProgressEvent {
                stage: "venv".to_string(),
                step: 3,
                total_steps,
                percent: 45.0,
                message: "Python-Umgebung bereits vorhanden.".to_string(),
                detail: venv_py.display().to_string(),
            });
        }

        // Step 4/5: PyTorch / Torchvision
        let torch_needs_install = force_reinstall
            || !venv_existed
            || prev_state
                .as_ref()
                .map(|s| s.accelerator != accelerator || s.deps_hash != manifest.deps_hash)
                .unwrap_or(true);

        if torch_needs_install && !manifest.torch_dependencies.is_empty() {
            let accel_label = match accelerator.as_str() {
                "cuda" => "PyTorch GPU (CUDA 12.6)",
                "mps" => "PyTorch Apple Metal (MPS)",
                _ => "PyTorch CPU",
            };
            emit(RuntimeProgressEvent {
                stage: "torch".to_string(),
                step: 4,
                total_steps,
                percent: 48.0,
                message: format!("Installiere {accel_label} – einmalig ca. 1–2 Min…"),
                detail: manifest.torch_dependencies.join(", "),
            });

            let mut args = vec![
                "pip".to_string(),
                "install".to_string(),
                "--python".to_string(),
                venv_py.display().to_string(),
            ];
            let index_url = match accelerator.as_str() {
                "cuda" => manifest.torch_indexes.cuda.as_deref(),
                "mps" => manifest.torch_indexes.mps.as_deref(),
                _ => manifest.torch_indexes.cpu.as_deref(),
            };
            if let Some(url) = index_url {
                args.push("--index-url".to_string());
                args.push(url.to_string());
                args.push("--extra-index-url".to_string());
                args.push("https://pypi.org/simple".to_string());
                args.push("--index-strategy".to_string());
                args.push("first-index".to_string());
            }
            args.extend(manifest.torch_dependencies.clone());

            run_uv_command(
                &uv_exe,
                &args,
                &uv_cache_dir,
                &python_install_dir,
                {
                    let emit = emit.clone();
                    let accel_label = accel_label.to_string();
                    move |line| {
                        emit(RuntimeProgressEvent {
                            stage: "torch".to_string(),
                            step: 4,
                            total_steps,
                            percent: 65.0,
                            message: format!("Installiere {accel_label}…"),
                            detail: line,
                        });
                    }
                },
            )
            .await?;
        }

        // Step 5/5: Remaining Python deps + verification
        if torch_needs_install && !manifest.base_dependencies.is_empty() {
            emit(RuntimeProgressEvent {
                stage: "deps".to_string(),
                step: 5,
                total_steps,
                percent: 80.0,
                message: "Installiere Spandrel & Videoverarbeitungs-Pakete…".to_string(),
                detail: manifest.base_dependencies.join(", "),
            });

            let mut args = vec![
                "pip".to_string(),
                "install".to_string(),
                "--python".to_string(),
                venv_py.display().to_string(),
            ];
            args.extend(manifest.base_dependencies.clone());

            run_uv_command(
                &uv_exe,
                &args,
                &uv_cache_dir,
                &python_install_dir,
                {
                    let emit = emit.clone();
                    move |line| {
                        emit(RuntimeProgressEvent {
                            stage: "deps".to_string(),
                            step: 5,
                            total_steps,
                            percent: 88.0,
                            message: "Installiere Spandrel & Video-Pakete…".to_string(),
                            detail: line,
                        });
                    }
                },
            )
            .await?;
        }

        emit(RuntimeProgressEvent {
            stage: "verify".to_string(),
            step: 5,
            total_steps,
            percent: 95.0,
            message: "Prüfe KI-Laufzeitumgebung…".to_string(),
            detail: "import ruby_ai, torch, spandrel".to_string(),
        });

        let py_clone = venv_py.clone();
        let src_clone = engine_src_dir.clone();
        tauri::async_runtime::spawn_blocking(move || verify_python_runtime(&py_clone, &src_clone))
            .await
            .map_err(|err| format!("Verifizierungs-Task fehlgeschlagen: {err}"))??;

        let next_state = RuntimeStateFile {
            deps_hash: manifest.deps_hash.clone(),
            accelerator: accelerator.clone(),
            python_version: manifest.python_version.clone(),
            engine_version: manifest.engine_version.clone(),
        };
        write_json_atomic(&self.state_file_path(), &next_state)?;

        emit(RuntimeProgressEvent {
            stage: "ready".to_string(),
            step: total_steps,
            total_steps,
            percent: 100.0,
            message: "ruby.ai Runtime ist einsatzbereit!".to_string(),
            detail: format!("Modus: {} · Python {}", accelerator.to_uppercase(), manifest.python_version),
        });

        Ok(self.inspect_status(None))
    }
}

fn resolve_runtime_root(app_local_data_dir: Option<PathBuf>) -> PathBuf {
    if let Ok(explicit) = std::env::var("RUBY_AI_RUNTIME_DIR") {
        let p = PathBuf::from(explicit);
        if !p.as_os_str().is_empty() {
            return normalize_windows_path(p);
        }
    }

    if let Some(dir) = app_local_data_dir {
        return normalize_windows_path(dir.join("runtime"));
    }

    #[cfg(target_os = "windows")]
    if let Ok(local_app_data) = std::env::var("LOCALAPPDATA") {
        return normalize_windows_path(
            PathBuf::from(local_app_data)
                .join("ai.ruby.desktop")
                .join("runtime"),
        );
    }

    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join(".ruby-ai").join("runtime");
    }

    PathBuf::from(".ruby-ai-runtime")
}

fn resolve_bundled_engine_dir(resource_dir: Option<PathBuf>) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();

    if let Ok(explicit) = std::env::var("RUBY_AI_BUNDLED_ENGINE") {
        candidates.push(PathBuf::from(explicit));
    }

    if let Some(res_dir) = resource_dir {
        let res = normalize_windows_path(res_dir);
        candidates.push(res.join("resources").join("engine"));
        candidates.push(res.join("engine"));
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            let clean_exe_dir = normalize_windows_path(exe_dir.to_path_buf());
            candidates.push(clean_exe_dir.join("resources").join("engine"));
            candidates.push(clean_exe_dir.join("engine"));
        }
    }

    if let Some(root) = find_repo_root() {
        candidates.push(
            root.join("desktop")
                .join("src-tauri")
                .join("resources")
                .join("engine"),
        );
    }

    candidates.into_iter().find(|dir| is_valid_engine_dir(dir))
}

fn is_valid_engine_dir(dir: &Path) -> bool {
    dir.join("runtime-manifest.json").is_file()
        && dir
            .join("src")
            .join("ruby_ai")
            .join("__init__.py")
            .is_file()
}

fn progress_fraction(downloaded: u64, total: Option<u64>) -> f64 {
    match total {
        Some(t) if t > 0 => (downloaded as f64 / t as f64).clamp(0.0, 1.0),
        _ => 0.5,
    }
}

fn format_download_detail(downloaded: u64, total: Option<u64>) -> String {
    let dl_mb = downloaded as f64 / (1024.0 * 1024.0);
    match total {
        Some(t) if t > 0 => {
            let total_mb = t as f64 / (1024.0 * 1024.0);
            format!("{dl_mb:.1} MB / {total_mb:.1} MB")
        }
        _ => format!("{dl_mb:.1} MB geladen"),
    }
}

async fn download_file_with_progress<F>(
    client: &reqwest::Client,
    url: &str,
    dest_path: &Path,
    expected_sha256: Option<&str>,
    on_progress: F,
) -> Result<(), String>
where
    F: Fn(u64, Option<u64>),
{
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|err| format!("Download fehlgeschlagen ({url}): {err}"))?;

    if !response.status().is_success() {
        return Err(format!(
            "Download-Server meldete HTTP {} für {url}",
            response.status()
        ));
    }

    let total = response.content_length();
    let part_path = dest_path.with_extension("part");
    if let Some(parent) = part_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|err| format!("Konnte Temp-Ordner nicht erstellen: {err}"))?;
    }

    let mut file = std::fs::File::create(&part_path)
        .map_err(|err| format!("Konnte Zieldatei nicht anlegen ({}): {err}", part_path.display()))?;
    let mut hasher = Sha256::new();
    let mut downloaded: u64 = 0;
    let mut last_emit_bytes: u64 = 0;
    let mut stream = response.bytes_stream();

    while let Some(chunk_res) = stream.next().await {
        let chunk = chunk_res.map_err(|err| format!("Stream-Fehler beim Download: {err}"))?;
        file.write_all(&chunk)
            .map_err(|err| format!("Schreibfehler beim Download: {err}"))?;
        hasher.update(&chunk);
        downloaded += chunk.len() as u64;
        if downloaded - last_emit_bytes >= 512 * 1024 || total.is_some_and(|t| downloaded >= t) {
            last_emit_bytes = downloaded;
            on_progress(downloaded, total);
        }
    }

    file.flush()
        .map_err(|err| format!("Flush-Fehler beim Download: {err}"))?;
    drop(file);

    if let Some(expected) = expected_sha256 {
        let actual = hex::encode(hasher.finalize());
        if !actual.eq_ignore_ascii_case(expected.trim()) {
            let _ = std::fs::remove_file(&part_path);
            return Err(format!(
                "SHA-256-Prüfsumme fehlgeschlagen für {url}: erwartet {expected}, erhalten {actual}"
            ));
        }
    }

    if dest_path.exists() {
        let _ = std::fs::remove_file(dest_path);
    }
    std::fs::rename(&part_path, dest_path)
        .map_err(|err| format!("Konnte Download nicht abschließen: {err}"))?;
    Ok(())
}

pub fn extract_binaries_from_zip(
    zip_path: &Path,
    dest_dir: &Path,
    wanted_names: &[&str],
) -> Result<Vec<PathBuf>, String> {
    std::fs::create_dir_all(dest_dir)
        .map_err(|err| format!("Konnte Zielordner {} nicht erstellen: {err}", dest_dir.display()))?;

    let file = std::fs::File::open(zip_path)
        .map_err(|err| format!("Konnte ZIP nicht öffnen ({}): {err}", zip_path.display()))?;
    let mut archive = ZipArchive::new(file)
        .map_err(|err| format!("Ungültiges ZIP-Archiv ({}): {err}", zip_path.display()))?;

    let wanted_lookup: HashMap<String, &str> = wanted_names
        .iter()
        .map(|name| (name.to_ascii_lowercase(), *name))
        .collect();
    let mut found: HashSet<String> = HashSet::new();
    let mut extracted_paths: Vec<PathBuf> = Vec::new();

    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .map_err(|err| format!("ZIP-Lesefehler: {err}"))?;
        if entry.is_dir() {
            continue;
        }
        let Some(file_name) = Path::new(entry.name()).file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        let lower = file_name.to_ascii_lowercase();
        let Some(&canonical_name) = wanted_lookup.get(&lower) else {
            continue;
        };
        if found.contains(&lower) {
            continue;
        }

        let target = dest_dir.join(canonical_name);
        let temp_target = dest_dir.join(format!("{canonical_name}.tmp"));
        {
            let mut out = std::fs::File::create(&temp_target)
                .map_err(|err| format!("Konnte {} nicht schreiben: {err}", temp_target.display()))?;
            let mut buf = [0_u8; 65536];
            loop {
                let n = entry
                    .read(&mut buf)
                    .map_err(|err| format!("Entpackfehler für {canonical_name}: {err}"))?;
                if n == 0 {
                    break;
                }
                out.write_all(&buf[..n])
                    .map_err(|err| format!("Schreibfehler für {canonical_name}: {err}"))?;
            }
            out.flush()
                .map_err(|err| format!("Flush-Fehler für {canonical_name}: {err}"))?;
        }

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Ok(meta) = std::fs::metadata(&temp_target) {
                let mut perms = meta.permissions();
                perms.set_mode(0o755);
                let _ = std::fs::set_permissions(&temp_target, perms);
            }
        }

        if target.exists() {
            let _ = std::fs::remove_file(&target);
        }
        std::fs::rename(&temp_target, &target)
            .map_err(|err| format!("Konnte {} nicht aktivieren: {err}", target.display()))?;

        found.insert(lower);
        extracted_paths.push(target);
    }

    let missing: Vec<&str> = wanted_names
        .iter()
        .copied()
        .filter(|name| !found.contains(&name.to_ascii_lowercase()))
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "Folgende Dateien fehlten im Archiv {}: {}",
            zip_path.display(),
            missing.join(", ")
        ));
    }

    Ok(extracted_paths)
}

async fn run_uv_command<F>(
    uv_exe: &Path,
    args: &[String],
    uv_cache_dir: &Path,
    python_install_dir: &Path,
    on_line: F,
) -> Result<(), String>
where
    F: Fn(String) + Send + Sync + 'static,
{
    let uv_exe = uv_exe.to_path_buf();
    let args = args.to_vec();
    let uv_cache_dir = uv_cache_dir.to_path_buf();
    let python_install_dir = python_install_dir.to_path_buf();
    let on_line = std::sync::Arc::new(on_line);

    tauri::async_runtime::spawn_blocking(move || {
        let mut cmd = Command::new(&uv_exe);
        cmd.args(&args)
            .env("UV_CACHE_DIR", &uv_cache_dir)
            .env("UV_PYTHON_INSTALL_DIR", &python_install_dir)
            .env("UV_LINK_MODE", "copy")
            .env("UV_NO_PROGRESS", "1")
            .env("PYTHONUTF8", "1")
            .env("PYTHONIOENCODING", "utf-8")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        #[cfg(target_os = "windows")]
        cmd.creation_flags(CREATE_NO_WINDOW);

        let mut child = cmd
            .spawn()
            .map_err(|err| format!("Konnte uv nicht starten ({}): {err}", uv_exe.display()))?;

        let stderr = child.stderr.take();
        let stdout = child.stdout.take();

        let err_cb = std::sync::Arc::clone(&on_line);
        let err_thread = stderr.map(|pipe| {
            std::thread::spawn(move || {
                let mut lines: Vec<String> = Vec::new();
                let reader = BufReader::new(pipe);
                for line in reader.lines().map_while(Result::ok) {
                    let trimmed = line.trim().to_string();
                    if !trimmed.is_empty() {
                        err_cb(trimmed.clone());
                        lines.push(trimmed);
                    }
                }
                lines
            })
        });

        if let Some(out_pipe) = stdout {
            let reader = BufReader::new(out_pipe);
            for line in reader.lines().map_while(Result::ok) {
                let trimmed = line.trim();
                if !trimmed.is_empty() {
                    on_line(trimmed.to_string());
                }
            }
        }

        let status = child
            .wait()
            .map_err(|err| format!("Fehler beim Warten auf uv: {err}"))?;
        let stderr_lines = err_thread
            .and_then(|h| h.join().ok())
            .unwrap_or_default();

        if !status.success() {
            let tail = stderr_lines
                .iter()
                .rev()
                .take(8)
                .cloned()
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect::<Vec<_>>()
                .join("\n");
            return Err(format!("uv fehlgeschlagen ({status}): {tail}"));
        }

        Ok(())
    })
    .await
    .map_err(|err| format!("Runtime-Setup-Task fehlgeschlagen: {err}"))?
}

fn verify_python_runtime(venv_py: &Path, engine_src_dir: &Path) -> Result<(), String> {
    let mut cmd = Command::new(venv_py);
    cmd.args([
        "-c",
        "import ruby_ai, torch, spandrel; print(ruby_ai.__version__)",
    ])
    .env("PYTHONPATH", engine_src_dir)
    .env("PYTHONUTF8", "1")
    .env("PYTHONIOENCODING", "utf-8");

    #[cfg(target_os = "windows")]
    cmd.creation_flags(CREATE_NO_WINDOW);

    let out = cmd
        .output()
        .map_err(|err| format!("Python-Prüfung fehlgeschlagen: {err}"))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(format!(
            "Python-Importprüfung fehlgeschlagen ({}): {}",
            out.status,
            stderr.trim()
        ));
    }
    Ok(())
}

fn write_json_atomic<T: Serialize>(dest: &Path, value: &T) -> Result<(), String> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|err| format!("Konnte Ordner {} nicht anlegen: {err}", parent.display()))?;
    }
    let tmp = dest.with_extension("json.tmp");
    let json = serde_json::to_string_pretty(value)
        .map_err(|err| format!("JSON-Serialisierung fehlgeschlagen: {err}"))?;
    std::fs::write(&tmp, format!("{json}\n"))
        .map_err(|err| format!("Konnte {} nicht schreiben: {err}", tmp.display()))?;
    if dest.exists() {
        let _ = std::fs::remove_file(dest);
    }
    std::fs::rename(&tmp, dest)
        .map_err(|err| format!("Konnte {} nicht speichern: {err}", dest.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn make_temp_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("ruby_ai_rt_test_{label}_{nanos}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn extracts_nested_binaries_case_insensitively() {
        let root = make_temp_dir("zip");
        let zip_path = root.join("bundle.zip");
        let dest_dir = root.join("bin");

        {
            let file = std::fs::File::create(&zip_path).unwrap();
            let mut writer = zip::ZipWriter::new(file);
            let opts = zip::write::SimpleFileOptions::default();
            writer.start_file("ffmpeg-master/bin/ffmpeg.exe", opts).unwrap();
            writer.write_all(b"dummy-ffmpeg").unwrap();
            writer.start_file("ffmpeg-master/bin/FFPROBE.EXE", opts).unwrap();
            writer.write_all(b"dummy-ffprobe").unwrap();
            writer.finish().unwrap();
        }

        let extracted =
            extract_binaries_from_zip(&zip_path, &dest_dir, &["ffmpeg.exe", "ffprobe.exe"])
                .unwrap();
        assert_eq!(extracted.len(), 2);
        assert_eq!(
            std::fs::read(dest_dir.join("ffmpeg.exe")).unwrap(),
            b"dummy-ffmpeg"
        );
        assert_eq!(
            std::fs::read(dest_dir.join("ffprobe.exe")).unwrap(),
            b"dummy-ffprobe"
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn manifest_round_trip_and_status_transitions() {
        let root = make_temp_dir("status");
        let runtime_dir = root.join("runtime");
        let engine_dir = root.join("engine");
        let ruby_src = engine_dir.join("src").join("ruby_ai");
        std::fs::create_dir_all(&ruby_src).unwrap();
        std::fs::write(ruby_src.join("__init__.py"), "__version__ = '0.1.0'\n").unwrap();

        let manifest = RuntimeManifest {
            schema_version: 1,
            engine_version: "0.1.0".to_string(),
            python_version: "3.12".to_string(),
            base_dependencies: vec!["numpy>=2.0".to_string()],
            torch_dependencies: vec!["torch>=2.5".to_string()],
            torch_indexes: TorchIndexMap {
                cuda: Some("https://download.pytorch.org/whl/cu126".to_string()),
                cpu: Some("https://download.pytorch.org/whl/cpu".to_string()),
                mps: None,
            },
            deps_hash: "abc123".to_string(),
        };
        write_json_atomic(&engine_dir.join("runtime-manifest.json"), &manifest).unwrap();

        let mgr = RuntimeManager::from_paths(runtime_dir.clone(), Some(engine_dir.clone()));

        // Fresh state without venv -> needs_setup (unless global tools missing/present)
        let s1 = mgr.inspect_status(None);
        assert_eq!(s1.mode, "managed");
        assert!(!s1.ready);
        assert!(s1.needs_setup);
        assert!(!s1.venv_installed);

        // Create managed ffmpeg + venv + matching state -> ready
        let bin_dir = mgr.bin_dir();
        std::fs::create_dir_all(&bin_dir).unwrap();
        std::fs::write(bin_dir.join("ffmpeg.exe"), b"exe").unwrap();
        std::fs::write(bin_dir.join("ffprobe.exe"), b"exe").unwrap();
        std::fs::create_dir_all(mgr.venv_python_path().parent().unwrap()).unwrap();
        std::fs::write(mgr.venv_python_path(), b"py").unwrap();

        let state = RuntimeStateFile {
            deps_hash: "abc123".to_string(),
            accelerator: mgr.detect_accelerator(),
            python_version: "3.12".to_string(),
            engine_version: "0.1.0".to_string(),
        };
        write_json_atomic(&mgr.state_file_path(), &state).unwrap();

        let s2 = mgr.inspect_status(None);
        assert!(s2.ready);
        assert!(!s2.needs_setup);
        assert!(!s2.needs_sync);

        // Change manifest hash -> needs_sync without needing full setup
        let mut updated_manifest = manifest;
        updated_manifest.deps_hash = "def456".to_string();
        write_json_atomic(&engine_dir.join("runtime-manifest.json"), &updated_manifest).unwrap();

        let s3 = mgr.inspect_status(None);
        assert!(!s3.ready);
        assert!(!s3.needs_setup);
        assert!(s3.needs_sync);

        let _ = std::fs::remove_dir_all(root);
    }
}
