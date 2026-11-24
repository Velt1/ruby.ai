use crate::runtime::RuntimeManager;
use serde_json::Value;
use std::ffi::OsString;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

#[derive(Clone, Debug)]
pub struct EngineLocator {
    pub mode: &'static str,
    repo_root: Option<PathBuf>,
    engine_src_dir: Option<PathBuf>,
    python_bin: PathBuf,
    extra_bin_dirs: Vec<PathBuf>,
}

impl EngineLocator {
    pub fn discover() -> Self {
        if let Some((root, py_bin)) = find_workspace_ready() {
            return Self::from_workspace(root, py_bin);
        }
        Self {
            mode: "workspace",
            repo_root: None,
            engine_src_dir: None,
            python_bin: resolve_python(None),
            extra_bin_dirs: Vec::new(),
        }
    }

    pub fn from_workspace(repo_root: PathBuf, python_bin: PathBuf) -> Self {
        let clean_root = normalize_windows_path(repo_root);
        let src_dir = clean_root.join("src");
        Self {
            mode: "workspace",
            repo_root: Some(clean_root),
            engine_src_dir: src_dir.is_dir().then_some(src_dir),
            python_bin: normalize_windows_path(python_bin),
            extra_bin_dirs: Vec::new(),
        }
    }

    pub fn from_managed(
        python_bin: PathBuf,
        engine_src_dir: PathBuf,
        runtime_bin_dir: PathBuf,
    ) -> Self {
        Self {
            mode: "managed",
            repo_root: None,
            engine_src_dir: Some(normalize_windows_path(engine_src_dir)),
            python_bin: normalize_windows_path(python_bin),
            extra_bin_dirs: vec![normalize_windows_path(runtime_bin_dir)],
        }
    }

    pub fn discover_with_runtime(runtime: &RuntimeManager) -> Self {
        let force_managed = std::env::var("RUBY_AI_FORCE_MANAGED_RUNTIME")
            .map(|v| v == "1")
            .unwrap_or(false);

        if !force_managed {
            if let Some((root, py_bin)) = find_workspace_ready() {
                return Self::from_workspace(root, py_bin);
            }
        }

        let managed_py = runtime.venv_python_path();
        let managed_src = runtime
            .bundled_engine_src_dir()
            .unwrap_or_else(|| runtime.runtime_root().join("src"));
        Self::from_managed(managed_py, managed_src, runtime.bin_dir())
    }

    pub fn run_sync_command(&self, subcommand: &str, payload: &Value) -> Result<Value, String> {
        let mut cmd = self.build_command(subcommand);
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = cmd.spawn().map_err(|err| {
            format!(
                "Failed to start Python bridge ({}): {err}",
                self.python_bin.display()
            )
        })?;

        if let Some(mut stdin) = child.stdin.take() {
            let serialized = serde_json::to_string(payload)
                .map_err(|err| format!("Failed to serialize bridge payload: {err}"))?;
            stdin
                .write_all(serialized.as_bytes())
                .map_err(|err| format!("Failed to write bridge stdin: {err}"))?;
        }

        let output = child
            .wait_with_output()
            .map_err(|err| format!("Failed waiting for bridge command '{subcommand}': {err}"))?;

        let stdout_str = String::from_utf8_lossy(&output.stdout);
        let stderr_str = String::from_utf8_lossy(&output.stderr);

        let mut last_result: Option<Value> = None;
        let mut last_error: Option<String> = None;

        for line in stdout_str.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Ok(val) = serde_json::from_str::<Value>(trimmed) {
                match val.get("type").and_then(Value::as_str) {
                    Some("result") => {
                        if let Some(data) = val.get("data") {
                            last_result = Some(data.clone());
                        }
                    }
                    Some("error") => {
                        if let Some(msg) = val.get("message").and_then(Value::as_str) {
                            last_error = Some(msg.to_string());
                        }
                    }
                    _ => {}
                }
            }
        }

        if let Some(err_msg) = last_error {
            return Err(err_msg);
        }
        if !output.status.success() {
            let detail = stderr_str.trim();
            return Err(format!(
                "Bridge '{subcommand}' exited with {}: {}",
                output.status,
                if detail.is_empty() {
                    stdout_str.trim()
                } else {
                    detail
                }
            ));
        }

        last_result.ok_or_else(|| {
            format!(
                "Bridge '{subcommand}' produced no result payload. stderr: {}",
                stderr_str.trim()
            )
        })
    }

    pub fn spawn_streaming_command(
        &self,
        subcommand: &str,
        payload: &Value,
    ) -> Result<Child, String> {
        let mut cmd = self.build_command(subcommand);
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = cmd.spawn().map_err(|err| {
            format!(
                "Failed to start Python bridge ({}): {err}",
                self.python_bin.display()
            )
        })?;

        if let Some(mut stdin) = child.stdin.take() {
            let serialized = serde_json::to_string(payload)
                .map_err(|err| format!("Failed to serialize bridge payload: {err}"))?;
            stdin
                .write_all(serialized.as_bytes())
                .map_err(|err| format!("Failed to write bridge stdin: {err}"))?;
        }

        Ok(child)
    }

    fn build_command(&self, subcommand: &str) -> Command {
        let mut cmd = Command::new(&self.python_bin);
        cmd.args(["-m", "ruby_ai.bridge", subcommand]);
        cmd.env("PYTHONUTF8", "1");
        cmd.env("PYTHONIOENCODING", "utf-8");
        cmd.env("HF_HUB_DISABLE_SYMLINKS_WARNING", "1");
        cmd.env("RUBY_AI_RUNTIME_MODE", self.mode);

        if let Some(root) = &self.repo_root {
            cmd.current_dir(root);
        }
        if let Some(src_dir) = &self.engine_src_dir {
            cmd.env("PYTHONPATH", src_dir);
        }

        cmd.env("PATH", build_augmented_path(&self.extra_bin_dirs));

        #[cfg(target_os = "windows")]
        {
            cmd.creation_flags(CREATE_NO_WINDOW);
            let pathext = std::env::var("PATHEXT").unwrap_or_default();
            if !pathext.to_ascii_uppercase().contains(".EXE") {
                cmd.env(
                    "PATHEXT",
                    ".COM;.EXE;.BAT;.CMD;.VBS;.VBE;.JS;.JSE;.WSF;.WSH;.MSC",
                );
            }
        }

        cmd
    }
}

pub fn normalize_windows_path(path: PathBuf) -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        if let Some(text) = path.to_str() {
            if let Some(stripped) = text.strip_prefix(r"\\?\") {
                if !stripped.starts_with("UNC\\") {
                    return PathBuf::from(stripped);
                }
            }
        }
    }
    path
}

fn is_ruby_repo_root(dir: &Path) -> bool {
    dir.join("pyproject.toml").is_file()
        && dir.join("src").join("ruby_ai").join("__init__.py").is_file()
}

pub fn find_repo_root() -> Option<PathBuf> {
    if let Ok(explicit) = std::env::var("RUBY_AI_ROOT") {
        let path = PathBuf::from(explicit);
        if is_ruby_repo_root(&path) {
            return Some(normalize_windows_path(path));
        }
    }

    if let Ok(cwd) = std::env::current_dir() {
        for ancestor in cwd.ancestors().take(8) {
            if is_ruby_repo_root(ancestor) {
                return Some(normalize_windows_path(ancestor.to_path_buf()));
            }
        }
    }

    if let Ok(exe) = std::env::current_exe() {
        for ancestor in exe.ancestors().take(8) {
            if is_ruby_repo_root(ancestor) {
                return Some(normalize_windows_path(ancestor.to_path_buf()));
            }
        }
    }

    None
}

pub fn find_workspace_ready() -> Option<(PathBuf, PathBuf)> {
    let root = find_repo_root()?;
    if let Ok(explicit) = std::env::var("RUBY_AI_PYTHON") {
        let p = PathBuf::from(explicit);
        if p.is_file() {
            return Some((root, normalize_windows_path(p)));
        }
    }

    #[cfg(target_os = "windows")]
    let venv_py = root.join(".venv").join("Scripts").join("python.exe");
    #[cfg(not(target_os = "windows"))]
    let venv_py = root.join(".venv").join("bin").join("python");

    if venv_py.is_file() {
        return Some((root, normalize_windows_path(venv_py)));
    }

    None
}

fn resolve_python(repo_root: Option<&Path>) -> PathBuf {
    if let Ok(explicit) = std::env::var("RUBY_AI_PYTHON") {
        let p = PathBuf::from(explicit);
        if p.is_file() {
            return normalize_windows_path(p);
        }
    }

    if let Some(root) = repo_root {
        #[cfg(target_os = "windows")]
        let venv_py = root.join(".venv").join("Scripts").join("python.exe");
        #[cfg(not(target_os = "windows"))]
        let venv_py = root.join(".venv").join("bin").join("python");

        if venv_py.is_file() {
            return normalize_windows_path(venv_py);
        }
    }

    #[cfg(target_os = "windows")]
    return PathBuf::from("python");
    #[cfg(not(target_os = "windows"))]
    return PathBuf::from("python3");
}

pub fn find_binary_on_path(binary_name: &str, extra_dirs: &[PathBuf]) -> Option<PathBuf> {
    let mut candidates = Vec::new();
    #[cfg(target_os = "windows")]
    {
        if binary_name.to_ascii_lowercase().ends_with(".exe") {
            candidates.push(binary_name.to_string());
        } else {
            candidates.push(format!("{binary_name}.exe"));
            candidates.push(binary_name.to_string());
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        candidates.push(binary_name.to_string());
    }

    let path_var = build_augmented_path(extra_dirs);
    for dir in std::env::split_paths(&path_var) {
        for cand in &candidates {
            let full = dir.join(cand);
            if full.is_file() {
                return Some(normalize_windows_path(full));
            }
        }
    }

    None
}

pub fn build_augmented_path(extra_dirs: &[PathBuf]) -> OsString {
    let mut entries: Vec<PathBuf> = extra_dirs
        .iter()
        .filter(|d| d.is_dir())
        .cloned()
        .collect();

    #[cfg(target_os = "windows")]
    {
        if let Ok(local_app_data) = std::env::var("LOCALAPPDATA") {
            let base = PathBuf::from(local_app_data).join("Microsoft").join("WinGet");
            let links = base.join("Links");
            if links.is_dir() {
                entries.push(links);
            }
            let packages = base.join("Packages");
            if let Ok(pkg_dirs) = std::fs::read_dir(&packages) {
                for pkg in pkg_dirs.flatten() {
                    let name = pkg.file_name().to_string_lossy().to_string();
                    let pkg_path = pkg.path();
                    if name.starts_with("Gyan.FFmpeg") {
                        if let Ok(children) = std::fs::read_dir(&pkg_path) {
                            for child in children.flatten() {
                                let bin = child.path().join("bin");
                                if bin.join("ffmpeg.exe").is_file() {
                                    entries.push(bin);
                                }
                            }
                        }
                    } else if name.starts_with("astral-sh.uv") && pkg_path.join("uv.exe").is_file() {
                        entries.push(pkg_path);
                    }
                }
            }
        }
    }

    if let Some(existing) = std::env::var_os("PATH") {
        entries.extend(std::env::split_paths(&existing));
    }

    std::env::join_paths(entries).unwrap_or_else(|_| std::env::var_os("PATH").unwrap_or_default())
}
