use std::collections::HashMap;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

#[derive(Clone)]
struct ActiveJob {
    pid: u32,
    cancelled: Arc<AtomicBool>,
}

#[derive(Default)]
pub struct JobManager {
    inner: Mutex<HashMap<String, ActiveJob>>,
}

impl JobManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&self, job_id: &str, pid: u32) -> Arc<AtomicBool> {
        let cancelled = Arc::new(AtomicBool::new(false));
        if let Ok(mut guard) = self.inner.lock() {
            guard.insert(
                job_id.to_string(),
                ActiveJob {
                    pid,
                    cancelled: Arc::clone(&cancelled),
                },
            );
        }
        cancelled
    }

    pub fn remove(&self, job_id: &str) {
        if let Ok(mut guard) = self.inner.lock() {
            guard.remove(job_id);
        }
    }

    pub fn cancel(&self, job_id: &str) -> bool {
        let maybe_job = self
            .inner
            .lock()
            .ok()
            .and_then(|guard| guard.get(job_id).cloned());

        if let Some(job) = maybe_job {
            job.cancelled.store(true, Ordering::SeqCst);
            kill_process_tree(job.pid);
            return true;
        }
        false
    }

    pub fn cancel_all(&self) {
        let jobs: Vec<ActiveJob> = self
            .inner
            .lock()
            .map(|guard| guard.values().cloned().collect())
            .unwrap_or_default();

        for job in jobs {
            job.cancelled.store(true, Ordering::SeqCst);
            kill_process_tree(job.pid);
        }
    }
}

fn kill_process_tree(pid: u32) {
    #[cfg(target_os = "windows")]
    {
        let mut cmd = Command::new("taskkill");
        cmd.args(["/PID", &pid.to_string(), "/T", "/F"]);
        cmd.creation_flags(CREATE_NO_WINDOW);
        let _ = cmd.status();
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = Command::new("kill")
            .args(["-TERM", &pid.to_string()])
            .status();
    }
}
