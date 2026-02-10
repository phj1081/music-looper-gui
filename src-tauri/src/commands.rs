use crate::sidecar::ServerState;
use serde::Serialize;
use std::env;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tauri::State;
use tokio::time::sleep;

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeInfo {
    pub app_version: String,
    pub app_build_id: String,
    pub app_binary_path: String,
    pub sidecar_binary_path: String,
}

#[tauri::command]
pub async fn get_server_port(state: State<'_, ServerState>) -> Result<u16, String> {
    const STARTUP_TIMEOUT: Duration = Duration::from_secs(120);
    const POLL_INTERVAL: Duration = Duration::from_millis(100);

    let start = Instant::now();

    loop {
        {
            let stored_port = state
                .port
                .lock()
                .map_err(|_| "analysis server state poisoned".to_string())?;
            if let Some(port) = *stored_port {
                return Ok(port);
            }
        }

        {
            let startup_error = state
                .startup_error
                .lock()
                .map_err(|_| "analysis server state poisoned".to_string())?;
            if let Some(err) = startup_error.clone() {
                return Err(format!("analysis server failed to start: {}", err));
            }
        }

        if start.elapsed() >= STARTUP_TIMEOUT {
            return Err("analysis server startup timed out".to_string());
        }

        sleep(POLL_INTERVAL).await;
    }
}

#[tauri::command]
pub fn get_runtime_info() -> Result<RuntimeInfo, String> {
    let current_exe =
        env::current_exe().map_err(|e| format!("failed to resolve current executable path: {e}"))?;

    let parent_dir = current_exe
        .parent()
        .ok_or_else(|| "failed to resolve executable parent directory".to_string())?;
    let sidecar_path: PathBuf = parent_dir.join("music-looper-sidecar");

    Ok(RuntimeInfo {
        app_version: env!("CARGO_PKG_VERSION").to_string(),
        app_build_id: option_env!("MUSIC_LOOPER_BUILD_ID")
            .unwrap_or("unknown")
            .to_string(),
        app_binary_path: current_exe.to_string_lossy().into_owned(),
        sidecar_binary_path: sidecar_path.to_string_lossy().into_owned(),
    })
}
