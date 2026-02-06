use crate::sidecar::ServerState;
use std::time::{Duration, Instant};
use tauri::State;
use tokio::time::sleep;

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
