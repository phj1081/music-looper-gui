mod commands;
mod sidecar;
use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            app.manage(sidecar::ServerState::default());
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                if let Err(e) = sidecar::spawn_server(&handle).await {
                    eprintln!("[ERROR] Failed to spawn sidecar: {}", e);
                    let state = handle.state::<sidecar::ServerState>();
                    let startup_error_lock = state.startup_error.lock();
                    if let Ok(mut startup_error) = startup_error_lock {
                        *startup_error = Some(e);
                    }
                }
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![commands::get_server_port])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
