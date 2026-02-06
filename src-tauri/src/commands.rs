use crate::sidecar::ServerState;
use tauri::State;

#[tauri::command]
pub fn get_server_port(state: State<'_, ServerState>) -> u16 {
    state.port
}
