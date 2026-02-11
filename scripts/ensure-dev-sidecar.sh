#!/bin/bash
# Ensure the sidecar is a dev wrapper script (not a PyInstaller binary).
# Called automatically by Tauri's beforeDevCommand.
set -e

cd "$(git rev-parse --show-toplevel)"

TARGET=$(rustc -vV | grep host | awk '{print $2}')
SIDECAR="src-tauri/binaries/music-looper-sidecar-${TARGET}"

mkdir -p "$(dirname "${SIDECAR}")"

# Skip if already a shell script
if file "${SIDECAR}" 2>/dev/null | grep -q "shell script"; then
  exit 0
fi

rm -rf "${SIDECAR}"
cat > "${SIDECAR}" << 'WRAPPER'
#!/bin/bash
cd "$(git rev-parse --show-toplevel)/backend"
exec uv run python http_server.py "$@"
WRAPPER
chmod +x "${SIDECAR}"
echo "Dev sidecar wrapper created at: ${SIDECAR}"
