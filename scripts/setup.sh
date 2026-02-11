#!/bin/bash
set -e

cd "$(git rev-parse --show-toplevel)"

echo "=== Music Looper Development Setup ==="

# 1. Backend dependencies
echo "1. Installing backend dependencies..."
cd backend
uv sync --all-extras \
  --no-build-isolation-package madmom \
  --no-build-isolation-package natten
cd ..

# 2. Create dev sidecar wrapper script
echo "2. Creating dev sidecar wrapper..."
TARGET=$(rustc -vV | grep host | awk '{print $2}')
SIDECAR_DST="src-tauri/binaries/music-looper-sidecar-${TARGET}"

mkdir -p "$(dirname "${SIDECAR_DST}")"
rm -rf "${SIDECAR_DST}"

cat > "${SIDECAR_DST}" << 'WRAPPER'
#!/bin/bash
cd "$(git rev-parse --show-toplevel)/backend"
exec uv run python http_server.py "$@"
WRAPPER
chmod +x "${SIDECAR_DST}"

echo "   Dev sidecar created at: ${SIDECAR_DST}"

# 3. Frontend dependencies
echo "3. Installing frontend dependencies..."
pnpm -C frontend install

echo ""
echo "Setup complete! Run 'pnpm dev' to start development."
echo "Python backend code changes are reflected on app restart (no rebuild needed)."
