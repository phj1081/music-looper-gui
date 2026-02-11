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

# 2. Build sidecar
echo "2. Building Python sidecar..."
cd backend
uv run --all-extras --with pyinstaller \
  pyinstaller MusicLooperSidecar.spec --noconfirm
cd ..

# 3. Copy sidecar to Tauri binaries
echo "3. Copying sidecar to Tauri binaries..."
TARGET=$(rustc -vV | grep host | awk '{print $2}')
SIDECAR_SRC="backend/dist/music-looper-sidecar"
SIDECAR_DST="src-tauri/binaries/music-looper-sidecar-${TARGET}"

mkdir -p "$(dirname "${SIDECAR_DST}")"
rm -rf "${SIDECAR_DST}"

if [ -d "${SIDECAR_SRC}" ]; then
    cp -r "${SIDECAR_SRC}" "${SIDECAR_DST}"
elif [ -f "${SIDECAR_SRC}" ]; then
    cp "${SIDECAR_SRC}" "${SIDECAR_DST}"
else
    echo "ERROR: Sidecar binary not found at ${SIDECAR_SRC}"
    exit 1
fi

echo "   Sidecar copied to: ${SIDECAR_DST}"

# 4. Frontend dependencies
echo "4. Installing frontend dependencies..."
pnpm -C frontend install

echo ""
echo "Setup complete! Run 'pnpm dev' to start development."
