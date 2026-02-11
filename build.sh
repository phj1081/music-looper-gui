#!/bin/bash
set -e

echo "=== Building Music Looper (Tauri) ==="

cd "$(dirname "$0")"

# 1. Build Python sidecar (headless JSON-RPC server)
echo "1. Building Python sidecar..."
cd backend
echo "   Syncing backend deps (including beat/structure extras)..."
uv sync --all-extras --no-build-isolation-package madmom --no-build-isolation-package natten

echo "   Running PyInstaller inside uv project env..."
uv run --all-extras --with pyinstaller pyinstaller MusicLooperSidecar.spec --noconfirm
cd ..

# 2. Copy sidecar binary to Tauri binaries directory
echo "2. Copying sidecar to Tauri binaries..."
TARGET=$(rustc -vV | grep host | awk '{print $2}')
SIDECAR_SRC="backend/dist/music-looper-sidecar"
SIDECAR_DST="src-tauri/binaries/music-looper-sidecar-${TARGET}"

mkdir -p "$(dirname "${SIDECAR_DST}")"
rm -rf "${SIDECAR_DST}"

if [ -d "${SIDECAR_SRC}" ]; then
    # onedir mode: copy the entire directory
    cp -r "${SIDECAR_SRC}" "${SIDECAR_DST}"
elif [ -f "${SIDECAR_SRC}" ]; then
    # onefile mode: copy the binary
    cp "${SIDECAR_SRC}" "${SIDECAR_DST}"
else
    echo "ERROR: Sidecar binary not found at ${SIDECAR_SRC}"
    exit 1
fi

echo "   Sidecar copied to: ${SIDECAR_DST}"

# 3. Install frontend dependencies
echo "3. Installing frontend dependencies..."
pnpm -C frontend install --frozen-lockfile

# 4. Build Tauri app (includes frontend build via beforeBuildCommand)
echo "4. Building Tauri application..."
./frontend/node_modules/.bin/tauri build

echo ""
echo "Build complete!"
echo ""
echo "Output: src-tauri/target/release/bundle/"
