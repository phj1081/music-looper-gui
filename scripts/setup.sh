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
bash scripts/ensure-dev-sidecar.sh

# 3. Frontend dependencies
echo "3. Installing frontend dependencies..."
pnpm -C frontend install

echo ""
echo "Setup complete! Run 'pnpm dev' to start development."
echo "Python backend code changes are reflected on app restart (no rebuild needed)."
