#!/bin/bash
set -e

echo "=== Building Music Looper Desktop App ==="

cd "$(dirname "$0")"

# Build frontend
echo "1. Building frontend..."
cd frontend
pnpm install --silent
pnpm build

# Copy static files to backend
echo "2. Copying static files..."
rm -rf ../backend/static
cp -r out ../backend/static

echo ""
echo "Build complete!"
echo ""
echo "To run:"
echo "  cd backend && python app.py"
echo ""
echo "To package as executable:"
echo "  cd backend"
echo "  pip install pyinstaller"
echo "  pyinstaller --onefile --windowed --add-data 'static:static' --name 'MusicLooper' app.py"
