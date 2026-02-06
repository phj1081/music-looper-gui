# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the headless sidecar (Tauri mode).
# Entry point: http_server.py (FastAPI HTTP server)
# No GUI, no static frontend assets.
# Uses onefile mode for clean Tauri bundling.

import importlib.util
import re
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files


OPTIONAL_PACKAGES = [
    # Optional enhancement deps (beat/structure)
    "madmom",
    "allin1",
    "natten",
    "demucs",
    "torchcodec",
]

hiddenimports = []
datas = []


def _discover_package_modules(package_name: str) -> list[str]:
    """Discover module names from installed package files without importing it."""
    spec = importlib.util.find_spec(package_name)
    if not spec or not spec.submodule_search_locations:
        return []

    package_root = Path(next(iter(spec.submodule_search_locations)))
    discovered: list[str] = []
    so_suffix = re.compile(r"\.cpython-[^.]+$")

    for file_path in package_root.rglob("*"):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix
        if suffix not in {".py", ".so"}:
            continue

        rel = file_path.relative_to(package_root.parent)
        parts = list(rel.parts)
        filename = parts[-1]

        if filename == "__init__.py":
            parts = parts[:-1]
        elif suffix == ".py":
            parts[-1] = filename[:-3]
        else:
            # Strip ABI tag from extension modules:
            # e.g. comb_filters.cpython-312-darwin.so -> comb_filters
            stem = filename[:-3]
            stem = so_suffix.sub("", stem)
            parts[-1] = stem

        if not parts:
            continue

        module_name = ".".join(parts)
        discovered.append(module_name)

    return sorted(set(discovered))


for package in OPTIONAL_PACKAGES:
    try:
        hiddenimports += _discover_package_modules(package)
    except Exception:
        pass
    try:
        datas += collect_data_files(package)
    except Exception:
        pass

a = Analysis(
    ['http_server.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=sorted(set(hiddenimports)),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='music-looper-sidecar',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
