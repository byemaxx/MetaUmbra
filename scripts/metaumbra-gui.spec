# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

SPEC_DIR = Path(SPECPATH).resolve()
ROOT_DIR = SPEC_DIR.parent
ENTRY_SCRIPT = SPEC_DIR / "pyinstaller_gui_entry.py"
SRC_DIR = ROOT_DIR / "src"
ICON_PATH = ROOT_DIR / "src" / "metaumbra" / "assets" / "metaumbra_icon.png"
ASSETS_DIR = ROOT_DIR / "src" / "metaumbra" / "assets"

datas = [(str(ASSETS_DIR / "*.png"), "metaumbra/assets")]

# `pyarrow` is imported at runtime (inside workflow functions), so PyInstaller's
# static analysis may miss parts of it unless we declare them explicitly.
def _pyarrow_submodule_filter(name: str) -> bool:
    return not (
        name.startswith("pyarrow.tests")
        or name.startswith("pyarrow.benchmark")
        or name.startswith("pyarrow.conftest")
    )


def _package_submodule_filter(name: str) -> bool:
    return not (
        ".tests" in name
        or ".testing" in name
        or ".bench" in name
        or ".benchmarks" in name
        or name.endswith(".tests")
        or name.endswith(".testing")
    )


def _collect_conda_runtime_dlls() -> list[tuple[str, str]]:
    conda_bin = Path(sys.prefix) / "Library" / "bin"
    if not conda_bin.is_dir():
        return []

    dll_names = {
        "ffi.dll",
        "ffi-7.dll",
        "ffi-8.dll",
        "libbz2.dll",
        "libcrypto-3-x64.dll",
        "libexpat.dll",
        "liblzma.dll",
        "libmpdec-4.dll",
        "libssl-3-x64.dll",
        "sqlite3.dll",
        "zstd.dll",
    }
    return [
        (str(conda_bin / name), ".")
        for name in sorted(dll_names)
        if (conda_bin / name).is_file()
    ]


hiddenimports = [
    "metaumbra.digest",
    "metaumbra.scoring",
]
hiddenimports += collect_submodules(
    "pandas",
    filter=_package_submodule_filter,
    on_error="ignore",
)
hiddenimports += collect_submodules(
    "rpg",
    filter=_package_submodule_filter,
    on_error="ignore",
)
hiddenimports += collect_submodules(
    "pyarrow",
    filter=_pyarrow_submodule_filter,
    on_error="ignore",
)

# Bundle Arrow/Parquet native libraries required on Windows.
binaries = collect_dynamic_libs("pyarrow")
binaries += _collect_conda_runtime_dlls()

# Include non-code package data that some pyarrow builds rely on.
datas += collect_data_files("pyarrow", include_py_files=False)
excludes = [
    "IPython",
    "ipykernel",
    "jupyter_client",
    "jupyter_core",
    "matplotlib",
    "tkinter",
    "_tkinter",
]


a = Analysis(
    [str(ENTRY_SCRIPT)],
    pathex=[str(SRC_DIR)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
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
    name="metaumbra-gui",
    icon=str(ICON_PATH),
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
