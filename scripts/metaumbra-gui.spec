# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files


SPEC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SPEC_DIR.parent
ENTRY_SCRIPT = SPEC_DIR / "pyinstaller_gui_entry.py"
SRC_DIR = ROOT_DIR / "src"
ICON_PATH = ROOT_DIR / "src" / "metaumbra" / "assets" / "metaumbra_icon.png"

datas = collect_data_files("metaumbra", includes=["assets/*.png"])


a = Analysis(
    [str(ENTRY_SCRIPT)],
    pathex=[str(SRC_DIR)],
    binaries=[],
    datas=datas,
    hiddenimports=[],
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
