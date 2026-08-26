# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the Avatar V2 Linux desktop app (onedir).
import os

from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules("webview") + ["bottle", "proxy_tools"]

a = Analysis(
    [os.path.join(SPECPATH, "desktop_entry.py")],
    pathex=[os.path.dirname(SPECPATH)],
    binaries=[],
    datas=[
        (os.path.join(os.path.dirname(SPECPATH), "avatar_v2", "ui", "index.html"), "avatar_v2/ui"),
        (os.path.join(os.path.dirname(SPECPATH), "workflows", "animatediff_sd15_api.json"), "workflows"),
        (os.path.join(os.path.dirname(SPECPATH), "workflows", "animatediff_sdxl_api.json"), "workflows"),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "pkg_resources", "setuptools"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AvatarV2",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="AvatarV2",
)
