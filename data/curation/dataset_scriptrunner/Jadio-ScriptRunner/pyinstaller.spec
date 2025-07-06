# -*- mode: python ; coding: utf-8 -*-
import sys
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

a = Analysis([
    'src/__main__.py',
],
    pathex=['src'],
    binaries=[],
    datas=[
        ('config', 'config'),
        ('src/styles.py', 'src'),
        ('LICENSE', '.'),
        ('requirements.txt', '.'),
        ('src/dialogs', 'src/dialogs'),
        ('src/buttons.py', 'src'),
        ('src/terminalpanel.py', 'src'),
        ('src/shells/base_shell.py', 'src/shells'),
        ('src/shells/powershell.py', 'src/shells'),
    ],
    hiddenimports=collect_submodules('src.dialogs') + collect_submodules('src.shells'),
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Code.AIgent',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon='icon.ico',
    version='version.txt',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Code.AIgent')
