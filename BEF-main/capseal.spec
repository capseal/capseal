# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for CapSeal single-binary distribution.

Build:
    pip install pyinstaller
    pyinstaller capseal.spec

The resulting binary will be in dist/capseal
"""

import sys
from pathlib import Path

# Get the project root
project_root = Path(SPECPATH)

a = Analysis(
    [str(project_root / 'bef_zk' / 'capsule' / 'cli' / '__main__.py')],
    pathex=[str(project_root)],
    binaries=[],
    datas=[
        # Include policy templates
        (str(project_root / 'policies' / 'demo_policy_v1.json'), 'policies'),
        # Include any other data files needed at runtime
    ],
    hiddenimports=[
        'bef_zk',
        'bef_zk.capsule',
        'bef_zk.capsule.cli',
        'bef_zk.capsule.cli.init_cmd',
        'bef_zk.capsule.cli.demo_cmd',
        'bef_zk.capsule.cli.explain_cmd',
        'bef_zk.capsule.cli.verify',
        'bef_zk.capsule.cli.run',
        'bef_zk.capsule.cli.replay',
        'bef_zk.capsule.cli.audit',
        'bef_zk.capsule.cli.fetch',
        'bef_zk.capsule.cli.doctor',
        'bef_zk.capsule.cli.inspect_cmd',
        'bef_zk.capsule.cli.emit',
        'bef_zk.capsule.cli.shell',
        'bef_zk.capsule.cli.sandbox_cmd',
        'bef_zk.capsule.contracts',
        'bef_zk.capsule.header',
        'bef_zk.capsule.payload',
        'bef_zk.sandbox',
        'bef_zk.sandbox.detect',
        'bef_zk.sandbox.runner',
        'bef_zk.codec',
        'click',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavy optional dependencies for minimal binary
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
        'torch',
        'tensorflow',
        # Exclude test frameworks
        'pytest',
        'unittest',
        # Exclude dev tools
        'mypy',
        'ruff',
        'black',
    ],
    noarchive=False,
    optimize=2,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='capseal',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path here if desired
)

# For directory distribution (larger but faster startup):
# coll = COLLECT(
#     exe,
#     a.binaries,
#     a.datas,
#     strip=True,
#     upx=True,
#     upx_exclude=[],
#     name='capseal',
# )
