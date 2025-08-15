# bhx_bridge.spec
# Run once: pyinstaller --name bhx_bridge --onefile --console client.py
# Then edit hiddenimports and build with: pyinstaller bhx_bridge.spec
block_cipher = None

from PyInstaller.utils.hooks import collect_submodules

hidden = ['sentry_sdk'] + collect_submodules('sentry_sdk')

a = Analysis(
    ['client.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='bhx_bridge',
    console=True,
)
coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    strip=False, upx=True, upx_exclude=[],
    name='bhx_bridge'
)
