# main.py
import os
import sys
import json
import asyncio
from pathlib import Path
from getpass import getpass
from websockets.sync.client import connect
import bhaptics_python as bh

# -------------------------
# Config loading/saving
# -------------------------
APP_FAMILY = "bhx-bridge"
DEFAULT_CONFIG_NAME = "config.yaml"

def is_windows() -> bool:
    return os.name == "nt"

def appdata_dir() -> Path:
    if is_windows():
        base = os.getenv("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / APP_FAMILY
    # Fallback for other OSes (useful during dev)
    return Path.home() / ".config" / APP_FAMILY

def bundled_base_dir() -> Path:
    # When running under PyInstaller --onefile, _MEIPASS points to temp extract dir
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).parent

def find_config_path() -> Path:
    # 1) %APPDATA%/bhx-bridge/config.yaml (preferred for user edits)
    p1 = appdata_dir() / DEFAULT_CONFIG_NAME
    if p1.exists():
        return p1
    # 2) next to the executable/script (zip distribution)
    p2 = bundled_base_dir() / DEFAULT_CONFIG_NAME
    return p2

def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        import yaml  # pip install pyyaml
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        # allow JSON as a fallback if user provided .json
        if path.suffix.lower() == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        raise

def dump_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml  # pip install pyyaml
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    except Exception:
        # fallback to JSON if yaml missing
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def env_override(cfg: dict) -> dict:
    # ENV overrides (don’t print these; especially api_key)
    cfg["app_id"]   = os.getenv("BHX_APP_ID",   cfg.get("app_id",   ""))
    cfg["api_key"]  = os.getenv("BHX_API_KEY",  cfg.get("api_key",  ""))
    cfg["app_name"] = os.getenv("BHX_APP_NAME", cfg.get("app_name", "bhx-bridge"))
    cfg["ws_url"]   = os.getenv("BHX_WS_URL",   cfg.get("ws_url",   ""))
    return cfg

def sanitize_websocket_url(url: str, id: str) -> str:
    url = url.strip().rstrip("/")
    if not (url.startswith("ws://") or url.startswith("wss://")):
        # Upgrade http(s) → wss by default
        url = url.replace("http://", "ws://").replace("https://", "wss://")
        if not (url.startswith("ws://") or url.startswith("wss://")):
            url = f"wss://{url}"
    if not url.endswith(f"/ws/listen/bHapticsClient_{id}"):
        url += f"/ws/listen/bHapticsClient_{id}"
    # Always append the suffix "/ws/listen/{id}"
    return url

def load_config_interactive() -> dict:
    cfg_path = find_config_path()
    cfg = load_yaml(cfg_path) if cfg_path.exists() else {}
    cfg = env_override(cfg)

    # Prompt for missing values
    if not cfg.get("ws_url"):
        ws = input("Enter WebSocket URL (e.g., wss://host:8765): ").strip()
        cfg["ws_url"] = sanitize_websocket_url(ws, cfg["app_id"])
    else:
        cfg["ws_url"] = sanitize_websocket_url(cfg["ws_url"], cfg["app_id"])

    if not cfg.get("app_id"):
        cfg["app_id"] = input("Enter bHaptics App ID: ").strip()

    if not cfg.get("api_key"):
        # hide input while typing
        cfg["api_key"] = getpass("Enter bHaptics API Key: ").strip()

    if not cfg.get("app_name"):
        app_name = input("Enter App Name [bhx-bridge]: ").strip() or "bhx-bridge"
        cfg["app_name"] = app_name

    # Save back to %APPDATA% for next run (never to the temp _MEIPASS dir)
    save_to = appdata_dir() / DEFAULT_CONFIG_NAME
    dump_yaml(save_to, cfg)
    print(f"Config saved to: {save_to}")
    return cfg

# -------------------------
# Frame conversion + playback
# -------------------------
class FrameConverter:
    """
    Expects a message shaped like:
    {
      "word1": [
        {
          "duration": 200,
          "frame_nodes": [
            {
              "node_index": [0, 5, 6],            # motor indices (0..31)
              "intensity":  [64, 255, 128]       # 0..255, will be scaled to 0..100
            },
            ...
          ]
        },
        ...
      ],
      ...
    }
    """
    def __init__(self, sentence: dict):
        self.sentence = sentence
        self._data = []
        self._parse_sentence()

    def _parse_sentence(self) -> None:
        for word in self.sentence:
            self._parse_frames(self.sentence[word])

    def _parse_frames(self, word) -> None:
        for frame in word:
            for frame_nodes in frame.get("frame_nodes", []):
                # Normalize intensity 0..255 → 0..100
                intensities = [max(0, min(100, int(v / 255 * 100))) for v in frame_nodes.get("intensity", [])]
                padded = [0] * 32  # TactSuit: 32 motors
                for i, idx in enumerate(frame_nodes.get("node_index", [])):
                    if 0 <= idx < 32 and i < len(intensities):
                        padded[idx] = intensities[i]
                self._data.append({"values": padded, "duration": int(frame.get("duration", 200))})

    async def play(self) -> None:
        # play sequentially
        for item in self._data:
            await bh.play_dot(position=0, duration=item["duration"], values=item["values"])

# -------------------------
# Runtime
# -------------------------
def websocket_loop(ws_url: str):
    # Using sync client for simplicity (single thread)
    print(f"Connecting to {ws_url} ...")
    with connect(ws_url) as websocket:
        print("Connected. Waiting for frames...")
        for data in websocket:
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                print("Received non-JSON message; ignoring.")
                continue
            fc = FrameConverter(sentence=payload)
            asyncio.run(fc.play())

async def init_bhaptics(app_id: str, api_key: str, app_name: str):
    # Requires bHaptics Player running locally with your app deployed
    ok = await bh.registry_and_initialize(app_id, api_key, app_name)
    print(f"bHaptics init: {ok}")

async def shutdown_bhaptics():
    try:
        await bh.stop_all()
    finally:
        await bh.close()

def main():
    cfg = load_config_interactive()
    try:
        asyncio.run(init_bhaptics(cfg["app_id"], cfg["api_key"], cfg["app_name"]))
    except Exception as e:
        print("Failed to initialize bHaptics. Is the Player running and the app deployed?")
        raise

    try:
        websocket_loop(cfg["ws_url"])
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        asyncio.run(shutdown_bhaptics())
        print("bHaptics closed.")

if __name__ == "__main__":
    main()
