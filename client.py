import os
import sys
import json
import time
import asyncio
import logging
from logging import Logger
from pathlib import Path
from getpass import getpass
from websockets.sync.client import connect
from websockets.exceptions import WebSocketException, ConnectionClosed
import bhaptics_python as bh
import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration

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

def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "on")

def env_override(cfg: dict) -> dict:
    # existing keys...
    cfg["app_id"]   = os.getenv("BHX_APP_ID",   cfg.get("app_id",   ""))
    cfg["api_key"]  = os.getenv("BHX_API_KEY",  cfg.get("api_key",  ""))
    cfg["app_name"] = os.getenv("BHX_APP_NAME", cfg.get("app_name", "bhx-bridge"))
    cfg["ws_url"]   = os.getenv("BHX_WS_URL",   cfg.get("ws_url",   ""))

    # debug from env (optional)
    if "BHX_DEBUG" in os.environ:
        cfg["debug"] = _as_bool(os.getenv("BHX_DEBUG"))

    # NEW: Glitchtip/Sentry DSN from env (prefer BHX_SENTRY_DSN; fallback BHX_GLITCHTIP_DSN)
    cfg["glitchtip_dsn"] = os.getenv("BHX_SENTRY_DSN",
                             os.getenv("BHX_GLITCHTIP_DSN",
                                      cfg.get("glitchtip_dsn", "")))
    cfg["inter_frame_gap_ms"] = int(os.getenv("BHX_INTERFRAME_GAP_MS", cfg.get("inter_frame_gap_ms", 0) or 0))
    cfg["respect_duration_sleep"] = _as_bool(os.getenv("BHX_RESPECT_DURATION_SLEEP", cfg.get("respect_duration_sleep", True)))
    return cfg

def sanitize_websocket_url(url: str, id: str) -> str:
    url = (url or "").strip().rstrip("/")
    if not (url.startswith("ws://") or url.startswith("wss://")):
        # Upgrade http(s) → wss by default
        url = url.replace("http://", "ws://").replace("https://", "wss://")
        if not (url.startswith("ws://") or url.startswith("wss://")):
            url = f"wss://{url}"
    suffix = f"/ws/listen/bHapticsClient_{id}" if id else "/ws/listen/bHapticsClient_"
    if not url.endswith(suffix):
        url += suffix
    return url

def load_config_interactive(logger: Logger) -> dict:
    cfg_path = find_config_path()
    cfg = load_yaml(cfg_path) if cfg_path.exists() else {}
    cfg = env_override(cfg)

    # Prompt so app_id is known before URL sanitization
    if not cfg.get("app_id"):
        cfg["app_id"] = input("Enter bHaptics App ID: ").strip()

    # WebSocket URL (sanitize with app_id suffix)
    if not cfg.get("ws_url"):
        ws = input("Enter WebSocket URL (e.g., wss://host:8765): ").strip()
        cfg["ws_url"] = sanitize_websocket_url(ws, cfg["app_id"])
    else:
        cfg["ws_url"] = sanitize_websocket_url(cfg["ws_url"], cfg["app_id"])

    if not cfg.get("api_key"):
        # hide input while typing
        cfg["api_key"] = getpass("Enter bHaptics API Key: ").strip()

    if not cfg.get("app_name"):
        app_name = input("Enter App Name [bhx-bridge]: ").strip() or "bhx-bridge"
        cfg["app_name"] = app_name

    # Debug flag (default false; don’t prompt—config/env driven)
    cfg["debug"] = _as_bool(cfg.get("debug", False))
    logger.info(f"Debug mode: {cfg['debug']}")

    # (Optional) Glitchtip DSN—read from file/env only; do not prompt on CLI
    if cfg.get("glitchtip_dsn"):
        logger.info("Glitchtip DSN found in config/env (errors will be reported).")
    else:
        logger.info("No Glitchtip DSN configured; error reporting is disabled.")

    # Save for next run
    save_to = appdata_dir() / DEFAULT_CONFIG_NAME
    dump_yaml(save_to, cfg)
    logger.info(f"Config saved to: {save_to}")
    return cfg

# -------------------------
# Logging
# -------------------------
def setup_logging() -> Logger:
    log_dir = appdata_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "latest.log"  # overwritten each run = "last execution"
    logger = logging.getLogger("bhx-bridge")
    logger.setLevel(logging.INFO)
    # Avoid duplicate handlers if main() runs twice (e.g., during reload)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Logging to: {log_file}")
    return logger

def setup_sentry(dsn: str, app_name: str, logger: Logger, environment: str = "production"):
    """
    Wire Python logging into Glitchtip/Sentry.
    Any logger.* at ERROR (and CRITICAL) becomes an event.
    INFO/DEBUG stay as breadcrumbs.
    """
    if not dsn:
        return

    sentry_logging = LoggingIntegration(
        level=logging.INFO,        # breadcrumbs at ≥ INFO
        event_level=logging.ERROR  # send events at ≥ ERROR (change to CRITICAL if you prefer)
    )
    try:
        sentry_sdk.init(
            dsn=dsn,
            integrations=[sentry_logging],
            environment=environment,
            release=f"{app_name}@1.0.0",  # adjust/version as you like
            traces_sample_rate=0.0,       # no performance traces
            send_default_pii=False,
        )
        logger.info("Glitchtip (Sentry) reporting enabled.")
    except Exception:
        logger.exception("Failed to initialize Glitchtip (Sentry). Continuing without error reporting.")


# -------------------------
# Frame conversion + playback
# -------------------------
class FrameConverter:
    """
    Flexible converter for server payloads. Accepts shapes like:

    {
      "word1": [
        { "duration": 200,
          "frame_nodes": [
            { "node_index": [0,5,6], "intensity": [64,255,128] }
          ]
        }
      ],
      "word2": { "duration": 150, "frame_nodes": { "node_index": 3, "intensity": 200 } },
      "pause": 300  # optional: treat as 300ms pause (all zeros)
    }

    Also ignores non-frame events such as:
    { "type": "ServerEventNameList", "message": ["required"] }
    """
    def __init__(self, sentence: dict, logger: Logger):
        self.sentence = sentence
        self._data = []
        self._logger = logger
        self._parse_sentence()

    def _as_list(self, x):
        # Turn scalars into single-item lists; keep lists/tuples as-is; None -> []
        if x is None:
            return []
        if isinstance(x, (list, tuple)):
            return list(x)
        return [x]

    def _coerce_int(self, x, default=0):
        try:
            return int(x)
        except Exception:
            return default
        
    def _append_step(self, values: list[int], duration_ms: int) -> None:
        self._data.append({"values": values, "duration": max(0, int(duration_ms))})
        
    def _merge_nodes_into_values(self, fns) -> list[int]:
        """Merge one or more frame_nodes dicts into a 40-long intensity list (0..100), using max() per motor."""
        padded = [0] * 40
        # Allow dict or list
        if isinstance(fns, dict):
            fns = [fns]
        elif not isinstance(fns, list):
            fns = []

        for fn in fns:
            if not isinstance(fn, dict):
                continue
            idxs = self._as_list(fn.get("node_index", []))
            raw  = self._as_list(fn.get("intensity", []))
            if len(raw) == 1 and len(idxs) > 1:
                raw = raw * len(idxs)

            for i, idx in enumerate(idxs):
                try:
                    idx = int(idx)
                except Exception:
                    continue
                if 0 <= idx < len(padded):
                    v = raw[i] if i < len(raw) else 0
                    try:
                        v = int(v)
                    except Exception:
                        v = 0
                    # Scale 0..255 -> 0..100 (ensure >0 maps to at least 1)
                    val = max(0, min(100, round(v * 100 / 255)))
                    if v > 0 and val == 0:
                        val = 1
                    padded[idx] = max(padded[idx], val)  # parallel merge via max
        return padded

    def _parse_sentence(self) -> None:
        # Ignore server “event” messages that aren’t frame payloads
        if isinstance(self.sentence, dict) and "type" in self.sentence and "message" in self.sentence:
            self._logger.info(f"Ignoring server event: type={self.sentence.get('type')} message={self.sentence.get('message')}")
            return

        if not isinstance(self.sentence, dict):
            # If the whole payload is a bare int: treat as pause frame.
            if isinstance(self.sentence, int):
                self._data.append({"values": [0]*40, "duration": max(0, self._coerce_int(self.sentence, 0))})
                return
            self._logger.warning(f"Unexpected payload type {type(self.sentence)}; ignoring.")
            return

        for key, val in self.sentence.items():
            # Accept int as "pause"
            if isinstance(val, int):
                self._data.append({"values": [0]*40, "duration": max(0, self._coerce_int(val, 0))})
                continue

            # A single frame dict or a list of frames
            if isinstance(val, dict):
                frames = [val]
            elif isinstance(val, list):
                frames = val
            else:
                self._logger.warning(f"Ignoring {key}: unexpected value type {type(val)}")
                continue

            self._parse_frames(frames, label=key)

    def _parse_frames(self, frames, label: str) -> None:
        # Normalize to list of dicts (allow a simple int to mean "pause")
        norm = []
        for f in frames:
            if isinstance(f, int):
                norm.append({"duration": int(f), "frame_nodes": []})
            elif isinstance(f, dict):
                norm.append(f)
            else:
                self._logger.warning(f"Ignoring non-dict frame in {label}: {type(f)}")

        if not norm:
            return

        # Detect whether we have an explicit sequence (any frame carries 'order')
        has_order = any(isinstance(f, dict) and ("order" in f) for f in norm)

        if not has_order:
            # === Parallel-only (legacy) ===
            # Previous behavior: treat each dict as its own frame, in list order.
            for idx_frame, frame in enumerate(norm):
                duration = self._coerce_int(frame.get("duration", 200), 200)
                fns      = frame.get("frame_nodes", [])
                values   = self._merge_nodes_into_values(fns)
                # If no nodes provided => pause of 'duration'
                self._append_step(values, duration)
            return

        # === Sequential mode (new) ===
        # Group by 'order'; within each order, merge all frames in parallel.
        steps: dict[int, list[dict]] = {}
        for frame in norm:
            try:
                step_idx = int(frame.get("order", 0))
            except Exception:
                step_idx = 0
            steps.setdefault(step_idx, []).append(frame)

        for step_idx in sorted(steps.keys()):
            group = steps[step_idx]
            # Merge all frame_nodes in this step (parallel within the step)
            merged_values = [0] * 40
            step_duration = 0

            for f in group:
                duration = self._coerce_int(f.get("duration", 200), 200)
                step_duration = max(step_duration, duration)

                fns = f.get("frame_nodes", [])
                vals = self._merge_nodes_into_values(fns)
                # Parallel combine by max per motor
                merged_values = [max(a, b) for a, b in zip(merged_values, vals)]

            # If the (first) item of the step is an explicit pause (empty frame_nodes), we still add a zero frame
            self._append_step(merged_values, step_duration)

        self._logger.info(f"{label}: built {len(steps)} sequential step(s) from ordered frames.")

    def _log_frames(self) -> None:
        total = len(self._data)
        self._logger.info(f"Converted {total} frames:")
        for n, item in enumerate(self._data, start=1):
            nz = {i: v for i, v in enumerate(item["values"]) if v}
            self._logger.info(f"  frame {n}: duration={item['duration']}ms motors={nz}")

    async def play(self, simulate: bool = False,
                   inter_frame_gap_ms: int = 0,
                   respect_duration_sleep: bool = True) -> None:
        """
        Plays frames sequentially. If respect_duration_sleep=True, we sleep for each frame's
        duration after sending it to ensure back-to-back timing. Optionally adds a small
        inter-frame gap to avoid edge overlaps in the Player.
        """
        if simulate:
            self._log_frames()
            return

        total = len(self._data)
        for idx, item in enumerate(self._data):
            duration_ms = max(0, int(item["duration"]))
            values = item["values"] or [0]*40
            is_pause = all(v == 0 for v in values)

            try:
                if is_pause:
                    # Pure pause: no need to call SDK; just wait.
                    if duration_ms > 0:
                        await asyncio.sleep(duration_ms / 1000.0)
                else:
                    # Send the frame to the Player
                    await bh.play_dot(0, duration_ms, values)
                    # Ensure the frame's duration is honored
                    if respect_duration_sleep and duration_ms > 0:
                        await asyncio.sleep(duration_ms / 1000.0)

                # Optional tiny pad between frames (except after the last)
                if inter_frame_gap_ms > 0 and idx < (total - 1):
                    await asyncio.sleep(inter_frame_gap_ms / 1000.0)

            except Exception:
                # Keep going even if one frame has an issue
                self._logger.exception("Error while playing a frame; continuing to next.")


# -------------------------
# Runtime
# -------------------------
def websocket_loop(ws_url: str, logger: Logger, debug: bool,
                   inter_frame_gap_ms: int, respect_duration_sleep: bool):
    """
    Sync client loop with:
    - Protocol pings (ping_interval/ping_timeout)
    - App-level __ping__/__pong__
    - Reconnect with backoff
    """
    backoff_s = 2
    max_backoff_s = 30

    while True:
        try:
            logger.info(f"Connecting to {ws_url} ...")
            with connect(
                ws_url,
                ping_interval=25,   # protocol-level keepalive
                ping_timeout=10,
                close_timeout=5,
            ) as websocket:
                logger.info("Connected. Waiting for frames...")
                if debug:
                    logger.info("DEBUG mode active: will log frames instead of sending to SDK.")
                backoff_s = 2  # reset backoff on successful connect

                for data in websocket:
                    if data is None:
                        continue

                    # websockets sync yields str; protect in case of bytes
                    if isinstance(data, (bytes, bytearray)):
                        try:
                            data = data.decode("utf-8")
                        except UnicodeDecodeError:
                            logger.warning("Received non-UTF8 binary; ignoring.")
                            continue

                    # App-level keepalive: respond to server's __ping__
                    if data == "__ping__":
                        try:
                            websocket.send("__pong__")
                            logger.debug("Replied to __ping__ with __pong__")
                        except Exception as e:
                            logger.warning(f"Failed to send __pong__: {e}")
                        continue

                    # Normal payload: expect JSON frames
                    try:
                        payload = json.loads(data)
                    except json.JSONDecodeError:
                        logger.warning("Received non-JSON message; ignoring.")
                        continue
                    
                    # NEW: ignore server status/error envelopes early
                    if isinstance(payload, dict) and "type" in payload and "message" in payload:
                        logger.info(f"Server event: type={payload.get('type')} message={payload.get('message')}")
                        continue
                    
                    try:
                        fc = FrameConverter(sentence=payload, logger=logger)
                        asyncio.run(fc.play(
                            simulate=debug,
                            inter_frame_gap_ms=inter_frame_gap_ms,
                            respect_duration_sleep=respect_duration_sleep
                        ))
                    except Exception:
                        logger.exception("Error during frame handling")
        except KeyboardInterrupt:
            logger.info("Interrupted by user. Exiting.")
            break
        except (ConnectionClosed, WebSocketException, ConnectionRefusedError, OSError) as e:
            logger.warning(f"WebSocket error: {e}. Reconnecting in {backoff_s}s ...")
            time.sleep(backoff_s)
            backoff_s = min(max_backoff_s, backoff_s * 2)
        except Exception:
            logger.exception(f"Unexpected error. Reconnecting in {backoff_s}s ...")
            time.sleep(backoff_s)
            backoff_s = min(max_backoff_s, backoff_s * 2)

async def init_bhaptics(app_id: str, api_key: str, app_name: str, logger: Logger, debug: bool):
    if debug:
        logger.info("DEBUG mode: skipping bHaptics initialization.")
        return
    # Requires bHaptics Player running locally with your app deployed
    ok = await bh.registry_and_initialize(app_id, api_key, app_name)
    logger.info(f"bHaptics init: {ok}")

async def shutdown_bhaptics(logger: Logger, debug: bool):
    if debug:
        logger.info("DEBUG mode: skipping bHaptics shutdown.")
        return
    try:
        await bh.stop_all()
    finally:
        await bh.close()
    logger.info("bHaptics closed.")

def main():
    logger = setup_logging()
    cfg = load_config_interactive(logger)
    debug = _as_bool(cfg.get("debug", False))

    # NEW: initialize Glitchtip/Sentry
    setup_sentry(cfg.get("glitchtip_dsn", ""), cfg.get("app_name", "bhx-bridge"), logger)

    try:
        asyncio.run(init_bhaptics(cfg["app_id"], cfg["api_key"], cfg["app_name"], logger, debug))
    except Exception:
        # This gets captured by Glitchtip because logger.exception logs at ERROR
        logger.exception("Failed to initialize bHaptics. Is the Player running and the app deployed?")
        raise

    try:
        websocket_loop(
            cfg["ws_url"], logger, debug,
            inter_frame_gap_ms=int(cfg.get("inter_frame_gap_ms", 0)),
            respect_duration_sleep=_as_bool(cfg.get("respect_duration_sleep", True)),
        )
    finally:
        asyncio.run(shutdown_bhaptics(logger, debug))


if __name__ == "__main__":
    main()
