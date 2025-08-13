# bHaptics WebSocket Bridge

This tool listens on a WebSocket for incoming haptic frame data and plays those frames on a bHaptics TactSuit (or compatible device) using the official `bhaptics-python` SDK.

It is intended for situations where haptic data is generated externally and streamed into a Windows machine running the bHaptics Player.

---

## 📋 Requirements

- **Windows 10/11** (x64)
- **bHaptics Player** installed and running
- A **paired** bHaptics TactSuit / compatible device via Bluetooth
- **Developer account** on the [bHaptics Developer Portal](https://developer.bhaptics.com)
- A deployed **Haptic App** with at least one event or pattern, so you can obtain:
  - **App ID**
  - **API Key**
  - **App Name** (arbitrary, but must match what you configured)
- The provided `.exe` build of this tool (or build it yourself with PyInstaller)

---

## 🛠 Setup Instructions

### 1. Install and set up bHaptics Player
1. Download bHaptics Player from the [official site](https://www.bhaptics.com/support/download).
2. Install and run it on your Windows PC.
3. Pair your bHaptics device in **Windows Bluetooth Settings**.
4. Verify the Player detects your device and shows it as **connected**.

---

### 2. Create and deploy an app in the Developer Portal
1. Log in to [bHaptics Developer Portal](https://developer.bhaptics.com).
2. Create a **new app** for this bridge.
3. Add and configure **events** and haptic patterns you plan to trigger.
4. Deploy the app.
5. Note down your **App ID** and **API Key** — you’ll enter these in the bridge.

---

### 3. Configuration file location
The bridge saves its configuration to:

`%APPDATA%\bhx-bridge\config.yaml`

Example contents:
```yaml
app_id: "YOUR_APP_ID"
api_key: "YOUR_API_KEY"
app_name: "bhx-bridge"
ws_url: "wss://example.com/ws/listen/bHapticsClient_YOUR_APP_ID"
```

## 🚀 Running the bridge

### First run

- Ensure bHaptics Player is running and the device is connected.
- Double-click bhx_bridge.exe (or run it from cmd/PowerShell).
- If %APPDATA%\bhx-bridge\config.yaml does not exist, you will be prompted for:
    - WebSocket URL (will be sanitized to /ws/listen/bHapticsClient_{AppID})
    - App ID
    - API Key
    - App Name (optional, defaults to bhx-bridge)

These values are saved for future runs.

### Subsequent runs
Just start the exe; it will:

- Initialize the SDK with your credentials
- Connect to the WebSocket URL
- Wait for incoming JSON haptic frames
- Play them sequentially on the device

## 🔄 JSON Frame Format
The bridge expects WebSocket messages shaped like:

```json
{
  "word1": [
    {
      "duration": 200,
      "frame_nodes": [
        {
          "node_index": [0, 5, 6],
          "intensity": [64, 255, 128]
        }
      ]
    }
  ]
}
```

- node_index: Motor indices (0–31 for TactSuit)
- intensity: Values 0–255 (scaled internally to 0–100)
- duration: Playback duration in milliseconds

## ⚠️ Notes

- The bridge requires bHaptics Player to be running locally; there is no direct BLE control from Python.
- The App ID and API Key must match a deployed app in the Developer Portal with your desired events.
- If the WebSocket connection drops, the program will attempt to reconnect only when restarted.
- If you build the exe yourself with PyInstaller, remember to use:


```powershell
pyinstaller --onefile --console --name bhx_bridge main.py
```