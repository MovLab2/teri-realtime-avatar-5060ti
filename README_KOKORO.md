# Optional: Sapphire + Kokoro Integration

This document describes an **optional add-on** to the base project. The base avatar still works exactly as documented in the main README — nothing here is required.

## What this adds

- **`webrtc_pr_parrot_kokoro.py`** — same as `webrtc_pr_parrot.py`, but instead of calling OpenAI's Nova TTS, it calls a local Kokoro TTS server (e.g. the one bundled with [Sapphire](https://github.com/ddxfish/sapphire)).
- **`viewer.py`** — a standalone Tkinter window that connects to your LiveKit room and displays the avatar with audio. Useful if you don't want to use a browser.

## Why you'd want this

- **Free voice** — Kokoro runs locally. No OpenAI TTS cost.
- **Sapphire integration** — if you run Sapphire, its replies can be routed to the avatar automatically.
- **Standalone viewer** — no Chrome needed.

## Why you might *not* want this

- **Requires Sapphire or another Kokoro TTS server** running on the same machine or network.
- **Hardcoded TTS URL** — the Kokoro variant calls `http://172.17.128.1:5012/tts` by default (WSL → Windows host). You'll need to change this if your setup differs.
- **Adds a webhook listener** on port `5013` for external text input.

## Prerequisites

1. The base project working (see main README).
2. A running Kokoro TTS HTTP server. Sapphire ships one — it starts on `http://localhost:5012` when Sapphire is running.
3. (Optional) Sapphire itself, if you want automatic routing of replies.

## Setup

### 1. Change the Kokoro TTS URL if needed

Open webrtc_pr_parrot_kokoro.py and find the requests.post call to http://172.17.128.1:5012/tts. Replace 172.17.128.1 with whatever address your Kokoro server listens on.

If WSL, find the Windows host IP with:
    ip route show | grep -i default | awk '{ print $3}'

If the viewer and bot run on the same machine, localhost:5012 also works.

### 2. Run the Kokoro variant

    python webrtc_pr_parrot_kokoro.py

It behaves exactly like webrtc_pr_parrot.py except voice comes from Kokoro instead of Nova.

### 3. (Optional) Connect Sapphire via plugin

If you want Sapphire's replies to drive the avatar automatically, install the teri-bridge plugin:

- Repo: https://github.com/MovLab2/teri-bridge
- Install via Sapphire's Plugins -> Install Plugin with the GitHub URL.

The plugin sends Sapphire's replies to http://<wsl-ip>:5013/say — which the Kokoro variant listens for.

### 4. (Optional) Use the standalone viewer

viewer.py connects to your LiveKit room and shows the avatar in a Tkinter window with audio.

Requirements:
    pip install livekit pillow sounddevice

Setup: open viewer.py and set:

    URL = "wss://your-project.livekit.cloud"
    TOKEN = "your-jwt-token"
    ROOM = "soulx-flashhead-room"

Generate a token via your existing setup (or lk token create).

Run:
    python viewer.py

## Audio caveats

- LiveKit's Python SDK delivers audio frames in a format that sounddevice sometimes struggles with. If audio is choppy, check frame_size_ms in viewer.py and try different values.
- For clean audio without any Python-side work, use LiveKit Meet in a browser instead — the WebRTC stack handles audio natively.

## Token expiry

LiveKit tokens expire (usually 24h). Regenerate before each session, or automate with a small .bat that calls lk token create.

## What's untouched

- webrtc_pr.py (AI mode) — unchanged
- webrtc_pr_parrot.py (parrot mode with Nova) — unchanged
- All base dependencies, configs, and models — unchanged

This add-on is purely additive. If you don't use it, nothing about the base project is affected.
