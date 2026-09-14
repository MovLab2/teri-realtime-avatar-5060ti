# Teri — Real-Time AI Avatar on an RTX 5060 Ti

A fully real-time AI talking-head avatar that runs entirely on a single consumer GPU: the **NVIDIA RTX 5060 Ti (Blackwell)**. Voice and text input, OpenAI-powered conversation (or a no-AI parrot mode), local speech-to-text, and diffusion-generated video streamed live over WebRTC via LiveKit.

**Verified working as of September 2026.**

## Demo

[Watch the 25-second demo video](assets/demo/teri_demo_2026-09-13.mp4) — the avatar speaking its intro, generated live on the RTX 5060 Ti.

## What this is

This is not a tutorial project. It is a working, tested real-time pipeline:

- **Voice input** via Moonshine — runs locally, free, no API calls
- **Text input** via terminal
- **AI replies** via OpenAI Chat (gpt-4o-mini)
- **Voice synthesis** via OpenAI TTS (nova, HD)
- **Video generation** via SoulX-FlashHead Lite (diffusion, ~0.79 s per 1.32 s chunk — 1.6× real-time)
- **Streaming** via LiveKit Cloud (WebRTC)
- **Idle motion** driven by a background ambient music loop

### Two Modes

**AI Mode** — `webrtc_pr.py`
Voice or text -> ChatGPT (gpt-4o-mini) thinks and replies -> Nova HD voice -> avatar lip-syncs.
Best for real conversation. ~7-8 second response time.

**Parrot Mode** — `webrtc_pr_parrot.py`
Voice or text -> Nova HD voice repeats what you said -> avatar lip-syncs.
No ChatGPT. Lower latency (~6s). Cheaper (no chat tokens). Best for demos, voice-over, or when you want the avatar to be *your* voice.

Both modes support voice input (Moonshine) and text input (terminal). Both run on the same SoulX-FlashHead model.

## Hardware tested

- **GPU:** RTX 5060 Ti 16 GB (Blackwell, sm_120)
- **CPU:** 12-core / 24-thread
- **RAM:** 16 GB
- **OS:** Windows 11 25H2 + WSL2 + Ubuntu 22.04
- **Driver:** NVIDIA 610.47 or newer

**Minimum:** 12 GB+ VRAM, Blackwell preferred (Ampere/Ada/Hopper also work). ~40 GB free disk.

## What you need

1. NVIDIA GPU with recent driver
2. Windows 11 with WSL2, or native Linux
3. LiveKit Cloud account (free tier: cloud.livekit.io)
4. OpenAI API key with billing (platform.openai.com)
5. HuggingFace account (huggingface.co)
6. A microphone (for voice input)

## Installation

### 0. Clone this repository

    git clone https://github.com/MovLab2/teri-realtime-avatar-5060ti.git
    cd teri-realtime-avatar-5060ti

### 1. WSL2 + Ubuntu 22.04 (Windows only)

In Windows PowerShell as administrator:

    wsl --install -d Ubuntu-22.04

Reboot when prompted. On first launch, Ubuntu asks for a UNIX username and password.

Verify GPU passthrough inside WSL:

    nvidia-smi

You should see the RTX 5060 Ti. If not, update your Windows NVIDIA driver.

### 2. Miniconda + Python 3.10 environment

    cd ~
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p ~/miniconda3
    ~/miniconda3/bin/conda init bash
    source ~/.bashrc
    conda create -n flashhead python=3.10 -y
    conda activate flashhead

### 3. PyTorch 2.7.1 with CUDA 12.8

    pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

Verify Blackwell support:

    python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"

Expected: 2.7.1+cu128 True (12, 0). The (12, 0) confirms Blackwell.

### 4. FlashAttention 2.8.0.post2 prebuilt wheel

    pip install ninja
    pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl --no-build-isolation

Verify:

    python -c "import torch; from flash_attn import flash_attn_func; x = torch.randn(1, 8, 32, 64, device='cuda', dtype=torch.float16); y = flash_attn_func(x, x, x); print('OK:', y.shape)"

If the wheel 404s, fall back to source build (30-90 min):

    pip install flash-attn --no-build-isolation

### 5. Install the remaining dependencies

    pip install -r requirements.txt
    pip install -r requirements_pipecat.txt
    pip install -r requirements_extra.txt

### 6. Download the models from HuggingFace

Install the HuggingFace CLI and log in:

    pip install "huggingface_hub[cli]"
    huggingface-cli login

Download the SoulX-FlashHead 1.3B model (~15 GB):

    huggingface-cli download Soul-AILab/SoulX-FlashHead-1_3B --local-dir models/SoulX-FlashHead-1_3B

Download the wav2vec2 audio encoder (~400 MB):

    huggingface-cli download facebook/wav2vec2-base-960h --local-dir models/wav2vec2-base-960h

Verify the layout:

    ls models/SoulX-FlashHead-1_3B/
    # Should show: Model_Lite/  Model_Pro/  VAE_LTX/  VAE_Wan/  config.json  ...

    ls models/wav2vec2-base-960h/
    # Should show: config.json  model.safetensors  preprocessor_config.json  ...

### 7. Create the .env file

    cat > .env << 'ENVEOF'
    LIVEKIT_URL=wss://your-project.livekit.cloud
    LIVEKIT_API_KEY=your_api_key
    LIVEKIT_API_SECRET=your_api_secret
    OPENAI_API_KEY=sk-your-openai-key
    ENVEOF

Replace with your real LiveKit and OpenAI credentials.

Get LiveKit creds: cloud.livekit.io -> Project -> Settings -> Keys
Get OpenAI key: platform.openai.com/api-keys

### 8. Microphone setup (for voice input)

Inside WSL:

    sudo apt update
    sudo apt install -y alsa-utils libasound2-plugins
    cat > ~/.asoundrc << 'EOF'
    pcm.default pulse
    pcm.!default pulse
    ctl.default pulse
    ctl.!default pulse
    EOF

Test the mic:

    arecord -d 5 -f cd /tmp/test.wav
    aplay /tmp/test.wav

You should hear yourself. If not, check Windows Sound settings, Input, ensure the mic is enabled and at 80-100 percent volume.

## Running the bot

    conda activate flashhead
    cd teri-realtime-avatar-5060ti

    # AI mode (ChatGPT-powered conversation)
    python webrtc_pr.py

    # OR parrot mode (repeats what you say in Nova's voice)
    python webrtc_pr_parrot.py

Wait for these log lines:

    SoulX Model fully loaded and GPU is pre-warmed.
    OpenAI client initialized (chat: gpt-4o-mini, tts: tts-1-hd, voice: nova)
    Ambient loop loaded: 480000 samples (30.0s)
    Moonshine STT starting...
    Moonshine STT listening (1.5s debounce)...
    Room connected. Type OR speak.

The first run does a torch.compile pre-warm (~30-60 seconds). You will see a few large "model denoise per step" values before it settles to about 0.10s steady state.

### Joining the room

Generate a LiveKit token:

    set -a; source .env; set +a
    lk token create \
      --api-key "$LIVEKIT_API_KEY" \
      --api-secret "$LIVEKIT_API_SECRET" \
      --join --room soulx-flashhead-room \
      --identity you --valid-for 24h

Open https://meet.livekit.io and enter:

- LiveKit URL: your LIVEKIT_URL
- Token: the JWT you just generated
- Username: anything

You will see the avatar tile.

## Using the bot

### Text input

Type at the You: prompt and press Enter. The bot replies with voice and video.

### Voice input

1. Unmute your mic (headset button)
2. Wait ~2 seconds for the audio pipeline to engage
3. Speak your full question
4. Mute your mic
5. Wait ~2 seconds for the debounce to fire

The bot does NOT publish your voice to the room. Other participants hear the AI, not you.

### Sharing with guests

Generate a token per guest:

    lk token create \
      --api-key "$LIVEKIT_API_KEY" \
      --api-secret "$LIVEKIT_API_SECRET" \
      --join --room soulx-flashhead-room \
      --identity guest1 --name "Guest" --valid-for 24h

Send them the URL, the LiveKit URL, the token, and any username. They join and see the avatar. They do not hear you speak to it.

## Configuration

Model type (in webrtc_pr.py):

    model_type="lite"   # faster, fits 12 GB VRAM
    # model_type="pro"  # higher quality, NOT real-time on 16 GB

Performance (in flash_head/configs/infer_params.yaml):

    frame_num: 33                # DO NOT change without custom Triton kernels
    tgt_fps: 25
    sample_rate: 16000
    cached_audio_duration: 8

Avatar face (examples/girl.png):
- Replace with any portrait (512x512+, face centered)
- Update the path in webrtc_pr.py if using a different filename

Ambient idle music (examples/ambient.wav):
- Any 16 kHz mono WAV, ~30 seconds
- Louder = more idle motion, quieter = more subtle

Voice (in webrtc_pr.py):

    self.tts_voice = "nova"        # alloy, echo, fable, onyx, nova, shimmer
    self.tts_model = "tts-1-hd"    # tts-1-hd (higher quality) or tts-1 (cheaper)

Personality (in webrtc_pr.py):

    self.system_prompt = "You are Teri, a friendly, concise voice assistant. ..."

## Performance on RTX 5060 Ti

- Per-chunk generation: ~0.79 seconds
- Each chunk: 33 frames = 1.32 seconds of video
- Effective speed: 1.6x faster than real-time
- VRAM usage: ~6.6 GB with the Lite model
- Sustained runtime: indefinite (no idle timeout)

## Known issues

- Chat integration is broken. Adding LiveKit chat handler caused video quality degradation. The chat version exists as webrtc_pr_CHAT_95_PERCENT.py but is not recommended.
- Mic is on the bot machine. Guests do not hear you speak to the AI. A second mic on the viewer machine, or code changes to publish the bot mic, would fix this.
- 512x512 native resolution. Higher resolutions need a different model or upscaling.

## Fixed

- **Mic-queue phantom cancellations** (fixed 2026-09-13). Stale Moonshine entries used to block typed input, causing apparent hangs. Now mic entries older than 3 seconds are dropped automatically. Applied to both `webrtc_pr.py` and `webrtc_pr_parrot.py`.

## File reference

| File | Purpose |
|---|---|
| webrtc_pr.py | AI mode (ChatGPT conversation, voice + text input) |
| webrtc_pr_parrot.py | Parrot mode (repeats what you say in Nova's voice, no ChatGPT) |
| webrtc_pr_FINAL_VOICE_DEBOUNCED.py | Same as webrtc_pr.py, labeled copy |
| webrtc_pr_KNOWN_GOOD.py | Text-only version (no voice) |
| webrtc_pr_FILE_ONLY_WORKING.py | Original file-driven test bot |
| webrtc_pr_CHAT_*.py | Experimental chat (choppy video, not recommended) |
| flash_head/ | Model engine (patched for Blackwell) |
| generate_video.py | Standalone offline video generation |
| examples/ | Avatar image, ambient music, test audio |
| requirements*.txt | Dependencies (see install order above) |

## Credits

This project builds on:

- Original wrapper: github.com/yepicaiaaron/soulx-livekit-avatar
- Base model: github.com/Soul-AILab/SoulX-FlashHead
- Key file: webrtc_pr.py comes from the unmerged PR feat-webrtc-openai-s2s on the original repo

The bot file, voice input integration, Blackwell compatibility fixes, and this documentation were developed and tested locally on an RTX 5060 Ti.

## Roadmap

- ~~Fix mic-queue phantom cancellations~~ DONE (2026-09-13)
- Fix LiveKit chat integration (video quality issue)
- Publish the bot mic to the room
- Optional: ElevenLabs TTS integration
- FlashAttention 3 upgrade for Hopper/Blackwell
- Custom Triton kernels for sub-360ms latency

## License

See LICENSE.

## Support

Open an issue on this repository.

---

Built and tested September 2026. Runs on an RTX 5060 Ti. Voice input via Moonshine. AI via OpenAI. Avatar via SoulX-FlashHead.
