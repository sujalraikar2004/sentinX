# SentinelX local model service

This service runs local inference on uploaded media. It exposes `POST /predict` for audio and `POST /analyze` for images or videos, matching the Express adapters. The default audio model is `Hemgg/Deepfake-audio-detection` (labels `AIVoice` and `HumanVoice`). The image model is `dima806/deepfake_vs_real_image_detection`; the service detects and crops a visible face before classification. Video analysis samples up to 12 frames at one frame per second and scores frames where a face is detected.

The default weights are public; no API key is required. The first inference for each modality downloads its model from Hugging Face, so internet access is needed once. Both checkpoints are around 90–95 million parameters; CPU inference can be slow and needs several GB of RAM. Install FFmpeg for audio decoding and video frame extraction. The service accepts audio up to 15 MB, media up to 25 MB, and clips up to 30 seconds.

On Ubuntu/Debian, install FFmpeg with `sudo apt install ffmpeg`. On other systems, install FFmpeg from its official packages and ensure both `ffmpeg` and `ffprobe` are on `PATH`.

## Install and start

From the repository root, use Python 3.10 or newer:

```bash
cd ml-service
python -m venv .venv
```

Activate the environment, then install:

```bash
# Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app:app --host 127.0.0.1 --port 8100
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app:app --host 127.0.0.1 --port 8100
```

Then, in the repository root `.env`, set:

```env
VOICE_DETECTOR_API_URL=http://127.0.0.1:8100
DEEPFAKE_DETECTOR_API_URL=http://127.0.0.1:8100
```

Restart `npm run dev` or `npm start` after changing the root `.env`. Keep both processes running. Check the model service at <http://127.0.0.1:8100/health>; it loads checkpoints on the first inference. Swagger API docs are at <http://127.0.0.1:8100/docs>.

## What the scores mean

The voice score is the configured model's estimated synthetic-audio class probability. The image model is a face deepfake classifier; its model card warns of concept drift because its training dataset is several years old. Video scores average sampled frame estimates and do not analyze audio or temporal consistency. These research models can make mistakes; use scores as signals for review, not proof. Model cards: [audio model](https://huggingface.co/Hemgg/Deepfake-audio-detection) and [image model](https://huggingface.co/dima806/deepfake_vs_real_image_detection). Review checkpoint licenses and intended-use terms before commercial use.

If the audio model returns generic labels, set `AUDIO_FAKE_LABELS` in `ml-service/.env` to the comma-separated label names that mean synthetic. The configured default expects the model label `AIVoice`.
