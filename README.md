# SentinelX

SentinelX is a React/Vite frontend with an Express API for checking suspicious text, URLs, QR payloads, audio metadata/samples, and image/video metadata. Text, URL, and QR checks use local rules. Audio and media use heuristic baselines unless optional external inference services are configured. Gemini is an optional explanation service; it does not run the detectors or determine the numeric risk score.

## Requirements

- Node.js 20.19+ or 22.12+ (Vite 8 requirement)
- npm
- Optional: a Google Gemini API key for generated explanations
- Optional: compatible voice/deepfake inference services for model-based analysis

## Run locally

1. In the project directory, install dependencies:

   ```bash
   npm install
   ```

2. (Optional) Set up environment variables. Copy `.env.example` to `.env` and edit it:

   ```bash
   cp .env.example .env
   ```

   Set `GEMINI_API_KEY` to a key from Google AI Studio if you want Gemini explanations. Keep it in `.env` on the server; do not put it in frontend code or a `VITE_` variable. Without it, SentinelX uses a local deterministic explanation. Gemini receives the report's evidence and a short submitted text/URL/QR snippet to write that explanation.

   For actual local voice and image/video model inference, follow [ml-service/README.md](ml-service/README.md) in a second terminal. It explains how to install the Python service, download model weights, and set the two service URLs in `.env`. No key is needed for its default public checkpoints. Without the model service, the app uses heuristic fallbacks; those are not trained voice/deepfake classifiers.

3. Start the development app (Vite frontend and Express API middleware share port 3000):

   ```bash
   npm run dev
   ```

   Open <http://localhost:3000>. Stop it with Ctrl+C. Keep the Python model service running in its own terminal if you enabled the voice/media models.

4. For a production-style local run:

   ```bash
   npm run build
   npm start
   ```

   This serves the built frontend and API from the same Express process. `PORT` can override port 3000.

## Checks

```bash
npm test       # detector, fusion and privacy behavior checks
npm run lint   # TypeScript checks across frontend, API and tests
npm run build  # production frontend bundle
```

## API overview

- `GET /api/health`
- `GET /api/system/status`
- `GET /api/detectors/status`
- `POST /api/analyze/text` with `{ "text": "...", "privacyMode": false }`
- `POST /api/analyze/url` with `{ "url": "https://...", "privacyMode": false }`
- `POST /api/analyze/qr` with `{ "payload": "...", "privacyMode": false }`
- `POST /api/analyze/audio`, `/api/analyze/image`, `/api/analyze/video`
- `POST /api/analyze/multimodal`
- `GET/DELETE /api/history`, `GET/DELETE /api/history/:id`
- `GET /api/analytics/summary`

Submitted URLs are analyzed statically; the service does not fetch or open them. History currently lives in server memory and clears when the server restarts. Privacy mode skips saving each report to that history. Browser preferences (including the privacy-mode switch) are stored in that browser's local storage.

## Important limits

The project provides assistive indicators, not a guarantee that content is safe or malicious. The optional local audio model scores synthetic speech; the image model estimates AI-generated imagery and may miss face swaps or other edits. Video analysis samples a limited number of frames and does not analyze temporal consistency. Read the model cards and service limitations before relying on scores. For production use, add authenticated access, persistent per-user storage, rate limiting, upload/content validation, and tested model services before exposing the API publicly.
