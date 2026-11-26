"""Local, file-backed voice and image/video classifier endpoints for SentinelX."""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from dotenv import load_dotenv
from PIL import Image

app = FastAPI(title="SentinelX Model Inference", version="1.0.0")
load_dotenv(Path(__file__).with_name(".env"))
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_AUDIO_SECONDS = 30
MAX_VIDEO_SECONDS = 30
MAX_VIDEO_FRAMES = 12

_audio_pipe = None
_image_pipe = None
_model_lock = threading.Lock()


def _read_upload(upload: UploadFile, max_bytes: int = MAX_FILE_BYTES) -> bytes:
    data = upload.file.read(max_bytes + 1)
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail="The uploaded file is too large.")
    return data


def _audio_classifier():
    global _audio_pipe
    if _audio_pipe is None:
        with _model_lock:
            if _audio_pipe is None:
                from transformers import pipeline

                model_id = os.getenv("AUDIO_MODEL_ID", "Hemgg/Deepfake-audio-detection")
                _audio_pipe = pipeline("audio-classification", model=model_id, device=-1)
    return _audio_pipe


def _image_classifier():
    global _image_pipe
    if _image_pipe is None:
        with _model_lock:
            if _image_pipe is None:
                from transformers import pipeline

                model_id = os.getenv("IMAGE_MODEL_ID", "dima806/deepfake_vs_real_image_detection")
                _image_pipe = pipeline("image-classification", model=model_id, device=-1)
    return _image_pipe


def _fake_label(label: str) -> bool | None:
    normalized = label.lower().replace(" ", "_").replace("-", "_")
    fake_words = ("fake", "spoof", "synthetic", "generated", "deepfake", "artificial", "aivoice")
    real_words = ("real", "bona_fide", "bonafide", "genuine", "human", "authentic")
    if any(word in normalized for word in fake_words):
        return True
    if any(word in normalized for word in real_words):
        return False

    # Generic LABEL_0/LABEL_1 classes need an explicit model-specific mapping.
    raw_mapping = os.getenv("AUDIO_FAKE_LABELS", "")
    if raw_mapping and normalized in {item.strip().lower() for item in raw_mapping.split(",") if item.strip()}:
        return True
    return None


def _decode_audio(data: bytes) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="sentinelx-audio-") as tmp:
        source = Path(tmp) / "input.bin"
        source.write_bytes(data)
        _check_duration(source, MAX_AUDIO_SECONDS)
        command = [
            "ffmpeg", "-v", "error", "-nostdin", "-i", str(source), "-t", str(MAX_AUDIO_SECONDS),
            "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1",
        ]
        try:
            converted = subprocess.run(command, check=True, capture_output=True, timeout=45).stdout
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail="ffmpeg is required by the model service.") from exc
        except subprocess.CalledProcessError as exc:
            raise HTTPException(status_code=400, detail="Could not decode the audio file. Try WAV, MP3, OGG, or AAC.") from exc
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(status_code=408, detail="Audio decoding timed out.") from exc
    if not converted:
        raise HTTPException(status_code=400, detail="No audio samples could be decoded.")
    return np.frombuffer(converted, dtype="<f4").copy()


def _check_duration(path: Path, limit_seconds: int) -> None:
    command = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=15)
        duration = float(result.stdout.strip())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="ffprobe is required by the model service.") from exc
    except (subprocess.CalledProcessError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Could not read the media duration.") from exc
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=408, detail="Media duration check timed out.") from exc
    if duration > limit_seconds:
        raise HTTPException(status_code=413, detail=f"Media must be {limit_seconds} seconds or shorter.")


def _classify_audio(data: bytes) -> dict[str, Any]:
    samples = _decode_audio(data)
    if len(samples) / 16000 > MAX_AUDIO_SECONDS:
        raise HTTPException(status_code=413, detail="Audio must be 30 seconds or shorter.")
    predictions = _audio_classifier()({"array": samples, "sampling_rate": 16000}, top_k=None)
    predictions = sorted(predictions, key=lambda item: float(item["score"]), reverse=True)
    mapped = [(item, _fake_label(str(item["label"]))) for item in predictions]
    fake = next(((item, is_fake) for item, is_fake in mapped if is_fake is True), None)
    real = next(((item, is_fake) for item, is_fake in mapped if is_fake is False), None)
    if fake is None or real is None:
        labels = ", ".join(str(item["label"]) for item in predictions)
        raise HTTPException(
            status_code=503,
            detail=f"Audio model returned labels that need mapping ({labels}). Set AUDIO_FAKE_LABELS in ml-service/.env.",
        )
    fake_score = float(fake[0]["score"])
    confidence = max(fake_score, float(real[0]["score"]))
    return {
        "riskScore": round(fake_score * 100),
        "confidence": confidence,
        "reasonCodes": ["MODEL_PREDICTS_SYNTHETIC_AUDIO" if fake_score >= 0.5 else "MODEL_PREDICTS_BONA_FIDE_AUDIO"],
        "evidence": [{
            "id": "ev-audio-model",
            "label": "Voice authenticity model result",
        "value": f"Classifier score: {fake_score * 100:.1f}/100 (uncalibrated)",
            "severity": "danger" if fake_score >= 0.7 else "warning" if fake_score >= 0.5 else "safe",
            "explanation": "A local audio classification model compared the uploaded waveform with its synthetic and bona fide classes. This is an uncalibrated classifier score, not a calibrated probability or proof.",
        }],
        "modelName": os.getenv("AUDIO_MODEL_ID", "Hemgg/Deepfake-audio-detection"),
    }


def _face_crop(image: Image.Image) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        raise HTTPException(status_code=422, detail="No clear frontal face was detected. Use an image or video frame with a visible face.")
    x, y, width, height = max(faces, key=lambda rect: int(rect[2]) * int(rect[3]))
    padding = int(max(width, height) * 0.15)
    left, top = max(0, x - padding), max(0, y - padding)
    right, bottom = min(rgb.shape[1], x + width + padding), min(rgb.shape[0], y + height + padding)
    return Image.fromarray(rgb[top:bottom, left:right])


def _predict_image(image: Image.Image) -> float:
    face = _face_crop(image)
    predictions = _image_classifier()(face, top_k=None)
    labels = {str(item["label"]).lower(): float(item["score"]) for item in predictions}
    fake_score = next((score for label, score in labels.items() if "fake" in label or "deepfake" in label), None)
    real_score = next((score for label, score in labels.items() if "real" in label or "genuine" in label), None)
    if fake_score is None or real_score is None:
        raise HTTPException(status_code=503, detail=f"Image model labels need fake/real names; got: {', '.join(labels)}")
    return fake_score


def _decode_video_frames(data: bytes) -> list[Image.Image]:
    with tempfile.TemporaryDirectory(prefix="sentinelx-video-") as tmp:
        root = Path(tmp)
        source = root / "input-video"
        source.write_bytes(data)
        _check_duration(source, MAX_VIDEO_SECONDS)
        pattern = root / "frame-%03d.jpg"
        command = [
            "ffmpeg", "-v", "error", "-nostdin", "-i", str(source), "-t", str(MAX_VIDEO_SECONDS),
            "-vf", "fps=1,scale=512:-1:force_original_aspect_ratio=decrease", "-frames:v", str(MAX_VIDEO_FRAMES),
            str(pattern),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, timeout=60)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail="ffmpeg is required for video analysis.") from exc
        except subprocess.CalledProcessError as exc:
            raise HTTPException(status_code=400, detail="Could not decode the video file. Try MP4 or WEBM.") from exc
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(status_code=408, detail="Video frame extraction timed out.") from exc
        frames = [Image.open(path).convert("RGB") for path in sorted(root.glob("frame-*.jpg"))]
        if not frames:
            raise HTTPException(status_code=400, detail="The video did not contain decodable frames.")
        return frames


@app.get("/health")
def health():
    return {"status": "ok", "audioModelLoaded": _audio_pipe is not None, "imageModelLoaded": _image_pipe is not None}


@app.post("/predict")
def predict_voice(file: UploadFile = File(...)):
    data = _read_upload(file, 15 * 1024 * 1024)
    return _classify_audio(data)


@app.post("/analyze")
def analyze_media(file: UploadFile = File(...), mediaType: str = Form("image")):
    data = _read_upload(file)
    if mediaType == "video":
        frames = _decode_video_frames(data)
        scores = []
        for frame in frames:
            try:
                scores.append(_predict_image(frame))
            except HTTPException as error:
                if error.status_code != 422:
                    raise
        if not scores:
            raise HTTPException(status_code=422, detail="No clear frontal face was detected in the sampled video frames.")
        # The median limits the influence of a single unusual/compressed frame.
        score = float(np.median(scores))
        frame_count = len(scores)
        detail = f"{frame_count} sampled frames with a detected face; median classifier score {score * 100:.1f}/100 (uncalibrated)"
    else:
        try:
            image = Image.open(BytesIO(data))
            image.verify()
            source_image = Image.open(BytesIO(data))
            if source_image.width * source_image.height > 40_000_000:
                raise HTTPException(status_code=413, detail="Image dimensions are too large to analyze safely.")
            image = source_image.convert("RGB")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail="The uploaded image is invalid or unsupported.") from exc
        score = _predict_image(image)
        frame_count = 1
        detail = f"Face classifier score {score * 100:.1f}/100 (uncalibrated)"

    return {
        "riskScore": round(score * 100),
        "confidence": max(score, 1 - score),
        "reasonCodes": ["MODEL_PREDICTS_SYNTHETIC_MEDIA" if score >= 0.5 else "MODEL_PREDICTS_REAL_MEDIA"],
        "evidence": [{
            "id": "ev-media-model",
            "label": "Image authenticity classifier score",
            "value": detail,
            "severity": "danger" if score >= 0.7 else "warning" if score >= 0.5 else "safe",
            "explanation": "This is the raw output of an image classifier, not a calibrated probability of deepfake risk. The configured model targets AI-generated image detection and is not a temporal video deepfake detector. It can misclassify genuine media, especially outside its training data; treat this result as an unverified signal, not proof.",
        }],
        "modelName": os.getenv("IMAGE_MODEL_ID", "dima806/deepfake_vs_real_image_detection"),
        "framesAnalyzed": frame_count,
    }
