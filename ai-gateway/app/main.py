"""AI Verification Gateway.

API Router & Orchestrator: every endpoint runs the same pipeline

    AI Detection Engine -> Internet Evidence Engine -> Evidence Fusion & Decision
    -> Report Generation & Signing

and returns the verdict, the evidence, the report and a signed attestation that the
frontend submits to the MediaRegistry smart contract.
"""

import hashlib
import io
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

from . import config, detectors, evidence, fusion, image_fusion, report, signer, urlfetch
from .claims import extract_claims

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(_app):
    # Load the (large) image models before serving, so the first request doesn't stall.
    detectors.load_image_models()
    if config.RELEVANCE == "embedding":
        # The embedding model takes ~45 s to load the first time; do it in the background now.
        import threading

        from .support import _embedder

        threading.Thread(target=_embedder, daemon=True, name="embedder-warmup").start()
    yield


app = FastAPI(title="AI Verification Gateway", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=50_000)


class UrlRequest(BaseModel):
    url: str = Field(min_length=8, max_length=2048)


# ------------------------------------------------------------------ pipeline

def verify(hashed: bytes, media_type: str, detection: detectors.Result, ev: dict, details: dict | None = None) -> dict:
    """Fuse detection with evidence, write and hash the report, sign the attestation."""
    is_model = not detection.model_id.startswith("heuristic")
    decision = fusion.decide(detection.fake_probability, is_model, ev)
    content_hash = "0x" + hashlib.sha256(hashed).hexdigest()
    model_version = f"{detection.model_id}|{fusion.FUSION_VERSION}"

    rep = report.build(
        contentHash=content_hash,
        mediaType=media_type,
        verdict=decision.verdict,
        aiScore=decision.score_bps,
        fakeProbability=round(decision.fake_probability, 4),
        confidence=decision.confidence,
        modelVersion=model_version,
        detection={
            "modelId": detection.model_id,
            "fakeProbability": round(detection.fake_probability, 4),
            "signals": detection.signals,
            **detection.extras,
        },
        evidence=ev,
        fusion=decision.steps,
        details=details or {},
        gateway=signer.gateway_address(),
    )
    report_hash = report.save(rep)
    try:
        attestation = signer.sign_attestation(
            content_hash, media_type, decision.score_bps, decision.verdict_id, report_hash, model_version
        )
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc

    return {
        "contentHash": content_hash,
        "mediaType": media_type,
        "mediaTypeId": signer.MEDIA_TYPES[media_type],
        "aiScore": decision.score_bps,
        "fakeProbability": round(decision.fake_probability, 4),
        "confidence": decision.confidence,
        "verdict": decision.verdict,
        "verdictId": decision.verdict_id,
        "modelVersion": model_version,
        "signals": detection.signals,
        "claims": ev.get("claims", []),
        "evidence": ev,
        "fusion": decision.steps,
        "reportHash": report_hash,
        "report": rep,
        "attestation": attestation,
    }


def _read_upload(file: UploadFile, max_mb: int) -> bytes:
    data = file.file.read(max_mb * 1024 * 1024 + 1)
    if len(data) > max_mb * 1024 * 1024:
        raise HTTPException(413, f"File larger than {max_mb} MB")
    if not data:
        raise HTTPException(400, "Empty file")
    return data


def _text_pipeline(text: str, hashed: bytes, media_type: str, title: str = "", source_url: str = "", details=None):
    # AI detection (local models) and evidence gathering (network: GDELT, article downloads, Claude)
    # are independent, so run them at the same time.
    claims = extract_claims(text, title)
    with ThreadPoolExecutor(max_workers=2) as pool:
        ev_job = pool.submit(evidence.gather, claims, source_url=source_url, text=text)
        detection = detectors.analyze_text(text)
        ev = ev_job.result()
    return verify(hashed, media_type, detection, ev, details)


def _image_pipeline(data: bytes, media_type: str = "image", source_url: str = "", details=None, hashed=None):
    try:
        detection = detectors.analyze_image(data)
        img = Image.open(io.BytesIO(data))
    except Exception as exc:
        raise HTTPException(400, f"Could not read image: {exc}") from exc
    ev = evidence.gather([], image=img, source_url=source_url)
    return verify(hashed or data, media_type, detection, ev, details)


def _video_pipeline(data: bytes, media_type: str = "video", source_url: str = "", details=None, hashed=None):
    try:
        detection, frames = detectors.analyze_video(data)
    except ValueError as exc:
        raise HTTPException(400, f"Could not read video: {exc}") from exc
    # Reverse-search the middle frame: re-used footage is a common form of video misinformation.
    key_frame = frames[len(frames) // 2] if frames else None
    ev = evidence.gather([], image=key_frame, source_url=source_url)
    return verify(hashed or data, media_type, detection, ev, details)


def _audio_pipeline(data: bytes, media_type: str = "audio", source_url: str = "", details=None, hashed=None):
    try:
        detection = detectors.analyze_audio(data)
    except Exception as exc:
        raise HTTPException(400, f"Could not read audio (WAV, FLAC, OGG or MP3): {exc}") from exc
    ev = evidence.gather([], source_url=source_url)
    return verify(hashed or data, media_type, detection, ev, details)


# ------------------------------------------------------------------ routes

@app.get("/health")
def health():
    return {
        "status": "ok",
        "gateway": signer.gateway_address(),
        "mode": config.DETECTOR_MODE,
        "textModel": config.TEXT_MODEL,
        "imageModel": config.IMAGE_MODEL,
        "imageForensics": {
            "univfd": config.UNIVFD_MODEL,
            "trufor": config.TRUFOR_MODEL,
            "fusion": "calibrated" if image_fusion.load() else "uncalibrated",
        },
        "audioModel": config.AUDIO_MODEL,
        "evidence": evidence.engines_status(),
    }


@app.post("/analyze/text")
def analyze_text(req: TextRequest):
    # The hash covers exactly the text sent; the frontend trims it before sending and hashing.
    return _text_pipeline(req.text, req.text.encode("utf-8"), "text")


@app.post("/analyze/image")
def analyze_image(file: UploadFile = File(...)):
    return _image_pipeline(_read_upload(file, config.MAX_UPLOAD_MB))


@app.post("/analyze/video")
def analyze_video(file: UploadFile = File(...)):
    return _video_pipeline(_read_upload(file, config.MAX_VIDEO_MB))


@app.post("/analyze/audio")
def analyze_audio(file: UploadFile = File(...)):
    return _audio_pipeline(_read_upload(file, config.MAX_UPLOAD_MB))


@app.post("/verify-url")
def verify_url(req: UrlRequest):
    """Fetch a news article (or a direct image / video / audio link) and verify it.
    The content hash is the SHA-256 of the URL exactly as submitted."""
    url = req.url.strip()
    try:
        page = urlfetch.fetch(url)
    except urlfetch.UnsafeURL as exc:
        raise HTTPException(400, f"URL not allowed: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(413, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, f"Could not fetch URL: {exc}") from exc

    hashed = url.encode("utf-8")
    details = {"url": url, "finalUrl": page.url, "contentType": page.content_type,
               "contentSha256": "0x" + hashlib.sha256(page.body).hexdigest()}
    # Direct links to media are analysed as media, but registered under the URL's hash.
    media = {"image": _image_pipeline, "video": _video_pipeline, "audio": _audio_pipeline}
    if pipeline := media.get(page.content_type.split("/")[0]):
        return pipeline(page.body, "url", page.url, details, hashed=hashed)

    text = "\n".join(t for t in (page.title, page.text) if t).strip()
    if len(text.split()) < 5:
        raise HTTPException(422, "Could not find article text on that page")
    details |= {"title": page.title, "published": page.published, "words": len(text.split())}
    return _text_pipeline(text, hashed, "url", page.title, page.url, details)


@app.get("/reports/{report_hash}")
def get_report(report_hash: str):
    data = report.load(report_hash)
    if data is None:
        raise HTTPException(404, "Report not found on this gateway")
    return Response(data, media_type="application/json")
