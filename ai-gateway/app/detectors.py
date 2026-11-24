"""Fake-news and deepfake detectors.

Each detector returns a fake probability (0..1), the model id used, and
human-readable signals explaining the score. When transformers/torch are not
installed (or DETECTOR_MODE=heuristic) a transparent heuristic fallback is used
so the system still runs end to end.
"""

import io
import logging
import re
from dataclasses import dataclass, field
from functools import partial

from PIL import ExifTags, Image, ImageChops

from . import config, image_fusion
from .textclean import clean_text

log = logging.getLogger("detectors")

FAKE_WORDS = ("fake", "false", "unreliable", "deepfake", "manipulated", "artificial", "ai", "spoof", "synthetic")
REAL_WORDS = ("real", "true", "reliable", "authentic", "human", "hum", "realism", "bonafide", "bona-fide", "genuine")


@dataclass
class Result:
    fake_probability: float
    model_id: str
    signals: list[str] = field(default_factory=list)
    extras: dict = field(default_factory=dict)  # added to the report's detection section

    @property
    def score_bps(self) -> int:
        return max(0, min(10000, round(self.fake_probability * 10000)))


# ----------------------------------------------------------------- ML models
# A predictor maps an input to (fake probability or None, explanation signals).

_predictors: dict = {}


def _predictor(src: str, loader):
    """Load a model once, or return None (heuristics only) if disabled or unavailable."""
    if config.DETECTOR_MODE == "heuristic":
        return None
    if src not in _predictors:
        try:
            log.info("Loading model %s (a Hub model is downloaded on first run)...", src)
            _predictors[src] = loader(config.resolve_model(src))
        except Exception as exc:  # missing deps, no internet, bad model id...
            log.warning("Could not load %s (%s); using heuristic fallback", src, exc)
            _predictors[src] = None
    return _predictors[src]


def _hf_predictor(task: str, path: str):
    """Hugging Face model: a Hub id or a folder saved by text_finetune.py / image_finetune.py."""
    from transformers import pipeline

    clf = pipeline(task, model=path)
    name = config.model_name(path)
    kwargs = {"truncation": True} if task == "text-classification" else {}

    def predict(x):
        preds = clf(x, top_k=None, **kwargs)
        summary = ", ".join(f"{p['label']}={p['score']:.2f}" for p in preds)
        prob = _fake_prob_from_labels(preds)
        return prob, [f"{name}: {summary}" if prob is not None else f"Unrecognised labels from {name}: {summary}"]

    return predict


def _tfidf_predictor(path: str):
    """Our TF-IDF + logistic regression model (training/text_baseline.py)."""
    import joblib

    pipe = joblib.load(path)["pipeline"]
    tfidf, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    names = tfidf.get_feature_names_out()

    def predict(text):
        vec = tfidf.transform([text])
        prob = float(clf.predict_proba(vec)[0, 1])
        # Each term's contribution to the decision; positive pushes towards fake.
        row = vec.multiply(clf.coef_).tocsr()
        ranked = sorted(zip(row.data, row.indices))
        signals = [f"Own TF-IDF model: {prob:.0%} fake"]
        if fake := [names[i] for v, i in reversed(ranked) if v > 0][:6]:
            signals.append("Words pointing to fake: " + ", ".join(fake))
        if real := [names[i] for v, i in ranked if v < 0][:6]:
            signals.append("Words pointing to real: " + ", ".join(real))
        return prob, signals

    return predict


def _cnn_predictor(path: str):
    """Our TruthNet CNN (training/image_cnn.py)."""
    import torch

    from .cnn import TruthNet, to_tensor

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = TruthNet(len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    fake_idx = ckpt["classes"].index("fake")

    def predict(img):
        with torch.no_grad():
            prob = float(model(to_tensor(img).unsqueeze(0)).softmax(1)[0, fake_idx])
        return prob, [f"Own TruthNet CNN: {prob:.0%} fake"]

    return predict


def _smaug_predictor(path: str):
    """General AI-generated image detector (see app/smaug.py)."""
    from .smaug import SmaugDetector

    detector = SmaugDetector(path, crops=config.SMAUG_CROPS)

    def predict(img):
        prob = detector(img)
        return prob, [f"Smaug AI-image detector: {prob:.0%} likely AI-generated"]

    return predict


def _univfd_predictor(path: str):
    """AI-generated image detector on CLIP features (see app/univfd.py)."""
    from .univfd import UnivFD

    detector = UnivFD(path, config.UNIVFD_CLIP)

    def predict(img):
        prob = detector(img)
        return prob, [f"UnivFD AI-image detector: {prob:.0%} likely AI-generated"], {}

    return predict


def _trufor_predictor(path: str):
    """Local manipulation detector with a tampering map (see app/trufor.py)."""
    from .trufor import TruFor, heatmap_data_uri

    detector = TruFor(path, max_side=config.TRUFOR_MAX_SIDE)

    def predict(img):
        out = detector(img)
        signals = [f"TruFor tampering score: {out.score:.0%} (map confidence {out.confidence:.0%})"]
        if out.scaled:
            signals.append(f"Image reduced to {config.TRUFOR_MAX_SIDE} px for TruFor, which can weaken its traces")
        return out.score, signals, {"heatmap": heatmap_data_uri(out.tamper_map)}

    return predict


def _image_loader(src: str):
    if src.endswith(".pt"):
        return _cnn_predictor
    if "smaug" in src.lower():
        return _smaug_predictor
    return partial(_hf_predictor, "image-classification")


def _text_models():
    src = config.TEXT_MODEL
    loader = _tfidf_predictor if src.endswith(".joblib") else partial(_hf_predictor, "text-classification")
    return [(src, _predictor(src, loader))]


def _image_models():
    """IMAGE_MODEL may list several models; each can cover generators the others miss."""
    return [(src, _predictor(src, _image_loader(src))) for src in config.IMAGE_MODELS]


FORENSIC_MODELS = {
    "univfd": (lambda: config.UNIVFD_MODEL, _univfd_predictor, "AI-generated probability"),
    "trufor": (lambda: config.TRUFOR_MODEL, _trufor_predictor, "Tampering score"),
}


def forensic_model(key: str):
    """Load UnivFD or TruFor once. These score uploaded images only, not video frames."""
    source, loader, _ = FORENSIC_MODELS[key]
    return _predictor(source(), loader) if source() else None


def load_image_models():
    """Load every image model, so the first upload does not stall."""
    _image_models()
    for key in FORENSIC_MODELS:
        forensic_model(key)


def _fake_prob_from_labels(preds: list[dict]) -> float | None:
    """Turn classifier output [{label, score}, ...] into P(fake), whatever the label names."""
    fake = real = 0.0
    for p in preds:
        label = p["label"].lower()
        if any(w == label or w in label.split() or label.startswith(w) for w in FAKE_WORDS):
            fake += p["score"]
        elif any(w == label or w in label.split() or label.startswith(w) for w in REAL_WORDS):
            real += p["score"]
    if fake + real == 0:
        return None
    if real == 0 or fake == 0:  # only one side's label recognised; scores are softmax
        return fake if real == 0 else 1 - real
    return fake / (fake + real)


# ---------------------------------------------------------------------- text

CLICKBAIT = (
    "shocking", "you won't believe", "miracle", "secret", "exposed", "they don't want you",
    "100%", "breaking", "urgent", "share before", "banned", "cure", "hoax", "conspiracy",
    "mainstream media", "wake up", "doctors hate",
)


def _text_heuristic(text: str) -> Result:
    signals, score = [], 0.15
    lower = text.lower()
    hits = [w for w in CLICKBAIT if w in lower]
    if hits:
        score += min(0.4, 0.1 * len(hits))
        signals.append(f"Sensational / clickbait phrases: {', '.join(hits[:5])}")
    letters = [c for c in text if c.isalpha()]
    caps = sum(c.isupper() for c in letters) / max(1, len(letters))
    if caps > 0.3:
        score += 0.15
        signals.append(f"Excessive capital letters ({caps:.0%})")
    if text.count("!") >= 3:
        score += 0.1
        signals.append(f"Many exclamation marks ({text.count('!')})")
    if not re.search(r"https?://|according to|reported|said|source", lower):
        score += 0.1
        signals.append("No sources, quotes or attribution found")
    else:
        signals.append("Contains attribution or sources")
    if len(text.split()) < 25:
        signals.append("Very short text - low confidence")
    return Result(min(score, 0.95), "heuristic-text-v1", signals)


def _combine(heuristic: Result, models, x, kind: str) -> Result:
    """Run every loaded model; the most suspicious one decides, since an item is fake
    if any detector recognises its generator. Heuristics nudge the score."""
    probs, names, signals = [], [], []
    for src, predict in models:
        if predict is None:
            continue
        prob, model_signals = predict(x)
        signals += model_signals
        if prob is not None:
            probs.append(prob)
            names.append(config.model_name(src))
    if not probs:
        heuristic.signals = signals + heuristic.signals
        return heuristic
    return Result(
        0.85 * max(probs) + 0.15 * heuristic.fake_probability,
        f"{kind}:{'+'.join(names)}",
        signals + heuristic.signals,
    )


SHORT_TEXT_WORDS = 25


def analyze_text(text: str) -> Result:
    result = _combine(_text_heuristic(text), _text_models(), clean_text(text), "text")
    if len(text.split()) < SHORT_TEXT_WORDS:
        # Style classifiers are unreliable on a headline-length text: halve their pull away from 50%.
        before = result.fake_probability
        result.fake_probability = 0.5 + (before - 0.5) * 0.5
        result.signals.append(f"Short text: detector score damped from {before:.0%} to {result.fake_probability:.0%}")
    return result


# --------------------------------------------------------------------- image

def _error_level_analysis(img: Image.Image) -> float:
    """Mean error level after JPEG re-compression. Edited regions recompress differently."""
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    resaved = Image.open(io.BytesIO(buf.getvalue()))
    diff = ImageChops.difference(img, resaved).convert("L")
    hist = diff.histogram()
    return sum(i * n for i, n in enumerate(hist)) / max(1, sum(hist))


def _image_heuristic(img: Image.Image, raw: Image.Image) -> Result:
    signals, score = [], 0.3
    exif = raw.getexif()
    if not exif:
        score += 0.15
        signals.append("No EXIF metadata (common for generated or re-processed images)")
    else:
        tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
        if "Make" in tags or "Model" in tags:
            score -= 0.15
            signals.append(f"Camera metadata present: {tags.get('Make', '')} {tags.get('Model', '')}".strip())
        software = str(tags.get("Software", ""))
        if software:
            signals.append(f"Processed with software: {software}")
            if re.search(r"photoshop|gimp|stable diffusion|midjourney|dall", software, re.I):
                score += 0.25
    ela = _error_level_analysis(img)
    signals.append(f"Error level analysis mean: {ela:.2f}")
    if ela > 6:
        score += 0.15
        signals.append("High error level - possible editing or compositing")
    w, h = img.size
    if w == h and w in (512, 768, 1024, 2048):
        score += 0.1
        signals.append(f"Square {w}x{h} resolution typical of AI image generators")
    return Result(max(0.02, min(score, 0.95)), "heuristic-image-v1", signals)


def _run_forensic_models(img: Image.Image, scores: dict, rows: list, names: list, signals: list) -> dict:
    """Run UnivFD and TruFor. One failing on a single image must not fail the whole upload."""
    extras = {}
    for key, (source, _, role) in FORENSIC_MODELS.items():
        predict = forensic_model(key)
        if predict is None:
            continue
        try:
            prob, model_signals, model_extras = predict(img)
        except Exception as exc:  # e.g. out of memory on a very large image
            log.warning("%s failed on this image: %s", key, exc)
            signals.append(f"{key} could not score this image")
            continue
        scores[key] = prob
        rows.append({"model": key, "role": role, "score": round(prob, 4)})
        names.append(config.model_name(source()))
        signals += model_signals
        extras.update(model_extras)
    return extras


def analyze_image(data: bytes) -> Result:
    raw = Image.open(io.BytesIO(data))
    img = raw.convert("RGB")
    heuristic = _image_heuristic(img, raw)

    scores, rows, names, signals, base_probs = {}, [], [], [], []
    for src, predict in _image_models():
        if predict is None:
            continue
        prob, model_signals = predict(img)
        signals += model_signals
        if prob is not None:
            key = "smaug" if "smaug" in src.lower() else config.model_name(src)
            scores[key] = prob
            base_probs.append(prob)
            rows.append({"model": key, "role": "AI-generated probability", "score": round(prob, 4)})
            names.append(config.model_name(src))
    if not base_probs:
        heuristic.signals = signals + heuristic.signals
        return heuristic

    extras = _run_forensic_models(img, scores, rows, names, signals)
    fused = image_fusion.fuse(scores)
    if fused is None:
        # No calibration yet: keep the established score. Raw scores are never averaged.
        prob = max(base_probs)
        for row in rows:
            row["counted"] = row["model"] not in FORENSIC_MODELS and row["score"] == round(prob, 4)
        if len(scores) > len(base_probs):
            signals.append("UnivFD and TruFor are shown but not counted: image fusion is not calibrated yet")
    else:
        prob = fused
        for row in rows:
            row["counted"] = True
        signals.append(f"Calibrated image fusion of {len(scores)} models: {fused:.0%}")
        names.append("calibrated")
    return Result(
        0.85 * prob + 0.15 * heuristic.fake_probability,
        f"image:{'+'.join(names)}",
        signals + heuristic.signals,
        {"models": rows, "imageFusion": "calibrated" if fused is not None else "uncalibrated", **extras},
    )


# --------------------------------------------------------------------- video

GENERATOR_SIZES = {(512, 512), (768, 768), (1024, 1024), (1280, 720), (720, 1280), (1920, 1080), (1080, 1920)}


def extract_frames(data: bytes, count: int) -> tuple[list[Image.Image], dict]:
    """Decode a video and return `count` frames spread evenly through it, plus metadata.
    Raises ImportError when OpenCV is missing and ValueError for undecodable files."""
    import os
    import tempfile

    import cv2

    # OpenCV only reads from a path, so spool the upload to a temporary file.
    fd, path = tempfile.mkstemp(suffix=".video")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError("unsupported or corrupt video")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        meta = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": round(fps, 2),
            "frames": total,
            "duration": round(total / fps, 2) if fps else None,
        }
        positions = [int(total * (i + 0.5) / count) for i in range(count)] if total else [None] * count
        frames = []
        for pos in positions:
            if pos is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, frame = cap.read()
            if ok:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
        if not frames:
            raise ValueError("no frames could be decoded")
        return frames, meta
    finally:
        os.unlink(path)


def _video_heuristic(meta: dict) -> Result:
    signals, score = [], 0.35
    w, h, dur = meta["width"], meta["height"], meta.get("duration")
    signals.append(f"Video {w}x{h}, {meta['fps']} fps, {dur if dur is not None else '?'} s")
    if (w, h) in GENERATOR_SIZES and dur is not None and dur <= 10.5:
        score += 0.1
        signals.append("Short clip at a resolution typical of AI video generators")
    if dur is not None and dur < 2:
        signals.append("Very short clip - low confidence")
    return Result(score, "heuristic-video-v1", signals)


def analyze_video(data: bytes) -> tuple[Result, list[Image.Image]]:
    """Score sampled frames with the image models. The video's score is the mean of the most
    suspicious third of frames, so a few manipulated frames are enough to raise it."""
    try:
        frames, meta = extract_frames(data, max(1, config.VIDEO_FRAMES))
    except ImportError:
        return Result(0.4, "heuristic-video-v1", ["Video decoding unavailable - install opencv-python-headless"]), []
    heuristic = _video_heuristic(meta)
    models = [(src, p) for src, p in _image_models() if p is not None]
    if not models:
        heuristic.signals.append("No image model loaded - heuristic estimate only")
        return heuristic, frames

    per_frame, names = [], set()
    for frame in frames:
        frame_probs = []
        for src, predict in models:
            prob, _ = predict(frame)
            if prob is not None:
                frame_probs.append(prob)
                names.add(config.model_name(src))
        if frame_probs:
            per_frame.append(max(frame_probs))
    if not per_frame:
        return heuristic, frames
    top = sorted(per_frame, reverse=True)[: max(1, len(per_frame) // 3)]
    prob = sum(top) / len(top)
    signals = [
        f"Deepfake / AI-video check on {len(per_frame)} frames: "
        f"max {max(per_frame):.0%}, mean {sum(per_frame) / len(per_frame):.0%}",
        "Per-frame AI probability: " + ", ".join(f"{p:.0%}" for p in per_frame),
    ] + heuristic.signals
    return Result(0.85 * prob + 0.15 * heuristic.fake_probability, f"video:{'+'.join(sorted(names))}", signals), frames


# --------------------------------------------------------------------- audio

AUDIO_RATE = 16_000
TTS_RATES = {16_000, 22_050, 24_000}


def load_audio(data: bytes) -> tuple["np.ndarray", int, int]:  # noqa: F821
    """Decode audio to mono float32 at 16 kHz. Returns (samples, 16000, original rate).
    Uses soundfile (WAV, FLAC, OGG, MP3) and falls back to the stdlib wave module."""
    import numpy as np

    try:
        import soundfile as sf

        samples, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        samples = samples.mean(axis=1)
    except ImportError:
        import wave

        with wave.open(io.BytesIO(data)) as w:
            rate, width, channels = w.getframerate(), w.getsampwidth(), w.getnchannels()
            raw = np.frombuffer(w.readframes(w.getnframes()), dtype={1: np.uint8, 2: np.int16, 4: np.int32}[width])
        samples = raw.astype("float32").reshape(-1, channels).mean(axis=1)
        samples = (samples - 128) / 128 if width == 1 else samples / float(2 ** (8 * width - 1))
    original = int(rate)
    if rate != AUDIO_RATE:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(rate), AUDIO_RATE)
        samples = resample_poly(samples, AUDIO_RATE // g, int(rate) // g).astype("float32")
    return samples, AUDIO_RATE, original


def _audio_heuristic(samples, original_rate: int) -> Result:
    import numpy as np

    signals, score = [], 0.35
    duration = len(samples) / AUDIO_RATE
    signals.append(f"Audio {duration:.1f} s, recorded at {original_rate} Hz")
    if original_rate in TTS_RATES:
        score += 0.1
        signals.append(f"{original_rate} Hz sample rate is common for text-to-speech output")
    frame = AUDIO_RATE // 50  # 20 ms windows
    if len(samples) >= frame * 10:
        rms = np.sqrt((samples[: len(samples) // frame * frame].reshape(-1, frame) ** 2).mean(axis=1))
        if (rms < 1e-5).mean() > 0.05:
            score += 0.1
            signals.append("Contains digital silence (no background noise) - common in synthetic speech")
    if duration < 3:
        signals.append("Very short clip - low confidence")
    return Result(score, "heuristic-audio-v1", signals)


def _audio_predictor(path: str):
    """Hugging Face audio-classification model (e.g. a wav2vec2 deepfake-voice detector).
    Long clips are scored in 10 s windows; the most suspicious window decides."""
    from transformers import pipeline

    clf = pipeline("audio-classification", model=path)
    name = config.model_name(path)
    window = AUDIO_RATE * 10

    def predict(samples):
        chunks = [samples[i : i + window] for i in range(0, max(1, len(samples)), window)][:6]
        probs, summary = [], ""
        for chunk in chunks:
            preds = clf({"raw": chunk, "sampling_rate": AUDIO_RATE}, top_k=None)
            summary = summary or ", ".join(f"{p['label']}={p['score']:.2f}" for p in preds)
            if (p := _fake_prob_from_labels(preds)) is not None:
                probs.append(p)
        if not probs:
            return None, [f"Unrecognised labels from {name}: {summary}"]
        return max(probs), [f"{name}: synthetic-voice probability {max(probs):.0%} over {len(probs)} window(s)"]

    return predict


def _audio_models():
    return [(config.AUDIO_MODEL, _predictor(config.AUDIO_MODEL, _audio_predictor))]


def analyze_audio(data: bytes) -> Result:
    samples, _, original = load_audio(data)
    if len(samples) == 0:
        raise ValueError("audio file contains no samples")
    return _combine(_audio_heuristic(samples, original), _audio_models(), samples, "audio")
