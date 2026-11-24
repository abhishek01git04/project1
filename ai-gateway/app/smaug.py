"""Adapter for abacusai/Smaug-AI-Image-Detector (Apache-2.0): a general detector for
AI-generated images (GPT-Image, Midjourney, FLUX, Nano Banana, DALL-E, Firefly, ...).

Mirrors the model's reference smaug_image_detect.py, so the Python file shipped in the
downloaded repo never has to be executed. The weights are safetensors.
"""

import math

import torch
from PIL import Image, ImageOps

CROP, SIDE, BUDGET = 528, 576, 140  # must match training: shorter side -> 576, 528 crops, 140 vision tokens


def views(img: Image.Image, crops: int) -> list[Image.Image]:
    """Shorter side to 576 px (upscaled to 528 if smaller), then 528 x 528 crops at native
    resolution: start, centre and end of the long side (crops=3) or the centre only (crops=1)."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    if min(w, h) > SIDE:
        s = SIDE / min(w, h)
        img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
    elif min(w, h) < CROP:
        s = CROP / min(w, h)
        img = img.resize((math.ceil(w * s), math.ceil(h * s)), Image.BICUBIC)
    w, h = img.size
    cx, cy = (w - CROP) // 2, (h - CROP) // 2
    if crops == 1:
        return [img.crop((cx, cy, cx + CROP, cy + CROP))]
    if w >= h:
        return [img.crop((x, cy, x + CROP, cy + CROP)) for x in (0, cx, w - CROP)]
    return [img.crop((cx, y, cx + CROP, y + CROP)) for y in (0, cy, h - CROP)]


class SmaugDetector:
    def __init__(self, name_or_path: str, crops: int = 3):
        from sentence_transformers import SentenceTransformer

        assert crops in (1, 3)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # bf16 on CUDA, float32 elsewhere (the model is unstable in fp16)
        self.st = SentenceTransformer(
            name_or_path, device=device, config_kwargs={"audio_config": None},
            model_kwargs={"torch_dtype": torch.bfloat16 if device == "cuda" else torch.float32},
        )
        ip = self.st[0].processor.image_processor
        ip.max_soft_tokens = ip.image_seq_length = BUDGET
        self.crops = crops

    @torch.no_grad()
    def __call__(self, img: Image.Image) -> float:
        """Probability that the image is AI-generated."""
        flat = [{"image": v} for v in views(img, self.crops)]
        logits = self.st.encode(flat, convert_to_tensor=True).float().view(1, -1)
        return float(torch.sigmoid(logits.mean(1))[0])  # average the crops' logits
