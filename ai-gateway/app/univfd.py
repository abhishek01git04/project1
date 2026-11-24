"""UnivFD (Ojha, Li and Lee, CVPR 2023; MIT licence): a detector for AI-generated images made of
one linear layer trained on top of CLIP ViT-L/14 image features.

fc_weights.pth holds only that linear layer; CLIP comes from the Hugging Face Hub. Preprocessing
follows the authors' test code (validate.py): a 224 x 224 centre crop at native resolution, with no
resizing, then CLIP normalisation. Resizing first would blur the traces the detector reads.
"""

import numpy as np
from PIL import Image

CROP_SIZE = 224
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def preprocess(img: Image.Image) -> np.ndarray:
    """Return a CHW float array ready for CLIP, cropped like torchvision's CenterCrop(224)."""
    img = img.convert("RGB")
    w, h = img.size
    if w < CROP_SIZE or h < CROP_SIZE:
        # CenterCrop pads a smaller image with black on both sides first.
        canvas = Image.new("RGB", (max(w, CROP_SIZE), max(h, CROP_SIZE)))
        canvas.paste(img, ((canvas.width - w) // 2, (canvas.height - h) // 2))
        img, (w, h) = canvas, canvas.size
    x, y = int(round((w - CROP_SIZE) / 2.0)), int(round((h - CROP_SIZE) / 2.0))
    pixels = np.asarray(img.crop((x, y, x + CROP_SIZE, y + CROP_SIZE)), dtype=np.float32) / 255.0
    return ((pixels - MEAN) / STD).transpose(2, 0, 1)


class UnivFD:
    def __init__(self, fc_path: str, clip_model: str):
        import torch
        from transformers import CLIPVisionModelWithProjection

        self.torch = torch
        self.clip = CLIPVisionModelWithProjection.from_pretrained(clip_model).eval()
        self.fc = torch.nn.Linear(self.clip.config.projection_dim, 1)
        self.fc.load_state_dict(torch.load(fc_path, map_location="cpu", weights_only=True))
        self.fc.eval()

    def __call__(self, img: Image.Image) -> float:
        """Probability that the image is AI-generated."""
        torch = self.torch
        with torch.no_grad():
            pixels = torch.from_numpy(preprocess(img)).unsqueeze(0)
            features = self.clip(pixel_values=pixels).image_embeds
            return float(torch.sigmoid(self.fc(features))[0, 0])
