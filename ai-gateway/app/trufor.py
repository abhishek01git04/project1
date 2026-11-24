"""TruFor (Guillaro et al., CVPR 2023; GRIP-UNINA, nonprofit use only): finds locally manipulated
regions. It returns an integrity score for the whole image, a map of suspected regions and a
confidence map. Inference follows the authors' test_docker/src/trufor_test.py.

The model code is in app/vendor/trufor. The weights go in models/image-trufor/trufor.pth.tar.
"""

import base64
import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

HEATMAP_SIDE = 512
HEATMAP_RGB = (239, 79, 95)
HEATMAP_MAX_ALPHA = 0.85


class _Cfg(dict):
    __getattr__ = dict.__getitem__


# Values from the authors' test_docker/src/trufor.yaml.
_CONFIG = _Cfg(
    MODEL=_Cfg(
        MODS=("RGB", "NP++"),
        PRETRAINED="",
        EXTRA=_Cfg(BACKBONE="mit_b2", DECODER="MLPDecoder", DECODER_EMBED_DIM=512, PREPRC="imagenet",
                   BN_EPS=0.001, BN_MOMENTUM=0.1, DETECTION="confpool", CONF=True),
    ),
    DATASET=_Cfg(NUM_CLASSES=2),
)


@dataclass
class TruForOutput:
    score: float            # 0..1, higher = more likely manipulated
    tamper_map: np.ndarray  # H x W, 0..1
    confidence: float       # mean of the confidence map, 0..1
    scaled: bool            # True when the image was reduced before scoring


def fit_side(img: Image.Image, max_side: int) -> tuple[Image.Image, bool]:
    """Keep memory bounded on CPU. Reducing an image weakens the traces TruFor reads, so it is reported."""
    if max(img.size) <= max_side:
        return img, False
    scale = max_side / max(img.size)
    w, h = img.size
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS), True


def heatmap_data_uri(tamper_map: np.ndarray) -> str:
    """A transparent PNG of the map alone. It holds no pixels of the image, because reports can be public."""
    alpha = (np.clip(tamper_map, 0, 1) * HEATMAP_MAX_ALPHA * 255).astype(np.uint8)
    layer = Image.fromarray(alpha)
    scale = min(1.0, HEATMAP_SIDE / max(layer.size))
    size = (max(1, round(layer.width * scale)), max(1, round(layer.height * scale)))
    layer = layer.resize(size, Image.BILINEAR)
    solid = [Image.new("L", size, value) for value in HEATMAP_RGB]
    buffer = io.BytesIO()
    Image.merge("RGBA", (*solid, layer)).save(buffer, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _load_checkpoint(path: str) -> dict:
    """The checkpoint stores a few numpy numbers next to its weights. Only those data types are
    allowed, so weights_only stays on and the file cannot run code when it is loaded."""
    import torch

    try:
        from numpy._core.multiarray import scalar
    except ImportError:  # numpy < 2
        from numpy.core.multiarray import scalar
    number_types = [type(np.dtype(name)) for name in ("float64", "float32", "int64")]
    with torch.serialization.safe_globals([(scalar, "numpy.core.multiarray.scalar"), np.dtype, *number_types]):
        return torch.load(path, map_location="cpu", weights_only=True)


class TruFor:
    def __init__(self, weights_path: str, max_side: int = 1280):
        import torch

        from .vendor.trufor.models.cmx.builder_np_conf import myEncoderDecoder

        self.torch = torch
        self.max_side = max_side
        self.model = myEncoderDecoder(cfg=_CONFIG)
        self.model.load_state_dict(_load_checkpoint(weights_path)["state_dict"])
        self.model.eval()

    def __call__(self, img: Image.Image) -> TruForOutput:
        torch = self.torch
        img, scaled = fit_side(img.convert("RGB"), self.max_side)
        pixels = np.asarray(img).transpose(2, 0, 1)
        rgb = torch.tensor(pixels, dtype=torch.float).unsqueeze(0) / 256.0  # the authors divide by 256
        with torch.no_grad():
            pred, conf, det, _ = self.model(rgb)
            score = torch.sigmoid(det).item()
            tamper_map = torch.softmax(pred, dim=1)[0, 1].numpy()
            confidence = torch.sigmoid(conf)[0, 0].mean().item()
        return TruForOutput(score, tamper_map, confidence, scaled)
