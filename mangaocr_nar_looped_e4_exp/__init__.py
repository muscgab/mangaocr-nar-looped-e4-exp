"""mangaocr-nar-looped-e4-exp: Japanese manga text-box OCR, non-autoregressive, looped decoder (experimental).

    from mangaocr_nar_looped_e4_exp import MangaOcrNarLoopedE4Exp
    ocr = MangaOcrNarLoopedE4Exp(loop=3)     # read out after loop 1, 2 or 3 (more loops = slower)
    ocr("crop.png")                          # one crop -> str
    ocr([crop1, crop2, ...])                 # several crops -> list[str], run as one batch
    ocr([crop1, crop2], loop="all")          # -> [[loop1, loop2, loop3], ...] at the cost of 3 loops

Crops may be file paths, PIL images or uint8 HWC RGB arrays. Plain PyTorch in fp32; device = CUDA if available, else CPU
(any torch device name can be passed)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

from .preprocess import empty_inputs, preprocess
from .vocab import Vocab

__version__ = "1.0.0"
__all__ = ["MangaOcrNarLoopedE4Exp"]
DEFAULT_DIR = Path(os.environ.get("MANGAOCR_NAR_LOOPED_E4_EXP_DIR", Path(__file__).resolve().parent.parent))
MAX_BATCH = 64


def _device(name):
    import torch
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pil(x) -> Image.Image:
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    if isinstance(x, np.ndarray):
        return Image.fromarray(np.ascontiguousarray(x if x.ndim == 3 else np.stack([x] * 3, -1), dtype=np.uint8))
    with Image.open(x) as im:
        im.load()
        return im.convert("RGB")


class MangaOcrNarLoopedE4Exp:
    def __init__(self, loop=3, device: str | None = None, model_dir=None):
        import torch
        from .model import LoopedE4Net
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_DIR
        if not (self.model_dir / "model.safetensors").exists():
            raise FileNotFoundError(f"model files not found in {self.model_dir}; pass model_dir= (the downloaded repository) "
                                    "or set MANGAOCR_NAR_LOOPED_E4_EXP_DIR")
        self.loop = self._check(loop)
        self.vocab = Vocab(self.model_dir / "nar_vocabulary_6817.json")
        self.device = _device(device)
        self.net = LoopedE4Net.load(self.model_dir, self.device)
        self._torch = torch

    @staticmethod
    def _check(loop):
        if loop not in (1, 2, 3, "all"):
            raise ValueError(f"loop must be 1, 2, 3 or 'all', got {loop!r}")
        return loop

    def ids(self, images, loops: int) -> list[np.ndarray]:
        """crops -> list over loops 1..`loops` of ids [n, 65]"""
        torch = self._torch
        out = [[] for _ in range(loops)]
        for i in range(0, len(images), MAX_BATCH):
            part = images[i:i + MAX_BATCH]
            x = empty_inputs(len(part))
            for j, im in enumerate(part):
                preprocess(im, {k: v[j:j + 1] if k in ("H", "W") else v[j] for k, v in x.items()})
            t = {k: torch.from_numpy(v).to(self.device) for k, v in x.items()}
            with torch.inference_mode():
                rows = self.net(t["P"], t["V"].bool(), t["R"].long(), t["C"].long(), t["H"].long(), t["W"].long(), loops)
            for n, r in enumerate(rows):
                out[n].append(r.cpu().numpy())
        return [np.concatenate(o) if o else np.zeros((0, 65), np.int64) for o in out]

    def __call__(self, images, loop=None):
        loop = self.loop if loop is None else self._check(loop)
        single = not isinstance(images, (list, tuple))
        crops = [_pil(x) for x in ([images] if single else images)]
        rows = self.ids(crops, 3 if loop == "all" else loop)
        if loop == "all":
            per_loop = [self.vocab.decode(r.tolist()) for r in rows]
            texts = [list(t) for t in zip(*per_loop)]
        else:
            texts = self.vocab.decode(rows[-1].tolist())
        return texts[0] if single else texts
