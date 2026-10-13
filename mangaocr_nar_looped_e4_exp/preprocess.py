"""CPU preprocessing (numpy + Pillow).
The crop is resized with its aspect ratio kept to fit a grid of at most 324 patches of 14 x 14 px (PIL bicubic), placed at the
top-left of the grid canvas, the rest filled with the median colour of the resized image's border, normalised and cut into patches."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image

BUDGET, PATCH, FEAT = 324, 14, 588
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


def pad_grid(height: int, width: int, budget: int = BUDGET):
    """Largest uniform scale s with ceil(H*s/14) * ceil(W*s/14) <= budget -> (s, grid h, grid w)."""
    lo, hi = 1e-4, 1000.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if math.ceil(height * mid / PATCH) * math.ceil(width * mid / PATCH) <= budget:
            lo = mid
        else:
            hi = mid
    return lo, math.ceil(height * lo / PATCH), math.ceil(width * lo / PATCH)


def resized_size(height: int, width: int):
    """-> (scale, grid h, grid w, resized width, resized height)"""
    s, h, w = pad_grid(height, width)
    return s, h, w, max(1, min(w * PATCH, round(width * s))), max(1, min(h * PATCH, round(height * s)))


def empty_inputs(batch: int | None = None) -> dict:
    lead = () if batch is None else (batch,)
    return {"P": np.zeros(lead + (BUDGET, FEAT), np.float32), "V": np.zeros(lead + (BUDGET,), np.float32),
            "R": np.zeros(lead + (BUDGET,), np.int32), "C": np.zeros(lead + (BUDGET,), np.int32),
            "H": np.zeros(lead + (1,) if batch is None else lead, np.int32), "W": np.zeros(lead + (1,) if batch is None else lead, np.int32)}


def preprocess(image: Image.Image, out: dict) -> None:
    """Writes one crop into the arrays of `out`: P [324, 588] patches, V [324] valid (1/0), R/C [324] patch row/column,
    H/W grid size (padded tail zero). The arrays may be views into model input buffers."""
    image = image.convert("RGB")
    _, h, w, rw, rh = resized_size(image.height, image.width)
    small = np.asarray(image.resize((rw, rh), Image.Resampling.BICUBIC), np.uint8)
    border = np.concatenate([small[0], small[-1], small[:, 0], small[:, -1]])
    canvas = np.empty((h * PATCH, w * PATCH, 3), np.uint8)
    canvas[:] = np.median(border, axis=0).astype(np.uint8)
    canvas[:rh, :rw] = small
    arr = (canvas.astype(np.float32) / 255.0 - MEAN) / STD
    n = h * w
    out["P"][:n] = arr.reshape(h, PATCH, w, PATCH, 3).transpose(0, 2, 4, 1, 3).reshape(n, FEAT)
    out["P"][n:] = 0
    out["V"][:n], out["V"][n:] = 1, 0
    idx = np.arange(n)
    out["R"][:n], out["R"][n:] = idx // w, 0
    out["C"][:n], out["C"][n:] = idx % w, 0
    out["H"][0], out["W"][0] = h, w
