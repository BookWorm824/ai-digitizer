"""Stage 2 — Analyze.

Find out what is in the picture before anything is redrawn: what kind of art it is, the key
elements, any text (exact strings), how many colors, how much fine detail. The vision model
answers that; OpenCV measures the dominant colors locally. With no AI available the local
measurements still come back, and `ai` is None.
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .. import ai


def png_of(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "PNG")
    return buf.getvalue()


def dominant_colors(rgb: np.ndarray, k: int = 6) -> list[dict]:
    """The picture's main colors and the share of the picture each covers."""
    small = cv2.resize(rgb, (96, 96), interpolation=cv2.INTER_AREA).reshape(-1, 3)
    data = small.astype(np.float32)
    k = min(k, len(np.unique(small, axis=0)))
    if k < 1:
        return []
    cv2.setRNGSeed(0)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=k)
    out = [{"rgb": tuple(int(v) for v in centers[i]), "share": float(counts[i] / counts.sum())}
           for i in np.argsort(-counts) if counts[i]]
    return out


def run(intake: dict, use_ai: bool = True, debug_dir: Path | None = None) -> dict:
    rgb = intake["rgb"]
    result = {"ai": None, "ai_problem": None, "text": [],
              "dominant_colors": dominant_colors(rgb)}
    if not use_ai:
        return result
    try:
        found = ai.analyze(png_of(rgb))
        result["ai"] = found
        result["text"] = [t.strip() for t in found.get("text", []) if t and t.strip()]
    except ai.AIUnavailable as e:
        result["ai_problem"] = e.reason
    return result
