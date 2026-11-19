"""Stage 1 — Intake.

Open the picture, turn it the right way up, deal with transparency, bring very small or very
large pictures to a sensible size, and crop away empty border so the design fills the frame.
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .common import PipelineError, save_debug

MAX_SIDE = 4096
MIN_SIDE = 800


def _border_background(rgb: np.ndarray):
    """If the picture's border is (almost) one flat color, return that color, else None."""
    h, w = rgb.shape[:2]
    t = max(2, int(round(min(h, w) * 0.01)))
    border = np.concatenate([
        rgb[:t].reshape(-1, 3), rgb[-t:].reshape(-1, 3),
        rgb[:, :t].reshape(-1, 3), rgb[:, -t:].reshape(-1, 3),
    ]).astype(np.int16)
    med = np.median(border, axis=0)
    close = np.abs(border - med).max(axis=1) <= 24
    if close.mean() >= 0.7:
        return tuple(int(v) for v in med)
    return None


def run(data: bytes, debug_dir: Path | None = None) -> dict:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, ValueError):
        raise PipelineError(
            "We couldn't open that picture. Please use a PNG, JPG or WEBP file.")

    img = ImageOps.exif_transpose(img).convert("RGBA")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    if max(img.size) < MIN_SIDE:
        f = MIN_SIDE / max(img.size)
        img = img.resize((max(1, round(img.width * f)), max(1, round(img.height * f))),
                         Image.LANCZOS)

    rgba = np.asarray(img)
    alpha = rgba[:, :, 3]
    has_alpha = bool((alpha < 250).mean() > 0.002)

    # Flatten onto white so see-through pixels have a defined color.
    a = (alpha.astype(np.float32) / 255.0)[:, :, None]
    rgb = (rgba[:, :, :3].astype(np.float32) * a + 255.0 * (1 - a)).round().astype(np.uint8)

    bg_rgb = None
    if has_alpha:
        content = alpha >= 128
    else:
        alpha = None
        bg_rgb = _border_background(rgb)
        if bg_rgb is not None:
            diff = np.abs(rgb.astype(np.int16) - np.array(bg_rgb, np.int16)).max(axis=2)
            content = diff > 28
        else:
            content = np.ones(rgb.shape[:2], bool)

    if content.mean() < 0.0005:
        raise PipelineError(
            "This picture looks blank. Try a picture with a clear design on a plain background.")

    # Crop to the design, leaving a small margin.
    ys, xs = np.nonzero(content)
    h, w = content.shape
    m = int(round(max(h, w) * 0.02)) + 2
    y0, y1 = max(0, ys.min() - m), min(h, ys.max() + 1 + m)
    x0, x1 = max(0, xs.min() - m), min(w, xs.max() + 1 + m)
    rgb = np.ascontiguousarray(rgb[y0:y1, x0:x1])
    if alpha is not None:
        alpha = np.ascontiguousarray(alpha[y0:y1, x0:x1])

    save_debug(debug_dir, "01_intake.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return {"rgb": rgb, "alpha": alpha, "bg_rgb": bg_rgb}
