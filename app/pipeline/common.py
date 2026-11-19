"""Small shared pieces for the pipeline stages."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon

BG = 255  # label value for "background, not stitched" in a label map


class PipelineError(Exception):
    """A problem the user can do something about. The message is shown to them as written,
    so it must be plain words and say what to do next."""


def polygons_of(geom) -> list[Polygon]:
    """Every non-empty Polygon inside any shapely geometry."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, (MultiPolygon, GeometryCollection)):
        out = []
        for g in geom.geoms:
            out.extend(polygons_of(g))
        return out
    return []


def dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def hex_of(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(v) for v in rgb)


def save_debug(debug_dir: Path | None, name: str, bgr: np.ndarray) -> None:
    """Write a stage's debug picture. Uses imencode so paths with special characters work."""
    if debug_dir is None:
        return
    debug_dir.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", bgr)
    if ok:
        (debug_dir / name).write_bytes(buf.tobytes())


def canvas_for(bounds, px_per_mm: float, pad_mm: float = 3.0, color=(245, 245, 245)):
    """A blank picture big enough for geometry with these mm bounds, and a function that
    converts mm points to pixel points on it."""
    minx, miny, maxx, maxy = bounds
    w = int(math.ceil((maxx - minx + 2 * pad_mm) * px_per_mm)) + 1
    h = int(math.ceil((maxy - miny + 2 * pad_mm) * px_per_mm)) + 1
    img = np.full((max(h, 2), max(w, 2), 3), color, np.uint8)

    def to_px(pts):
        a = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        out = np.empty_like(a)
        out[:, 0] = (a[:, 0] - minx + pad_mm) * px_per_mm
        out[:, 1] = (a[:, 1] - miny + pad_mm) * px_per_mm
        return out

    return img, to_px
