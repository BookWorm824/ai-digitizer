"""Stage 5 — Vectorize.

Trace each color of the label map into outlines (polygons with holes), smooth the pixel
stair-steps away, and scale everything to real millimetres, centred on (0, 0).

Tracing uses OpenCV's contour follower on the label map rather than vtracer/potrace: every
color is traced from the same map, so neighbouring colors line up exactly with no gaps.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union

from .common import BG, PipelineError, canvas_for, polygons_of, save_debug


def _trace(mask: np.ndarray) -> list[Polygon]:
    """Polygons (with holes) for one color's mask, in pixel units."""
    contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None:
        return []
    hier = hier[0]
    out = []
    for i, cnt in enumerate(contours):
        if hier[i][3] != -1 or len(cnt) < 3:      # only outer rings start a polygon
            continue
        holes = []
        child = hier[i][2]
        while child != -1:
            if len(contours[child]) >= 3:
                holes.append(contours[child].reshape(-1, 2))
            child = hier[child][0]
        poly = Polygon(cnt.reshape(-1, 2), holes)
        out.extend(polygons_of(poly.buffer(0)))
    return out


def run(clean: dict, fit_w_mm: float, fit_h_mm: float, rules: dict,
        debug_dir: Path | None = None) -> dict:
    labels, palette, ppm = clean["labels"], clean["palette"], clean["ppm"]
    px = 1.0 / ppm
    layers = []
    for i, rgb in enumerate(palette):
        mask = (labels == i).astype(np.uint8)
        if not mask.any():
            continue
        polys = []
        for p in _trace(mask):
            p = affinity.scale(p, px, px, origin=(0, 0))
            # Contours run through pixel centres, half a pixel inside the true edge. Grow
            # back by that much, then straighten the stair-steps.
            p = p.buffer(0.5 * px, join_style=2, mitre_limit=2.0)
            p = p.simplify(0.75 * px, preserve_topology=True)
            polys.extend(q for q in polygons_of(p.buffer(0)) if q.area > 0.05)
        if polys:
            layers.append({"color": i, "rgb": rgb, "polys": polys})

    if not layers:
        raise PipelineError(
            "We couldn't find any shapes to stitch in this picture. "
            "Try a picture with a clear design on a plain background.")

    # Scale so the whole design exactly fits the requested size, leaving room for the small
    # amount shapes are grown later, and centre it.
    everything = unary_union([p for layer in layers for p in layer["polys"]])
    minx, miny, maxx, maxy = everything.bounds
    grow = rules["pull_comp_mm"]  # worst case growth on each side, see classify
    f = min((fit_w_mm - grow) / (maxx - minx), (fit_h_mm - grow) / (maxy - miny))
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    for layer in layers:
        moved = []
        for p in layer["polys"]:
            p = affinity.translate(p, -cx, -cy)
            moved.append(affinity.scale(p, f, f, origin=(0, 0)))
        layer["polys"] = moved

    bounds = ((minx - cx) * f, (miny - cy) * f, (maxx - cx) * f, (maxy - cy) * f)
    if debug_dir is not None:
        img, to_px = canvas_for(bounds, 8.0, color=(255, 255, 255))
        for layer in layers:
            bgr = tuple(int(v) for v in layer["rgb"][::-1])
            for p in layer["polys"]:
                rings = [to_px(p.exterior.coords).astype(np.int32)]
                rings += [to_px(r.coords).astype(np.int32) for r in p.interiors]
                cv2.fillPoly(img, rings, bgr, lineType=cv2.LINE_AA)
                cv2.polylines(img, rings, True, (0, 0, 0), 1, cv2.LINE_AA)
        save_debug(debug_dir, "05_vectorize.png", img)

    return {"layers": layers, "bounds": bounds}
