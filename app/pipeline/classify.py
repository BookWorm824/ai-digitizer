"""Stage 6 — Classify.

Decide how each traced shape will be sewn, by measuring how wide it is:
  * thin lines (narrower than `thin_line_mm`)  -> a running stitch along the centre line
    (a triple "bean" run when the line is bold enough to need it)
  * everything else                            -> a fill

A shape can be both: a wide body with thin tails is split into a fill plus runs.
Satin columns arrive in Phase 3; until then narrow columns are sewn as fills.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union
from skimage.morphology import skeletonize

from .common import canvas_for, dist, polygons_of, save_debug

SKEL_PPM = 12.0          # resolution used to find centre lines
MIN_RUN_MM = 1.5         # centre-line pieces shorter than this are spurs, not lines
MIN_THIN_WIDTH_MM = 0.2  # slivers thinner than this are tracing noise, not lines
WIDE_FACTOR = 1.3        # a part must reach this many times the thin-line width to be a fill
MIN_WIDE_AREA_MM2 = 8.0  # smaller wide spots (where two lines cross, say) stay part of the line


def _trace_skeleton(skel: np.ndarray) -> list[list[tuple[int, int]]]:
    """Follow a one-pixel-wide skeleton and return its branches as pixel paths (y, x)."""
    ys, xs = np.nonzero(skel)
    pts = set(zip(ys.tolist(), xs.tolist()))

    def nbrs(p):
        y, x = p
        return [(y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                if (dy or dx) and (y + dy, x + dx) in pts]

    nb = {p: nbrs(p) for p in pts}
    nodes = {p for p, n in nb.items() if len(n) != 2}
    seen = set()

    def key(a, b):
        return (a, b) if a < b else (b, a)

    paths = []
    for n in nodes:
        for q in nb[n]:
            if key(n, q) in seen:
                continue
            seen.add(key(n, q))
            path, prev, cur = [n], n, q
            while True:
                path.append(cur)
                if cur in nodes:
                    break
                nxt = [r for r in nb[cur] if r != prev]
                if not nxt or key(cur, nxt[0]) in seen:
                    break
                seen.add(key(cur, nxt[0]))
                prev, cur = cur, nxt[0]
            paths.append(path)

    # closed loops with no branch points
    loose = {p for p in pts if p not in nodes and not any(key(p, q) in seen for q in nb[p])}
    while loose:
        start = loose.pop()
        path, prev, cur = [start], None, start
        while True:
            nxt = [r for r in nb[cur] if r != prev and r in loose]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            loose.discard(cur)
            path.append(cur)
        if len(path) > 2:
            path.append(start)
        paths.append(path)
    return paths


def _join_paths(paths: list[list], tol: float = 2.5) -> list[list]:
    """Join branches whose ends touch, so a long line isn't chopped up at every junction."""
    paths = [p for p in paths if len(p) >= 2]
    changed = True
    while changed:
        changed = False
        for i in range(len(paths)):
            for j in range(len(paths)):
                if i == j:
                    continue
                a, b = paths[i], paths[j]
                if dist(a[-1], b[0]) <= tol:
                    paths[i] = a + b
                elif dist(a[-1], b[-1]) <= tol:
                    paths[i] = a + b[::-1]
                elif dist(a[0], b[0]) <= tol:
                    paths[i] = a[::-1] + b
                else:
                    continue
                del paths[j]
                changed = True
                break
            if changed:
                break
    return paths


def centerlines(piece: Polygon) -> tuple[list[LineString], float]:
    """Centre lines of a thin shape, in mm, plus the shape's average width in mm."""
    minx, miny, maxx, maxy = piece.bounds
    pad = 3
    w = int(math.ceil((maxx - minx) * SKEL_PPM)) + 2 * pad
    h = int(math.ceil((maxy - miny) * SKEL_PPM)) + 2 * pad

    def to_px(coords):
        a = np.asarray(coords)[:, :2]
        return np.round(np.column_stack([(a[:, 0] - minx) * SKEL_PPM + pad,
                                         (a[:, 1] - miny) * SKEL_PPM + pad])).astype(np.int32)

    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [to_px(piece.exterior.coords)], 1)
    for ring in piece.interiors:
        cv2.fillPoly(mask, [to_px(ring.coords)], 0)
    if not mask.any():
        return [], 0.0
    skel = skeletonize(mask.astype(bool))
    if not skel.any():
        return [], 0.0
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    width = float(2.0 * dt[skel].mean() / SKEL_PPM)

    lines = []
    for path in _join_paths(_trace_skeleton(skel)):
        if len(path) < 2:
            continue
        pts = [((x - pad) / SKEL_PPM + minx, (y - pad) / SKEL_PPM + miny) for y, x in path]
        line = LineString(pts).simplify(0.1)
        if line.length >= MIN_RUN_MM:
            lines.append(line)
    return lines, width


def run(vec: dict, rules: dict, debug_dir: Path | None = None) -> dict:
    half = rules["thin_line_mm"] / 2.0
    min_area = 0.6 * rules["min_detail_mm"] ** 2
    # Phase 1 stand-in for pull compensation: grow every fill evenly by half the preset value
    # (so widths grow by the full value). Phase 3 replaces this with growth along the stitch
    # direction only.
    grow = rules["pull_comp_mm"] / 2.0

    objects, dropped = [], 0
    for layer in vec["layers"]:
        color = layer["color"]
        for poly in layer["polys"]:
            # "Wide" parts survive being shrunk by half the thin-line width. To stop a line
            # whose width hovers around the limit from flipping between fill and run along
            # its length, a wide part only counts if it is clearly wide somewhere.
            opened = poly.buffer(-half).buffer(half * 1.02)
            seeds = poly.buffer(-half * WIDE_FACTOR)
            wide = [c for c in polygons_of(opened)
                    if not seeds.is_empty and c.intersects(seeds) and c.area >= MIN_WIDE_AREA_MM2]
            thin_pieces = polygons_of(poly.difference(unary_union(wide))) if wide else [poly]
            lines_here, thin_used = [], []
            for piece in thin_pieces:
                if piece.area < 0.15:
                    continue
                lines, width = centerlines(piece)
                if width >= MIN_THIN_WIDTH_MM and sum(l.length for l in lines) >= 2.0:
                    thin_used.append(piece)
                    for line in lines:
                        lines_here.append((line, width))

            body = poly
            if thin_used:
                body = poly.difference(unary_union([p.buffer(0.05) for p in thin_used]))
            for part in polygons_of(body):
                if part.area < min_area:
                    dropped += 1
                    continue
                grown = polygons_of(part.buffer(grow, quad_segs=4)) if grow > 0 else [part]
                for g in grown:
                    objects.append({"kind": "fill", "color": color, "poly": g,
                                    "area_mm2": float(g.area)})
            for line, width in lines_here:
                objects.append({"kind": "run", "color": color, "line": line,
                                "bean": bool(width >= rules["bean_line_mm"]),
                                "width_mm": width})

    if debug_dir is not None and objects:
        geoms = [o["poly"] if o["kind"] == "fill" else o["line"] for o in objects]
        img, to_px = canvas_for(unary_union(geoms).bounds, 8.0, color=(255, 255, 255))
        for o in objects:        # fills = green, runs = red (bean runs thicker)
            if o["kind"] == "fill":
                rings = [to_px(o["poly"].exterior.coords).astype(np.int32)]
                rings += [to_px(r.coords).astype(np.int32) for r in o["poly"].interiors]
                cv2.fillPoly(img, rings, (170, 220, 170), lineType=cv2.LINE_AA)
                cv2.polylines(img, rings, True, (60, 140, 60), 1, cv2.LINE_AA)
        for o in objects:
            if o["kind"] == "run":
                cv2.polylines(img, [to_px(o["line"].coords).astype(np.int32)], False,
                              (40, 40, 220), 3 if o["bean"] else 1, cv2.LINE_AA)
        save_debug(debug_dir, "06_classify.png", img)

    return {"objects": objects, "dropped_details": dropped}
