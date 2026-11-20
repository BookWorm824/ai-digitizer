"""Stage 6 — Classify.

Decide how each traced shape will be sewn, by measuring how wide it is:

  * thin lines   (narrower than `thin_line_mm`)        -> a running stitch along the centre
                                                           line (triple "bean" run when bold)
  * narrow columns (`thin_line_mm` .. `satin_max_mm`)  -> satin: stitches zig-zag across the
                                                           column from edge to edge
  * wide areas                                          -> a fill

A shape can be a mix: a wide body with thin tails is split into a fill plus runs. A shape only
becomes satin when it is narrow all over and columns along its centre lines really do cover
it; otherwise it stays a fill, which always works.

This stage also decides the sewing order of the colors (biggest area first) and lets each
color reach a little way under the colors sewn after it, so no gap opens where colors meet.
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
MIN_RUN_MM = 1.2         # centre-line pieces shorter than this are spurs, not lines
MIN_THIN_WIDTH_MM = 0.3  # slivers thinner than this are tracing noise, not lines
MIN_THIN_LENGTH_MM = 3.0 # ...and so are thin bits shorter than this
LINE_FACTOR = 0.85       # a narrow shape this close to the thin-line width is sewn as satin
FRINGE_MM = 0.5          # thin bits narrower than this on the edge of a solid shape are noise
WIDE_FACTOR = 1.15       # a part must reach this many times the thin-line width to count as wide
MIN_WIDE_AREA_MM2 = 8.0  # smaller wide spots (where two lines cross, say) stay part of the line
SATIN_STEP_MM = 0.25     # spacing of the sample points along a satin column's centre line
SATIN_COVER = 0.8        # columns must cover this share of a shape for it to be sewn as satin
SATIN_SPILL = 0.25       # ...and may not spill outside it by more than this share


# ------------------------------------------------------------------------------ skeletons

def _raster(piece: Polygon):
    """A shape as a small black-and-white picture, plus the way back to millimetres."""
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

    def to_mm(y, x):
        return ((x - pad) / SKEL_PPM + minx, (y - pad) / SKEL_PPM + miny)

    return mask, to_mm


def _trace_skeleton(skel: np.ndarray):
    """Follow a one-pixel-wide skeleton. Returns (branches as pixel paths (y, x), the set of
    pixels where branches meet or end)."""
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
    ends = {p for p in nodes if len(nb[p]) == 1}
    return paths, ends


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
    mask, to_mm = _raster(piece)
    if not mask.any():
        return [], 0.0
    skel = skeletonize(mask.astype(bool))
    if not skel.any():
        return [], 0.0
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    width = float(2.0 * dt[skel].mean() / SKEL_PPM)
    paths, _ends = _trace_skeleton(skel)
    lines = []
    for path in _join_paths(paths):
        if len(path) < 2:
            continue
        line = LineString([to_mm(y, x) for y, x in path]).simplify(0.1)
        if line.length >= MIN_RUN_MM:
            lines.append(line)
    return lines, width


# ---------------------------------------------------------------------------------- satin

def _smooth(a: np.ndarray, k: int = 5) -> np.ndarray:
    if len(a) < k:
        return a
    pad = k // 2
    padded = np.concatenate([np.repeat(a[:1], pad, axis=0), a, np.repeat(a[-1:], pad, axis=0)])
    kernel = np.ones(k) / k
    if a.ndim == 1:
        return np.convolve(padded, kernel, mode="valid")
    return np.column_stack([np.convolve(padded[:, i], kernel, mode="valid")
                            for i in range(a.shape[1])])


def column_outline(center: np.ndarray, half: np.ndarray) -> Polygon:
    """The area a satin column covers: the band `half` wide on each side of its centre line.
    A column that closes on itself (a ring, the outline of a letter) covers a band with a
    hole in the middle, not the whole disc."""
    left, right = column_rails(center, half)
    if len(center) > 8 and dist(center[0], center[-1]) < 3 * SATIN_STEP_MM:
        a, b = Polygon(left), Polygon(right)
        if not a.is_valid:
            a = a.buffer(0)
        if not b.is_valid:
            b = b.buffer(0)
        outer, inner = (a, b) if a.area >= b.area else (b, a)
        band = outer.difference(inner)
        if not band.is_empty:
            return band
    poly = Polygon(np.vstack([left, right[::-1]]))
    return poly if poly.is_valid else poly.buffer(0)


def column_rails(center: np.ndarray, half: np.ndarray):
    """The two edges of a satin column."""
    t = np.gradient(center, axis=0)
    n = np.linalg.norm(t, axis=1, keepdims=True)
    t = t / np.where(n < 1e-9, 1.0, n)
    normal = np.column_stack([-t[:, 1], t[:, 0]])
    return center + normal * half[:, None], center - normal * half[:, None]


def columns(piece: Polygon) -> list[dict]:
    """Satin columns along the centre lines of a narrow shape. Each is
    {"center": Nx2 array (mm), "half": N half-widths (mm), "outline": Polygon}."""
    mask, to_mm = _raster(piece)
    if not mask.any():
        return []
    skel = skeletonize(mask.astype(bool))
    if not skel.any():
        return []
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    paths, ends = _trace_skeleton(skel)

    # Thinning leaves short spurs running into corners. A branch that ends in mid-air and is
    # no longer than the column is wide where it starts is one of those, not a real arm.
    kept = []
    for path in paths:
        a_end, b_end = path[0] in ends, path[-1] in ends
        if a_end != b_end and len(paths) > 1:
            root = path[-1] if a_end else path[0]
            if len(path) <= 1.6 * dt[root]:
                continue
        kept.append(path)

    out = []
    for path in _join_paths(kept):
        if len(path) < 4:
            continue
        free_end = (tuple(path[0]) in ends, tuple(path[-1]) in ends)
        pts = np.array([to_mm(y, x) for y, x in path])
        rad = np.array([dt[y, x] for y, x in path]) / SKEL_PPM
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        if cum[-1] < 1.0:
            continue
        # even spacing along the line, then smooth away the pixel wobble
        s = np.linspace(0.0, cum[-1], max(4, int(cum[-1] / SATIN_STEP_MM) + 1))
        center = _smooth(np.column_stack([np.interp(s, cum, pts[:, 0]),
                                          np.interp(s, cum, pts[:, 1])]))
        half = np.interp(s, cum, rad)
        half = _smooth(np.minimum(half, 1.25 * np.median(half)))   # no bulge where arms meet
        closed = dist(center[0], center[-1]) < 2 * SATIN_STEP_MM and cum[-1] > 4.0
        if not closed:
            # The centre line stops about half a width short of a free end: carry it on.
            # (Not where it stops at a junction: the column it meets covers that.)
            for end in (0, -1):
                if not free_end[end]:
                    continue
                # The last half-width of a centre line tends to hook toward a corner, which
                # would splay the stitches. Cut it off and carry on straight instead, in the
                # direction the column was heading just before.
                cut = int(half[end] / SATIN_STEP_MM)
                if len(center) > 3 * cut + 6 and cut > 0:
                    center = center[cut:] if end == 0 else center[:-cut]
                    half = half[cut:] if end == 0 else half[:-cut]
                else:
                    cut = 0
                k = min(len(center) - 1, max(4, cut))
                d = center[end] - center[k if end == 0 else -1 - k]
                n = np.linalg.norm(d)
                if n < 1e-9:
                    continue
                steps = cut + int(0.9 * half[end] / SATIN_STEP_MM)
                extra = np.array([center[end] + d / n * SATIN_STEP_MM * (j + 1)
                                  for j in range(steps)]).reshape(-1, 2)
                if end == 0:
                    center = np.vstack([extra[::-1], center])
                    half = np.concatenate([np.full(steps, half[0]), half])
                else:
                    center = np.vstack([center, extra])
                    half = np.concatenate([half, np.full(steps, half[-1])])
        if len(center) < 4:
            continue
        out.append({"center": center, "half": half, "outline": column_outline(center, half)})
    return out


def try_satin(piece: Polygon, rules: dict) -> list[dict] | None:
    """Satin columns for a narrow shape, or None if columns wouldn't reproduce it well."""
    if piece.area < 2.0:
        return None
    cols = []
    for c in columns(piece):
        length = float(np.linalg.norm(np.diff(c["center"], axis=0), axis=1).sum())
        width = 2 * float(np.median(c["half"]))
        if width > rules["satin_max_mm"] * 1.1:
            return None
        # Not columns: stubs where arms meet, hair-thin spurs, and wedges or lenses whose
        # width keeps changing. Real columns nearby cover the first two; a shape made of the
        # third fails the coverage test below and is filled instead.
        even = np.percentile(c["half"], 25) >= 0.6 * np.median(c["half"])
        if length >= 1.2 * width and width >= 0.6 * rules["thin_line_mm"] and even:
            cols.append(c)
    if not cols:
        return None
    try:
        cover = unary_union([c["outline"] for c in cols])
        covered = cover.intersection(piece).area
        spill = cover.difference(piece.buffer(0.3)).area
    except Exception:  # noqa: BLE001 - odd geometry: just fill it instead
        return None
    if covered < SATIN_COVER * piece.area or spill > SATIN_SPILL * piece.area:
        return None
    return cols


# ------------------------------------------------------------------------------------ run

def sewing_order(layers: list, satin_max_mm: float = 7.0) -> list[int]:
    """Colors in the order they are sewn. Colors made mostly of wide areas go first, biggest
    first, so they form the background. Colors made mostly of narrow shapes go after them,
    widest first, so lettering lands on the areas and the thinnest outlines land on top of
    everything."""
    half = satin_max_mm / 2.0
    area, wide, width = {}, {}, {}
    for layer in layers:
        c = layer["color"]
        area[c] = sum(p.area for p in layer["polys"])
        wide[c] = sum(p.buffer(-half).buffer(half).area for p in layer["polys"])
        edge = sum(p.length for p in layer["polys"])
        width[c] = 2.0 * area[c] / edge if edge else 0.0     # a strip's width = 2 x area / edge
    def key(c):
        if wide[c] >= 0.3 * area[c]:
            return (0, -area[c])
        return (1, -width[c])
    return sorted(area, key=key)


def underlap(layers: list, order: list[int], overlap: float) -> dict:
    """Let each color reach `overlap` mm under the colors sewn after it, where they touch.
    It never reaches to within half a millimetre of the far side of the later shape, so a
    color can't show through on the other side of a thin outline."""
    by_color = {layer["color"]: layer["polys"] for layer in layers}
    out = {}
    later = None                      # everything sewn after the current color
    for color in reversed(order):
        polys = by_color[color]
        if later is None or overlap <= 0:
            out[color] = list(polys)
        else:
            grown = []
            for p in polys:
                reach = p.buffer(overlap, quad_segs=4).intersection(later)
                if not reach.is_empty:
                    far_side = later.boundary.difference(p.buffer(0.05))
                    reach = reach.difference(far_side.buffer(overlap + 0.1, quad_segs=4))
                grown.extend(polygons_of(p.union(reach)) or [p])
            out[color] = grown
        mine = unary_union(polys)
        later = mine if later is None else later.union(mine)
    return out


def run(vec: dict, rules: dict, debug_dir: Path | None = None) -> dict:
    half = rules["thin_line_mm"] / 2.0
    min_area = 0.6 * rules["min_detail_mm"] ** 2
    satin_half = rules["satin_max_mm"] / 2.0

    order = sewing_order(vec["layers"], rules["satin_max_mm"])
    shapes = underlap(vec["layers"], order, rules["overlap_mm"])

    objects, dropped = [], 0
    def add_satin(cols, color):
        for c in cols:
            objects.append({"kind": "satin", "color": color, "center": c["center"],
                            "half": c["half"], "outline": c["outline"],
                            "width_mm": 2 * float(np.median(c["half"])),
                            "area_mm2": float(c["outline"].area)})

    for color in order:
        for poly in shapes[color]:
            # A shape that is narrow all over (an outline, a letter stroke) is sewn one way
            # along its whole length, so it can't flip between satin and a thin line where
            # its width wobbles around the limit. Its typical width (2 x area / edge length)
            # decides which.
            narrow = poly.buffer(-satin_half).is_empty
            typical = 2.0 * poly.area / poly.length if poly.length else 0.0
            if narrow and rules.get("satin", True) and typical >= LINE_FACTOR * rules["thin_line_mm"]:
                cols = try_satin(poly, rules)
                if cols:
                    add_satin(cols, color)
                    continue
            # "Wide" parts survive being shrunk by half the thin-line width. To stop a line
            # whose width hovers around the limit from flipping between fill and run along
            # its length, a wide part only counts if it is clearly wide somewhere.
            opened = poly.buffer(-half).buffer(half * 1.02)
            seeds = poly.buffer(-half * WIDE_FACTOR)
            wide = [c for c in polygons_of(opened)
                    if not seeds.is_empty and c.intersects(seeds) and c.area >= MIN_WIDE_AREA_MM2]
            if narrow and typical < LINE_FACTOR * rules["thin_line_mm"]:
                wide = []                                  # a thin line all the way along
            thin_pieces = polygons_of(poly.difference(unary_union(wide))) if wide else [poly]
            lines_here, thin_used = [], []
            for piece in thin_pieces:
                if piece.area < 0.15:
                    continue
                lines, width = centerlines(piece)
                if wide and width < FRINGE_MM:
                    continue        # a hair-thin fringe hanging off a solid shape: tracing noise
                if width >= MIN_THIN_WIDTH_MM and sum(l.length for l in lines) >= MIN_THIN_LENGTH_MM:
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
                cols = None
                if rules.get("satin", True) and part.buffer(-satin_half).is_empty:
                    cols = try_satin(part, rules)      # narrow all over: a satin candidate
                if cols:
                    add_satin(cols, color)
                else:
                    objects.append({"kind": "fill", "color": color, "poly": part,
                                    "area_mm2": float(part.area)})
            for line, width in lines_here:
                objects.append({"kind": "run", "color": color, "line": line,
                                "bean": bool(width >= rules["bean_line_mm"]),
                                "width_mm": width})

    if debug_dir is not None and objects:
        geoms = [o["poly"] if o["kind"] == "fill" else o["outline"] if o["kind"] == "satin"
                 else o["line"] for o in objects]
        img, to_px = canvas_for(unary_union(geoms).bounds, 8.0, color=(255, 255, 255))
        # fills = green, satin = blue with its centre line, runs = red (bean runs thicker)
        for o in objects:
            if o["kind"] == "fill":
                rings = [to_px(o["poly"].exterior.coords).astype(np.int32)]
                rings += [to_px(r.coords).astype(np.int32) for r in o["poly"].interiors]
                cv2.fillPoly(img, rings, (170, 220, 170), lineType=cv2.LINE_AA)
                cv2.polylines(img, rings, True, (60, 140, 60), 1, cv2.LINE_AA)
        for o in objects:
            if o["kind"] == "satin":
                for p in polygons_of(o["outline"]):
                    ring = to_px(p.exterior.coords).astype(np.int32)
                    cv2.fillPoly(img, [ring], (240, 200, 150), lineType=cv2.LINE_AA)
                    cv2.polylines(img, [ring], True, (200, 120, 40), 1, cv2.LINE_AA)
                cv2.polylines(img, [to_px(o["center"]).astype(np.int32)], False,
                              (160, 60, 0), 1, cv2.LINE_AA)
        for o in objects:
            if o["kind"] == "run":
                cv2.polylines(img, [to_px(o["line"].coords).astype(np.int32)], False,
                              (40, 40, 220), 3 if o["bean"] else 1, cv2.LINE_AA)
        save_debug(debug_dir, "06_classify.png", img)

    return {"objects": objects, "dropped_details": dropped, "order": order}
