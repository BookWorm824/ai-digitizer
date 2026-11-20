"""Stage 6 — Classify.

Decide how each traced shape will be sewn. The rules come from how professional digitizers
work (see reports/Embroidery digitizing rules.md):

  * Satin is only for column-shaped parts: at least `satin_min_mm` (1.2) wide, at most
    `satin_max_mm` (7) wide, longer than they are wide, and of fairly even width. "Satin
    everything" is wrong: blobs, stubby shapes, wedges and webs of thin outlines are not
    columns.
  * Lines thinner than satin can carry are sewn as a bean (triple) stitch along their centre,
    the standard for line art. Hair-thin tracing noise is dropped.
  * Areas wider than satin can carry are filled.
  * A web of thin outlines with many openings (the outline drawing of a building, say) is sewn
    as bean lines along its centre, not as one huge fill riddled with holes.
  * Lettering is found as rows of similar-sized, stroke-like shapes, and each row is sewn ONE
    way: the eye notices instantly when one letter is satin and the next is fill. Row height
    then sets the small-text rules (bean lettering under `text_run_mm`, no underlay under 5 mm,
    lighter density).

Satin columns are built along each shape's centre line, but every stitch end is snapped to the
shape's real edge, so edges are crisp instead of wobbling with the centre line. Columns are
split at sharp bends (like the corners of an E) rather than fanning round them.

This stage also decides the sewing order of the colors and lets each color reach a little way
under the colors sewn after it, so no gap opens where colors meet.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from skimage.morphology import skeletonize

from .common import canvas_for, dist, polygons_of, save_debug

SKEL_PPM = 12.0          # resolution used to find centre lines
MIN_RUN_MM = 1.2         # centre-line pieces shorter than this are spurs, not lines
NOISE_WIDTH_MM = 0.3     # lines thinner than this are tracing noise, not drawing
MIN_THIN_LENGTH_MM = 3.0 # ...and so are thin bits shorter than this
FRINGE_MM = 0.6          # thin slivers on the edge of a solid shape are part of that shape
SATIN_STEP_MM = 0.25     # spacing of the sample points along a satin column's centre line
SATIN_COVER = 0.8        # columns must cover this share of a shape for it to be sewn as satin
SATIN_SPILL = 0.25       # ...and may not spill outside it by more than this share
CORNER_DEG = 70.0        # a centre line turning this sharply is split into two columns
                         # (Wilcom/Soft Sight: interior angles under ~110 deg are mitred)
JOIN_STRAIGHT_DEG = 45.0 # at a junction, branches this close to straight are one stroke
END_MIN_MM = 0.6         # satin stops where the column gets narrower than this (rule 26)
GLYPH_MAX_COLUMNS = 6    # a letter needing more columns than this is not clean satin
ROW_MEAN_COLUMNS = 4.0   # a word whose letters average more columns than this looks patched
GLYPH_MAX_OVERLAP = 1.35 # ...nor one whose columns pile up on each other this much
ROW_SATIN_SHARE = 0.85   # a word is satin only if nearly every letter is clean satin
SERIF_SMALL_MM = 8.0     # under this cap height, hairline serifs are dropped (rule 5)
NETWORK_HOLES = 3        # a narrow shape with this many openings is a web of outlines
NETWORK_MAX_MM = 2.0     # ...and is sewn as lines when its strokes are thinner than this
GLYPH_STROKE_RATIO = 0.42  # stroke width / height of a letter (a solid square is 0.5)


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


def _end_dir(path: list, end: int) -> np.ndarray:
    """Outward direction of a pixel path at one of its ends (0 = start, 1 = finish)."""
    k = min(len(path) - 1, 6)
    a = np.array(path[0] if end == 0 else path[-1], float)
    b = np.array(path[k] if end == 0 else path[-1 - k], float)
    d = a - b
    n = np.linalg.norm(d)
    return d / n if n > 1e-9 else d


def _join_paths(paths: list[list], tol: float = 2.5, max_turn: float = 180.0) -> list[list]:
    """Join branches whose ends meet. Where several meet (a junction), the pair that runs
    most nearly straight through is joined first, so a T becomes one bar plus one stem rather
    than an arbitrary bend. Joins turning more than `max_turn` degrees are not made."""
    paths = [list(p) for p in paths if len(p) >= 2]
    if len(paths) > 1:
        # A junction is often a little cluster of pixels joined by 2-3 pixel stubs; those
        # stubs are not strokes. Their ends are within `tol`, so the strokes still meet.
        longest = max(len(p) for p in paths)
        paths = [p for p in paths if len(p) > 3 or longest <= 3]
    n = len(paths)
    if n < 2:
        return paths
    ends = np.array([paths[i][0] if e == 0 else paths[i][-1] for i in range(n) for e in (0, 1)], float)
    dirs = np.array([_end_dir(paths[i], e) for i in range(n) for e in (0, 1)])
    d2 = ((ends[:, None, :] - ends[None, :, :]) ** 2).sum(axis=2)
    cand = []
    for a, b in zip(*np.nonzero(d2 <= tol * tol)):
        if a < b and a // 2 != b // 2:
            cos = float(np.clip(-(dirs[a] @ dirs[b]), -1.0, 1.0))  # straight through: dirs opposite
            turn = math.degrees(math.acos(cos))
            if turn <= max_turn:
                cand.append((turn, a, b))
    cand.sort()
    link = {}                       # end id -> end id it is joined to
    chain = list(range(n))          # union-find over paths, so joins never close a loop

    def root(x):
        while chain[x] != x:
            chain[x] = chain[chain[x]]
            x = chain[x]
        return x

    for _turn, a, b in cand:
        if a in link or b in link or root(a // 2) == root(b // 2):
            continue
        link[a], link[b] = b, a
        chain[root(a // 2)] = root(b // 2)
    out, used = [], set()
    for start in range(n):
        if start in used:
            continue
        # walk to one free end of this chain
        cur, end, seen = start, 0, set()
        while (2 * cur + end) in link and cur not in seen:
            seen.add(cur)
            nxt = link[2 * cur + end]
            cur, end = nxt // 2, 1 - (nxt % 2)
        # now walk forward from that free end, concatenating
        merged, enter = [], end
        while True:
            used.add(cur)
            seg = paths[cur] if enter == 0 else paths[cur][::-1]
            merged.extend(seg if not merged else seg[1:])
            exit_end = 2 * cur + (1 - enter)
            if exit_end not in link or link[exit_end] // 2 in used:
                break
            nxt = link[exit_end]
            cur, enter = nxt // 2, nxt % 2
        out.append(merged)
    return out


def _skeleton(piece: Polygon, prune_hairlines: bool = True):
    """(pruned branches in pixels, branch-end pixels, distance map, to_mm) or None."""
    mask, to_mm = _raster(piece)
    if not mask.any():
        return None
    skel = skeletonize(mask.astype(bool))
    if not skel.any():
        return None
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    paths, ends = _trace_skeleton(skel)
    # Thinning leaves short spurs running into corners and serifs. A branch that ends in
    # mid-air and is no longer than the stroke is wide where it starts is one of those.
    # Also prune hairline side branches (serifs): much thinner than the stroke they hang off
    # and not much longer than it is wide (research rule 40).
    kept = []
    for path in paths:
        a_end, b_end = path[0] in ends, path[-1] in ends
        if a_end != b_end and len(paths) > 1:
            root = path[-1] if a_end else path[0]
            r = max(dt[root], 1.0)
            if len(path) <= 1.6 * r:
                continue
            if prune_hairlines:
                thick = float(np.median([dt[y, x] for y, x in path]))
                if thick < 0.65 * r and len(path) <= 3.5 * r:
                    continue
        kept.append(path)
    # After pruning, a junction whose side branches were all spurs (a stem meeting its
    # serifs) is really a free end: count any branch end that no other kept branch reaches.
    tips = [(p[0], p[-1]) for p in kept if len(p) > 3]
    free = set(ends)
    for n, (a, b) in enumerate(tips):
        for pt in (a, b):
            meets = any(dist(pt, q) <= 2.5 for m, (c, d) in enumerate(tips) if m != n for q in (c, d))
            if not meets:
                free.add(pt)
    return kept, free, dt, to_mm


def centerlines(piece: Polygon, min_len: float = MIN_RUN_MM) -> tuple[list[LineString], float]:
    """Centre lines of a thin shape, in mm, plus the shape's average width in mm. Every real
    branch is kept (thin lines have no serifs to drop)."""
    sk = _skeleton(piece, prune_hairlines=False)
    if sk is None:
        return [], 0.0
    paths, _ends, dt, to_mm = sk
    on = [dt[y, x] for p in paths for y, x in p]
    width = float(2.0 * np.mean(on) / SKEL_PPM) if on else 0.0
    lines = []
    for path in _join_paths(paths):
        if len(path) < 2:
            continue
        line = LineString([to_mm(y, x) for y, x in path]).simplify(0.1)
        if line.length >= min_len:
            lines.append(line)
    return lines, width


def typical_width(poly) -> float:
    """The width of a strip-like shape: 2 x area / outline length."""
    return 2.0 * poly.area / poly.length if poly.length else 0.0


# ---------------------------------------------------------------------------------- satin

def _smooth(a: np.ndarray, k: int = 5) -> np.ndarray:
    if len(a) < k or k < 2:
        return a
    pad = k // 2
    padded = np.concatenate([np.repeat(a[:1], pad, axis=0), a, np.repeat(a[-1:], pad, axis=0)])
    kernel = np.ones(2 * pad + 1) / (2 * pad + 1)
    if a.ndim == 1:
        return np.convolve(padded, kernel, mode="valid")
    return np.column_stack([np.convolve(padded[:, i], kernel, mode="valid")
                            for i in range(a.shape[1])])


def _normals(center: np.ndarray, window: int) -> np.ndarray:
    """Unit normals along a centre line, from a well-smoothed direction so stitch angles turn
    gently instead of following every pixel step."""
    t = np.gradient(_smooth(center, window), axis=0)
    n = np.linalg.norm(t, axis=1, keepdims=True)
    t = t / np.where(n < 1e-9, 1.0, n)
    return np.column_stack([-t[:, 1], t[:, 0]])


def column_rails(center: np.ndarray, half: np.ndarray):
    """The two edges of a satin column from its centre line and half-widths."""
    normal = _normals(center, 3)
    return center + normal * half[:, None], center - normal * half[:, None]


def column_outline(center: np.ndarray, half: np.ndarray, left=None, right=None) -> Polygon:
    """The area a satin column covers. A column that closes on itself (a ring, the outline of
    a letter) covers a band with a hole in the middle, not the whole disc."""
    if left is None or right is None:
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


def _snap_to_edges(center: np.ndarray, half: np.ndarray, piece: Polygon):
    """For each point of a centre line, find where the line across the column (along the
    normal) meets the shape's real edges. Returns (left, right, ok) where ok marks the points
    that found both edges at a believable distance."""
    med = float(np.median(half))
    window = max(5, int(1.2 * 2 * med / SATIN_STEP_MM) | 1)
    normal = _normals(center, window)
    reach = 1.45 * np.maximum(half, med) + 0.35
    a = center - normal * reach[:, None]
    b = center + normal * reach[:, None]
    lines = shapely.linestrings(np.stack([a, b], axis=1))
    hits = shapely.intersection(lines, piece)
    left = center + normal * half[:, None]
    right = center - normal * half[:, None]
    ok = np.zeros(len(center), bool)
    for i, g in enumerate(hits):
        if g.is_empty:
            continue
        parts = [g] if g.geom_type == "LineString" else [p for p in getattr(g, "geoms", [])
                                                        if p.geom_type == "LineString"]
        c = Point(center[i])
        best = min(parts, key=lambda p: p.distance(c), default=None)
        if best is None or best.distance(c) > 0.25:
            continue
        (x0, y0), (x1, y1) = best.coords[0], best.coords[-1]
        p0, p1 = np.array([x0, y0]), np.array([x1, y1])
        s0, s1 = float((p0 - center[i]) @ normal[i]), float((p1 - center[i]) @ normal[i])
        if s0 < s1:
            p0, p1, s0, s1 = p1, p0, s1, s0
        width = s0 - s1
        if width < 0.2:
            continue
        if width > 1.6 * 2 * max(half[i], med):
            ok[i] = True                  # ran into a crossing stroke or serif: still inside
            continue                      # the shape, so keep the estimated edges
        left[i], right[i], ok[i] = p0, p1, True
    return left, right, ok


def _split_at_corners(center: np.ndarray, med_half: float) -> list[tuple[int, int]]:
    """Index ranges of a centre line, split where it bends by CORNER_DEG or more within about
    one stroke width. Each piece runs a little past the corner so the two overlap there."""
    n = len(center)
    k = max(3, int(round(2 * med_half / SATIN_STEP_MM)))
    if n < 2 * k + 3:
        return [(0, n)]
    head = np.arctan2(*np.gradient(_smooth(center, 3), axis=0).T[::-1])
    turn = np.zeros(n)
    for i in range(k, n - k):
        d = math.degrees(abs((head[i + k] - head[i - k] + math.pi) % (2 * math.pi) - math.pi))
        turn[i] = d
    cuts = []
    i = k
    while i < n - k:
        if turn[i] >= CORNER_DEG:
            j = i
            while j + 1 < n - k and turn[j + 1] >= turn[j]:
                j += 1                     # walk to the sharpest point of this bend
            cuts.append(j)
            i = j + 2 * k
        else:
            i += 1
    if not cuts:
        return [(0, n)]
    over = max(2, int(round(med_half / SATIN_STEP_MM)))
    edges = [0] + cuts + [n]
    return [(max(0, edges[q] - (over if q else 0)), min(n, edges[q + 1] + (over if q + 1 < len(edges) - 1 else 0)))
            for q in range(len(edges) - 1)]


def columns(piece: Polygon) -> list[dict]:
    """Satin columns along the centre lines of a narrow shape. Each is
    {"center", "half", "left", "right": Nx2 / N arrays (mm), "outline": Polygon}."""
    sk = _skeleton(piece)
    if sk is None:
        return []
    paths, ends, dt, to_mm = sk
    out = []
    for path in _join_paths(paths, max_turn=JOIN_STRAIGHT_DEG):
        if len(path) < 4:
            continue
        free_end = (tuple(path[0]) in ends, tuple(path[-1]) in ends)
        pts = np.array([to_mm(y, x) for y, x in path])
        rad = np.array([dt[y, x] for y, x in path]) / SKEL_PPM
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        if cum[-1] < 1.0:
            continue
        s = np.linspace(0.0, cum[-1], max(4, int(cum[-1] / SATIN_STEP_MM) + 1))
        center = _smooth(np.column_stack([np.interp(s, cum, pts[:, 0]),
                                          np.interp(s, cum, pts[:, 1])]))
        half = np.interp(s, cum, rad)
        half = _smooth(np.minimum(half, 1.25 * np.median(half)))   # no bulge where arms meet
        closed = dist(center[0], center[-1]) < 2 * SATIN_STEP_MM and cum[-1] > 4.0
        if not closed:
            # The centre line stops about half a width short of a free end: carry it on
            # straight (the last bit tends to hook toward a corner). Edge snapping then stops
            # the column exactly where the shape ends.
            for end in (0, -1):
                if not free_end[end]:
                    continue
                cut = int(half[end] / SATIN_STEP_MM)
                if len(center) > 3 * cut + 6 and cut > 0:
                    center = center[cut:] if end == 0 else center[:-cut]
                    half = half[cut:] if end == 0 else half[:-cut]
                else:
                    cut = 0
                k = min(len(center) - 1, max(4, cut))
                d = center[end] - center[k if end == 0 else -1 - k]
                nrm = np.linalg.norm(d)
                if nrm < 1e-9:
                    continue
                steps = cut + int(1.2 * half[end] / SATIN_STEP_MM)
                extra = np.array([center[end] + d / nrm * SATIN_STEP_MM * (j + 1)
                                  for j in range(steps)]).reshape(-1, 2)
                if end == 0:
                    center = np.vstack([extra[::-1], center])
                    half = np.concatenate([np.full(steps, half[0]), half])
                else:
                    center = np.vstack([center, extra])
                    half = np.concatenate([half, np.full(steps, half[-1])])
        med = float(np.median(half))
        ranges = [(0, len(center))] if closed else _split_at_corners(center, med)
        for a, b in ranges:
            c, h = center[a:b], half[a:b]
            if len(c) < 4:
                continue
            left, right, ok = _snap_to_edges(c, h, piece)
            # where the line across ran off the end of the shape, or the column has thinned
            # to a point, the column ends: satin never tapers to nothing
            ok &= np.linalg.norm(left - right, axis=1) >= END_MIN_MM
            idx = np.nonzero(ok)[0]
            if len(idx) < 3:
                continue
            lo, hi = idx[0], idx[-1] + 1
            c, h, left, right = c[lo:hi], h[lo:hi], left[lo:hi], right[lo:hi]
            left, right = _smooth(left, 3), _smooth(right, 3)
            h = np.linalg.norm(left - right, axis=1) / 2
            out.append({"center": c, "half": h, "left": left, "right": right,
                        "outline": column_outline(c, h, left, right)})
    return out


def try_satin(piece: Polygon, rules: dict, drop_hairlines: bool = False) -> list[dict] | None:
    """Satin columns for a narrow shape, or None if columns wouldn't reproduce it well.
    With `drop_hairlines`, hair-thin parts (the serifs of small letters) don't count against
    it: at small sizes digitizers drop hairline serifs rather than stitch them (rule 5)."""
    if piece.area < 2.0:
        return None
    target = piece
    if drop_hairlines:
        r = 0.35 * rules["satin_min_mm"]
        core = piece.buffer(-r).buffer(r * 1.05).intersection(piece)
        if not core.is_empty and core.area > 0.6 * piece.area:
            target = core
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
        if length >= 1.2 * width and width >= 0.6 * rules["satin_min_mm"] and even:
            cols.append(c)
    if not cols:
        return None
    try:
        cover = unary_union([c["outline"] for c in cols])
        covered = cover.intersection(target).area
        spill = cover.difference(piece.buffer(0.3)).area
    except Exception:  # noqa: BLE001 - odd geometry: just fill it instead
        return None
    if covered < SATIN_COVER * target.area or spill > SATIN_SPILL * piece.area:
        return None
    return cols


# ------------------------------------------------------------------------------ lettering

def text_rows(shapes: dict, order: list, rules: dict) -> dict:
    """Find lettering: rows of at least three similar-sized, stroke-like shapes of one color
    sitting side by side. Returns {(color, index): {"row": n, "height": mm}}."""
    out, row_id = {}, 0
    for color in order:
        cands = []
        for i, p in enumerate(shapes[color]):
            minx, miny, maxx, maxy = p.bounds
            h = maxy - miny
            if not (1.2 <= h <= 60.0) or p.area < 0.3:
                continue
            if typical_width(p) > GLYPH_STROKE_RATIO * h:
                continue                   # solid blobs and squares are not letters
            cands.append((i, minx, maxx, miny, maxy, h))
        if len(cands) < 3:
            continue
        parent = list(range(len(cands)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for a in range(len(cands)):
            for b in range(a + 1, len(cands)):
                _, ax0, ax1, ay0, ay1, ah = cands[a]
                _, bx0, bx1, by0, by1, bh = cands[b]
                if abs(ah - bh) > 0.35 * max(ah, bh):
                    continue
                overlap = min(ay1, by1) - max(ay0, by0)
                if overlap < 0.6 * min(ah, bh):
                    continue
                gap = max(bx0 - ax1, ax0 - bx1)
                if gap > 1.6 * max(ah, bh):
                    continue
                parent[find(a)] = find(b)
        groups = {}
        for a in range(len(cands)):
            groups.setdefault(find(a), []).append(cands[a])
        for members in groups.values():
            if len(members) < 3:
                continue
            height = float(np.median([m[5] for m in members]))
            for m in members:
                out[(color, m[0])] = {"row": row_id, "height": height}
            row_id += 1
    return out


def _text_style(height: float, rules: dict) -> dict:
    """Underlay and density for lettering of this cap height (Wilcom, Hatch, Melco): no
    underlay under 5 mm, a centre run up to 10 mm, the normal ladder above; small letters are
    sewn a little more openly so they don't fill in."""
    style = {"underlay": None, "spacing_scale": 1.0, "comp_scale": 1.0}
    if height < 5.0:
        style.update(underlay="none", spacing_scale=1.12, comp_scale=0.6)
    elif height < 10.0:
        style.update(underlay="center", spacing_scale=1.06, comp_scale=0.8)
    return style


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
    satin_min = rules["satin_min_mm"]
    satin_half = rules["satin_max_mm"] / 2.0
    min_area = 0.6 * rules["min_detail_mm"] ** 2
    use_satin = rules.get("satin", True)

    order = sewing_order(vec["layers"], rules["satin_max_mm"])
    shapes = underlap(vec["layers"], order, rules["overlap_mm"])
    rows = text_rows(shapes, order, rules)

    objects, dropped = [], 0

    def add_satin(cols, color, style=None):
        for c in cols:
            o = {"kind": "satin", "color": color, "center": c["center"], "half": c["half"],
                 "left": c.get("left"), "right": c.get("right"), "outline": c["outline"],
                 "width_mm": 2 * float(np.median(c["half"])), "area_mm2": float(c["outline"].area)}
            if style:
                o.update(style)
            objects.append(o)

    def add_fill(poly, color, angle=None, style=None):
        o = {"kind": "fill", "color": color, "poly": poly, "area_mm2": float(poly.area)}
        if angle is not None:
            o["angle"] = angle
        if style:
            o.update(style)
        objects.append(o)

    def add_lines(piece, color, stitch_mm=None, min_len=MIN_THIN_LENGTH_MM):
        """Bean-stitch lines along the centre of a thin shape. Returns False if it had none."""
        lines, width = centerlines(piece, min(MIN_RUN_MM, min_len))
        if width < NOISE_WIDTH_MM or sum(ln.length for ln in lines) < min_len:
            return False
        for ln in lines:
            o = {"kind": "run", "color": color, "line": ln, "width_mm": width,
                 "bean": bool(width >= rules["bean_line_mm"])}
            if stitch_mm:
                o["stitch_mm"] = stitch_mm
            objects.append(o)
        return True

    def sew_narrow(part, color):
        """A part that is narrow all over: satin if it is a real column, bean lines if it is
        thin or a web of outlines, otherwise a fill."""
        nonlocal dropped
        w = typical_width(part)
        if w < satin_min or (len(part.interiors) >= NETWORK_HOLES and w < NETWORK_MAX_MM):
            if add_lines(part, color):
                return
            if part.area >= min_area and w >= satin_min * 0.8:
                add_fill(part, color)
            else:
                dropped += 1
            return
        cols = try_satin(part, rules) if use_satin else None
        if cols:
            add_satin(cols, color)
        elif part.area >= min_area:
            add_fill(part, color)
        else:
            dropped += 1

    # ---- letter counters: blank areas sewn because they are shut inside the design are left
    # open when they sit inside a letter (the inside of an O is fabric, not thread)
    done = set()
    inside = vec.get("inside_color")
    if inside is not None and inside in shapes and rows:
        letters = [Polygon(shapes[c][i].exterior).buffer(0.3) for c, i in rows]
        for i, blank in enumerate(shapes[inside]):
            if any(L.contains(blank) for L in letters):
                done.add((inside, i))
    by_row = {}
    for key, info in rows.items():
        by_row.setdefault(info["row"], []).append(key)
    for row, keys in by_row.items():
        height = rows[keys[0]]["height"]
        if height < rules.get("drop_text_mm", 0.0):
            done.update(keys)          # the user chose to leave this tiny text out
            continue
        style = _text_style(height, rules)
        glyphs = [(c, i, shapes[c][i]) for c, i in sorted(keys, key=lambda k: shapes[k[0]][k[1]].bounds[0])]
        if height < rules["text_run_mm"]:
            # too small for satin: run-stitch lettering, short stitches so curves keep shape
            for c, i, g in glyphs:
                if add_lines(g, c, stitch_mm=1.5, min_len=0.8):
                    done.add((c, i))
            continue
        tried = []
        small = height < SERIF_SMALL_MM
        for c, i, g in glyphs:
            cols = try_satin(g, rules, drop_hairlines=small) if use_satin else None
            if cols and (len(cols) > GLYPH_MAX_COLUMNS
                         or sum(col["outline"].area for col in cols) > GLYPH_MAX_OVERLAP * g.area):
                cols = None              # a patchwork of short columns: fill looks cleaner
            tried.append((c, i, g, cols))
        satin_share = sum(1 for t in tried if t[3]) / len(tried)
        passing = [len(t[3]) for t in tried if t[3]]
        if passing and np.mean(passing) > ROW_MEAN_COLUMNS:
            satin_share = 0.0            # heavy serif lettering: a clean fill beats a patchwork
        angle = 45.0 if row % 2 == 0 else 135.0
        # One way for the whole word: satin when (nearly) every letter is a clean set of
        # columns; otherwise small letters become run lettering (a fill across 1 mm strokes
        # looks ragged) and large letters are all filled at one shared angle.
        if satin_share >= ROW_SATIN_SHARE:
            mode = "satin"
        elif small:
            mode = "lines"
        else:
            mode = "fill"
        for c, i, g, cols in tried:
            if mode == "satin" and cols:
                add_satin(cols, c, style)
            elif mode == "lines" and add_lines(g, c, stitch_mm=1.5, min_len=0.8):
                pass
            else:
                add_fill(g, c, angle, style)
            done.add((c, i))

    # ---- everything else
    for color in order:
        if color not in shapes:
            continue
        for i, poly in enumerate(shapes[color]):
            if (color, i) in done:
                continue
            if poly.buffer(-satin_half).is_empty:          # narrow all over
                sew_narrow(poly, color)
                continue
            # Parts wider than satin can carry are filled. What is left over are narrow arms
            # and outlines hanging off them: sewn as satin, bean lines or fill on their own.
            core = poly.buffer(-satin_half).buffer(satin_half * 1.02)
            core = core.intersection(poly)
            rest = poly.difference(core.buffer(0.05))
            arms = []
            edge = core.buffer(0.15)
            for piece in polygons_of(rest):
                # Corner trimmings and edge slivers hug the area along much of their outline;
                # a real arm or outline only touches it at one end.
                hug = piece.boundary.intersection(edge).length / max(piece.length, 1e-9)
                if not (typical_width(piece) < FRINGE_MM or piece.area < min_area or hug > 0.3):
                    arms.append(piece)
            # the area is everything that isn't a real arm (corner trimmings stay with it)
            core = poly.difference(unary_union(arms).buffer(0.05)) if arms else poly
            for part in polygons_of(core.buffer(0)):
                if part.area >= min_area:
                    add_fill(part, color)
                else:
                    dropped += 1
            for piece in arms:
                sew_narrow(piece, color)

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

    heights = sorted({round(rows[k[0]]["height"], 2) for k in by_row.values()})
    kept = [h for h in heights if h >= rules.get("drop_text_mm", 0.0)]
    return {"objects": objects, "dropped_details": dropped, "order": order,
            "text_rows": len(by_row), "text_heights": kept}
