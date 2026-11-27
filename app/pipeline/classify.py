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
import re
from pathlib import Path

import cv2
import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import substring, unary_union
from shapely.prepared import prep
from skimage.morphology import skeletonize

from . import aiplan, geometry, lettering, linenet, rings
from .common import canvas_for, dist, polygons_of, save_debug

SKEL_PPM = 12.0          # resolution used to find centre lines
MIN_RUN_MM = 1.2         # centre-line pieces shorter than this are spurs, not lines
NOISE_WIDTH_MM = 0.3     # lines thinner than this are tracing noise, not drawing
MIN_THIN_LENGTH_MM = 3.0 # ...and so are thin bits shorter than this
FRINGE_MM = 0.6          # thin slivers on the edge of a solid shape are part of that shape
SATIN_STEP_MM = 0.25     # spacing of the sample points along a satin column's centre line
SATIN_COVER = 0.8        # columns must cover this share of a shape for it to be sewn as satin
RING_EVEN_RATIO = 1.6    # a satin ring round a hole may vary this much in width, no more
COVERED_EDGE_SHARE = 0.8 # a column edge this much under later colors gets no pull comp
PATCH_MIN_MM2 = 0.8      # a piece of a satin shape the columns miss this big is filled
PATCH_MIN_WIDTH_MM = 0.5 # ... if it is at least this wide
PATCH_TUCK_MM = 0.3      # ... reaching this far under the column ends
SATIN_SPILL = 0.25       # ...and may not spill outside it by more than this share
CORNER_SHARP_SHARE = 0.7 # ... if most of that turn is within half the distance (a corner)
CORNER_DEG = 70.0        # a centre line turning this sharply is split into two columns
                         # (Wilcom/Soft Sight: interior angles under ~110 deg are mitred)
JOIN_STRAIGHT_DEG = 45.0 # at a junction, branches this close to straight are one stroke
END_MIN_MM = 0.6         # satin stops where the column gets narrower than this (rule 26)
GLYPH_MAX_COLUMNS = 6    # a letter needing more columns than this is not clean satin
ROW_MEAN_COLUMNS = 4.0   # a word whose letters average more columns than this looks patched
GLYPH_MAX_OVERLAP = 1.35 # ...nor one whose columns pile up on each other this much
ROW_SATIN_SHARE = 1.0    # a word is satin only if every letter is clean satin: one fill
                         # letter among satin ones is the first thing the eye catches
HEAVY_LETTER_RATIO = 0.31   # strokes heavier than this x the letter height: fill, not satin
FILL_TEXT_MM = 2.3       # traced letters with strokes at least this wide fill cleanly, so
                         # they keep the picture's own lettering; thinner ones are retyped
SERIF_SMALL_MM = 8.0     # under this cap height, hairline serifs are dropped (rule 5)
NETWORK_HOLES = 3        # a narrow shape with this many openings is a web of outlines
NETWORK_MAX_MM = 2.0     # ...and is sewn as lines when its strokes are thinner than this
GLYPH_STROKE_RATIO = 0.42  # stroke width / height of a letter (a solid square is 0.5)
BOLD_TEXT_STROKE = 0.2   # traced letters with strokes this share of their height are bold
BIG_RETYPE_STROKE_MM = 1.2  # words of RETYPE_AUTO_MM and up that would be filled are retyped
                            # only when their strokes are thinner than this
RETYPE_AUTO_MM = 8.0     # "auto" retypes known words shorter than this (research rules 5, 6)


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
    sharp = np.zeros(n)
    k2 = max(2, k // 2)

    def bend(i, w):
        return math.degrees(abs((head[i + w] - head[i - w] + math.pi) % (2 * math.pi) - math.pi))

    for i in range(k, n - k):
        turn[i] = bend(i, k)
        sharp[i] = bend(i, k2)
    cuts = []
    i = k
    while i < n - k:
        if turn[i] >= CORNER_DEG:
            j = i
            while j + 1 < n - k and turn[j + 1] >= turn[j]:
                j += 1                     # walk to the sharpest point of this bend
            # a corner, not a smooth tight curve (a narrow ring round an ear): most of the
            # turn happens within half the distance. A ring is one column, or its pieces'
            # ends fan where they meet.
            if sharp[max(k, i - k):min(n - k, j + k) + 1].max() >= CORNER_SHARP_SHARE * turn[j]:
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
    return _end_under_through(out)


JOINT_TUCK_MM = 0.3      # a column ending at a joint reaches this far under the through column


def _end_under_through(cols: list) -> list:
    """Where a column ends at a joint that another column runs straight through (branches
    running on are joined into one column), it stops just under that column's edge instead
    of reaching its middle: every branch reaching the middle stacks a layer of satin per
    branch (a knot where a palm's fronds meet). Training session "Line art at left chest":
    one stroke continuous on top, the rest ending under it, at most two layers at a point."""
    if len(cols) < 2:
        return cols
    lines = [LineString(c["center"]) for c in cols]
    for k, col in enumerate(cols):
        c, h = col["center"], col["half"]
        if len(c) < 6:
            continue
        cut = [0, len(c)]
        for end in (0, -1):
            P = Point(c[end])
            best = None
            for j, other in enumerate(cols):
                if j == k:
                    continue
                d = lines[j].distance(P)
                oh = float(np.median(other["half"]))
                ends_far = min(Point(other["center"][0]).distance(P),
                               Point(other["center"][-1]).distance(P)) > 2.0 * oh
                if d < 0.5 * oh and ends_far and (best is None or oh > best):
                    best = oh
            if best is None:
                continue
            back = best - JOINT_TUCK_MM
            seg = np.linalg.norm(np.diff(c, axis=0), axis=1)
            run = np.concatenate([[0.0], np.cumsum(seg if end == 0 else seg[::-1])])
            n = int(np.searchsorted(run, back))
            if end == 0:
                cut[0] = max(cut[0], n)
            else:
                cut[1] = min(cut[1], len(c) - n)
        if cut != [0, len(c)] and cut[1] - cut[0] >= 4:
            a, b = cut
            col.update(center=c[a:b], half=h[a:b], left=col["left"][a:b],
                       right=col["right"][a:b])
            col["outline"] = column_outline(col["center"], col["half"], col["left"], col["right"])
    return cols


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
    for c in cols:
        ring = dist(c["center"][0], c["center"][-1]) < 2 * SATIN_STEP_MM
        lo, hi = np.percentile(c["half"], [10, 90])
        if ring and hi > RING_EVEN_RATIO * lo:
            # a band all the way round a hole that is wide in one place and narrow in
            # another (a solid badge with a shape cut out of it): satin turning round it
            # fans and bunches; a fill with the hole in it sews it cleanly
            return None
    if len(cols) == 1:
        c = cols[0]
        length = float(np.linalg.norm(np.diff(c["center"], axis=0), axis=1).sum())
        if length < 2.0 * 2 * float(np.median(c["half"])):
            return None          # a small block (a window), not a column: a fill looks cleaner
    try:
        cover = unary_union([c["outline"] for c in cols])
        covered = cover.intersection(target).area
        spill = cover.difference(piece.buffer(0.3)).area
    except Exception:  # noqa: BLE001 - odd geometry: just fill it instead
        return None
    if covered < SATIN_COVER * target.area or spill > SATIN_SPILL * piece.area:
        return None
    # What the columns miss and is big enough to see (the slanted end of a ring cut off by
    # a shape in front of it, a corner): a small patch of fill under the column ends, or
    # the fabric shows through there.
    try:
        missed = target.difference(cover.buffer(0.05))
    except Exception:  # noqa: BLE001
        missed = None
    patches = []
    for gap in polygons_of(missed) if missed is not None else []:
        if gap.area >= PATCH_MIN_MM2 and not gap.buffer(-PATCH_MIN_WIDTH_MM / 2).is_empty:
            patches.extend(polygons_of(gap.buffer(PATCH_TUCK_MM).intersection(piece)))
    if patches:
        cols[0] = dict(cols[0], patches=patches)
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


def _no_slivers(p: Polygon) -> Polygon:
    """Without the hairline holes the overlap can leave inside a shape (a seam where two
    reaches under a thin outline meet, area near zero). They aren't openings; counted as
    one they make a solid roof measure as a hollow outline."""
    keep = [h for h in p.interiors if Polygon(h).area >= 0.05]
    return p if len(keep) == len(p.interiors) else Polygon(p.exterior, keep)


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
                grown.extend(_no_slivers(q) for q in (polygons_of(p.union(reach)) or [p]))
            out[color] = grown
        mine = unary_union(polys)
        later = mine if later is None else later.union(mine)
    return out


SEW_ON_TOP_STROKE_MM = 1.5      # holes for later details thinner than this are not cut
SEW_ON_TOP_COVER = 0.85         # ... when later colors cover at least this share of the hole


def sew_on_top(shapes: dict, layers: list, order: list[int]) -> dict:
    """Close the holes a fill has for small, thin details sewn after it (small lettering, a
    thin outline drawn inside a colored area): the fill runs on unbroken and the detail is
    sewn on top. Cut out, such holes split the fill into many small sections with cuts and
    travel between them, and the slightest shift on the fabric opens a gap between the
    letters and the ground; pros knock out only lettering with strokes clearly over a
    millimetre (Hatch / Wilcom knockout guides, Ink/Stitch forum on text over fills).
    Pieces of the same color inside a closed hole (the counter of an O) join the fill."""
    by_color = {layer["color"]: layer["polys"] for layer in layers}
    out = dict(shapes)
    for k, color in enumerate(order):
        after = [p for c in order[k + 1:] for p in by_color.get(c, [])]
        if not after:
            continue
        later = unary_union(after)
        grown, closed = [], []
        for p in out[color]:
            keep = []
            for ring in p.interiors:
                hole = Polygon(ring)
                if hole.area < 0.05:
                    keep.append(ring)
                    continue
                covered = hole.intersection(later).area / hole.area
                stroke = 2.0 * hole.area / max(hole.length, 1e-6)
                if covered >= SEW_ON_TOP_COVER and stroke < SEW_ON_TOP_STROKE_MM:
                    closed.append(hole)
                else:
                    keep.append(ring)
            grown.append(p if len(keep) == len(p.interiors) else Polygon(p.exterior, keep))
        if closed:
            shut = unary_union(closed).buffer(0.05)
            grown = [p for p in grown if not shut.contains(p)]
        out[color] = grown
    return out


def _line_width(seg: np.ndarray, piece) -> float:
    """How wide a line of a web was drawn: the typical length of the cut straight across it
    through the drawn shape, measured away from its ends (junctions are fatter). Measuring
    across, not from the centre line outward, gives the same answer when straightening has
    moved the centre line a little to one side."""
    ls = LineString(seg)
    if ls.length < 0.6:
        return 0.0
    lo, hi = 0.2 * ls.length, 0.8 * ls.length
    n = max(3, int((hi - lo) / 0.5) + 1)
    out = []
    for k in range(n):
        s = lo + (hi - lo) * k / (n - 1)
        p, q = ls.interpolate(max(0.0, s - 0.15)), ls.interpolate(min(ls.length, s + 0.15))
        tx, ty = q.x - p.x, q.y - p.y
        t = math.hypot(tx, ty)
        if t < 1e-9:
            continue
        c = ls.interpolate(s)
        nx, ny = -ty / t, tx / t
        cut = LineString([(c.x - nx * 4.0, c.y - ny * 4.0), (c.x + nx * 4.0, c.y + ny * 4.0)])
        parts = cut.intersection(piece)
        parts = [g for g in getattr(parts, "geoms", [parts]) if g.geom_type == "LineString"]
        if not parts:
            continue
        near = min(parts, key=lambda g: g.distance(c))
        if near.distance(c) <= 1.0:
            out.append(near.length)
    # (the lower quarter: a cut that happens to run along a crossing line reads far too
    # wide, and never too narrow)
    return float(np.percentile(out, 25)) if out else 0.0


def _web_weights(route: list, piece, least: float, bold: float,
                 axis: float | None = None, drawn: float | None = None, detail_runs: bool = True) -> list | None:
    """Line weights for a web of outlines, or None if it is thin line work (running stitch).
    A web has one weight, its typical drawn width, so every wall is the same thickness (a
    line here and there measuring a little over or under must not come out bold beside
    thin ones). Only lines drawn at least twice as heavy are sewn heavier (1.5 times the
    web's), and lines drawn clearly finer stay running stitch (0). A web drawn bold (from
    `bold` mm) but under the narrowest satin (`least`) is thickened to it, as a digitizer
    thickens hairlines; finer webs are running stitch."""
    widths = np.array([_line_width(seg, piece) for seg in route])
    lengths = np.array([LineString(seg).length for seg in route])
    if drawn:
        # a drawing redrawn for its size (simplify.py): every line was drawn this wide
        out = [max(drawn, least) if w >= 0.35 * drawn else 0.0 for w in widths]
        if drawn >= DETAIL_RUN_FROM_MM and detail_runs:
            # a detailed drawing kept whole at a bigger size (not rebuilt): its short
            # detail lines (window frames and bars, bricks) are runs, only the structure
            # is satin; satin on every little line crowds and piles up at the joins
            # (training session "Line art at left chest": three weights by role, small
            # detail never satin)
            out = [0.0 if LineString(seg).length < DETAIL_LEN_MM else v
                   for seg, v in zip(route, out)]
        return _uncrowd(route, out)
    twins = _twins(route, axis)
    for k, t in twins.items():                      # the two sides measure alike
        widths[k] = widths[t] = max(widths[k], widths[t])
    ok = widths > 0
    if not ok.any():
        return None
    order = np.argsort(widths[ok])
    cum = np.cumsum(lengths[ok][order])
    typical = float(widths[ok][order][int(np.searchsorted(cum, cum[-1] / 2.0))])
    if typical < bold:
        return None
    weight = max(typical, least)          # bold lines under the narrowest satin are thickened
    out = []
    for w in widths:
        if w < 0.5 * typical:
            out.append(0.0)
        elif w <= 2.0 * typical:
            out.append(weight)
        else:
            out.append(float(max(weight, 1.5 * typical)))
    out = _uncrowd(route, out)
    for k, t in twins.items():                      # ... and are sewn alike
        if out[k] != out[t]:
            out[k] = out[t] = 0.0 if min(out[k], out[t]) == 0.0 else max(out[k], out[t])
    # A drawing where much of the line work is too fine or too crowded for satin (bricks,
    # panes, hatching) is fine line art: all of it is running stitch. Bold lines mixed
    # through a tangle of thin ones only make the tangle heavier.
    fine = float(lengths[np.asarray(out) == 0.0].sum()) / max(float(lengths.sum()), 1e-9)
    if fine > FINE_WEB_SHARE:
        return None
    return out


def _twins(route: list, axis: float | None) -> dict:
    """For a symmetric drawing: each line's mirror twin (by position in the route)."""
    if axis is None:
        return {}
    mids = np.array([(float(np.mean([seg[0][0], seg[-1][0]])), float(np.mean([seg[0][1], seg[-1][1]])))
                     for seg in route])
    lens = np.array([LineString(seg).length for seg in route])
    out = {}
    for k, (x, y) in enumerate(mids):
        d = np.hypot(mids[:, 0] - (2.0 * axis - x), mids[:, 1] - y)
        d[np.abs(lens - lens[k]) > 0.3 + 0.1 * lens[k]] = np.inf
        t = int(d.argmin())
        if d[t] <= 0.4 and abs(x - axis) > 0.2:
            out[k] = t
    return out


CLEAR_MM = 0.5        # satin lines closer than this (edge to edge) sew as one blob
FINE_WEB_SHARE = 0.3  # more of a web than this unfit for satin: sew it all as lines


def _uncrowd(route: list, weights: list) -> list:
    """Lines too close together to sew as separate satin lines (the rings of a doorway arch,
    a small window's frame) are demoted to running stitch, shortest first, so the long main
    lines stay bold and the crowded detail inside them stays readable instead of closing
    up into a solid lump."""
    keys, lines = {}, {}
    for k, seg in enumerate(route):
        a, b = tuple(np.round(seg[0], 3)), tuple(np.round(seg[-1], 3))
        key = (a, b, len(seg)) if a <= b else (b, a, len(seg))
        keys[k] = key
        lines.setdefault(key, LineString(seg))
    weight = {keys[k]: w for k, w in enumerate(weights)}
    kept = []                                       # (line, weight) of lines staying satin
    for key in sorted(lines, key=lambda q: -lines[q].length):
        w, ln = weight[key], lines[key]
        if w <= 0:
            continue
        if ln.length <= 2.0 * w:
            kept.append((ln, w))
            continue
        # the middle of the line (its ends meet other lines at the junctions)
        mid = substring(ln, min(w, 0.25 * ln.length), ln.length - min(w, 0.25 * ln.length))
        near = [(o, ow) for o, ow in kept if o.distance(mid) < ow / 2.0 + w / 2.0 + CLEAR_MM]
        if near and _alongside(mid, near, w) > 0.5:
            weight[key] = 0.0
        else:
            kept.append((ln, w))
    return [weight[keys[k]] for k in range(len(route))]


def _tangent(line: LineString, at: float) -> float:
    a = line.interpolate(max(0.0, at - 0.25))
    b = line.interpolate(min(line.length, at + 0.25))
    return math.atan2(b.y - a.y, b.x - a.x)


def _alongside(mid: LineString, near: list, w: float) -> float:
    """How much of a line runs alongside other lines, too close for both to be satin:
    the share of points along it with another line within reach AND running the same way
    (within 30 degrees). Two lines that meet at an angle (a roof's slope and its gable)
    are close only near where they meet; that is a junction, not crowding."""
    n = max(3, int(mid.length / 0.3) + 1)
    hits = 0
    for k in range(n):
        s = mid.length * k / (n - 1)
        pt = mid.interpolate(s)
        t = _tangent(mid, s)
        for o, ow in near:
            if o.distance(pt) >= ow / 2.0 + w / 2.0 + CLEAR_MM:
                continue
            u = _tangent(o, o.project(pt))
            diff = abs((t - u + math.pi / 2) % math.pi - math.pi / 2)
            if diff < math.radians(30):
                hits += 1
                break
    return hits / n


def _round_openings(piece: Polygon) -> list:
    """Round openings in a shape (a clock face inside its ring): (x, y, radius) of each."""
    out = []
    for ring in piece.interiors:
        hole = Polygon(ring)
        if hole.area <= 0:
            continue
        r = math.sqrt(hole.area / math.pi)
        if not 0.8 <= r <= ROUND_MAX_R_MM:
            continue
        if 4.0 * math.pi * hole.area / max(hole.length ** 2, 1e-9) < 0.86:
            continue
        c = hole.centroid
        far = max(Point(x, y).distance(c) for x, y in ring.coords)
        near = min(Point(x, y).distance(c) for x, y in ring.coords)
        if far <= 1.2 * r and near >= 0.8 * r:
            out.append((c.x, c.y, r))
    return out


def _web_blobs(poly: Polygon) -> list:
    """Solid masses in a web of thin outlines: parts at least BLOB_WIDTHS line-widths across
    (and at least BLOB_MIN_MM), e.g. a tree drawn touching a building, or a letter of a
    monogram overlapping its ring. A web's own junctions are only about 1.4 line-widths
    thick, so they never count. Empty for an ordinary web."""
    if len(poly.interiors) < NETWORK_HOLES:
        return []
    tw = typical_width(poly)
    r = max(BLOB_MIN_MM / 2.0, BLOB_WIDTHS / 2.0 * tw)
    core = poly.buffer(-r, quad_segs=4)
    if core.is_empty:
        return []
    blobs = core.buffer(r * 1.05, quad_segs=4).intersection(poly)
    return [b for b in polygons_of(blobs.buffer(0)) if b.area >= BLOB_MIN_MM2]


SPECK_MM = 1.6          # a mark no bigger across than this...
SPECK_INSIDE_MM = 3.0   # ...and no stitching around it within this distance
SPECK_CLEAR_MM = 1.0    # ...with nothing else within this distance is a speck, not design
SCRAP_REACH = 15.0      # a left-out line's stray bits are looked for this many heights away
ROUND_MAX_R_MM = 150.0  # round openings up to this radius get a true circle (a badge ring
                        # 40 mm across was sewn as a polygon of flat sides)
BLOB_WIDTHS = 3.0       # a part this many line-widths thick inside a web is a solid mass...
BLOB_MIN_MM = 2.0       # ...and at least this thick...
BLOB_MIN_MM2 = 6.0     # ...and this big
FRAME_MIN_MM2 = 150.0   # a shape this big can be a frame (a building) that holds other shapes

FRAME_ROW_MAX = 5          # only a short row inside a big outline is taken for part of it
                           # (the panels of a door); a whole word inside a border is a word
MIXED_ROW_RATIO = 3.0      # a row whose tallest shape is this many times its shortest is mixed
CELL_ROW_MAX = 3           # a row this short sitting snugly in a drawing's opening is part of it
BAND_MAX_ELONGATION = 8.0  # ... and no longer than this many times its width (a long thin
                           # bar on its own is a line: running stitch)
BAND_MAX_COLUMNS = 4   # a small solid shape this many satin columns or fewer is sewn as satin
BLOCK_MIN_MM = 0.9     # the narrowest block that can be filled (two or three rows of thread)


def _elongation(poly: Polygon) -> float:
    """Length over width of the smallest rectangle around a shape."""
    rect = poly.minimum_rotated_rectangle
    c = list(rect.exterior.coords)
    if len(c) < 4:
        return float("inf")
    a, b = math.dist(c[0], c[1]), math.dist(c[1], c[2])
    return max(a, b) / max(min(a, b), 1e-9)


def _compact(poly: Polygon) -> bool:
    """Is this a block (a window, a square, a short bar) and not a line? No holes, and less
    than about three times as long as it is wide."""
    if poly.interiors:
        return False
    rect = poly.minimum_rotated_rectangle
    if rect.area <= 0 or poly.area < 0.6 * rect.area:
        return False
    c = list(rect.exterior.coords)
    a, b = math.dist(c[0], c[1]), math.dist(c[1], c[2])
    return max(a, b) < 3.0 * max(min(a, b), 1e-9)


def _dot(poly: Polygon, satin_max: float) -> dict | None:
    """A small round dot as one satin column sewn straight across it: its two edges are the
    two halves of the circle, so it sews as a clean round dot. (A tatami fill of a 3 mm dot
    is a few one-stitch rows: it sews as a lumpy lozenge.)"""
    if poly.interiors or poly.area <= 0:
        return None
    r = math.sqrt(poly.area / math.pi)
    round_ = 4.0 * math.pi * poly.area / max(poly.length ** 2, 1e-9)
    if round_ < 0.82 or not 0.6 <= r <= satin_max / 2.0:
        return None
    c = poly.centroid
    if max(Point(x, y).distance(c) for x, y in poly.exterior.coords) > 1.25 * r:
        return None
    reach = max(0.6 * r, r - 0.15)                  # column ends push out about 0.15 mm
    xs = np.linspace(-reach, reach, max(5, int(math.ceil(2 * reach / 0.25)) + 1))
    half = np.sqrt(np.maximum(r * r - xs * xs, 0.0))
    center = np.column_stack([c.x + xs, np.full_like(xs, c.y)])
    left = np.column_stack([c.x + xs, c.y - half])
    right = np.column_stack([c.x + xs, c.y + half])
    return {"center": center, "half": half, "left": left, "right": right,
            "outline": Point(c.x, c.y).buffer(r, quad_segs=16)}


SMALL_BLOCK_MM = 3.5      # a square-ish block no longer than this is one satin bar
BLOCK_SHRINK_FROM_MM = 1.3  # blocks this wide or more are sewn a little inside their edges
BAR_MIN_MM = 0.75         # ... at least this wide (a narrow satin, as small lettering uses)


def _block(poly: Polygon) -> dict | None:
    """A small rectangular block (a window 1-3 mm across) as one satin column along its
    long side, stitches straight across it. 2 x area / outline calls a 1.5 mm square only
    0.75 mm wide, too thin for anything, and the building lost every window; filled, a
    block that small is a few one-stitch rows."""
    if poly.interiors or poly.area <= 0:
        return None
    rect = poly.minimum_rotated_rectangle
    if rect.area <= 0 or poly.area < 0.8 * rect.area:
        return None
    c = np.asarray(rect.exterior.coords)[:4]
    e1, e2 = c[1] - c[0], c[2] - c[1]
    l1, l2 = float(np.hypot(*e1)), float(np.hypot(*e2))
    if l1 < l2:
        e1, e2, l1, l2 = e2, e1, l2, l1
    if not (BAR_MIN_MM <= l2 and l1 <= SMALL_BLOCK_MM):
        return None
    if l1 < 1.3 * l2 and abs(e1[0]) > abs(e1[1]):
        e1, e2, l1, l2 = e2, e1, l2, l1           # a near-square block runs top to bottom
    u, n = e1 / l1, e2 / l2                       # along the long side, across it
    mid = c.mean(axis=0)
    # (sewn a little inside its edges: thread spreads and satin ends push out, and the
    # half-millimetre walls between windows must stay open)
    # (only blocks of a good size: a 1 mm window shrunk the same way sews as a dash, and
    # the walls around a block that small were already kept open when it was drawn)
    small = l2 < BLOCK_SHRINK_FROM_MM
    reach = l1 / 2.0 - (0.08 if small else 0.25)
    ts = np.linspace(-reach, reach, max(4, int(math.ceil(2 * reach / 0.25)) + 1))
    center = mid + ts[:, None] * u
    half = np.full(len(ts), l2 / 2.0 - (0.0 if small else 0.12))
    return {"center": center, "half": half, "left": center - half[:, None] * n,
            "right": center + half[:, None] * n, "outline": rect}


def _fit_fills_to_lines(objects: list) -> None:
    """Where a color area is bordered by line art (a roof inside the building's outline), it
    is made to end exactly under those lines. The traced area stops at the edge of the
    drawn line and the line is straightened and sewn a thread wide, so otherwise the color
    falls short of the line on one side and pokes out past it on the other."""
    from shapely.ops import polygonize               # noqa: PLC0415
    from shapely.strtree import STRtree              # noqa: PLC0415
    lines = [LineString(seg) for o in objects if o["kind"] == "run" and o.get("route")
             and o.get("seq") is None for seg in o["route"] if len(seg) >= 2]
    fills = [o for o in objects if o["kind"] == "fill" and not o.get("texture")
             and o.get("text_row") is None and o["poly"].geom_type == "Polygon"]
    if not lines or not fills:
        return
    try:
        faces = [f for f in polygonize(unary_union(lines)) if f.area >= 1.0]
    except Exception:  # noqa: BLE001 - odd geometry: leave the fills as traced
        return
    if not faces:
        return
    tree = STRtree(faces)
    for o in fills:
        poly = o["poly"]
        take = [faces[i] for i in tree.query(poly)
                if faces[i].intersection(poly).area >= 0.6 * faces[i].area]
        if not take:
            continue
        new = unary_union(take).buffer(0)
        if new.geom_type != "Polygon":
            continue
        if new.intersection(poly).area < 0.75 * poly.area or new.area > 1.5 * poly.area:
            continue
        o["poly"], o["area_mm2"] = new, float(new.area)


def _ground(shapes: dict, inside: int, skip: set, reach: float) -> list:
    """The blank areas that are sewn, as a digitizer lays them: one ground fill running under
    the outlines and small details that sit on it (anything up to 2 x `reach` wide), not a
    patchwork of separate pieces butted against every line. It stays half a millimetre
    inside the design's outer edge, so it never peeks out past an outline, and keeps holes
    where wide areas of another color will be filled."""
    blanks = [p for i, p in enumerate(shapes[inside]) if (inside, i) not in skip]
    if not blanks:
        return []
    others = [p for c, polys in shapes.items() if c != inside for p in polys]
    merged = unary_union(blanks).buffer(reach, quad_segs=4).buffer(-reach, quad_segs=4)
    if others:
        whole = unary_union(blanks + others).buffer(-0.5, quad_segs=4)
        merged = merged.intersection(whole)
    return [p for p in polygons_of(merged.buffer(0)) if p.area >= 4.0]


def _row_scraps(shapes: dict, info: dict, keys: list, done: set) -> set:
    """Small shapes of a row's color lying within the row's band: pieces of its letters."""
    color = keys[0][0]
    glyphs = [shapes[c][i] for c, i in keys]
    x0 = min(g.bounds[0] for g in glyphs)
    x1 = max(g.bounds[2] for g in glyphs)
    y0 = min(g.bounds[1] for g in glyphs)
    y1 = max(g.bounds[3] for g in glyphs)
    h = max(y1 - y0, 1e-6)
    # (well beyond the row's ends: the rest of the line, a phone number beside a web
    # address, can lie away from the letters that formed the row)
    band = box(x0 - SCRAP_REACH * h, y0 - 0.3 * h, x1 + SCRAP_REACH * h, y1 + 0.3 * h)
    big = 3.0 * float(np.median([g.area for g in glyphs]))
    out = set()
    for i, poly in enumerate(shapes.get(color, [])):
        if (color, i) in done or poly.area > big:
            continue
        if band.contains(poly.centroid) and poly.bounds[3] - poly.bounds[1] < 1.6 * h:
            out.add((color, i))
    return out


def _cover_text_scraps(objects: list, shapes: dict, done: set, fits: dict) -> None:
    """Pieces of traced letters that never joined their row (a merged "BI", the dot of a
    small "i") would be sewn as solid blobs over the retyped words. A leftover shape of the
    lettering's color that the retyped letters mostly cover is part of that text: skip it."""
    if not fits:
        return
    areas = {}
    for o in objects:
        if o.get("text_row") not in fits:
            continue
        g = o.get("outline") if o["kind"] == "satin" else o.get("poly")
        if g is None and o.get("line") is not None:
            g = o["line"].buffer(0.6)
        if g is not None:
            areas.setdefault(o["color"], []).append(g.buffer(0.4))
    for color, parts in areas.items():
        cover = unary_union(parts)
        # the band the retyped words fill: a small scrap of the old letters left inside it
        # (a serif, the broken tail of a Y) would be sewn as a stray mark beside them
        bands = [box(*g.bounds).buffer(0.5) for g in polygons_of(cover.buffer(1.5))]
        small = 0.5 * float(np.median([g.area for g in parts])) if parts else 0.0
        for i, poly in enumerate(shapes.get(color, [])):
            if (color, i) in done or poly.area <= 0:
                continue
            if poly.intersection(cover).area >= 0.6 * poly.area:
                done.add((color, i))
            elif poly.area <= small and any(b.contains(poly) for b in bands):
                done.add((color, i))


LINE_GAP_HEIGHTS = 4.5   # pieces of one line of text lie within this many letter heights
PART_WHOLE_SHARE = 0.75   # a piece with this share of its word's letters stands for the word


def _complete_word(fit: dict, text: list[str], box: list):
    """A piece of a long word found alone (MICROBIOLOG..., ICROBIOLOGY when the M merged into
    a border at cap size): when it holds most of the word, (the whole word, the row's box
    widened on the side the missing letters belong). Otherwise None."""
    line, a, _b = fit["words"]
    ww = fit.get("whole_words")
    if not ww or ww[0] != line or not ww[1] <= a < ww[2]:
        return None              # the row as a whole looks like other words: not this one
    words = [w for w in re.split(r"\s+", text[line].strip()) if w]
    if a >= len(words):
        return None
    word, piece = words[a].upper(), fit["text"].upper()
    if len(piece) < PART_WHOLE_SHARE * len(word):
        return None
    missing = len(word) - len(piece)
    pitch = (box[2] - box[0]) / max(len(piece), 1)
    x0, x1 = box[0], box[2]
    if word.startswith(piece):
        x1 += missing * pitch
    elif word.endswith(piece):
        x0 -= missing * pitch
    else:
        return None
    return words[a], (x0, box[1], x1, box[3])


def _whole_lines(fits: dict, row_info: list, text: list[str]) -> set:
    """Small lettering often traces badly: letters merge, a line breaks into several rows and
    only some of them match a word. When a small row matches words of a known line, the
    rows of the same color lined up beside it (same height and baseline, close by) are taken
    as the rest of that line, and the whole line is set evenly across them. Rows whose
    letters were squeezed into each other are re-set evenly too. Returns the rows taken."""
    infos = {i["row"]: i for i in row_info}
    taken: set = set()
    for row, fit in sorted(list(fits.items()), key=lambda kv: -kv[1]["score"]):
        if row in taken or row not in fits:
            continue
        info = infos[row]
        h = info["height"]
        if h >= RETYPE_AUTO_MM:
            continue
        line = fit["words"][0]
        style = fit.get("style", "block")
        box = list(unary_union(info["glyphs"]).bounds)
        cy = (box[1] + box[3]) / 2
        members = [row]
        grew = True
        while grew:
            grew = False
            for other in row_info:
                r2 = other["row"]
                if r2 in members or r2 in taken or other["color"] != info["color"]:
                    continue
                if r2 in fits and fits[r2]["words"][0] != line:
                    continue
                if not 0.7 <= other["height"] / max(h, 1e-6) <= 1.4:
                    continue
                ob = unary_union(other["glyphs"]).bounds
                if abs((ob[1] + ob[3]) / 2 - cy) > 0.5 * h:
                    continue
                if max(ob[0] - box[2], box[0] - ob[2]) > LINE_GAP_HEIGHTS * h:
                    continue          # (" - " separators leave gaps of several letters)
                box = [min(box[0], ob[0]), min(box[1], ob[1]),
                       max(box[2], ob[2]), max(box[3], ob[3])]
                members.append(r2)
                grew = True
        pls = fit["placements"]
        if len(members) > 1 and fit.get("part"):
            # a piece of a word that gathered neighbours: together they must look most like
            # that word's line, or the piece matched by chance (a scrap of KNOXVILLE read as
            # part of TENNESSEE)
            together = [g for r2 in members for g in infos[r2]["glyphs"]]
            check = lettering.fit_row(together, text, style, min_score=0.0)
            if not check or check["words"][0] != line:
                fits.pop(row)
                info["lone_piece"] = True
                continue
        if len(members) == 1 and fit.get("part"):
            whole = _complete_word(fit, text, box)
            if whole is None:
                fits.pop(row)            # a piece of a word with no rest of it beside it
                info["lone_piece"] = True
            else:
                new_text, wbox = whole   # nearly all of it: a letter or two merged away
                H = float(np.median([pl.H for pl in pls]))
                w = float(np.median([pl.w for pl in pls]))
                run = sum(pl.run for pl in pls) * 2 >= len(pls)
                fit["placements"] = lettering.fit_line(new_text, wbox, H, w, style, run)
                fit["text"], fit["part"] = new_text, False
            continue
        if len(members) == 1 and not lettering.overlapping(pls, style):
            continue
        new_text = text[line] if len(members) > 1 else fit["text"]
        H = float(np.median([pl.H for pl in pls]))
        w = float(np.median([pl.w for pl in pls]))
        run = sum(pl.run for pl in pls) * 2 >= len(pls)
        fit["placements"] = lettering.fit_line(new_text, tuple(box), H, w, style, run)
        fit["text"] = new_text
        if len(members) > 1:
            fit["words"] = (line, 0, len(text[line].split()))
        for r2 in members[1:]:
            fits.pop(r2, None)
            taken.add(r2)
    return taken


def run(vec: dict, rules: dict, debug_dir: Path | None = None,
        text: list[str] | None = None, letters: str = "auto", plan: dict | None = None) -> dict:
    """Choose how every shape is sewn. `text` is the words known to be in the picture (read
    by ChatGPT); `letters` is auto | retype | trace (see the lettering rows below)."""
    satin_min = rules["satin_min_mm"]
    satin_half = rules["satin_max_mm"] / 2.0
    min_area = 0.6 * rules["min_detail_mm"] ** 2
    use_satin = rules.get("satin", True)

    order = sewing_order(vec["layers"], rules["satin_max_mm"])
    if vec.get("inside_color") in order:
        # blank areas that are sewn are the design's ground: first, under everything else
        order.remove(vec["inside_color"])
        order.insert(0, vec["inside_color"])
    shapes = underlap(vec["layers"], order, rules["overlap_mm"])
    shapes = sew_on_top(shapes, vec["layers"], order)
    _b = vec.get("bounds") or unary_union([q for layer in vec["layers"] for q in layer["polys"]]).bounds
    design_mm = max(_b[2] - _b[0], _b[3] - _b[1])
    # rings and round borders: their own objects, sewn after what crosses them (rings.py)
    ring_objects = []
    for color in order:
        found = rings.find_rings(shapes.get(color, []))
        if found:
            shapes[color] = rings.split(shapes[color], found)
            # (rings follow the line rules: drawn bold, a satin at least the narrowest
            # net satin; finer, a run. Training: satin borders 1-1.2 mm at the least)
            ring_objects += rings.objects(found, color, rules.get("satin_net_min_mm", 1.1),
                                          rules.get("line_bold_mm", 0.8))
    # colors that draw lines (a web of outlines somewhere in them): their narrow parts are
    # lines. In any other color a narrow part is a solid shape that happens to be small.
    line_colors = {c for c, polys in shapes.items()
                   if any(len(q.interiors) >= NETWORK_HOLES and typical_width(q) < NETWORK_MAX_MM
                          for q in polys)}
    textures = _foliage(shapes, plan)
    rows = text_rows(shapes, order, rules)
    inside = vec.get("inside_color")
    if inside is not None and rows:
        # the insides of big letters (sewn blank areas) can look like a row of letters too
        hulls = [Polygon(shapes[c][i].exterior).buffer(0.3) for c, i in rows if c != inside]
        groups = {}
        for (c, i), info in rows.items():
            if c == inside:
                groups.setdefault(info["row"], []).append((c, i))
        for keys in groups.values():
            held = sum(1 for c, i in keys if any(h.contains(shapes[c][i]) for h in hulls))
            if held >= 0.6 * len(keys):
                for k in keys:
                    del rows[k]

    objects, dropped = [], 0

    later_of: dict = {}

    def under_later(color, rail) -> bool:
        """Does this edge of a column lie under colors sewn after it (an outline drawn over
        it)? Pull compensation there only pushes the edge out past the outline."""
        if rail is None or color not in order:
            return False
        if color not in later_of:
            k = order.index(color)
            after = [q for layer in vec["layers"] if layer["color"] in order[k + 1:]
                     for q in layer["polys"]]
            later_of[color] = prep(unary_union(after).buffer(0.15)) if after else None
        cover = later_of[color]
        if cover is None:
            return False
        pts = np.asarray(rail)[:: max(1, len(rail) // 40)]
        inside = sum(1 for x, y in pts if cover.contains(Point(float(x), float(y))))
        return inside >= COVERED_EDGE_SHARE * len(pts)

    def add_satin(cols, color, style=None):
        for c in cols:
            for patch in c.get("patches", ()):
                objects.append({"kind": "fill", "color": color, "poly": patch,
                                "area_mm2": float(patch.area), "patch": True})
            o = {"kind": "satin", "color": color, "center": c["center"], "half": c["half"],
                 "left": c.get("left"), "right": c.get("right"), "outline": c["outline"],
                 "width_mm": 2 * float(np.median(c["half"])), "area_mm2": float(c["outline"].area)}
            if under_later(color, c.get("left")):
                o["comp_left"] = 0.0
            if under_later(color, c.get("right")):
                o["comp_right"] = 0.0
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
        lines = geometry.clean_lines(lines, vec.get("axes", [0.0, 90.0]))
        if width < NOISE_WIDTH_MM or sum(ln.length for ln in lines) < min_len:
            return False
        for ln in lines:
            o = {"kind": "run", "color": color, "line": ln, "width_mm": width,
                 "bean": bool(width >= rules["bean_line_mm"])}
            if stitch_mm:
                o["stitch_mm"] = stitch_mm
            objects.append(o)
        return True

    def add_texture(poly, color, angle=None, ink=1.0):
        """Foliage: one soft, textured mass (stitch.stitch_texture). Small gaps inside it
        are filled; only real openings stay open."""
        mass = Polygon(poly.exterior, [h for h in poly.interiors if Polygon(h).area > 3.0])
        if mass.area < min_area:
            return
        objects.append({"kind": "fill", "texture": True, "color": color, "poly": mass,
                        "area_mm2": float(mass.area), "angle": 70.0 if angle is None else angle,
                        "ink": float(ink)})

    def add_network(piece, color, bold=None):
        """Line art: the centre lines as one clean network, straightened and lined up, sewn
        as one continuous thread with two passes on every line (linenet.py). Returns False
        if there was nothing worth sewing."""
        # An outline drawn round shapes of other colors (a colored letter's border) is the
        # heaviest line of a design: from about half a millimetre drawn it is a satin at the
        # narrowest satin width, covering the fill's edge; as a single running line the fill
        # shows past it (training session "Line art at left chest": outer outlines satin
        # 1-1.5 mm)
        bold_from = rules["line_bold_mm"]
        hull = Polygon(piece.exterior)
        inside_other = sum(q.intersection(hull).area for c2, qs in shapes.items() if c2 != color
                           for q in qs if q.intersects(hull))
        # (an outer outline only: a line inside another color's shape, the muzzle line in a
        # bear's face, is an inner line and keeps the lighter weight of the drawing)
        held = sum(Polygon(q.exterior).intersection(piece).area for c2, qs in shapes.items()
                   if c2 != color for q in qs if q.intersects(piece))
        outlines_others = inside_other > OUTLINE_FILLS_SHARE * max(hull.area - piece.area, 1e-9)
        if outlines_others and held < OUTLINE_INNER_SHARE * piece.area:
            bold_from = min(bold_from, OUTLINE_BOLD_MM)
        sk = _skeleton(piece, prune_hairlines=False)
        if sk is None:
            return False
        paths, _ends, dt, to_mm = sk
        on = [dt[y, x] for pth in paths for y, x in pth]
        width = float(2.0 * np.mean(on) / SKEL_PPM) if on else 0.0
        if width < NOISE_WIDTH_MM:
            return False
        mm = [np.array([to_mm(y, x) for y, x in pth]) for pth in paths if len(pth) >= 2]
        net = {"circles": _round_openings(piece)}
        routes = linenet.network(mm, vec.get("axes", [0.0, 90.0]), spur=rules["min_detail_mm"],
                                 info=net)
        routes = [r for r in routes if linenet.length(r) >= MIN_THIN_LENGTH_MM / 2]
        if not routes:
            return False
        # Running-stitch weight by drawn width (research: running_stitch.md): thin lines get
        # two even passes; lines from 0.8 mm (or any line on pile, where runs sink) get four:
        # out once and back as a bean, which reads as a bold line without a skinny satin.
        if bold is None:
            bold = width >= rules["line_bold_mm"] or rules.get("knockdown", False)
        # Lines drawn about a millimetre wide or more are satin, as in production files
        # (measured with tools/study.py: satin from about 1 mm, almost no bean): each line
        # of the web is a narrow satin over its own running-stitch underlay, the whole web
        # one thread. Thinner lines stay running stitch.
        for route in routes:
            line = LineString(np.vstack(route))
            o = {"kind": "run", "color": color, "line": line, "route": route,
                 "width_mm": width, "bean": False, "bold": bool(bold)}
            if use_satin and not rules.get("knockdown", False):
                weights = _web_weights(route, piece, rules.get("satin_net_min_mm", 1.1),
                                       bold_from, net.get("axis"),
                                       rules.get("drawn_line_mm"),
                                       detail_runs=design_mm <= DETAIL_RUNS_MAX_MM)
                if weights:
                    o["satin_w"] = weights
            objects.append(o)
        return True

    def sew_narrow(part, color, alone=True, blobs_done=False):
        """A part that is narrow all over: satin if it is a real column, bean lines if it is
        thin or a web of outlines, otherwise a fill."""
        nonlocal dropped
        blobs = [] if blobs_done else _web_blobs(part)
        if blobs:
            # a web of outlines with solid masses joined to it (a tree touching a building, a
            # monogram letter over its ring): the masses are filled, the outlines around them
            # sewn on their own (every piece of them: a piece that still looked like it held
            # a mass used to be skipped, and a building lost its lines)
            for b in blobs:
                add_fill(b, color)
            for rest in polygons_of(part.difference(unary_union(blobs).buffer(0.05))):
                if rest.area >= min_area:
                    sew_narrow(rest, color, alone, blobs_done=True)
            return
        dot = (_dot(part, rules["satin_max_mm"]) or _block(part)) if use_satin else None
        if dot:
            # (little pull compensation: a dot stretched along its stitches sews as an oval)
            add_satin([dot], color, {"comp_scale": 0.5, "underlay": "none", "block": True})
            return
        w = typical_width(part)
        if (alone and use_satin and color not in line_colors and not part.interiors
                and BLOCK_MIN_MM <= w < satin_min and not rules.get("knockdown", False)
                and _elongation(part) <= BAND_MAX_ELONGATION):
            # A small solid shape of a color that draws no lines (a roof a millimetre deep
            # at cap size) keeps its mass as a narrow satin band, as production files sew
            # anything from about 1 mm. As an outline it reads as a hollow box.
            cols = try_satin(part, rules)
            if cols and len(cols) <= BAND_MAX_COLUMNS:
                add_satin(cols, color, {"comp_scale": 0.5})
                return
            if part.area >= min_area:
                add_fill(part, color)          # (a small gable triangle: solid, not a frame)
                return
        if w < satin_min and w >= BLOCK_MIN_MM and part.area >= min_area and _compact(part):
            # A small solid block (a window a millimetre or so wide) is still a block: sewn
            # as one line it loses its mass and the building loses its windows.
            add_fill(part, color)
            return
        if w < satin_min or (len(part.interiors) >= NETWORK_HOLES and w < NETWORK_MAX_MM):
            if add_network(part, color) or add_lines(part, color):
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
        # (blank shapes that line up in a row themselves are not letters to sit inside:
        # each would be "inside" its own outline and none would ever be sewn)
        hulls = [Polygon(shapes[c][i].exterior).buffer(0.3) for c, i in rows if c != inside]
        for i, blank in enumerate(shapes[inside]):
            if any(h.contains(blank) for h in hulls):
                done.add((inside, i))

    plans = {}

    def plan_row(row, keys, height):
        """How a traced row would be sewn: (mode, tried) with mode satin | lines | fill."""
        if row in plans:
            return plans[row]
        glyphs = [(c, i, shapes[c][i]) for c, i in keys]
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
        stroke = float(np.median([typical_width(g) for _c, _i, g in glyphs]))
        if stroke > HEAVY_LETTER_RATIO * height and not small:
            # Extra-bold letters are more junction than stroke: columns cut out of them
            # come out as stubby wedges (an M like a starburst). They are filled instead.
            satin_share = 0.0
        if satin_share >= ROW_SATIN_SHARE:
            mode = "satin"
        elif small:
            mode = "lines"
        else:
            mode = "fill"
        plans[row] = (mode, tried)
        return plans[row]

    def trace_row(row, keys, height):
        """Sew a row of traced letters one consistent way: satin when (nearly) every letter
        is a clean set of columns; otherwise small letters become run lettering (a fill
        across 1 mm strokes looks ragged) and large letters are all filled at one angle."""
        style = _text_style(height, rules)
        if height < rules["text_run_mm"]:
            # too small for satin: run-stitch lettering, short stitches so curves keep shape
            for c, i in keys:
                if add_lines(shapes[c][i], c, stitch_mm=1.5, min_len=0.8):
                    done.add((c, i))
            return
        mode, tried = plan_row(row, keys, height)
        angle = 18.0 if row % 2 == 0 else 162.0       # deep-dive rule 38
        for c, i, g, cols in tried:
            if mode == "satin" and cols:
                add_satin(cols, c, style)
            elif mode == "lines" and add_lines(g, c, stitch_mm=1.5, min_len=0.8):
                pass
            else:
                # A crisp edge on fill lettering: a bean run around the letter, sewn straight
                # after its fill in the same thread, hides the uneven ends of the fill rows
                # (Jeff Fuller: fill letters get an outline sewn with them; rule 24).
                add_fill(g, c, angle, dict(style or {}, edge=True))
            done.add((c, i))

    def follow_plan(poly, color) -> bool:
        """Sew a shape the way ChatGPT planned it. Returns False to use the app's own rules
        (no plan for it, "auto", or a choice that doesn't suit the shape)."""
        nonlocal dropped
        dv = aiplan.directive_for(plan, color, poly)
        if not dv or dv["stitch"] == "auto":
            return False
        st, angle = dv["stitch"], (dv["angle"] if dv["angle"] >= 0 else None)
        if st == "omit":
            dropped += 1
            return True
        if st in ("run", "bean"):
            return bool(add_network(poly, color, bold=(st == "bean")) or add_lines(poly, color))
        if st == "texture":
            add_texture(poly, color, angle)
            return True
        if st == "satin":
            cols = try_satin(poly, rules) if use_satin else None
            if cols:
                add_satin(cols, color)
                return True
            return False
        if st == "fill" and poly.area >= min_area:
            # Guardrail: a web of thin strokes is line art however dark it looks; filling
            # across 1 mm walls is the mess professionals avoid (deep-dive rule 18).
            if typical_width(poly) < satin_min:
                return bool(add_network(poly, color) or add_lines(poly, color))
            add_fill(poly, color, angle)
            return True
        return False

    # ---- lettering, one row at a time
    by_row = {}
    for key, info in rows.items():
        by_row.setdefault(info["row"], []).append(key)
    row_info, row_keys = [], {}
    for row, keys in by_row.items():
        keys = sorted(keys, key=lambda k: shapes[k[0]][k[1]].bounds[0])
        row_keys[row] = keys
        row_info.append({"row": row, "color": keys[0][0], "height": rows[keys[0]]["height"],
                         "text": None, "glyphs": [shapes[c][i] for c, i in keys]})
    # Research rule 6: retype known words in the embroidery font rather than trace them.
    # "auto" does it where tracing struggles (letters under RETYPE_AUTO_MM, where serifs and
    # fine detail fill in); "retype" does it for every row it can match. Each known word can
    # be claimed by one row only: the best match wins.
    fits, absorbed = {}, set()
    if letters != "trace" and text:
        # Lettering is never inside a drawing: shapes lined up inside a building's outline
        # (the panels of a door) are part of the building, even when they happen to look
        # like a short word.
        frames = [Polygon(q.exterior) for polys in shapes.values() for q in polys
                  if q.area >= FRAME_MIN_MM2]
        # the openings of line drawings (a web of outlines, not a single box or ring around
        # a word): what sits snugly inside one is part of the drawing
        cells = [Polygon(h) for polys in shapes.values() for q in polys
                 if len(q.interiors) >= NETWORK_HOLES for h in q.interiors]
        for info in row_info:
            h = info["height"]
            if h < rules.get("drop_text_mm", 0.0):
                continue
            ug = unary_union(info["glyphs"])
            if len(info["glyphs"]) <= FRAME_ROW_MAX and any(
                    f.contains(ug.centroid) and not any(f.equals(Polygon(g.exterior))
                                                        for g in info["glyphs"])
                    and f.area > 8.0 * ug.convex_hull.area for f in frames):
                info["not_text"] = True
                continue
            if len(info["glyphs"]) <= CELL_ROW_MAX and any(
                    cell.contains(ug.centroid) and cell.area < 6.0 * ug.convex_hull.area
                    for cell in cells):
                # (at cap size a building's ink is too little to count as a frame, but the
                # door bars still sit snugly in one opening of its drawing. Only a few
                # shapes: a long word inside a border drawn round it is still a word.)
                info["not_text"] = True
                continue
            style = "block"
            want = letters == "retype" or h < RETYPE_AUTO_MM
            if not want and h >= rules["text_run_mm"]:
                # Bigger words are kept as drawn when they make clean satin. When they would
                # be filled although their strokes are satin-width, they are retyped instead:
                # fill lettering is only for strokes over about 10 mm (deep-dive rule 25),
                # and a satin font beats ragged fill on 2-6 mm strokes.
                mode, _tried = plan_row(info["row"], row_keys[info["row"]], h)
                stroke = float(np.median([typical_width(g) for g in info["glyphs"]]))
                # (only thin-stroked ones: big heavy lettering, a serif TENNESSEE at left
                # chest, keeps its own character as a fill; the block font made it plain)
                want = mode == "fill" and stroke < BIG_RETYPE_STROKE_MM
            if not want:
                continue
            if h >= lettering.SERIF_MIN_MM and lettering.has_serifs(info["glyphs"]):
                style = "varsity"          # the closest embroidery style to serif lettering
            fit = lettering.fit_row(info["glyphs"], text, style, rules["satin_max_mm"],
                                    parts=h < RETYPE_AUTO_MM)
            if fit:
                fit["style"] = style
                fits[info["row"]] = fit
        claimed = []
        # (whole words claim first: a piece of a word matched by chance in a row of trees
        # and windows must never take the word from the row that really spells it)
        for row, fit in sorted(fits.items(), key=lambda kv: (bool(kv[1].get("part")),
                                                              -kv[1]["score"])):
            line, a, b = fit["words"]
            if any(line == l2 and a < b2 and a2 < b for l2, a2, b2 in claimed):
                if fit.get("part"):
                    # a piece of a word another row really spells: matched by chance
                    next(i for i in row_info if i["row"] == row)["lone_piece"] = True
                del fits[row]
            else:
                claimed.append((line, a, b))
        absorbed = _whole_lines(fits, row_info, text)
        # words from one line of text are one piece of lettering: same height, same weight,
        # sewn the same way, even when they were traced as separate rows
        by_line = {}
        for fit in fits.values():
            by_line.setdefault(fit["words"][0], []).append(fit)
        for group in by_line.values():
            if len(group) > 1:
                lettering.unify([f["placements"] for f in group])
    for color, mass, ink in textures:
        add_texture(mass, color, ink=ink)
    for info in row_info:
        if info["row"] in fits or info["row"] in absorbed:
            continue
        # Shapes that line up like letters are not always letters: the walls between a
        # building's windows, a row of windows. A row of plain blocks is not lettering. And
        # when the words are known, a row of blank areas that spells none of them, or a
        # row of blocks that spells none of them, is not lettering either. Such rows are
        # sewn as ordinary shapes (a window stays a solid block, not a letter-thin line).
        blocks = len(info["glyphs"]) >= 2 and all(_compact(g) and g.area >= 0.85
                                                  * g.minimum_rotated_rectangle.area
                                                  for g in info["glyphs"])
        # (any row: trees and window blocks lined up across a building's wings are not a
        # word either, and as "letters" they were sewn as thin outlines. Letters of a word are
        # much the same height; a tree beside millimetre windows is not. A row that spells
        # at least a piece of a known word and is even in height may still be a broken word.)
        hs = [g.bounds[3] - g.bounds[1] for g in info["glyphs"]]
        mixed = len(hs) >= 2 and max(hs) > MIXED_ROW_RATIO * min(hs)
        unspelled = bool(text) and (
            info.get("lone_piece")       # its only match was a stray piece of a word
            or not lettering.fit_row(info["glyphs"], text, "block", rules["satin_max_mm"],
                                     parts=True)
            or (mixed and not lettering.fit_row(info["glyphs"], text, "block",
                                                rules["satin_max_mm"])))
        if unspelled or (blocks and not text):
            info["text"] = None
            info["not_text"] = True
    if inside is not None and inside in shapes:
        lettering_keys = {k for i in row_info if i["color"] == inside and not i.get("not_text")
                          for k in row_keys[i["row"]]}
        for part in _ground(shapes, inside, done | lettering_keys, satin_half):
            add_fill(part, inside)
        done.update((inside, i) for i in range(len(shapes[inside]))
                    if (inside, i) not in lettering_keys)
    # lines too small to sew are left out with the bits of them that didn't join their row
    # (a broken "5", a dash): sewn alone they are stray marks where the line was
    for info in row_info:
        if info["height"] < rules.get("drop_text_mm", 0.0) and not info.get("not_text")                 and text and lettering.fit_row(info["glyphs"], text, "block",
                                               rules["satin_max_mm"], parts=True):
            # (only rows that really are some of the words: rows of small windows or
            # stonework in a drawing are not lettering, and their neighbours are lines)
            done.update(_row_scraps(shapes, info, row_keys[info["row"]], done))
    for info in row_info:
        row, height = info["row"], info["height"]
        keys = row_keys[row]
        if info.get("not_text"):
            continue
        if height < rules.get("drop_text_mm", 0.0) or row in absorbed:
            done.update(keys)          # left out by choice, or set as part of a whole line
            continue
        start = len(objects)
        fit = fits.get(row)
        if fit:
            info["text"] = fit["text"]
            pls = fit["placements"]
            stroke = float(np.median([lettering.stroke_width(g) for g in info["glyphs"]]))
            for pl in pls:
                pl.heavy = stroke >= BOLD_TEXT_STROKE * height
            objects.extend(lettering.objects(pls, info["color"], rules, fit.get("style", "block"),
                                             lettering.text_style(pls[0].H), group=row, row=row))
            done.update(keys)
            continue
        if all(k in done for k in keys):
            continue                   # every piece is a scrap of a line left out
        trace_row(row, keys, height)
        for o in objects[start:]:
            o["text_row"] = row
    _cover_text_scraps(objects, shapes, done, fits)

    # ---- monograms: a tall centre letter with smaller side letters (rings.py took any
    # ring out first). Not a word row (heights differ) and the letters often merge.
    mono, mono_keys = _monogram(shapes, done, text, order, rules)
    objects.extend(mono)
    done.update(mono_keys)

    # ---- everything else
    for color in order:
        if color not in shapes:
            continue
        for i, poly in enumerate(shapes[color]):
            if (color, i) in done:
                continue
            if follow_plan(poly, color):
                continue
            if color == inside and typical_width(poly) < satin_min:
                dropped += 1            # a thin gap between parts is open fabric, not thread
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
                touch = piece.boundary.intersection(edge)
                hug = touch.length / max(piece.length, 1e-9)
                if typical_width(piece) < FRINGE_MM or piece.area < min_area or hug > 0.3:
                    continue
                # A narrow part that joins the area in two places, or has holes in it (the
                # strips around letters knocked out of a banner), is part of the area: one
                # fill with holes. Cut into satin columns it sews as a patchwork of scraps.
                # Only a clean single band (a handle) is still sewn as its own column.
                # (Thin line work hanging off an area, a building's outlines beside its solid
                # trees, is not this case: it is sewn as lines whatever it joins.)
                joins = len(polygons_of(touch.buffer(0.4)))
                wp = typical_width(piece)
                lines = wp < satin_min or (len(piece.interiors) >= NETWORK_HOLES
                                           and wp < NETWORK_MAX_MM)
                if (joins >= 2 or piece.interiors) and not lines:
                    cols = try_satin(piece, rules) if use_satin else None
                    if not cols or len(cols) > 1:
                        continue
                arms.append(piece)
            # the area is everything that isn't a real arm (corner trimmings stay with it)
            core = poly.difference(unary_union(arms).buffer(0.05)) if arms else poly
            for part in polygons_of(core.buffer(0)):
                if part.area >= min_area:
                    add_fill(part, color)
                else:
                    dropped += 1
            for piece in arms:
                sew_narrow(piece, color, alone=False)

    bounds = vec.get("bounds") or unary_union(
        [q for layer in vec["layers"] for q in layer["polys"]]).bounds
    objects.extend(ring_objects)
    dropped += _drop_specks(objects)
    _match_mirrors(objects, bounds)
    _match_mirror_lines(objects, bounds, rules.get("satin_net_min_mm", 1.1))
    _same_weight_rows(objects)
    _fit_fills_to_lines(objects)

    if debug_dir is not None and objects:
        debug_picture(objects, debug_dir)

    row_info = [r for r in row_info if not r.get("not_text")]
    by_row = {r: k for r, k in by_row.items() if r in {i["row"] for i in row_info}}
    heights = sorted({round(rows[k[0]]["height"], 2) for k in by_row.values()})
    kept = [h for h in heights if h >= rules.get("drop_text_mm", 0.0)]
    left_out = [h for h in heights if h < rules.get("drop_text_mm", 0.0)]
    return {"objects": objects, "dropped_details": dropped, "order": order,
            "text_rows": len(by_row), "text_heights": kept, "text_left_out": left_out,
            "rows": row_info,
            "retyped": sum(1 for r in row_info if r["text"])}


def _match_mirrors(objects: list, bounds) -> None:
    """Matching parts on the two sides of a design (two roofs, two towers) are sewn the same
    way: if mirror-image parts of one color came out one satin and one fill, the satin one
    becomes a fill too. Mismatched halves are the first thing a customer notices."""
    minx, _miny, maxx, _maxy = bounds
    cx = (minx + maxx) / 2
    span = max(maxx - minx, 1e-6)
    cands = [k for k, o in enumerate(objects)
             if o["kind"] in ("satin", "fill") and o.get("text_row") is None
             and not o.get("texture") and o.get("seq") is None]

    def geom(o):
        return o["poly"] if o["kind"] == "fill" else o["outline"]
    for i in cands:
        a = objects[i]
        if a["kind"] != "satin":
            continue
        ga = geom(a)
        ax0, ay0, ax1, ay1 = ga.bounds
        for j in cands:
            b = objects[j]
            if b["kind"] != "fill" or b["color"] != a["color"]:
                continue
            bx0, by0, bx1, by1 = geom(b).bounds
            wa, ha, wb, hb = ax1 - ax0, ay1 - ay0, bx1 - bx0, by1 - by0
            if not (abs(wa - wb) <= 0.2 * max(wa, wb) and abs(ha - hb) <= 0.2 * max(ha, hb)):
                continue
            if abs((ay0 + ay1) / 2 - (by0 + by1) / 2) > 0.25 * max(ha, hb):
                continue
            if abs(((ax0 + ax1) / 2 - cx) + ((bx0 + bx1) / 2 - cx)) > 0.04 * span:
                continue
            objects[i] = {"kind": "fill", "color": a["color"], "poly": ga.buffer(0),
                          "area_mm2": float(ga.area)}
            break


def _drop_specks(objects: list) -> int:
    """Very short marks standing alone (bits of lettering too small to sew, left after the
    line was dropped) are not part of the design: a short line that belongs to a drawing (a
    clock hand) sits beside other stitching. Removed in place; returns how many."""
    from shapely.strtree import STRtree           # noqa: PLC0415

    def geom(o):
        if o["kind"] == "run":
            return o["line"]
        return o.get("outline") if o["kind"] == "satin" else o.get("poly")
    gs = [geom(o) for o in objects]
    idx = [k for k, g in enumerate(gs) if g is not None]
    if len(idx) < 2:
        return 0
    tree = STRtree([gs[k] for k in idx])
    small = set()
    for k, o in enumerate(objects):
        if o["kind"] != "run" or o.get("seq") is not None or o.get("route") is None and                 o.get("text_row") is not None:
            continue
        g = gs[k]
        if g is None:
            continue
        x0, y0, x1, y1 = g.bounds         # (its size, not its path: tours go out and back)
        if max(x1 - x0, y1 - y0) < SPECK_MM:
            small.add(k)
    drop = set()
    for k in small:
        g = gs[k]
        near = [idx[i] for i in tree.query(g.buffer(SPECK_CLEAR_MM))]
        # (beside real stitching it is part of something; beside other specks it is not)
        if any(j != k and j not in small and gs[j].distance(g) < SPECK_CLEAR_MM for j in near):
            continue
        # (inside other stitching, the hands of a clock inside its ring, it is a detail)
        c = g.centroid
        around = [idx[i] for i in tree.query(g.buffer(SPECK_INSIDE_MM))]
        if any(j != k and j not in small and gs[j].convex_hull.contains(c) for j in around):
            continue
        drop.add(k)
    for k in sorted(drop, reverse=True):
        del objects[k]
    return len(drop)


ROW_SIDE_OVERLAP = 0.35  # pieces overlapping sideways more than this share are not a row
OUTLINE_BOLD_MM = 0.5     # a line outlining other colors is satin from this drawn width
OUTLINE_INNER_SHARE = 0.6 # ... unless this much of the line lies inside another color's shape
OUTLINE_FILLS_SHARE = 0.5 # ... when other colors fill this share of what it encloses
DETAIL_LEN_MM = 3.0       # in a detailed drawing, lines shorter than this are runs...
DETAIL_RUNS_MAX_MM = 120.0 # ... on designs up to about 4.7 in (on a full back they are satin)
DETAIL_RUN_FROM_MM = 0.95 # ... when it was redrawn whole (lines this wide), not rebuilt
SAME_WEIGHT_TOL = 0.3    # lines drawn within this share of each other's width are "alike"


def _same_weight_rows(objects: list) -> None:
    """Strokes drawn alike, side by side in a row (the letters of a word traced as line
    art, script lettering at cap size), are sewn alike. A letter with a closed bowl (a B)
    is a web of lines and can come out as satin while the open strokes beside it stay
    running stitch: one heavy letter in a thin word. The row is sewn the way that covers
    most of its length."""
    lines = [k for k, o in enumerate(objects) if o["kind"] == "run" and o.get("route")
             and o.get("seq") is None and o.get("width_mm") and not o.get("ring")]
    parent = {k: k for k in lines}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in lines:
        a = objects[i]
        ax0, ay0, ax1, ay1 = a["line"].bounds
        for j in lines:
            if j <= i or objects[j]["color"] != a["color"]:
                continue
            b = objects[j]
            bx0, by0, bx1, by1 = b["line"].bounds
            h = min(ay1 - ay0, by1 - by0)
            if h <= 0.5:
                continue
            over = min(ay1, by1) - max(ay0, by0)
            gap = max(bx0 - ax1, ax0 - bx1)
            side = min(ax1 - ax0, bx1 - bx0)
            if -gap > ROW_SIDE_OVERLAP * max(side, 0.1):
                continue        # one inside the other (a mouth in a muzzle): not a row
            wa, wb = a["width_mm"], b["width_mm"]
            alike = abs(wa - wb) <= SAME_WEIGHT_TOL * max(wa, wb)
            if over >= 0.6 * h and gap <= max(ay1 - ay0, by1 - by0) and alike:
                parent[find(i)] = find(j)
    groups: dict = {}
    for k in lines:
        groups.setdefault(find(k), []).append(k)
    for ks in groups.values():
        if len(ks) < 2:
            continue
        sat = [k for k in ks if objects[k].get("satin_w")]
        if not sat or len(sat) == len(ks):
            continue
        sat_len = sum(objects[k]["line"].length for k in sat)
        all_len = sum(objects[k]["line"].length for k in ks)
        widths = [v for k in sat for v in objects[k]["satin_w"] if v > 0]
        if sat_len * 2 >= all_len and widths:
            w = float(np.median(widths))
            for k in ks:
                if not objects[k].get("satin_w"):
                    objects[k]["satin_w"] = [w] * len(objects[k]["route"])
        else:
            for k in sat:
                objects[k].pop("satin_w", None)
                objects[k]["bold"] = True


MONO_SIDE = (0.4, 0.8)    # side letters of a monogram are this share of the centre's height
MONO_MIN_IOU = 0.55       # retyped monogram letters must cover the art this well
MONO_FILL = 0.2           # a letter covers at least this share of its box (a ring doesn't)
MONO_MIN_H_MM = 5.0       # monograms smaller than this are left as drawn


def _monogram_tokens(text: list[str] | None) -> list[str]:
    """Monograms in the known words: two or three capitals written together (JMR) or
    apart (J M R)."""
    out = []
    for line in text or []:
        compact = re.sub(r"[\s.]+", "", line.strip())
        letters = lettering.letters_of(compact)
        if 2 <= len(letters) <= 3 and len(letters) == len(compact) and compact.isupper():
            out.append(compact)
    return list(dict.fromkeys(out))


def _monogram(shapes: dict, done: set, text, order: list, rules: dict):
    """Monogram letters retyped one by one (training session "Monogram letters": Erich
    Campbell Takeup ep. 126, Wilcom monogram help). A monogram is one tall centre letter
    with side letters about half its height, centred on it; merged letters (a side letter
    overlapping the centre one) are split by fitting each letter at its own size and place.
    Kept only if the letters together cover the traced shapes closely. The centre letter is
    sewn first and the side letters over it."""
    tokens = _monogram_tokens(text)
    if not tokens:
        return [], set()
    for color in order:
        pieces = [(i, p) for i, p in enumerate(shapes.get(color, []))
                  if (color, i) not in done and p.area > 1.0]
        # (not a frame, ring or arc: a letter fills a fair share of its box)
        solid = [(i, p) for i, p in pieces
                 if p.area > MONO_FILL * (p.bounds[2] - p.bounds[0]) * (p.bounds[3] - p.bounds[1])]
        if not solid:
            continue
        tall_i, tall = max(solid, key=lambda ip: ip[1].bounds[3] - ip[1].bounds[1])
        tx0, ty0, tx1, ty1 = tall.bounds
        Ht = ty1 - ty0
        if Ht < MONO_MIN_H_MM:
            continue
        # the letters: big pieces level with the tall one and near it
        group = [(i, p) for i, p in solid
                 if p.bounds[1] < ty1 and p.bounds[3] > ty0
                 and p.bounds[0] > tx0 - 2.5 * Ht and p.bounds[2] < tx1 + 2.5 * Ht
                 and p.bounds[3] - p.bounds[1] > 0.3 * Ht]
        art = unary_union([p for _i, p in group])
        ux0, _uy0, ux1, _uy1 = art.bounds
        side_hs = sorted({round(p.bounds[3] - p.bounds[1], 2) for _i, p in group
                          if MONO_SIDE[0] * Ht <= p.bounds[3] - p.bounds[1] <= MONO_SIDE[1] * Ht})
        side_hs = side_hs or [0.5 * Ht, 0.6 * Ht, 0.7 * Ht]
        w0 = float(np.median([lettering.stroke_width(p) for _i, p in group]))
        style = "varsity" if lettering.has_serifs([p for _i, p in group]) else "block"
        best = None
        for tok in tokens:
            chars = lettering.letters_of(tok)
            mid = len(chars) // 2 if len(chars) == 3 else None
            for hs in side_hs:
                for align in ("middle", "bottom", "top"):
                    pls = _monogram_place(chars, mid, Ht, hs, (ux0, ty0, ux1, ty1), w0, style,
                                          align, rules)
                    if not pls:
                        continue
                    ours = unary_union([lettering.outline_of(pl, style) for pl in pls])
                    try:
                        iou = ours.intersection(art).area / max(ours.union(art).area, 1e-9)
                    except Exception:  # noqa: BLE001 - odd geometry: not this one
                        continue
                    if best is None or iou > best[0]:
                        best = (iou, tok, pls, mid)
        if best is None or best[0] < MONO_MIN_IOU:
            continue
        iou, tok, pls, mid = best
        cover = unary_union([lettering.outline_of(pl, style) for pl in pls]).buffer(0.6)
        keys = {(color, i) for i, _p in group}
        keys |= {(color, i) for i, p in pieces
                 if (color, i) not in keys and p.intersection(cover).area > 0.5 * p.area}
        # sewing order: the centre letter first, the side letters over it
        order_idx = ([mid] if mid is not None else []) + [k for k in range(len(pls)) if k != mid]
        out = []
        for li, k in enumerate(order_idx):
            pl = pls[k]
            for o in lettering.objects([pl], color, rules, style, lettering.text_style(pl.H),
                                       group=MONO_GROUP + color, row=None):
                o["seq"] = (MONO_GROUP + color, li, o["seq"][2])
                out.append(o)
        return out, keys
    return [], set()


MONO_GROUP = 9000              # lettering group numbers used for monogram letters


def _monogram_place(chars, mid, Ht, hs, box, w0, style, align, rules):
    """Placements for monogram letters across `box` (x0, y0, x1, y1 of the centre letter's
    height): side letters at the outer edges, the centre letter centred."""
    x0, y0, x1, y1 = box
    ymid = (y0 + y1) / 2.0
    pls = []
    for k, ch in enumerate(chars):
        H = Ht if k == mid else hs
        w = lettering.width_for(H, style, weight_mm=w0, satin_max=rules.get("satin_max_mm", 7.0))
        pl = lettering.Placement(ch, 0.0, 0.0, H, w)
        if align == "middle":
            by = ymid + H / 2.0
        elif align == "bottom":
            by = y1
        else:
            by = y0 + H
        pl.by = by
        lo, hi = lettering.extent(pl, style)
        if mid is None:
            pl.bx = (x0 - lo) if k == 0 else (x1 - hi)
        elif k == 0:
            pl.bx = x0 - lo
        elif k == len(chars) - 1:
            pl.bx = x1 - hi
        else:
            pl.bx = (x0 + x1) / 2.0 - (lo + hi) / 2.0
        pls.append(pl)
    return pls


def _match_mirror_lines(objects: list, bounds, satin_from: float) -> None:
    """Lines on the two sides of a design drawn alike (the rules either side of a dot) are
    sewn alike. Measured separately, one can come out a hair over the satin width and the
    other a hair under: one heavy satin bar beside one thin running line. Both get the
    weight their average width calls for."""
    minx, _miny, maxx, _maxy = bounds
    cx = (minx + maxx) / 2
    span = max(maxx - minx, 1e-6)
    lines = [k for k, o in enumerate(objects) if o["kind"] == "run" and o.get("route")
             and o.get("seq") is None and o.get("text_row") is None and not o.get("ring")]
    used = set()
    for i in lines:
        if i in used:
            continue
        a = objects[i]
        ax0, ay0, ax1, ay1 = a["line"].bounds
        for j in lines:
            if j == i or j in used or objects[j]["color"] != a["color"]:
                continue
            b = objects[j]
            bx0, by0, bx1, by1 = b["line"].bounds
            tol = max(0.6, 0.04 * span)
            if abs((ax0 + ax1) - 2 * cx + (bx0 + bx1) - 2 * cx) / 2 > tol:
                continue
            if abs((ay0 + ay1) - (by0 + by1)) / 2 > tol or abs((ax1 - ax0) - (bx1 - bx0)) > tol                     or abs((ay1 - ay0) - (by1 - by0)) > tol:
                continue
            if abs((ax0 + ax1) / 2 - cx) < 0.5:
                continue                          # on the centre line: its own mirror
            used.update((i, j))
            sa, sb = a.get("satin_w"), b.get("satin_w")
            if bool(sa and any(sa)) == bool(sb and any(sb)):
                break
            w = (a.get("width_mm", 0.0) + b.get("width_mm", 0.0)) / 2.0
            have, lack = (a, b) if sa and any(sa) else (b, a)
            if w >= satin_from and len(have["route"]) == len(lack["route"]):
                lack["satin_w"] = list(have["satin_w"])
            else:
                have.pop("satin_w", None)
            break


def _foliage(shapes: dict, plan: dict | None) -> list:
    """Cut foliage out of the shapes: inside every foliage box from the plan, ink that is
    dense (a tree drawn as a web or scribble of lines, or a dark blob) becomes one mass to be
    sewn as a soft texture; lines that merely pass through stay lines. Returns
    [(color, mass)]."""
    if not plan or not plan.get("foliage"):
        return []
    out = []
    # In each box only the main ink colors are foliage (a roof corner that pokes into the
    # box stays what it is).
    ink_in = {}
    for k, bx in enumerate(plan["foliage"]):
        for color in shapes:
            ink_in[(k, color)] = sum(p.intersection(bx).area for p in shapes[color] if p.intersects(bx))
    for color in list(shapes):
        masses = []
        for k, bx in enumerate(plan["foliage"]):
            total = sum(v for (kk, _c), v in ink_in.items() if kk == k)
            if total <= 0 or ink_in[(k, color)] < 0.25 * total:
                continue
            inside = [p.intersection(bx) for p in shapes[color] if p.intersects(bx)]
            region = unary_union(inside)
            kept = []
            for r in (1.0, 2.0, 3.0):       # close small gaps; wider only for sparse drawing
                closed = region.buffer(r, quad_segs=4).buffer(-r, quad_segs=4).intersection(bx)
                found = []
                for m in polygons_of(closed):
                    ink = region.intersection(m).area / max(m.area, 1e-9)
                    if m.area >= 15.0 and ink >= 0.2:   # not a line merely passing through
                        found.append((m, ink))
                if not found:
                    continue
                # the tree is the main mass; small separate bits (a window beside it) stay lines
                biggest = max(m.area for m, _ in found)
                kept = [(m, ink) for m, ink in found if m.area >= 0.3 * biggest]
                covered = sum(region.intersection(m).area for m, _ in kept)
                if covered >= 0.85 * region.area and len(kept) <= 3:
                    break                       # a few cohesive masses: that's the tree
            for m, ink in kept:
                masses.append(m)
                out.append((color, m, ink))
        if masses:
            cut = unary_union(masses).buffer(0.2)
            shapes[color] = [q for p in shapes[color] for q in polygons_of(p.difference(cut))
                             if q.area > 0.05]
    return out


def debug_picture(objects: list, debug_dir: Path) -> None:
    """06_classify.png: fills green, satin blue with its centre line, runs red."""
    geoms = [o["poly"] if o["kind"] == "fill" else o["outline"] if o["kind"] == "satin"
             else o["line"] for o in objects]
    img, to_px = canvas_for(unary_union(geoms).bounds, 8.0, color=(255, 255, 255))
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
