"""Rings and round borders, found in a color's traced shapes and sewn as objects of their own.

Training session "Monograms rings and borders" (Erich Campbell's Takeup ep. 197, Romero
Threads satin borders, Wilcom help): one color field is almost never one object. Where
letters touch or cross a ring of the same color, the traced art is one merged shape, and
sewing it as drawn mangles both (an R's bowl melts into a thin ring, the ring breaks into
bits). Pros split it: each ring is a true circle with a fixed width in millimetres, sewn as
its own satin (or a run when it is a hairline), continuous where letters of its own color
cross it and sewn after them, the letters' ends tucked under it. A ring's width does not
shrink with the design below a sewable satin.

A ring is found from the edges of the shapes: circles fitted to their outline points
(RANSAC, then a least-squares fit), concentric circles paired into bands, and a band is a
ring where the color really fills it. Only the arcs the color covers are kept: where
another color covers the ring (a banner in front of a badge's border) it stays broken.
"""

from __future__ import annotations

import math

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from .common import polygons_of

RING_MIN_R_MM = 5.0        # circles smaller than this are details (eyes, dots), not rings
FIT_TOL_MM = 0.2           # outline points this near a circle lie on it
SWEEP_SHARE = 0.6          # a circle needs outline points round this share of it
CONCENTRIC_MM = 0.8        # circles whose centres are this close are one family
MAX_BAND_MM = 6.0          # wider bands are discs or fields, not rings
BAND_FILL = 0.75           # a stretch of band is ring where the color fills this much of it
RING_SHARE = 0.5           # ... and a ring needs that round at least this share of it
BINS = 90                  # stretches the band is checked in (4 degrees each)
RUN_BELOW_MM = 0.6         # a ring drawn thinner than this is sewn as a run
TUCK_MM = 0.4              # letters reach this far under a ring sewn over them
STEP_MM = 0.3              # spacing of outline points
SLIVER_PAD_MM = 0.3        # a ring is cut out this much wider than its fitted band
NEAR_MM = 1.0              # leftovers within this of a ring...
SLIVER_W_MM = 0.6          # ... and thinner than this are slivers of its edge, dropped
MAX_GAP_BINS = 2           # breaks this short (8 degrees) in a traced ring are closed
THIN_REACH_MM = 1.0        # a hairline ring is measured this far either side of its circle
THIN_MAX_MM = 1.2          # ... and is no wider than this


def _points(polys: list) -> np.ndarray:
    pts = []
    for p in polys:
        for ring in (p.exterior, *p.interiors):
            ls = LineString(ring.coords)
            n = max(4, int(ls.length / STEP_MM))
            pts.extend(ls.interpolate(ls.length * k / n).coords[0] for k in range(n))
    return np.asarray(pts, float) if pts else np.zeros((0, 2))


def _circle3(a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-9:
        return None
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay)
          + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx)
          + (cx * cx + cy * cy) * (bx - ax)) / d
    return ux, uy, math.hypot(ax - ux, ay - uy)


def _kasa(pts: np.ndarray):
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    sol, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    return cx, cy, math.sqrt(max(sol[2] + cx * cx + cy * cy, 0.0))


def _sweep(pts: np.ndarray, cx: float, cy: float) -> float:
    ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    bins = np.unique(((ang + math.pi) / (2 * math.pi) * BINS).astype(int) % BINS)
    return len(bins) / BINS


def find_circles(pts: np.ndarray, max_circles: int = 8, tries: int = 500) -> list:
    """Circles that many outline points lie on, all the way round (or most of it)."""
    rng = np.random.default_rng(0)
    left = pts.copy()
    out = []
    for _ in range(max_circles):
        if len(left) < 20:
            break
        best = None
        for _t in range(tries):
            i = rng.choice(len(left), 3, replace=False)
            c = _circle3(*left[i])
            if c is None or c[2] < RING_MIN_R_MM:
                continue
            d = np.abs(np.hypot(left[:, 0] - c[0], left[:, 1] - c[1]) - c[2])
            n = int((d < FIT_TOL_MM).sum())
            if best is None or n > best[0]:
                best = (n, c)
        if best is None:
            break
        cx, cy, r = best[1]
        for _r in range(2):                          # refine on its own points
            d = np.abs(np.hypot(left[:, 0] - cx, left[:, 1] - cy) - r)
            on = d < FIT_TOL_MM
            if on.sum() < 10:
                break
            cx, cy, r = _kasa(left[on])
        d = np.abs(np.hypot(left[:, 0] - cx, left[:, 1] - cy) - r)
        on = d < FIT_TOL_MM
        if r < RING_MIN_R_MM or on.sum() < 2 * math.pi * r / STEP_MM * SWEEP_SHARE * 0.5 \
                or _sweep(left[on], cx, cy) < SWEEP_SHARE:
            break
        out.append((cx, cy, r))
        left = left[~on]
    return out


def _sector(cx, cy, r1, r2, a0, a1) -> Polygon:
    n = max(2, int(abs(a1 - a0) / math.radians(2)) + 1)
    angs = np.linspace(a0, a1, n)
    outer = [(cx + r2 * math.cos(a), cy + r2 * math.sin(a)) for a in angs]
    inner = [(cx + r1 * math.cos(a), cy + r1 * math.sin(a)) for a in angs[::-1]]
    return Polygon(outer + inner)


def find_rings(polys: list) -> list[dict]:
    """Rings in one color's shapes: {"cx", "cy", "r1", "r2", "arcs": [(a0, a1)...] or None
    when it goes all the way round}."""
    if not polys:
        return []
    circles = find_circles(_points(polys))
    if len(circles) < 2:
        return []
    shape = unary_union(polys)
    families: list = []
    for c in sorted(circles, key=lambda c: c[2]):
        for fam in families:
            if math.hypot(fam[0][0] - c[0], fam[0][1] - c[1]) < CONCENTRIC_MM:
                fam.append(c)
                break
        else:
            families.append([c])
    rings, used = [], set()
    for fam in families:
        fam.sort(key=lambda c: c[2])
        for (x1, y1, r1), (x2, y2, r2) in zip(fam, fam[1:]):
            if r2 - r1 > MAX_BAND_MM or r2 - r1 < 0.15:
                continue
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            covered = []
            for k in range(BINS):
                a0 = -math.pi + 2 * math.pi * k / BINS
                sec = _sector(cx, cy, r1, r2, a0, a0 + 2 * math.pi / BINS)
                covered.append(sec.area > 0 and shape.intersection(sec).area >= BAND_FILL * sec.area)
            if sum(covered) < RING_SHARE * BINS:
                continue
            used.update({(x1, y1, r1), (x2, y2, r2)})
            rings.append({"cx": cx, "cy": cy, "r1": r1, "r2": r2, "arcs": _arcs(covered)})
    # a hairline ring: both its edges lie within the fitting tolerance of one circle, so
    # it was found once. Its width is measured straight across it, all the way round.
    for c in circles:
        if c in used:
            continue
        cx, cy, r = c
        widths, covered = [], []
        for k in range(BINS):
            a = -math.pi + 2 * math.pi * (k + 0.5) / BINS
            ux, uy = math.cos(a), math.sin(a)
            cut = LineString([(cx + (r - THIN_REACH_MM) * ux, cy + (r - THIN_REACH_MM) * uy),
                              (cx + (r + THIN_REACH_MM) * ux, cy + (r + THIN_REACH_MM) * uy)])
            on = Point(cx + r * ux, cy + r * uy)
            piece = [g for g in polygons_of(shape.buffer(0)) if g.distance(on) < 0.05]
            hit = cut.intersection(piece[0]) if piece else None
            segs = [] if hit is None or hit.is_empty else \
                [g for g in getattr(hit, "geoms", [hit]) if g.geom_type == "LineString"]
            seg = min(segs, key=lambda g: g.distance(on)) if segs else None
            ok = seg is not None and seg.distance(on) < 0.05
            covered.append(ok)            # (a letter crossing it widens it there: still ring)
            if ok and seg.length < THIN_MAX_MM:
                widths.append(seg.length)
        if sum(covered) < RING_SHARE * BINS or len(widths) < RING_SHARE * BINS / 2:
            continue
        w = float(np.median(widths))
        if any(math.hypot(g["cx"] - cx, g["cy"] - cy) < CONCENTRIC_MM
               and g["r1"] - w < r < g["r2"] + w for g in rings):
            continue                     # the other edge of a ring already found
        rings.append({"cx": cx, "cy": cy, "r1": r - w / 2, "r2": r + w / 2,
                      "arcs": _arcs(covered)})
    return rings


def _arcs(covered: list):
    """The covered stretches of a ring as (start, end) angles, or None if it is whole.
    Breaks of a bin or two (a ragged traced edge) are closed."""
    covered = list(covered)
    n = len(covered)
    for k in range(n):
        if not covered[k]:
            gap = 0
            while gap < n and not covered[(k + gap) % n]:
                gap += 1
            if gap <= MAX_GAP_BINS and covered[(k - 1) % n] and covered[(k + gap) % n]:
                for g in range(gap):
                    covered[(k + g) % n] = True
    if all(covered):
        return None
    arcs, k0, run = [], covered.index(False), None
    for step in range(BINS + 1):
        kk = (k0 + step) % BINS
        if covered[kk] and run is None and step < BINS:
            run = step
        if (not covered[kk] or step == BINS) and run is not None:
            a0 = -math.pi + 2 * math.pi * ((k0 + run) % BINS) / BINS
            arcs.append((a0, a0 + 2 * math.pi * (step - run) / BINS))
            run = None
    return arcs


def band(ring: dict, pad: float = 0.0):
    """The ring's band (where it is drawn), grown by `pad` mm across."""
    cx, cy, r1, r2 = ring["cx"], ring["cy"], ring["r1"] - pad, ring["r2"] + pad
    if ring["arcs"] is None:
        return Point(cx, cy).buffer(r2, 256).difference(Point(cx, cy).buffer(max(r1, 0.01), 256))
    return unary_union([_sector(cx, cy, r1, r2, a0, a1) for a0, a1 in ring["arcs"]])


def split(polys: list, rings: list) -> list:
    """The color's shapes with its rings taken out, and what touched a ring reaching
    TUCK_MM under it (the ring is sewn after them and covers their ends)."""
    if not rings:
        return polys
    cut = unary_union([band(r) for r in rings])
    # (a little past the fitted circles: the traced edges wander a few tenths either side,
    # and what is left beside a ring would be sewn as a stray line along it)
    wide = unary_union([band(r, SLIVER_PAD_MM) for r in rings])
    near = unary_union([band(r, NEAR_MM) for r in rings])
    rest = unary_union(polys).difference(wide)
    if rest.is_empty:
        return []
    out = []
    for p in polygons_of(rest):
        if p.area < 0.05:
            continue
        if p.intersection(near).area > 0.8 * p.area and 2.0 * p.area / max(p.length, 1e-9) < SLIVER_W_MM:
            continue                                  # a sliver along a ring's edge
        if p.buffer(0.05).intersects(wide):
            p = p.union(p.buffer(TUCK_MM + SLIVER_PAD_MM).intersection(wide))
            p = max(polygons_of(p), key=lambda q: q.area) if polygons_of(p) else p
        out.append(p)
    return out


def objects(rings: list, color: int, satin_min: float, bold_from: float = RUN_BELOW_MM) -> list[dict]:
    """Each ring as one object: a satin column along its middle at its drawn width (never
    under the narrowest satin), or a run when it is a hairline. Sewn after the shapes
    of its color ("sew_last")."""
    out = []
    for ring in rings:
        cx, cy = ring["cx"], ring["cy"]
        w = ring["r2"] - ring["r1"]
        rm = (ring["r1"] + ring["r2"]) / 2.0
        spans = [(-math.pi, math.pi)] if ring["arcs"] is None else ring["arcs"]
        for a0, a1 in spans:
            n = max(8, int(abs(a1 - a0) * rm / 0.25))
            angs = np.linspace(a0, a1, n + 1)
            mid = np.column_stack([cx + rm * np.cos(angs), cy + rm * np.sin(angs)])
            if w < bold_from:
                # (a hairline ring is a run all the way round; a ring drawn bold is a satin
                # at least `satin_min` wide, as the app sews bold lines)
                line = LineString(mid)
                out.append({"kind": "run", "color": color, "line": line,
                            "route": [[tuple(p) for p in mid]], "width_mm": float(w),
                            "bean": False, "bold": True, "sew_last": True, "ring": True})
                continue
            W = max(w, satin_min)
            half = np.full(len(mid), W / 2.0)
            outer = np.column_stack([cx + (rm + W / 2) * np.cos(angs), cy + (rm + W / 2) * np.sin(angs)])
            inner = np.column_stack([cx + (rm - W / 2) * np.cos(angs), cy + (rm - W / 2) * np.sin(angs)])
            outline = Polygon(list(map(tuple, outer)) + list(map(tuple, inner[::-1]))).buffer(0)
            out.append({"kind": "satin", "color": color, "center": mid, "half": half,
                        "left": outer, "right": inner, "outline": outline,
                        "width_mm": float(W), "area_mm2": float(outline.area),
                        "sew_last": True, "ring": True})
    return out
