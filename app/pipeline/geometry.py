"""Clean geometry: redraw traced outlines and centre lines the way a digitizer would.

A professional never feeds traced pixel edges to the stitch generator. They redraw the art at
its final size with a few deliberate points: corners pinned, straight runs dead straight,
lines that should be level or upright snapped exactly to that direction, and curves smooth
instead of following every pixel step (reports/Embroidery digitizing deep dive.md, rules 5-7:
Embroidery Legacy's straight vs curve points and Shift-snapping, Hatch's 3 to 4 nodes per
circle, "judge at 1:1").

What this module does to every ring or line, all in millimetres at the final size:
  1. resample evenly and smooth away the pixel stair-steps
  2. find real corners: a sharp turn concentrated in a short stretch (a curve turns as much
     over a longer stretch, a corner doesn't)
  3. between corners, a stretch within TOL_MM of its chord becomes one straight segment;
     anything else stays a smooth curve with few points
  4. straight segments close to the design's main directions (level, upright, or the
     building's own axes) are turned exactly onto them, and corners are re-made where the
     straightened lines meet

Our own code; the corner test and tolerances are our engineering choices (marked inferred in
the report).
"""

from __future__ import annotations

import math

import numpy as np
from shapely.geometry import LineString, Polygon

from .common import polygons_of

STEP_MM = 0.1            # resampling step
SMOOTH_MM = 0.25         # half-width of the smoothing window (pixel stair-steps are ~0.1 mm)
TOL_MM = 0.15            # a stretch this close to its chord is straight (under half a thread)
CORNER_DEG = 35.0        # turn needed within CORNER_WINDOW_MM to count as a corner
CORNER_WINDOW_MM = 0.5
CURVE_TOL_MM = 0.04      # how closely the few points kept on a curve follow it
SNAP_DEG = 3.0           # straight segments within this of a main direction are snapped
SNAP_MAX_DEG = 7.0       # (short segments are allowed a little more, their angle is noisier)
AXIS_SHARE = 0.12        # a direction carrying this share of all straight length is a main one


# -------------------------------------------------------------------------------- helpers

def _resample(pts: np.ndarray, closed: bool, step: float = STEP_MM) -> np.ndarray:
    if closed:
        pts = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-9])
    pts = pts[keep]
    if len(pts) < 2:
        return pts
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    n = max(3, int(round(cum[-1] / step)))
    s = np.linspace(0.0, cum[-1], n + 1)
    out = np.column_stack([np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])])
    return out[:-1] if closed else out


def _smooth(pts: np.ndarray, closed: bool, half: int) -> np.ndarray:
    if half < 1 or len(pts) < 2 * half + 3:
        return pts
    k = np.ones(2 * half + 1) / (2 * half + 1)
    if closed:
        pad = np.vstack([pts[-half:], pts, pts[:half]])
        return np.column_stack([np.convolve(pad[:, i], k, "valid") for i in range(2)])
    out = pts.copy()
    pad = np.vstack([np.repeat(pts[:1], half, 0), pts, np.repeat(pts[-1:], half, 0)])
    sm = np.column_stack([np.convolve(pad[:, i], k, "valid") for i in range(2)])
    out[1:-1] = sm[1:-1]                        # open lines keep their true end points
    return out


def _turn(pts: np.ndarray, closed: bool, k: int) -> np.ndarray:
    """Turning angle (degrees) at every point, between the chords k points back and ahead."""
    n = len(pts)
    idx = np.arange(n)
    if closed:
        a, b = pts[(idx - k) % n], pts[(idx + k) % n]
    else:
        a, b = pts[np.clip(idx - k, 0, n - 1)], pts[np.clip(idx + k, 0, n - 1)]
    v1, v2 = pts - a, b - pts
    n1, n2 = np.linalg.norm(v1, axis=1), np.linalg.norm(v2, axis=1)
    cos = np.einsum("ij,ij->i", v1, v2) / np.maximum(n1 * n2, 1e-12)
    ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    if not closed:
        ang[:k] = 0.0
        ang[-k:] = 0.0
    return ang


def _corners(pts: np.ndarray, closed: bool) -> list[int]:
    k = max(2, int(round(CORNER_WINDOW_MM / STEP_MM)))
    t1, t2 = _turn(pts, closed, k), _turn(pts, closed, 2 * k)
    # A corner turns sharply in a short stretch and by about as much over a longer one. A
    # curve keeps turning over the longer stretch (t2 much bigger); pixel noise averages out
    # over it (t2 much smaller).
    cand = (t1 >= CORNER_DEG) & (t2 >= 0.8 * CORNER_DEG) & (t1 >= 0.7 * t2)
    out, n = [], len(pts)
    i = 0
    while i < n:
        if cand[i]:
            j = i
            while j + 1 < n and cand[j + 1]:
                j += 1
            seg = range(i, j + 1)
            out.append(max(seg, key=lambda q: t1[q]))
            i = j + 1
        else:
            i += 1
    if closed and len(out) >= 2 and cand[0] and cand[-1]:   # a corner run across the seam
        first, last = out[0], out[-1]
        out = out[1:-1] + [first if t1[first] >= t1[last] else last]
    return sorted(out)


def _dp(pts: np.ndarray, tol: float) -> list[int]:
    """Douglas-Peucker: indices of the points kept."""
    keep = [0, len(pts) - 1]
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        p, q = pts[a], pts[b]
        d = q - p
        L = math.hypot(*d)
        seg = pts[a + 1:b]
        if L < 1e-12:
            dev = np.linalg.norm(seg - p, axis=1)
        else:
            dev = np.abs(d[0] * (seg[:, 1] - p[1]) - d[1] * (seg[:, 0] - p[0])) / L
        m = int(np.argmax(dev))
        if dev[m] > tol:
            c = a + 1 + m
            keep.append(c)
            stack += [(a, c), (c, b)]
    return sorted(set(keep))


def _straight(span: np.ndarray, tol: float):
    """If a stretch is straight, the end points of its best-fit line, else None. Straight
    means every point within `tol` of the line, or scattered noise: within 2 x tol with a
    typical (RMS) offset under tol / 2."""
    if len(span) < 3 or np.linalg.norm(span[-1] - span[0]) <= 0.3:
        return None
    c = span.mean(axis=0)
    _, _, vt = np.linalg.svd(span - c, full_matrices=False)
    d = vt[0]
    off = (span - c) @ np.array([-d[1], d[0]])
    worst, rms = float(np.abs(off).max()), float(np.sqrt(np.mean(off ** 2)))
    # A short arc (part of a small circle: a clock face, a letter's bowl) lies within `tol`
    # of a line too, but it bows steadily to one side. Straightening such arcs turns a
    # circle into a hexagon. Noise doesn't bow: its middle sits where its ends do.
    third = max(1, len(off) // 3)
    bow = abs(float(off[third:-third].mean() if len(off) > 2 * third else off.mean())
              - float(np.concatenate([off[:third], off[-third:]]).mean()))
    if bow > 0.1 and bow > 0.04 * float(np.linalg.norm(span[-1] - span[0])):
        return None
    if worst <= tol or (worst <= 2 * tol and rms <= tol / 2):
        t0, t1 = float((span[0] - c) @ d), float((span[-1] - c) @ d)
        return np.vstack([c + d * t0, c + d * t1])
    return None


def _chord_dev(span: np.ndarray) -> float:
    p, q = span[0], span[-1]
    d = q - p
    L = math.hypot(*d)
    if L < 1e-9:
        return float(np.max(np.linalg.norm(span - p, axis=1)))
    return float(np.max(np.abs(d[0] * (span[:, 1] - p[1]) - d[1] * (span[:, 0] - p[0])) / L))


# ------------------------------------------------------------------------------- fitting

def fit(coords, closed: bool, tol: float = TOL_MM) -> list[dict]:
    """Fit a ring (closed) or a line (open) as a list of pieces, each
    {"kind": "line" | "curve", "pts": array}. Consecutive pieces share their end points."""
    pts = np.asarray(coords, float)
    if closed and len(pts) > 1 and np.linalg.norm(pts[0] - pts[-1]) < 1e-9:
        pts = pts[:-1]
    if len(pts) < (3 if closed else 2):
        return []
    raw = _resample(pts, closed)
    if len(raw) < (4 if closed else 2):
        return [{"kind": "line", "pts": np.asarray(coords, float)}]
    sm = _smooth(raw, closed, max(1, int(round(SMOOTH_MM / STEP_MM))))
    cs = _corners(sm, closed)
    n = len(sm)
    if closed:
        if not cs:
            spans = [np.vstack([sm, sm[:1]])]
            kinds = None
        else:
            spans = []
            for a, b in zip(cs, cs[1:] + [cs[0] + n]):
                idx = np.arange(a, b + 1) % n
                spans.append(sm[idx])
    else:
        cuts = [0] + cs + [n - 1]
        spans = [sm[a:b + 1] for a, b in zip(cuts, cuts[1:]) if b > a]
    pieces = []
    for span in spans:
        if len(span) < 2:
            continue
        line = _straight(span, tol)
        if line is not None:
            pieces.append({"kind": "line", "pts": line})
            continue
        # A curve, or a run of straight parts joined by gentle bends: keep few points, and
        # mark long straight parts as lines so they can be snapped too.
        keep = _dp(span, CURVE_TOL_MM)
        coarse = _dp(span, tol)
        long_lines = {(coarse[i], coarse[i + 1]) for i in range(len(coarse) - 1)
                      if np.linalg.norm(span[coarse[i + 1]] - span[coarse[i]]) >= 2.5}
        cur = [keep[0]]
        for i in range(len(coarse) - 1):
            a, b = coarse[i], coarse[i + 1]
            if (a, b) in long_lines:
                if len(cur) > 1:
                    pieces.append({"kind": "curve", "pts": span[cur]})
                pieces.append({"kind": "line", "pts": np.vstack([span[a], span[b]])})
                cur = [b]
            else:
                cur += [q for q in keep if a < q <= b]
        if len(cur) > 1:
            pieces.append({"kind": "curve", "pts": span[cur]})
    return pieces


def _angle(p, q) -> float:
    return math.degrees(math.atan2(q[1] - p[1], q[0] - p[0])) % 180.0


def main_axes(piece_lists: list[list[dict]]) -> list[float]:
    """The design's main directions: level and upright always, plus any direction that
    carries a good share of all straight length (a building drawn at an angle)."""
    hist = np.zeros(180)
    for pieces in piece_lists:
        for pc in pieces:
            if pc["kind"] != "line":
                continue
            p, q = pc["pts"][0], pc["pts"][-1]
            L = float(np.linalg.norm(q - p))
            if L >= 1.0:
                hist[int(_angle(p, q)) % 180] += L
    axes = [0.0, 90.0]
    total = hist.sum()
    if total > 0:
        sm = np.array([hist[(np.arange(i - 2, i + 3)) % 180].sum() for i in range(180)])
        for i in np.argsort(sm)[::-1][:4]:
            if sm[i] >= AXIS_SHARE * total and all(
                    min(abs(i - a) % 180, 180 - abs(i - a) % 180) > 6 for a in axes):
                axes.append(float(i))
    return axes


def _snap(pieces: list[dict], axes: list[float], closed: bool,
          snap_deg: float = SNAP_DEG, snap_max_deg: float = SNAP_MAX_DEG) -> list[dict]:
    """Turn near-axis straight pieces exactly onto the axis, then re-make the corners where
    neighbouring straight pieces meet."""
    out = []
    for pc in pieces:
        if pc["kind"] != "line":
            out.append(pc)
            continue
        p, q = pc["pts"][0], pc["pts"][-1]
        L = float(np.linalg.norm(q - p))
        ang = _angle(p, q)
        allow = min(snap_max_deg, max(snap_deg, math.degrees(math.atan2(0.15, max(L, 1e-6)))))
        best = None
        for a in axes:
            d = (ang - a + 90.0) % 180.0 - 90.0
            if abs(d) <= allow and (best is None or abs(d) < abs(best[1])):
                best = (a, d)
        if best is None:
            out.append(pc)
            continue
        m = (p + q) / 2
        t = math.radians(best[0])
        u = np.array([math.cos(t), math.sin(t)])
        if np.dot(q - p, u) < 0:
            u = -u
        out.append({"kind": "line", "pts": np.vstack([m - u * L / 2, m + u * L / 2]),
                    "snapped": True})
    # re-make shared corners
    n = len(out)
    pairs = range(n) if closed else range(n - 1)
    for i in pairs:
        a, b = out[i], out[(i + 1) % n]
        if a["kind"] == "line" and b["kind"] == "line":
            p1, p2 = a["pts"][0], a["pts"][-1]
            p3, p4 = b["pts"][0], b["pts"][-1]
            d1, d2 = p2 - p1, p4 - p3
            den = d1[0] * d2[1] - d1[1] * d2[0]
            n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
            if n1 > 1e-9 and n2 > 1e-9 and abs(den) / (n1 * n2) > math.sin(math.radians(15)):
                s = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / den
                x = p1 + d1 * s
                if np.linalg.norm(x - p2) < 1.0 and np.linalg.norm(x - p3) < 1.0:
                    a["pts"] = np.vstack([a["pts"][0], x])
                    b["pts"] = np.vstack([x, b["pts"][-1]])
                    continue
            mid = (p2 + p3) / 2
            a["pts"] = np.vstack([a["pts"][0], mid])
            b["pts"] = np.vstack([mid, b["pts"][-1]])
        elif a["kind"] == "line" or b["kind"] == "line":
            # a straight piece next to a curve: the curve follows the straight piece's end
            if a["kind"] == "line":
                b["pts"] = np.vstack([a["pts"][-1], b["pts"][1:]])
            else:
                a["pts"] = np.vstack([a["pts"][:-1], b["pts"][0]])
    return out


def _join(pieces: list[dict]) -> np.ndarray:
    pts = []
    for pc in pieces:
        p = pc["pts"]
        if pts and np.linalg.norm(pts[-1] - p[0]) < 1e-9:
            pts.extend(p[1:])
        else:
            pts.extend(p)
    return np.asarray(pts)


def clean_ring(coords, axes: list[float], pieces: list[dict] | None = None) -> np.ndarray | None:
    pieces = pieces if pieces is not None else fit(coords, True)
    if not pieces:
        return None
    pts = _join(_snap([dict(p) for p in pieces], axes, True))
    return pts if len(pts) >= 3 else None


def clean_line(coords, axes: list[float]) -> np.ndarray:
    pieces = fit(coords, False)
    if not pieces:
        return np.asarray(coords, float)
    return _join(_snap(pieces, axes, False))


def clean_polygons(polys: list[Polygon], axes: list[float] | None = None) -> list[Polygon]:
    """Redraw polygons (with holes) cleanly. Shapes that collapse are kept as they were."""
    fits = [(p, [fit(p.exterior.coords, True)] + [fit(r.coords, True) for r in p.interiors])
            for p in polys]
    if axes is None:
        axes = main_axes([pl for _, f in fits for pl in f])
    out = []
    for p, f in fits:
        ext = clean_ring(p.exterior.coords, axes, f[0])
        if ext is None:
            out.append(p)
            continue
        holes = []
        for r, fr in zip(p.interiors, f[1:]):
            h = clean_ring(r.coords, axes, fr)
            if h is not None:
                holes.append(h)
        q = Polygon(ext, holes).buffer(0)
        parts = [g for g in polygons_of(q) if g.area > 0.02]
        if not parts or abs(sum(g.area for g in parts) - p.area) > 0.25 * p.area + 0.5:
            out.append(p)                        # fitting went wrong: keep the trace
        else:
            out.extend(parts)
    return out


def all_pieces(polys: list[Polygon]) -> list[list[dict]]:
    return [fit(r.coords, True) for p in polys for r in [p.exterior, *p.interiors]]


def clean_lines(lines: list[LineString], axes: list[float]) -> list[LineString]:
    out = []
    for ln in lines:
        pts = clean_line(ln.coords, axes)
        if len(pts) >= 2 and LineString(pts).length > 0.05:
            out.append(LineString(pts))
        else:
            out.append(ln)
    return out
