"""Embroidery lettering: words set in a stitch font instead of traced from a picture.

Professional digitizers rarely trace text. They retype it in an embroidery font, because a
traced letter inherits every flaw of the picture (research report, rule 6). This module is
our own embroidery font and the code that sews it.

The font
  Every letter is drawn as the centre lines of its strokes (pen strokes, not outlines), on a
  grid where the baseline is 0, the cap height 1, the x-height 0.7 and descenders reach
  -0.25. A stroke becomes one satin column of an even width; small letters become a bean
  (triple) run along the same lines. Our own design, written for this app: plain geometric
  block letters (in the spirit of the "Block" fonts on home machines), nothing copied.

How a stroke turns into a satin column (research report, rules 5, 15, 16, 24, 36)
  * Ends that sit on the baseline, cap line or x-height are cut flat along that line, the
    way embroidery block fonts cut the feet of A, V and M.
  * Two strokes that meet end to end (the corner of an L) are mitred along the bisector.
  * A stroke that ends on another stroke (the arms of an E) reaches under it by half a
    stroke width and is sewn first, so the crossing stroke covers its end cleanly.
  * Strokes are sewn left to right through the letter, letters left to right.
  * Cap height under 4 mm: bean-run lettering. 4-6 mm: satin of at least 1 mm with light
    settings. Columns that would be wider than satin can carry become a fill.

Styles
  block   even strokes about 14 % of the cap height
  bold    heavier strokes, about 20 %
  varsity bold with flat slab serifs on stem ends (only at 8 mm and up: serifs fill in on
          smaller letters, research rule 5)
"""

from __future__ import annotations

import math
import re

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import substring, unary_union

from .common import polygons_of

STEP = 0.25                  # mm between centre-line points (as for traced satin)
GUIDES = (0.0, 1.0, 0.7, -0.25)
SPACE = 0.42                 # width of a space, in cap heights
TRACKING = 0.13              # gap between letters, in cap heights
STYLES = {
    "block": {"name": "Block", "weight": 0.14},
    "bold": {"name": "Bold", "weight": 0.20},
    "varsity": {"name": "Varsity", "weight": 0.19, "serifs": True},
}
RUN_BELOW_MM = 4.0           # cap height under which letters are bean runs (rule 5)
MIN_SATIN_MM = 1.0           # narrowest satin stroke for small letters (rule 5)
SATIN_MAX_RATIO = 0.2        # a satin stroke wider than this x the cap height closes the
                             # letters up, so 1 mm satin needs letters of at least 5 mm
SATIN_TEXT_MM = 6.0          # satin lettering starts at this cap height; smaller letters are
                             # runs (running_stitch.md: 4-6 mm double run or bean)
NARROW_TEXT_MM = 4.0         # from here up to SATIN_TEXT_MM, letters are a narrow satin
NARROW_MIN_MM = 0.8          # (0.8-1.2 mm): bean letters sink into twill, pique and caps and
                             # look pale from 3 ft; a narrow satin reads bolder (Embroidery
                             # Legacy group, members' advice on small left-chest lettering)
SPLIT_SATIN_MAX = 1.7        # split satin serves strokes up to about 12 mm (1.7 x the
                             # longest stitch); wider strokes are filled
LIGHT_MM = 0.85              # lettering traced thinner than this is retyped as a fine line
LETTER_MAX_RATIO = 0.3       # traced "letters" heavier than this x their height aren't text
SERIF_MIN_MM = 8.0

# ---------------------------------------------------------------------------------- the font
# M x y = start a stroke, L x y = straight line, A cx cy rx ry a0 a1 = elliptical arc from
# angle a0 to a1 in degrees (counter-clockwise when a1 > a0), D x y = a dot, | = next stroke.
FONT = {
    "A": "M 0 0 L .32 1 | M .64 0 L .32 1 | M .1 .3 L .54 .3",
    "B": "M 0 0 L 0 1 | M 0 1 L .3 1 A .3 .76 .24 .24 90 -90 L 0 .52 "
         "| M 0 .52 L .32 .52 A .32 .26 .26 .26 90 -90 L 0 0",
    "C": "A .36 .5 .36 .5 50 310",
    "D": "M 0 0 L 0 1 | M 0 1 L .24 1 A .24 .5 .36 .5 90 -90 L 0 0",
    "E": "M 0 0 L 0 1 | M 0 1 L .52 1 | M 0 .52 L .44 .52 | M 0 0 L .54 0",
    "F": "M 0 0 L 0 1 | M 0 1 L .52 1 | M 0 .5 L .44 .5",
    "G": "A .37 .5 .37 .5 48 360 | M .4 .44 L .70 .44",
    "H": "M 0 0 L 0 1 | M .6 0 L .6 1 | M 0 .52 L .6 .52",
    "I": "M 0 0 L 0 1",
    "J": "M .42 1 L .42 .25 A .21 .25 .21 .25 0 -160",
    "K": "M 0 0 L 0 1 | M .56 1 L 0 .36 | M .2 .58 L .58 0",
    "L": "M 0 1 L 0 0 | M 0 0 L .5 0",
    "M": "M 0 0 L 0 1 | M 0 1 L .4 0 | M .4 0 L .8 1 | M .8 1 L .8 0",
    "N": "M 0 0 L 0 1 | M 0 1 L .6 0 | M .6 0 L .6 1",
    "O": "A .38 .5 .38 .5 90 450",
    "P": "M 0 0 L 0 1 | M 0 1 L .3 1 A .3 .74 .26 .26 90 -90 L 0 .48",
    "Q": "A .38 .5 .38 .5 90 450 | M .46 .24 L .76 -.04",
    "R": "M 0 0 L 0 1 | M 0 1 L .3 1 A .3 .74 .26 .26 90 -90 L 0 .48 | M .3 .48 L .6 0",
    "S": "A .3 .75 .28 .25 30 270 A .3 .25 .3 .25 90 -150",
    "T": "M .3 0 L .3 1 | M 0 1 L .6 1",
    "U": "M 0 1 L 0 .3 A .3 .3 .3 .3 180 360 L .6 1",
    "V": "M 0 1 L .32 0 | M .64 1 L .32 0",
    "W": "M 0 1 L .22 0 | M .22 0 L .44 1 | M .44 1 L .66 0 | M .66 0 L .88 1",
    "X": "M 0 1 L .6 0 | M 0 0 L .6 1",
    "Y": "M 0 1 L .3 .5 | M .6 1 L .3 .5 | M .3 .5 L .3 0",
    "Z": "M 0 1 L .56 1 | M .56 1 L 0 0 | M 0 0 L .56 0",
    "a": "A .26 .35 .26 .35 90 450 | M .52 .7 L .52 0",
    "b": "M 0 1 L 0 0 | A .27 .35 .27 .35 90 450",
    "c": "A .27 .35 .27 .35 45 315",
    "d": "A .27 .35 .27 .35 90 450 | M .54 1 L .54 0",
    "e": "M 0 .35 L .54 .35 | A .27 .35 .27 .35 0 318",
    "f": "M .14 0 L .14 .78 A .36 .78 .22 .2 180 60 | M 0 .68 L .38 .68",
    "g": "A .27 .35 .27 .35 90 450 | M .54 .7 L .54 -.05 A .27 -.05 .27 .2 0 -160",
    "h": "M 0 1 L 0 0 | A .26 .44 .26 .26 150 0 L .52 0",
    "i": "M 0 0 L 0 .7 | D 0 .9",
    "j": "M .2 .7 L .2 -.05 A 0 -.05 .2 .2 0 -150 | D .2 .9",
    "k": "M 0 1 L 0 0 | M .44 .7 L 0 .26 | M .17 .43 L .48 0",
    "l": "M 0 1 L 0 0",
    "m": "M 0 .7 L 0 0 | A .2 .46 .2 .24 150 0 L .4 0 | A .6 .46 .2 .24 150 0 L .8 0",
    "n": "M 0 .7 L 0 0 | A .26 .44 .26 .26 150 0 L .52 0",
    "o": "A .28 .35 .28 .35 90 450",
    "p": "M 0 .7 L 0 -.25 | A .27 .35 .27 .35 90 450",
    "q": "A .27 .35 .27 .35 90 450 | M .54 .7 L .54 -.25",
    "r": "M 0 .7 L 0 0 | A .26 .44 .26 .26 150 75",
    "s": "A .24 .53 .22 .17 25 270 A .24 .18 .24 .18 90 -155",
    "t": "M .16 .95 L .16 .16 A .36 .16 .2 .16 180 300 | M 0 .68 L .42 .68",
    "u": "M 0 .7 L 0 .27 A .26 .27 .26 .27 180 330 | M .52 .7 L .52 0",
    "v": "M 0 .7 L .26 0 | M .52 .7 L .26 0",
    "w": "M 0 .7 L .18 0 | M .18 0 L .36 .7 | M .36 .7 L .54 0 | M .54 0 L .72 .7",
    "x": "M 0 .7 L .5 0 | M 0 0 L .5 .7",
    "y": "M .54 .7 L .14 -.25 | M 0 .7 L .25 0",
    "z": "M 0 .7 L .48 .7 | M .48 .7 L 0 0 | M 0 0 L .5 0",
    "0": "A .3 .5 .3 .5 90 450",
    "1": "M .3 0 L .3 1 | M .3 1 L .06 .78",
    "2": "A .28 .72 .28 .28 155 -35 L 0 0 | M 0 0 L .58 0",
    "3": "A .27 .75 .25 .25 150 -90 | A .27 .26 .3 .26 90 -150",
    "4": "M .46 0 L .46 1 | M .46 1 L 0 .3 | M 0 .3 L .62 .3",
    "5": "M .54 1 L .07 1 | M .07 1 L .068 .546 | A .28 .32 .3 .32 135 -150",
    "6": "A .3 .3 .3 .3 90 450 | A .62 .32 .62 .66 100 180",
    "7": "M 0 1 L .56 1 | M .56 1 L .18 0",
    "8": "A .29 .76 .24 .24 90 450 | A .29 .26 .29 .26 90 450",
    "9": "A .3 .7 .3 .3 270 630 | A -.02 .68 .62 .66 0 -80",
    ".": "D 0 0",
    ",": "D 0 0 | M .02 -.02 L -.06 -.18",
    "!": "M 0 .3 L 0 1 | D 0 0",
    "?": "A .25 .74 .25 .26 160 -60 L .25 .32 | D .25 0",
    "-": "M 0 .42 L .34 .42",
    "'": "M 0 1 L 0 .74",
    '"': "M 0 1 L 0 .74 | M .2 1 L .2 .74",
    ":": "D 0 0 | D 0 .55",
    ";": "D 0 .55 | D 0 0 | M .02 -.02 L -.06 -.18",
    "/": "M 0 0 L .45 1",
    "+": "M 0 .42 L .5 .42 | M .25 .17 L .25 .67",
    "&": "A .22 .78 .18 .2 340 120 A .22 .78 .18 .2 120 220 L .62 0 "
         "| M .56 .32 A .3 .26 .26 .26 0 -90 A .3 .26 .3 .26 -90 -230 L .28 .62",
    "#": "M .18 0 L .26 1 | M .44 0 L .52 1 | M 0 .34 L .66 .34 | M .04 .68 L .7 .68",
    "(": "A .3 .5 .3 .6 125 235",
    ")": "A 0 .5 .3 .6 55 -55",
    "$": "A .3 .75 .28 .25 30 270 A .3 .25 .3 .25 90 -150 | M .3 1.1 L .3 -.1",
}


def _arc(cx, cy, rx, ry, a0, a1):
    n = max(6, int(math.ceil(abs(a1 - a0) / 3.0)))
    t = np.radians(np.linspace(a0, a1, n + 1))
    return list(zip(cx + rx * np.cos(t), cy + ry * np.sin(t)))


def _parse(spec: str) -> list[dict]:
    """The strokes of one letter: [{"pts": [(x, y), ...], "dot": bool}]."""
    strokes = []
    for part in spec.split("|"):
        toks = part.split()
        pts, dots, i = [], [], 0
        while i < len(toks):
            cmd = toks[i]
            if cmd in ("M", "L"):
                pts.append((float(toks[i + 1]), float(toks[i + 2])))
                i += 3
            elif cmd == "A":
                arc = _arc(*(float(v) for v in toks[i + 1:i + 7]))
                pts.extend(arc[1:] if pts and math.dist(pts[-1], arc[0]) < 1e-3 else arc)
                i += 7
            elif cmd == "D":
                dots.append((float(toks[i + 1]), float(toks[i + 2])))
                i += 3
            else:
                raise ValueError(f"bad font command {cmd!r}")
        if len(pts) >= 2:
            strokes.append({"pts": pts, "dot": False})
        strokes.extend({"pts": [d], "dot": True} for d in dots)
    return strokes


def _build_font() -> dict:
    font = {}
    for ch, spec in FONT.items():
        strokes = _parse(spec)
        xs = [x for s in strokes for x, _ in s["pts"]]
        x0 = min(xs)
        for s in strokes:
            s["pts"] = [(x - x0, y) for x, y in s["pts"]]
        body = [p for s in strokes if not s["dot"] for p in s["pts"]] or \
               [p for s in strokes for p in s["pts"]]
        font[ch] = {"strokes": strokes, "width": max(xs) - x0,
                    "y0": min(y for _, y in body), "y1": max(y for _, y in body)}
    return font


GLYPHS = _build_font()


def supported(ch: str) -> bool:
    return ch in GLYPHS or ch.isspace()


def letters_of(text: str) -> list[str]:
    """The characters of `text` that are sewn as their own main shape (letters and digits)."""
    return [ch for ch in text if ch.isalnum() and ch in GLYPHS]


# ----------------------------------------------------------------------- geometry helpers

def _unit(v):
    n = math.hypot(v[0], v[1])
    return (0.0, 0.0) if n < 1e-12 else (v[0] / n, v[1] / n)


def _out_dir(pts, end: int):
    """Unit direction pointing out of a polyline at its start (end=0) or finish (end=1)."""
    if end == 0:
        a = pts[0]
        b = next((p for p in pts[1:] if math.dist(p, a) > 1e-6), pts[-1])
        return _unit((a[0] - b[0], a[1] - b[1]))
    a = pts[-1]
    b = next((p for p in reversed(pts[:-1]) if math.dist(p, a) > 1e-6), pts[0])
    return _unit((a[0] - b[0], a[1] - b[1]))


def _resample(pts: np.ndarray, step: float = STEP) -> np.ndarray:
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-9])
    pts = pts[keep]
    if len(pts) < 2:
        return pts
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    n = max(2, int(math.ceil(cum[-1] / step)) + 1)
    s = np.linspace(0.0, cum[-1], n)
    return np.column_stack([np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])])


def _normals(c: np.ndarray, closed: bool) -> np.ndarray:
    if closed:
        t = np.roll(c, -1, axis=0) - np.roll(c, 1, axis=0)
    else:
        t = np.gradient(c, axis=0)
    # a little smoothing so stitch angles turn gently on the corners of the polyline
    k = np.array([0.25, 0.5, 0.25])
    if len(t) >= 3:
        if closed:
            t = np.column_stack([np.convolve(np.concatenate([t[-1:, i], t[:, i], t[:1, i]]), k, "valid")
                                 for i in range(2)])
        else:
            pad = np.vstack([t[:1], t, t[-1:]])
            t = np.column_stack([np.convolve(pad[:, i], k, "valid") for i in range(2)])
    n = np.linalg.norm(t, axis=1, keepdims=True)
    t = t / np.where(n < 1e-12, 1.0, n)
    return np.column_stack([-t[:, 1], t[:, 0]])


def _cross_at(rail: np.ndarray, P, u, sign: float, start: int, forward: bool):
    """Walk a rail from index `start` (inside the cut) toward one end and return
    (last inside index, crossing point), or None if it never crosses the cut line."""
    def f(p):
        return sign * (u[0] * (p[1] - P[1]) - u[1] * (p[0] - P[0]))
    idx = range(start, len(rail)) if forward else range(start, -1, -1)
    prev = None
    for i in idx:
        v = f(rail[i])
        if v < 0:
            if prev is None:
                return None
            j, pv = prev
            t = pv / (pv - v)
            return j, rail[j] + (rail[i] - rail[j]) * t
        prev = (i, v)
    return None


def _repair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two rail pieces resampled to the same number of points by their own lengths."""
    def lengths(p):
        return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    la, lb = lengths(a), lengths(b)
    n = max(2, int(math.ceil(max(la[-1], lb[-1]) / STEP)) + 1)

    def resample(p, cum):
        if cum[-1] < 1e-9:
            return np.repeat(p[:1], n, axis=0)
        s = np.linspace(0.0, cum[-1], n)
        return np.column_stack([np.interp(s, cum, p[:, 0]), np.interp(s, cum, p[:, 1])])
    return resample(a, la), resample(b, lb)


def column(pts, width: float, start_cut=None, end_cut=None) -> dict | None:
    """A satin column along a centre line, `width` mm wide. Each cut is None (square end),
    ("ext", mm) (square end that far past the point) or ("line", point, direction) (the end
    is cut along that line, after extending the stroke enough to reach it).
    Returns {"center", "half", "left", "right", "outline"} or None if too short."""
    pts = np.asarray(pts, float)
    closed = len(pts) > 3 and np.linalg.norm(pts[0] - pts[-1]) < 1e-6
    c = _resample(pts)
    if len(c) < 2:
        return None
    half = width / 2.0
    if closed:
        n = _normals(c[:-1], True)
        n = np.vstack([n, n[:1]])
        left, right = c + n * half, c - n * half
        return _finish(left, right)

    cuts = [start_cut, end_cut]
    ext = [0.0, 0.0]
    for e, cut in enumerate(cuts):
        if cut is None:
            continue
        ext[e] = cut[1] if cut[0] == "ext" else 2.2 * width
    head, tail = [], []
    if ext[0] > 0:
        d = _out_dir([tuple(p) for p in c], 0)
        m = max(1, int(math.ceil(ext[0] / STEP)))
        head = [c[0] + np.array(d) * ext[0] * k / m for k in range(m, 0, -1)]
    if ext[1] > 0:
        d = _out_dir([tuple(p) for p in c], 1)
        m = max(1, int(math.ceil(ext[1] / STEP)))
        tail = [c[-1] + np.array(d) * ext[1] * k / m for k in range(1, m + 1)]
    i0 = len(head)                                   # index of the original first point
    i1 = i0 + len(c) - 1                             # ...and the original last point
    full = np.vstack([np.array(head).reshape(-1, 2), c, np.array(tail).reshape(-1, 2)])
    n = _normals(full, False)
    L, R = full + n * half, full - n * half

    # clip each rail where it crosses a cut line; lo/hi = first/last kept index, plus the
    # crossing points themselves
    clip = {"L": [0, len(full) - 1, None, None], "R": [0, len(full) - 1, None, None]}
    back = max(1, int(round(width / STEP)))
    for e, cut in enumerate(cuts):
        if not cut or cut[0] != "line":
            continue
        P, u = np.asarray(cut[1], float), _unit(cut[2])
        ref_i = min(i1, i0 + back) if e == 0 else max(i0, i1 - back)
        ref = full[ref_i]
        sign = 1.0 if (u[0] * (ref[1] - P[1]) - u[1] * (ref[0] - P[0])) >= 0 else -1.0
        for name, rail in (("L", L), ("R", R)):
            hit = _cross_at(rail, P, u, sign, ref_i, forward=(e == 1))
            if hit is None:
                continue
            j, q = hit
            if e == 1:
                clip[name][1], clip[name][3] = j, q
            else:
                clip[name][0], clip[name][2] = j, q

    def piece(rail, name, a, b):
        """The rail between original indices a..b, with crossing points at the ends."""
        lo, hi, qs, qe = clip[name]
        a, b = max(a, lo), min(b, hi)
        mid = rail[a:b + 1]
        parts = []
        if qs is not None and a == lo:
            parts.append(qs[None])
        parts.append(mid)
        if qe is not None and b == hi:
            parts.append(qe[None])
        return np.vstack(parts)

    # body pairs by index; each end zone is re-paired by length so a slanted cut fans
    # gently instead of bunching stitches at one corner
    zone = max(2, int(round(1.5 * width / STEP)))
    m0 = max(clip["L"][0], clip["R"][0]) + (zone if cuts[0] and cuts[0][0] == "line" else 0)
    m1 = min(clip["L"][1], clip["R"][1]) - (zone if cuts[1] and cuts[1][0] == "line" else 0)
    if m1 - m0 < 2:
        a, b = _repair(piece(L, "L", 0, len(full) - 1), piece(R, "R", 0, len(full) - 1))
        return _finish(a, b)
    hl, hr = _repair(piece(L, "L", 0, m0), piece(R, "R", 0, m0))
    tl, tr = _repair(piece(L, "L", m1, len(full) - 1), piece(R, "R", m1, len(full) - 1))
    left = np.vstack([hl[:-1], L[m0:m1], tl])
    right = np.vstack([hr[:-1], R[m0:m1], tr])
    return _finish(left, right)


def _finish(left: np.ndarray, right: np.ndarray) -> dict | None:
    center = (left + right) / 2.0
    if len(center) < 2 or float(np.sum(np.linalg.norm(np.diff(center, axis=0), axis=1))) < 0.4:
        return None
    half = np.linalg.norm(left - right, axis=1) / 2.0
    closed = np.linalg.norm(center[0] - center[-1]) < 1e-6 and len(center) > 8
    if closed:
        a, b = Polygon(left).buffer(0), Polygon(right).buffer(0)
        outer, inner = (a, b) if a.area >= b.area else (b, a)
        outline = outer.difference(inner)
    else:
        outline = Polygon(np.vstack([left, right[::-1]])).buffer(0)
    return {"center": center, "half": half, "left": left, "right": right, "outline": outline}


# ------------------------------------------------------------------------------- layout

class Placement:
    """Where one letter goes: its baseline-left corner (bx, by) in mm (y grows downward),
    cap height H, stroke width w and a horizontal stretch k (1 = as designed)."""

    def __init__(self, ch, bx, by, H, w, k=1.0, run=False):
        self.ch, self.bx, self.by, self.H, self.w, self.k = ch, bx, by, H, w, k
        self.run = run          # sew as a fine bean line (light lettering) even if big enough
        self.heavy = False      # retyped from bold lettering: a running-stitch letter is sewn bold
        self.s = max(H - w, 0.3 * H)

    def xy(self, x, y):
        return (self.bx + self.w / 2 + x * self.s * self.k, self.by - (self.w / 2 + y * self.s))

    def outer_width(self):
        return GLYPHS[self.ch]["width"] * self.s * self.k + self.w


def _endpoint_kind(strokes, si, end, wu):
    """How a stroke end meets the rest of the letter (all in font units)."""
    pts = strokes[si]["pts"]
    p = pts[0] if end == 0 else pts[-1]
    d = _out_dir(pts, end)
    for g in GUIDES:
        if abs(p[1] - g) < 1e-3 and abs(d[1]) >= 0.5:
            # cut flat on the line; if it ends on another stroke (the stem of a T under its
            # bar) it is still sewn first, so the other stroke covers its end
            under = None
            for sj, other in enumerate(strokes):
                op = other["pts"]
                if sj == si or other["dot"] or min(math.dist(p, op[0]), math.dist(p, op[-1])) < 0.03:
                    continue
                if LineString(op).distance(Point(p)) < wu * 0.55:
                    under = sj
            return ("guide", g, 1.0 if d[1] > 0 else -1.0), under
    meet, inside = [], None
    for sj, other in enumerate(strokes):
        if sj == si or other["dot"]:
            continue
        op = other["pts"]
        ends = [] if math.dist(op[0], op[-1]) < 1e-6 else [(0, op[0]), (1, op[-1])]
        hit_end = [(e2, q) for e2, q in ends if math.dist(p, q) < 0.03]
        if hit_end:
            meet.append((sj, hit_end[0][0]))
            continue
        if LineString(op).distance(Point(p)) < wu * 0.55:
            inside = sj
    if len(meet) == 1 and inside is None:
        sj, e2 = meet[0]
        return ("miter", _out_dir(strokes[sj]["pts"], e2)), None
    if meet or inside is not None:
        return ("tuck",), inside
    return ("free",), None


def _touches(strokes, si, end, wu) -> bool:
    """Does this stroke end meet another stroke (a corner, or the end of a T's stem)?"""
    p = strokes[si]["pts"][0 if end == 0 else -1]
    for sj, other in enumerate(strokes):
        if sj != si and not other["dot"] and LineString(other["pts"]).distance(Point(p)) < wu * 0.55:
            return True
    return False


def _serifs(strokes, kinds, wu):
    """Flat slab serifs on the free ends of stems that sit on the baseline or cap line."""
    out = []
    for si, s in enumerate(strokes):
        if s["dot"]:
            continue
        for end in (0, 1):
            kind = kinds[si][end]
            if kind[0] != "guide" or kind[1] not in (0.0, 1.0):
                continue
            d = _out_dir(s["pts"], end)
            if abs(d[1]) < 0.97 or _touches(strokes, si, end, wu):
                continue                             # only free ends of (nearly) upright stems
            x, y = (s["pts"][0] if end == 0 else s["pts"][-1])
            yy = y + kind[2] * wu * 0.12             # sits on the line, inside the letter
            reach = 0.17
            out.append({"pts": [(x - reach, yy), (x + reach, yy)], "dot": False, "serif": True})
    return out


_PLAN_CACHE: dict = {}


JOIN_MM = 1.0     # a connecting stitch this short may show (sequence.ALWAYS_JOIN_MM)


def _plan(lines: tuple, dots: tuple, waits: tuple, wu: float, width: float,
          join: float = 0.0) -> list:
    """The order, direction and way to sew a letter's strokes so that the thread is not cut
    inside the letter. Each stroke can be sewn from either end, and either straight through
    (it ends at its far end) or out along its centre as a run and back as satin (it ends
    where it began). A move to the next stroke costs nothing when it lies under strokes
    still to be sewn (they hide the travelling thread) and counts as a trim when it would
    lie on open fabric or on finished satin (unless it is a very short connecting stitch,
    `join` font units at most). The cheapest plan is found by trying every
    order that respects which strokes tuck under which (letters have few strokes).
    Returns [(stroke, reversed, return)] with dots last."""
    key = (lines, dots, waits, wu, round(width, 3), join)
    if key in _PLAN_CACHE:
        return _PLAN_CACHE[key]
    idx = [i for i in range(len(lines)) if not dots[i]
           and math.dist(lines[i][0], lines[i][-1]) > 1e-6]
    rest = [i for i in range(len(lines)) if i not in idx]
    n = len(idx)
    if n == 0 or n > 10:
        plan = [(i, False, False) for i in range(len(lines))]
        _PLAN_CACHE[key] = plan
        return plan
    foot = [LineString(lines[i]).buffer(wu / 2.0 + 0.02, cap_style=2) for i in idx]
    ends = [(lines[i][0], lines[i][-1]) for i in idx]
    pos = {i: k for k, i in enumerate(idx)}
    need = [sum(1 << pos[j] for j in waits[i] if j in pos) for i in idx]
    covers: dict = {}

    def hidden(P, Q, done_mask):
        if math.dist(P, Q) < 1e-6:
            return True
        if done_mask not in covers:
            later = [foot[k] for k in range(n) if not done_mask >> k & 1]
            covers[done_mask] = unary_union(later) if later else None
        cov = covers[done_mask]
        return cov is not None and cov.contains(LineString([P, Q]))

    # states: (strokes sewn, where the needle is) -> (cost, plan so far)
    start, finish = (0.0, 0.0), (width, 0.0)
    best = {(0, None): (0.0, [])}
    frontier = [(0, None)]
    full = (1 << n) - 1
    for _step in range(n):
        nxt: dict = {}
        for state in frontier:
            mask, at = state
            cost, plan = best[state]
            for k in range(n):
                if mask >> k & 1 or (need[k] & mask) != need[k]:
                    continue
                for rev in (False, True):
                    a, b_ = (ends[k][1], ends[k][0]) if rev else ends[k]
                    if at is None:
                        # begin at the letter's left side (low if there is a choice):
                        # the thread arrives from the letter before
                        move = 1.0 * abs(a[0] - start[0]) + 0.1 * abs(a[1] - start[1])
                    elif hidden(at, a, mask):
                        move = 0.05 * math.dist(at, a)
                    elif math.dist(at, a) <= join:
                        move = 0.5 + 0.2 * math.dist(at, a)     # a short connector that shows
                    else:
                        move = 4.0 + math.dist(at, a)           # a trim
                    for ret in (False, True):
                        end = a if ret else b_
                        c = cost + move + (0.05 if ret else 0.0)
                        if mask | 1 << k == full:
                            # ... and end at its right side, beside the next letter, with
                            # none of the letter's other strokes in the way (a connector
                            # from inside a G would cross its finished bowl)
                            c += 1.5 * abs(end[0] - finish[0]) + 0.1 * abs(end[1] - finish[1])
                            way = LineString([end, (width + 0.3, end[1])])
                            if any(way.intersects(foot[j]) for j in range(n) if j != k):
                                c += 3.0
                        st2 = (mask | 1 << k, (round(end[0], 4), round(end[1], 4)))
                        if st2 not in nxt or c < nxt[st2][0]:
                            nxt[st2] = (c, plan + [(idx[k], rev, ret)])
        best, frontier = nxt, list(nxt)
    plan = min(best.values(), key=lambda v: v[0])[1] if best else [(i, False, False) for i in idx]
    plan = plan + [(i, False, False) for i in rest]
    if len(_PLAN_CACHE) > 4000:
        _PLAN_CACHE.clear()
    _PLAN_CACHE[key] = plan
    return plan


def letter_strokes(pl: Placement, style: str = "block") -> list[dict]:
    """The strokes of one placed letter, in sewing order, each as
    {"pts": [(x, y) mm], "cut0", "cut1", "dot"} ready for column()."""
    g = GLYPHS[pl.ch]
    strokes = [dict(s) for s in g["strokes"]]
    wu = pl.w / pl.s
    kinds, deps = [], []
    for si, s in enumerate(strokes):
        if s["dot"] or math.dist(s["pts"][0], s["pts"][-1]) < 1e-6:
            kinds.append([("free",), ("free",)])
            deps.append(set())
            continue
        k0, d0 = _endpoint_kind(strokes, si, 0, wu)
        k1, d1 = _endpoint_kind(strokes, si, 1, wu)
        kinds.append([k0, k1])
        deps.append({d for d in (d0, d1) if d is not None})
    if STYLES.get(style, {}).get("serifs") and pl.H >= SERIF_MIN_MM and pl.ch.isalnum():
        extra = _serifs(strokes, kinds, wu)
        for s in extra:
            strokes.append(s)
            kinds.append([("free",), ("free",)])
            deps.append(set())

    # order: a stroke that tucks under another is sewn before it; within that, the order,
    # the direction and the way each stroke is sewn are planned so the thread never has to
    # be cut inside the letter (see _plan)
    wait = [set() for _ in strokes]                  # strokes that must be sewn before this one
    for i, d in enumerate(deps):
        for j in d:
            wait[j].add(i)
    order = _plan(tuple(tuple(map(tuple, s["pts"])) for s in strokes),
                  tuple(bool(s["dot"]) for s in strokes),
                  tuple(frozenset(w) for w in wait), round(wu, 3), g["width"],
                  round(JOIN_MM / max(pl.s, 1e-6), 2))

    out = []
    for i, rev, ret in order:
        s = strokes[i]
        pts, ks = list(s["pts"]), list(kinds[i])
        if rev:
            pts, ks = pts[::-1], ks[::-1]
        if s["dot"]:
            x, y = pts[0]
            h = 1.1 * pl.w / pl.s / 2
            if y > 0.75:          # the dot of an i or j keeps a clear gap above the stem
                y = max(y, 0.7 + 1.8 * wu)
            out.append({"pts": [pl.xy(x, y - h), pl.xy(x, y + h)], "cut0": None, "cut1": None,
                        "dot": True})
            continue
        mm = [pl.xy(x, y) for x, y in pts]
        cuts = []
        for e, kind in enumerate(ks):
            end_pt = pts[0] if e == 0 else pts[-1]
            if kind[0] == "guide":
                _, gy, sgn = kind
                P = pl.xy(end_pt[0], gy + sgn * wu / 2)
                Q = pl.xy(end_pt[0] + 1.0, gy + sgn * wu / 2)
                cuts.append(("line", P, (Q[0] - P[0], Q[1] - P[1])))
            elif kind[0] == "miter":
                d1 = _out_dir(pts, e)
                d2 = kind[1]
                m1 = pl.xy(end_pt[0] + d1[0], end_pt[1] + d1[1])
                m2 = pl.xy(end_pt[0] + d2[0], end_pt[1] + d2[1])
                E = pl.xy(*end_pt)
                v1, v2 = _unit((m1[0] - E[0], m1[1] - E[1])), _unit((m2[0] - E[0], m2[1] - E[1]))
                cosang = v1[0] * v2[0] + v1[1] * v2[1]
                # Mitre only right-angle-ish corners. A sharp joint would grow a long spike
                # (Wilcom caps sharp corners instead), and a straight-on one needs no cut.
                if cosang > 0.5 or cosang < -0.85:
                    cuts.append(None)
                else:
                    cuts.append(("line", E, (v1[0] + v2[0], v1[1] + v2[1])))
            elif kind[0] == "tuck":
                cuts.append(("ext", pl.w / 2))
            else:
                cuts.append(None)
        out.append({"pts": mm, "cut0": cuts[0], "cut1": cuts[1], "dot": False, "ret": ret})
    return out


def width_for(H: float, style: str = "block", weight_mm: float | None = None,
              satin_max: float = 7.0) -> float:
    """Stroke width for a cap height: the style's weight (or the original lettering's own
    weight, when retyping), kept between 1 mm and what still reads as a letter."""
    w = weight_mm if weight_mm else STYLES.get(style, STYLES["block"])["weight"] * H
    lo, hi = max(MIN_SATIN_MM, 0.1 * H), SATIN_MAX_RATIO * H
    return float(min(max(w, lo), max(hi, lo)))


_EXTENT_CACHE: dict = {}


def extent(pl: Placement, style: str = "block") -> tuple[float, float]:
    """Left and right edges of a placed letter's stitches, measured from pl.bx (slanted cuts
    and mitres can reach a little past the centre lines plus half a stroke)."""
    key = (pl.ch, style, round(pl.H, 2), round(pl.w, 3), round(pl.k, 3))
    if key not in _EXTENT_CACHE:
        probe = Placement(pl.ch, 0.0, 0.0, pl.H, pl.w, pl.k)
        o = outline_of(probe, style)
        _EXTENT_CACHE[key] = (o.bounds[0], o.bounds[2]) if not o.is_empty else (0.0, probe.outer_width())
        if len(_EXTENT_CACHE) > 5000:
            _EXTENT_CACHE.clear()
    return _EXTENT_CACHE[key]


def layout(text: str, H: float, x: float, y: float, style: str = "block",
           anchor: str = "center", w: float | None = None) -> list[Placement]:
    """Set a line of text at cap height H. (x, y) is the middle of the line (anchor
    "center") or the left end of its baseline (anchor "baseline")."""
    w = w or width_for(H, style)
    pls, cur = [], 0.0
    for ch in text:
        if ch.isspace() or ch not in GLYPHS:
            cur += SPACE * H
            continue
        pl = Placement(ch, 0.0, 0.0, H, w)
        lo, hi = extent(pl, style)
        pl.bx = cur - lo
        pls.append(pl)
        cur += (hi - lo) + TRACKING * H
    total = cur - TRACKING * H if pls else 0.0
    if anchor == "center":
        dx, by = x - total / 2, y + H / 2
    else:
        dx, by = x, y
    for pl in pls:
        pl.bx += dx
        pl.by = by
    return pls


def outline_of(pl: Placement, style: str = "block") -> Polygon:
    parts = []
    for st in letter_strokes(pl, style):
        col = column(st["pts"], pl.w, st["cut0"], st["cut1"])
        if col:
            parts.append(col["outline"])
    return unary_union(parts) if parts else Polygon()


# --------------------------------------------------------------------- stitch objects

def objects(pls: list[Placement], color: int, rules: dict, style: str = "block",
            text_style: dict | None = None, group=0, row=None) -> list[dict]:
    """Classified objects (satin columns, bean runs or fills) for placed letters, in sewing
    order. Each carries "seq" so the sequencer sews them letter by letter, stroke by stroke."""
    out = []
    narrow_from = max(rules.get("text_run_mm", RUN_BELOW_MM),
                      rules.get("text_narrow_satin_mm", NARROW_TEXT_MM))
    for li, pl in enumerate(pls):
        satin_from = max(rules.get("text_run_mm", RUN_BELOW_MM),
                         MIN_SATIN_MM / SATIN_MAX_RATIO, SATIN_TEXT_MM)
        narrow = not pl.run and narrow_from <= pl.H < satin_from
        if narrow:
            pl = _as_narrow(pl)
        elif pl.run or pl.H < satin_from:
            pl = _as_line(pl)
            out.extend(_run_letter(pl, style, color, group, li, row))
            continue
        strokes = letter_strokes(pl, style)
        # wide satin is split across the column when sewn (stitch.split_long), so letters
        # stay satin up to twice the longest stitch; only wider strokes become fill
        wide = pl.w > SPLIT_SATIN_MAX * rules.get("satin_max_mm", 7.0)
        cols = []
        for si, st in enumerate(strokes):
            seq = (group, li, si)
            col = column(st["pts"], pl.w, st["cut0"], st["cut1"])
            if not col:
                continue
            if wide:
                cols.append(col["outline"])
                continue
            o = {"kind": "satin", "color": color, "center": col["center"], "half": col["half"],
                 "left": col["left"], "right": col["right"], "outline": col["outline"],
                 "width_mm": float(pl.w), "area_mm2": float(col["outline"].area), "seq": seq}
            if text_style:
                o.update(text_style)
            # The plan says where each stroke ends: at its far end, or (sewn out along its
            # centre as a run and back as satin) where it began, beside the stroke that
            # comes next, which the thread reaches travelling under strokes sewn later.
            # No trim. (Embroidery fonts are built this way: the exit of one stroke is the
            # entry of the next.)
            o["finish"] = "start" if st.get("ret") else "far"
            if row is not None:
                o["text_row"] = row
            out.append(o)
        if wide and cols:
            for p in polygons_of(unary_union(cols)):
                o = {"kind": "fill", "color": color, "poly": p, "area_mm2": float(p.area),
                     "angle": 45.0, "seq": (group, li, 0)}
                if row is not None:
                    o["text_row"] = row
                out.append(o)
    return out


LINE_W_MM = 0.45     # a running-stitch letter's line is about one thread wide


def _as_narrow(pl: Placement) -> Placement:
    """The same letter as a narrow satin (0.8-1.2 mm, about a fifth of the cap height),
    keeping its box so the spacing of the word doesn't change."""
    w = float(np.clip(SATIN_MAX_RATIO * pl.H, NARROW_MIN_MM, 1.2))
    if abs(w - pl.w) < 1e-6:
        return pl
    q = Placement(pl.ch, 0.0, pl.by, pl.H, w, pl.k, pl.run)
    gw = GLYPHS[pl.ch]["width"]
    if gw > 0.05:
        inner = gw * pl.s * pl.k + pl.w - w
        q.k = float(np.clip(inner / (gw * q.s), 0.6, 1.5))
    cx = pl.bx + pl.outer_width() / 2
    q.bx = cx - q.outer_width() / 2
    return q


def _as_line(pl: Placement) -> Placement:
    """The same letter for running stitch: its lines are a thread wide, so they reach the
    full height instead of leaving room for a satin stroke, and it keeps its box."""
    if pl.w <= LINE_W_MM + 1e-6:
        return pl
    q = Placement(pl.ch, 0.0, pl.by, pl.H, LINE_W_MM, pl.k, True)
    q.heavy = pl.heavy
    gw = GLYPHS[pl.ch]["width"]
    if gw > 0.05:
        # Half the room the satin stroke took is given back to the gap between letters:
        # thin line letters set as close as bold ones run into each other (a 9 into a 4).
        inner = gw * pl.s * pl.k + 0.5 * (pl.w - LINE_W_MM)
        q.k = float(np.clip(inner / (gw * q.s), 0.6, 1.5))
    cx = pl.bx + pl.outer_width() / 2
    q.bx = cx - q.outer_width() / 2
    return q


def _letter_tours(lines: list, tol: float = 0.08) -> list:
    """A letter's strokes as one continuous thread (or as few as possible): strokes are split
    where another stroke meets them, joined into a network, and walked out and back along
    every stroke (linenet.tours), so a running-stitch letter needs no trims inside it."""
    from . import linenet                    # noqa: PLC0415 - avoid a circular import
    ends = [np.asarray(l[0]) for l in lines] + [np.asarray(l[-1]) for l in lines]
    pieces = []
    for l in lines:
        ls = LineString(l)
        cuts = sorted({round(ls.project(Point(e)), 6) for e in ends
                       if ls.distance(Point(e)) < tol and 1e-3 < ls.project(Point(e)) < ls.length - 1e-3})
        marks = [0.0] + cuts + [ls.length]
        for a, b in zip(marks, marks[1:]):
            if b - a > 1e-3:
                pieces.append(np.asarray(substring(ls, a, b).coords))
    nodes, edges = [], []

    def node(p):
        for k, q in enumerate(nodes):
            if math.dist(p, q) < tol:
                return k
        nodes.append(tuple(p))
        return len(nodes) - 1
    for pc in pieces:
        a, b = node(pc[0]), node(pc[-1])
        pc = pc.copy()
        pc[0], pc[-1] = nodes[a], nodes[b]
        edges.append({"a": a, "b": b, "pts": pc})
    if not edges:
        return []
    return linenet.tours(np.asarray(nodes, float), edges)


def _run_letter(pl: Placement, style: str, color: int, group, li: int, row) -> list:
    """One running-stitch letter: its strokes as one continuous out-and-back thread (two even
    passes; from 4.5 mm up the way back is a bean, for a bolder line), plus round knots for
    dots."""
    strokes = letter_strokes(pl, style)
    lines = [np.asarray(st["pts"], float) for st in strokes
             if not st["dot"] and LineString(st["pts"]).length >= 0.3]
    out = []
    for k, route in enumerate(_letter_tours(lines)):
        o = {"kind": "run", "color": color, "line": LineString(np.vstack(route)), "route": route,
             # (out once and back as a bean: from 4.5 mm, or smaller when the art's own
             # lettering is bold, which as a plain two-pass line loses all its weight)
             "width_mm": 0.6, "bean": False, "bold": pl.H >= 4.5 or pl.heavy,
             # tiny letters need short stitches or their curves turn into corners (a 2 mm
             # "9" sewn in 1.25 mm stitches looks like ">")
             # (a 3 mm "O" in 1.2 mm stitches is a hexagon)
             "stitch_mm": float(np.clip(0.3 * pl.H, 0.7, 1.25)) if pl.H < 4.0 else 1.6,
             "seq": (group, li, k),
             "punct": not pl.ch.isalnum()}
        if row is not None:
            o["text_row"] = row
        out.append(o)
    for k, st in enumerate(s for s in strokes if s["dot"]):
        (x0, y0), (x1, y1) = st["pts"]
        my, a = (y0 + y1) / 2, 0.25
        # a small crossed knot, which sews as a round dot (a short dash looks like a comma)
        knot = LineString([(x0 - a, my - a), (x0 + a, my + a), (x0 + a, my - a), (x0 - a, my + a)])
        o = {"kind": "run", "color": color, "line": knot, "width_mm": 0.8, "bean": True,
             "stitch_mm": 1.25, "seq": (group, li, 100 + k), "punct": not pl.ch.isalnum()}
        if row is not None:
            o["text_row"] = row
        out.append(o)
    return out


def text_style(H: float) -> dict:
    """Underlay and density for lettering of this cap height (as classify._text_style)."""
    if H < 5.0:
        return {"underlay": "none", "spacing_scale": 1.12, "comp_scale": 0.6}
    if H < 10.0:
        return {"underlay": "center", "spacing_scale": 1.06, "comp_scale": 0.8}
    return {}


# --------------------------------------------------------------- retyping traced words

def _candidates(lines: list[str]) -> list[tuple]:
    """Every run of consecutive words in the known text, as candidates for one row:
    (text, line number, first word, last word + 1), as written and in capitals."""
    out = []
    for n, line in enumerate(lines or []):
        words = [w for w in re.split(r"\s+", line.strip()) if w]
        for a in range(len(words)):
            for b in range(a + 1, min(len(words), a + 6) + 1):
                out.append((" ".join(words[a:b]), n, a, b))
    seen, uniq = set(), []
    for c in out + [(t.upper(), n, a, b) for t, n, a, b in out]:
        if c not in seen and letters_of(c[0]):
            seen.add(c)
            uniq.append(c)
    return uniq


def _words(lines: list[str]) -> list[str]:
    return list(dict.fromkeys(c[0] for c in _candidates(lines)))


def _typical_width(p) -> float:
    return 2.0 * p.area / max(p.length, 1e-9)


def stroke_width(p) -> float:
    """The width of a letter's main strokes: how far it can be shrunk from both sides before
    almost nothing is left. Unlike 2 x area / outline, serifs and corners don't drag it down."""
    lo, hi = 0.0, max(0.2, min(p.bounds[2] - p.bounds[0], p.bounds[3] - p.bounds[1]))
    if p.area <= 0:
        return 0.0
    for _ in range(12):
        mid = (lo + hi) / 2
        if p.buffer(-mid / 2).area > 0.05 * p.area:
            lo = mid
        else:
            hi = mid
    return 0.85 * (lo + hi) / 2       # the shrink test lands on the thickest strokes


def place_on_glyphs(text: str, glyphs: list, style: str = "block",
                    satin_max: float = 7.0) -> list[Placement] | None:
    """Put each letter of `text` exactly where a traced letter was: same height, baseline,
    width and stroke weight. `glyphs` are the traced letter shapes, left to right.
    None if the counts don't match."""
    chars = letters_of(text)
    if len(chars) != len(glyphs) or not chars:
        return None
    w0 = float(np.median([stroke_width(g) for g in glyphs]))
    Hs = []
    for ch, g in zip(chars, glyphs):
        m = GLYPHS[ch]
        span = max(m["y1"] - m["y0"], 0.2)
        h = g.bounds[3] - g.bounds[1]
        Hs.append((h - w0) / span + w0)
    H = float(np.median(Hs))
    w = width_for(H, style, weight_mm=w0, satin_max=satin_max)
    s = max(H - w, 0.3 * H)
    bys = [g.bounds[3] + GLYPHS[ch]["y0"] * s for ch, g in zip(chars, glyphs)]
    by = float(np.median(bys))
    light = w0 < LIGHT_MM
    pls, it = [], iter(glyphs)
    for ch in text:
        if ch not in GLYPHS or ch.isspace():
            continue
        if not ch.isalnum():
            # punctuation is too small to have been traced as a letter: it sits just after
            # the letter before it
            if pls:
                prev = pls[-1]
                lo, _hi = extent(Placement(ch, 0.0, 0.0, H, w), style)
                pl = Placement(ch, 0.0, by, H, w, 1.0, light)
                pl.bx = prev.bx + extent(prev, style)[1] + max(0.18 * H, 0.6) - lo
                pls.append(pl)
            continue
        g = next(it)
        gw = GLYPHS[ch]["width"]
        target = g.bounds[2] - g.bounds[0]
        k = 1.0
        if gw > 0.05:
            k = float(np.clip((target - w) / (gw * s), 0.75, 1.15))   # keep letters recognisable
        pl = Placement(ch, 0.0, by, H, w, k, light)
        cx = (g.bounds[0] + g.bounds[2]) / 2
        pl.bx = cx - pl.outer_width() / 2
        pls.append(pl)
    return pls


def has_serifs(glyphs: list) -> bool:
    """Do traced letters have serifs? Serif letters are wider at their feet and tops than
    through the middle of their stems (an I, N or T with little feet)."""
    ratios = []
    for g in glyphs:
        minx, miny, maxx, maxy = g.bounds
        h = maxy - miny
        if h <= 0:
            continue

        def across(y):
            cut = g.intersection(LineString([(minx - 1, y), (maxx + 1, y)]))
            return cut.length

        mid = across(miny + 0.5 * h)
        ends = max(across(miny + 0.06 * h), across(maxy - 0.06 * h))
        if mid > 1e-6:
            ratios.append(ends / mid)
    return bool(ratios) and float(np.median(ratios)) >= 1.5


def unify(rows: list[list[Placement]]) -> None:
    """Give several rows of placed letters (parts of one line of text) one cap height, one
    stroke weight, one baseline and one way of sewing."""
    pls = [pl for row in rows for pl in row]
    if not pls:
        return
    H = float(np.median([pl.H for pl in pls]))
    w = float(np.median([pl.w for pl in pls]))
    run = sum(pl.run for pl in pls) * 2 >= len(pls)
    by = float(np.median([pl.by for pl in pls]))
    for pl in pls:
        cx = pl.bx + pl.outer_width() / 2
        pl.H, pl.w, pl.run, pl.by = H, w, run, by
        pl.s = max(H - w, 0.3 * H)
        pl.bx = cx - pl.outer_width() / 2


def overlapping(pls: list[Placement], style: str = "block", gap: float = 0.04) -> bool:
    """Do neighbouring letters run into each other (less than `gap` x the cap height apart)?
    It happens when small traced letters merge into one shape and the fitter squeezes two
    letters into one box."""
    for a, b in zip(pls, pls[1:]):
        if abs(a.by - b.by) > 0.3 * a.H:
            continue
        if b.bx + extent(b, style)[0] - (a.bx + extent(a, style)[1]) < gap * a.H:
            return True
    return False


def fit_line(text: str, box: tuple, H: float, w: float, style: str = "block",
             run: bool = False) -> list[Placement]:
    """Set a whole line of text evenly across a box (minx, miny, maxx, maxy): the line's
    cap height and weight, centred on the box, stretched or squeezed (within 0.8-1.25) to
    its width. Used when small traced letters are too merged to place one by one."""
    minx, miny, maxx, maxy = box
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    pls = layout(text, H, cx, cy, style, w=w)
    if not pls:
        return pls
    lo = pls[0].bx + extent(pls[0], style)[0]
    hi = pls[-1].bx + extent(pls[-1], style)[1]
    want = (maxx - minx) / max(hi - lo, 1e-6)
    f = float(np.clip(want, 0.8, 1.25))
    for pl in pls:
        c = pl.bx + pl.outer_width() / 2
        pl.k *= f
        pl.run = run
        pl.bx = cx + (c - cx) * f - pl.outer_width() / 2
    if want > 1.25 and len(pls) > 1:
        # widely spaced lettering (M I C R O B I O L O G Y): the rest of the width goes
        # between the letters, as drawn, rather than stretching the letters out of shape
        gap = ((maxx - minx) - (hi - lo) * f) / (len(pls) - 1)
        for i, pl in enumerate(pls):
            pl.bx += (i - (len(pls) - 1) / 2.0) * gap
    return pls


def score(pls: list[Placement], glyphs: list, style: str = "block") -> float:
    """How well retyped letters cover the traced ones: overlap / union over the whole row,
    lowered when the number of letters doesn't match the number of traced shapes (two
    touching letters trace as one shape, so a small difference is allowed)."""
    ours = unary_union([outline_of(pl, style) for pl in pls])
    theirs = unary_union(glyphs)
    # traced letters drawn much thinner than the font's strokes are compared at the font's
    # weight: otherwise even the right word covers little of them
    t = float(np.median([stroke_width(g) for g in glyphs]))
    w = float(np.median([pl.w for pl in pls]))
    if w > 1.3 * t:
        theirs = theirs.buffer((w - t) / 2.0, quad_segs=4)
    union = ours.union(theirs).area
    if not union:
        return 0.0
    iou = ours.intersection(theirs).area / union
    n, m = len(pls), len(glyphs)
    return iou * (1.0 - 0.6 * abs(n - m) / max(n, m))


PART_MIN_LETTERS = 3      # the shortest piece of a long word a broken row may be matched to


def _parts(lines: list[str]) -> list[tuple]:
    """The front and back pieces of every long word (MICRO + BIOLOGY, MICROB + IOLOGY...).
    Small lettering often traces as two rows split inside a word, where one letter merged
    with something else; neither piece spells a whole word."""
    out = []
    for n, line in enumerate(lines or []):
        words = [w for w in re.split(r"\s+", line.strip()) if w]
        for a, word in enumerate(words):
            w = word.upper()
            if len(letters_of(w)) < 2 * PART_MIN_LETTERS:
                continue
            for k in range(1, len(w)):
                if k >= PART_MIN_LETTERS:
                    out.append((w[:k], n, a, a + 1))
                if len(w) - k >= PART_MIN_LETTERS:
                    out.append((w[k:], n, a, a + 1))
    return list(dict.fromkeys(out))


def fit_row(glyphs: list, lines: list[str], style: str = "block",
            satin_max: float = 7.0, min_score: float = 0.3, parts: bool = False) -> dict | None:
    """Find which of the known words a traced row of letters spells, by setting every
    candidate where the row is and keeping the one whose letters best cover the traced
    ones. Returns {"text", "placements", "score"} or None. With `parts`, a piece of a long
    word may match too (the result then has "part": True: only usable with the rest of the
    word beside it)."""
    best = None
    n = len(glyphs)
    if not n:
        return None
    w0 = float(np.median([_typical_width(g) for g in glyphs]))
    h = float(np.median([g.bounds[3] - g.bounds[1] for g in glyphs]))
    if w0 > LETTER_MAX_RATIO * h:
        return None                  # solid blobs in a row (windows, say), not letters
    whole_best = None
    whole = [(c, False) for c in _candidates(lines)]
    for (cand, line, a, b), part in whole + ([(c, True) for c in _parts(lines)] if parts else []):
        m = len(letters_of(cand))
        if not (0.6 * n <= m <= 1.5 * n + 1):
            continue
        pls = fit_box(cand, glyphs, style, satin_max)
        if not pls:
            continue
        sc = score(pls, glyphs, style)
        if not part and (whole_best is None or sc > whole_best[0]):
            whole_best = (sc, (line, a, b))
        if best is None or sc > best["score"]:
            best = {"text": cand, "placements": pls, "score": sc, "words": (line, a, b),
                    "part": part}
    if best is None or best["score"] < min_score:
        return None
    # which known words the row looks most like as whole words: a piece is only trusted
    # as one of those (eight letters of KNOXVILLE must not become ENNESSEE, then TENNESSEE)
    best["whole_words"] = whole_best[1] if whole_best else None
    return best


def fit_box(text: str, glyphs: list, style: str = "block", satin_max: float = 7.0):
    """Set `text` where a traced row was. Letter for letter when the counts match (same
    height, baseline, width and weight as each traced letter); otherwise across the row's
    space: the row's top and bottom fix the cap height and baseline (allowing for capitals,
    ascenders and descenders in `text`), stretched or squeezed a little to its width."""
    if not letters_of(text):
        return []
    pls = place_on_glyphs(text, glyphs, style, satin_max)
    if pls is not None:
        return pls
    x0 = min(g.bounds[0] for g in glyphs)
    x1 = max(g.bounds[2] for g in glyphs)
    top = min(g.bounds[1] for g in glyphs)
    bottom = max(g.bounds[3] for g in glyphs)
    w0 = float(np.median([stroke_width(g) for g in glyphs]))
    chars = [ch for ch in text if ch in GLYPHS and not ch.isspace()]
    y0 = min(GLYPHS[ch]["y0"] for ch in chars)
    y1 = max(GLYPHS[ch]["y1"] for ch in chars)
    span = max(y1 - y0, 0.3)
    H = (bottom - top - w0) / span + w0
    w = width_for(H, style, weight_mm=w0, satin_max=satin_max)
    H = (bottom - top - w) / span + w
    s = max(H - w, 0.3 * H)
    by = bottom + y0 * s
    pls = layout(text, H, x0, by, style, anchor="baseline", w=w)
    if not pls:
        return []
    for pl in pls:
        pl.run = w0 < LIGHT_MM
    natural = pls[-1].bx + pls[-1].outer_width() - x0
    k = float(np.clip((x1 - x0) / max(natural, 1e-6), 0.6, 1.5))
    for pl in pls:
        pl.k = k
        pl.bx = x0 + (pl.bx - x0) * k
    span = pls[-1].bx + pls[-1].outer_width() - x0
    shift = ((x1 - x0) - span) / 2
    for pl in pls:
        pl.bx += shift
    return pls
