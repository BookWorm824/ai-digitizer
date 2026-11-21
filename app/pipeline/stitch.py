"""Stage 7 — Stitch.

Turn each classified object into needle points (millimetres).

  * run    evenly spaced stitches along a line (shorter on tight curves), optionally tripled
  * satin  stitches zig-zagging across a narrow column, over an underlay chosen by its width
           (under 2 mm none, 2-4 mm a centre run, wider an edge run plus a loose zig-zag)
  * fill   back-and-forth rows at an angle (a "tatami" fill) with needle points staggered row
           to row, over an edge run and a loose fill sewn across it. Rows are lengthened a
           little at both ends because stitches pull a shape narrower as they tighten.

Underlay always goes down first: it fastens the fabric to the stabilizer so the top stitches
sit flat. Thread that has to move inside a shape is routed where later stitches will cover it.

This is our own code. Nothing here is copied from Ink/Stitch.

Every object comes back as a list of "segments". A segment is one unbroken run of thread;
between segments the machine has to jump.
"""

from __future__ import annotations

import heapq
import math
from pathlib import Path

import cv2
import numpy as np
from shapely import affinity
from shapely.geometry import LineString, MultiPoint, Point, Polygon
from shapely.ops import substring, unary_union, voronoi_diagram
from shapely.prepared import prep

from .classify import column_rails
from .linenet import route_points
from .common import canvas_for, dist, polygons_of, save_debug

STAGGER = (0.0, 0.5, 0.25, 0.75)   # where needle points fall along a row, row after row
MIN_ROW_MM = 0.3                   # fill rows shorter than this are skipped
INSIDE_TOL_MM = 0.15               # how far outside a shape a hidden travel stitch may stray
UNDERLAY_STITCH_MM = 2.0           # stitch length of underlay runs


# ----------------------------------------------------------------------------- running stitch

TIGHT_CURVE_STITCH_MM = 1.0     # shortest run stitch, used only round small circles
                                # (the floor professionals give for small rings)
SMALL_SHAPE_MM = 25.0           # ... on lines no longer than this (a 8 mm circle)
MAX_TURN_RAD = math.radians(30)  # a curve may turn this much from one stitch to the next


def run_points(line: LineString, rules: dict) -> list[tuple[float, float]]:
    """Needle points along a line: corners are kept, tight curves get shorter stitches."""
    coords = [(float(x), float(y)) for x, y in line.coords]
    if len(coords) < 2:
        return coords
    # split at sharp corners so a stitch always lands on the corner
    pieces, cur = [], [coords[0]]
    for i in range(1, len(coords) - 1):
        cur.append(coords[i])
        a, b, c = coords[i - 1], coords[i], coords[i + 1]
        v1 = (b[0] - a[0], b[1] - a[1])
        v2 = (c[0] - b[0], c[1] - b[1])
        n1, n2 = math.hypot(*v1), math.hypot(*v2)
        if n1 > 1e-9 and n2 > 1e-9:
            cosang = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
            if math.degrees(math.acos(cosang)) > 35.0:
                pieces.append(cur)
                cur = [coords[i]]
    cur.append(coords[-1])
    pieces.append(cur)

    out = [coords[0]]
    for piece in pieces:
        sub = LineString(piece)
        if sub.length < 1e-6:
            continue
        # Embroidery Legacy: 2.5 mm is the sweet spot, shorter on curves, never under about
        # 1.25 mm (shorter stitches pile on top of each other and sink). Tiny letters ask for
        # shorter stitches themselves (tip 48: shorten the run to smooth small objects).
        base = rules["run_stitch_mm"]
        steps = [base, min(base, rules["run_stitch_curve_mm"]), min(base, 1.25)]
        if sub.length <= SMALL_SHAPE_MM:          # only small shapes: a long line with one
            steps.append(min(base, TIGHT_CURVE_STITCH_MM))   # tight bend keeps normal stitches
        for step in steps:
            # equal stitches no shorter than `step` (rounding the count down, not up)
            n = max(1, math.floor(sub.length / step + 1e-9))
            pts = [sub.interpolate(sub.length * k / n) for k in range(n + 1)]
            # does any straight stitch stray from the curve it replaces?
            worst = 0.0
            for k in range(n):
                mid_on_path = sub.interpolate(sub.length * (k + 0.5) / n)
                mx = (pts[k].x + pts[k + 1].x) / 2
                my = (pts[k].y + pts[k + 1].y) / 2
                worst = max(worst, math.hypot(mid_on_path.x - mx, mid_on_path.y - my))
            # ... and does the line turn so much from one stitch to the next that a small
            # circle shows its corners (a clock face sewn as a hexagon)?
            turn = 0.0
            for k in range(1, n):
                a1 = math.atan2(pts[k].y - pts[k - 1].y, pts[k].x - pts[k - 1].x)
                a2 = math.atan2(pts[k + 1].y - pts[k].y, pts[k + 1].x - pts[k].x)
                turn = max(turn, abs((a2 - a1 + math.pi) % (2 * math.pi) - math.pi))
            if worst <= 0.12 and turn <= MAX_TURN_RAD:
                break
        out.extend((p.x, p.y) for p in pts[1:])
    return out


def _outline_satin(route: list, widths: list, rules: dict):
    """The satin for the return pass of each edge of an outline web (see
    linenet.route_points). Each column runs from junction to junction along the cleaned-up
    centre line, at the width the line was drawn. Ends are carried past the junction so
    corners close and joins lap: half a width at a corner, half a millimetre where a line
    butts into others, nothing at a loose end."""
    from . import lettering                       # noqa: PLC0415 - avoid a circular import
    least = rules.get("satin_net_min_mm", 1.1)
    most = rules.get("satin_max_mm", 7.0)
    seen, degree = set(), {}
    for poly in route:
        a, b = tuple(np.round(poly[0], 3)), tuple(np.round(poly[-1], 3))
        if (a, b) in seen or (b, a) in seen:
            continue
        seen.add((a, b))
        for e in (a, b):
            degree[e] = degree.get(e, 0) + 1

    def past(end, w):
        d = degree.get(tuple(np.round(end, 3)), 1)
        return 0.0 if d <= 1 else (w / 2.0 if d == 2 else min(w / 2.0, 0.5))

    def satin(k, poly):
        w = widths[k] if k < len(widths) else 0.0
        if w <= 0 or len(poly) < 2:
            return None
        w = min(w, most)
        loose = min(degree.get(tuple(np.round(poly[0], 3)), 1),
                    degree.get(tuple(np.round(poly[-1], 3)), 1)) <= 1
        if loose and LineString(poly).length < 2.5 * w:
            return None                 # a stub: as satin it would be a blob, not a line
        # A line that turns a corner is one column per leg, each carried half a width past
        # the corner so the legs lap: a square corner, as drawn. One column bent round the
        # corner fans out into a rounded hump (Wilcom: lapped corners for 45-110 degrees).
        legs = _legs(np.asarray(poly, float))
        pts: list = []
        for j, leg in enumerate(legs):
            c0 = ("ext", w / 2.0) if j > 0 else ("ext", past(poly[0], w))
            c1 = ("ext", w / 2.0) if j < len(legs) - 1 else ("ext", past(poly[-1], w))
            col = lettering.column(leg, w, c0, c1)
            if not col:
                continue
            # (half the usual pull compensation: full compensation on every line of a web
            # makes the whole drawing heavier than it was drawn)
            segs = stitch_satin(dict(col, underlay="none", comp_scale=0.5), rules)
            if segs and len(segs[0]) > 3:
                pts += segs[0]
        return pts if len(pts) > 3 else None
    return satin


CORNER_TURN_DEG = 50.0     # a turn this sharp in a line is a corner, not a bend


def _legs(poly: np.ndarray) -> list:
    """Split a polyline at its corners (turns sharper than CORNER_TURN_DEG)."""
    if len(poly) < 3:
        return [poly]
    cut = [0]
    for i in range(1, len(poly) - 1):
        a, b = poly[i] - poly[i - 1], poly[i + 1] - poly[i]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            continue
        turn = math.degrees(math.acos(max(-1.0, min(1.0, float(a @ b) / (na * nb)))))
        if turn > CORNER_TURN_DEG:
            cut.append(i)
    cut.append(len(poly) - 1)
    return [poly[a:b + 1] for a, b in zip(cut, cut[1:]) if b > a]


def bean(points: list) -> list:
    """Triple run: forward, back, forward over every stitch, for a bolder line."""
    if len(points) < 2:
        return list(points)
    out = [points[0]]
    for i in range(1, len(points)):
        out += [points[i], points[i - 1], points[i]]
    return out


# ------------------------------------------------------------------------------------ satin

def _along(arr: np.ndarray, cum: np.ndarray, s) -> np.ndarray:
    """Points at distances `s` along a polyline given with its cumulative lengths."""
    return np.column_stack([np.interp(s, cum, arr[:, 0]), np.interp(s, cum, arr[:, 1])])


def _tuples(a: np.ndarray) -> list:
    return [(float(x), float(y)) for x, y in a]


def satin_spacing(width: float, rules: dict) -> float:
    """Row spacing for a satin column of this width (research rule 12): narrow columns are sewn
    more openly, wide ones denser, as Wilcom's auto spacing does. 1.2-2 mm: 0.45; 2-4 mm: 0.40;
    4-7 mm: 0.37. The fabric preset nudges the whole table (a T-shirt's 0.45 fill makes satin
    about 7 % more open than a twill's 0.42)."""
    if width < 2.0:
        table = 0.45
    elif width < 4.0:
        table = 0.40
    else:
        table = 0.37
    nudge = min(1.15, max(0.9, rules["fill_spacing_mm"] / 0.42))
    return table * nudge


def stitch_satin(obj: dict, rules: dict) -> list:
    """One satin column: underlay to suit its width, then the zig-zag top, as one segment."""
    center, half = np.asarray(obj["center"], float), np.asarray(obj["half"], float)
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(center, axis=0), axis=1))])
    total = float(cum[-1])
    if total < 0.5:
        return []
    width = 2 * float(np.median(half))
    if obj.get("left") is not None and obj.get("right") is not None:
        edge_l, edge_r = np.asarray(obj["left"], float), np.asarray(obj["right"], float)
    else:
        edge_l, edge_r = column_rails(center, half)
    across = edge_l - edge_r
    unit = across / np.maximum(np.linalg.norm(across, axis=1, keepdims=True), 1e-9)
    # Stitches pull a column narrower, so each side is pushed out (Wilcom: 0.2-0.4 mm per
    # side). A very narrow column gets less, or a fine outline would come out much too fat.
    # Research rule 18: fabric base + about 5 % of the width, at most 0.5 mm per side, and
    # scaled down on columns under 2.5 mm so fine outlines and small letters stay fine.
    grow = min(0.5, rules["pull_comp_mm"] + 0.05 * width) * min(1.0, width / 2.5)
    grow *= obj.get("comp_scale", 1.0)
    left, right = edge_l + unit * grow, edge_r - unit * grow
    inset = np.minimum(rules.get("underlay_inset_mm", 0.5), half * 0.3)[:, None]
    in_left, in_right = edge_l - unit * inset, edge_r + unit * inset

    def run_along(arr, forward=True):
        n = max(1, math.ceil(total / UNDERLAY_STITCH_MM))
        s = np.linspace(0.0, total, n + 1)
        return _tuples(_along(arr, cum, s if forward else s[::-1]))

    def zigzag(a, b, spacing, forward=True, short=False):
        """Stitches crossing from rail a to rail b and back. They are spaced by whichever
        rail is moving faster, so on a curve the outside edge stays solid instead of fanning
        open. With `short`, every other stitch stops short on the crowded inside of a tight
        curve, so thread doesn't pile up there."""
        da = np.linalg.norm(np.diff(a, axis=0), axis=1)
        db = np.linalg.norm(np.diff(b, axis=0), axis=1)
        step = np.maximum(np.maximum(da, db), 1e-9)
        cu = np.concatenate([[0.0], np.cumsum(step)])
        span = float(cu[-1])
        n = max(1, math.ceil(span / spacing))
        u = np.linspace(0.0, span, n + 1)
        mid = np.minimum(u + span / n / 2, span)
        if not forward:
            u = u[::-1]
            mid = np.maximum(u - span / n / 2, 0.0)
        pa, pb = _along(a, cu, u), _along(b, cu, mid)
        if short and len(da) > 2:
            where = (cu[:-1] + cu[1:]) / 2
            crowd_a = np.interp(u, where, da / step)       # near 0 = a is the inside of a curve
            crowd_b = np.interp(mid, where, db / step)
            idx = np.arange(len(u))
            pull_a = (crowd_a < 0.5) & (idx % 2 == 1)
            pull_b = (crowd_b < 0.5) & (idx % 2 == 0)
            pa[pull_a] += 0.35 * (pb[pull_a] - pa[pull_a])
            pb[pull_b] += 0.35 * (pa[pull_b] - pb[pull_b])
        out = np.empty((2 * len(u), 2))
        out[0::2], out[1::2] = pa, pb
        return _tuples(out)

    # Underlay by width (Wilcom, Hatch, Melco): none on very narrow columns, a centre run on
    # medium ones, edge runs plus a loose zig-zag on wide ones. Lettering can ask for less.
    # Research rule 16: under 1.5 mm none; 1.5-2.5 mm a centre run; 2.5-4 mm edge runs;
    # over 4 mm edge runs plus a loose zig-zag. Lettering may ask for less (never edge runs
    # under small letters).
    kind = obj.get("underlay")
    if kind is None:
        if not rules.get("underlay") or width < rules["satin_center_run_mm"]:
            kind = "none"
        elif width < rules["satin_edge_run_mm"]:
            kind = "center"
        elif width < rules["satin_zigzag_mm"]:
            kind = "edge"
        else:
            kind = "edge_zigzag"
    elif kind == "center" and width < rules["satin_center_run_mm"]:
        kind = "none"
    pts: list = []
    top_forward = True
    mid = (edge_l + edge_r) / 2
    if kind in ("edge", "edge_zigzag"):
        pts += run_along(in_left, True) + run_along(in_right, False)
        if kind == "edge_zigzag":
            pts += zigzag(in_left, in_right, 2.0, True)
        else:
            top_forward = True                       # edge runs end back where they began
        top_forward = kind != "edge_zigzag"
    elif kind == "center":
        pts += run_along(mid, True)
        top_forward = False
    # Lettering plans where each stroke must end ("finish": at its "far" end, or back at
    # its "start") so the thread can go straight on to the next stroke. One more underlay
    # pass along the centre puts the needle at the right end for the top stitching.
    finish = obj.get("finish")
    if finish == "start" and top_forward:
        pts += run_along(mid, True)
        top_forward = False
    elif finish == "far" and not top_forward:
        pts += run_along(mid, False)
        top_forward = True
    spacing = satin_spacing(width, rules) * obj.get("spacing_scale", 1.0)
    pts += zigzag(left, right, spacing, top_forward, short=True)
    satin_max = rules.get("satin_max_mm", 7.0)
    if width + 2 * grow > satin_max:
        # every crossing of the column (underlay, the step to the top, the top) is split
        pts = split_long(pts, satin_max)
    return [pts]


def split_long(pts: list, max_len: float) -> list:
    """Auto-split a wide satin (Embroidery Legacy tip 99, Wilcom auto split): a stitch longer
    than `max_len` is broken by needle points across the column, staggered from row to row
    like brickwork, so it keeps the satin look without long loose stitches that snag and
    without a visible line down the middle."""
    out = [pts[0]]
    for i, (a, b) in enumerate(zip(pts, pts[1:])):
        d = dist(a, b)
        if d > max_len:
            n = math.ceil(d / (max_len * 0.75))
            shift = -0.22 if (i // 2) % 2 == 0 else 0.22
            for k in range(1, n):
                f = (k + shift) / n
                out.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
        out.append(b)
    return out


# ------------------------------------------------------------------------------------- fill

def _rows(rot: Polygon, spacing: float, extend: float = 0.0) -> list:
    """Cut a polygon into horizontal rows. Returns [(row_index, y, [(x_start, x_end), ...])].
    Row positions sit on a fixed grid so the texture is the same in every shape. `extend`
    lengthens every row by that much at both ends (pull compensation)."""
    edges = []
    for ring in [rot.exterior, *rot.interiors]:
        c = np.asarray(ring.coords)[:, :2]
        edges.append(np.hstack([c[:-1], c[1:]]))
    e = np.vstack(edges)
    x0, y0, x1, y1 = e[:, 0], e[:, 1], e[:, 2], e[:, 3]
    miny, maxy = rot.bounds[1], rot.bounds[3]
    rows = []
    i = math.ceil(miny / spacing - 0.5)
    while (i + 0.5) * spacing < maxy:
        y = (i + 0.5) * spacing
        m = ((y0 <= y) & (y1 > y)) | ((y1 <= y) & (y0 > y))
        pairs = []
        if m.any():
            xs = np.sort(x0[m] + (y - y0[m]) * (x1[m] - x0[m]) / (y1[m] - y0[m]))
            for j in range(0, len(xs) - 1, 2):
                a, b = float(xs[j]), float(xs[j + 1])
                if b - a < MIN_ROW_MM:
                    continue
                a, b = a - extend, b + extend
                if pairs and a <= pairs[-1][1]:          # lengthened rows met: make them one
                    pairs[-1] = (pairs[-1][0], b)
                else:
                    pairs.append((a, b))
        rows.append((i, y, pairs))
        i += 1
    return rows


def _sections(rows: list) -> list:
    """Group row pieces into sections that can each be sewn back and forth without lifting
    the needle: a section ends wherever the shape splits or merges (around a hole, say)."""
    sections, open_ = [], []
    for i, y, pairs in rows:
        new_open = []
        for a, b in pairs:
            over = [k for k, (_, pa, pb) in enumerate(open_) if pa < b and pb > a]
            cont = None
            if len(over) == 1:
                s, pa, pb = open_[over[0]]
                if sum(1 for c, d in pairs if pa < d and pb > c) == 1:
                    cont = s
            if cont is None:
                sections.append([])
                cont = len(sections) - 1
            sections[cont].append((i, y, a, b))
            new_open.append((cont, a, b))
        open_ = new_open
    return sections


def _row_xs(a: float, b: float, i: int, length: float, rng=None, jitter: float = 0.0) -> list:
    """Needle positions along one row from a to b, staggered by row number. With `rng`, the
    stagger and every stitch length vary at random by up to `jitter` (a textured fill)."""
    off = (rng.random() if rng is not None else STAGGER[i % len(STAGGER)]) * length
    gap = max(0.8, 0.25 * length)        # keep clear of the ends so no stitch is tiny
    xs = [a]
    x = off + math.ceil((a + gap - off) / length) * length
    while x < b - gap:
        xs.append(x)
        x += length * (1.0 + (rng.uniform(-jitter, jitter) if rng is not None else 0.0))
    if len(xs) == 1 and b - a > length:      # no grid point landed in a longish row: halve it
        xs.append((a + b) / 2)
    xs.append(b)
    return xs


def _section_points(sec: list, top_first: bool, left_first: bool, length: float,
                    rng=None, jitter: float = 0.0, inside=None) -> list:
    """Needle points for a section, back and forth along its rows. The step from one row to
    the next is normally one stitch. Where the next row begins far along from where this
    one ended (the shape steps in or out, or widely spaced underlay rows straddle a notch)
    a straight stitch could lie across open fabric: the thread then runs back along the row
    just sewn, crosses where both rows exist, and runs out along the new row."""
    rows = sec if top_first else sec[::-1]
    pts, left = [], left_first
    prev = None                                    # (y, a, b) of the row before
    for i, y, a, b in rows:
        xs = _row_xs(a, b, i, length, rng, jitter)
        if not left:
            xs = xs[::-1]
        if prev is not None and pts:
            (ex, ey), sx = pts[-1], xs[0]
            dy = abs(y - ey)
            if abs(sx - ex) > max(1.5, 2.0 * dy):
                if inside is not None and inside.contains(LineString([(ex, ey), (sx, y)])):
                    # a straight step stays on the shape: just keep its stitches short
                    n = math.ceil(math.hypot(sx - ex, y - ey) / length)
                    pts.extend((ex + (sx - ex) * k / n, ey + (y - ey) * k / n)
                               for k in range(1, n))
                else:
                    _py, pa, pb = prev
                    xw = min(max(sx, pa), pb)      # on the row just sewn, nearest the new start
                    xw = min(max(xw, a), b)        # and on the new row
                    for x0, x1, yy in ((ex, xw, ey), (xw, sx, y)):
                        if abs(x1 - x0) > 0.3:
                            n = math.ceil(abs(x1 - x0) / length)
                            pts.extend((x0 + (x1 - x0) * k / n, yy) for k in range(1, n + 1))
                    if pts[-1] == (xs[0], y):
                        xs = xs[1:]
        pts.extend((x, y) for x in xs)
        prev = (y, a, b)
        left = not left
    return pts


def _entry(sec: list, top_first: bool, left_first: bool) -> tuple:
    i, y, a, b = sec[0] if top_first else sec[-1]
    return (a if left_first else b, y)


class _Router:
    """Finds a path between two points that stays inside a shape, for when a straight line
    would cross a hole or leave the shape. Works on a coarse grid of the shape, and prefers
    ground that hasn't been filled yet, so the thread it lays is covered by later stitches.
    With `light` (an open, half-density fill such as foliage) no row covers the thread fully,
    so travel keeps to the shape's edge from the start ("travel on edge": Embroidery Legacy
    tip 52, travel through the middle shows in light-density fills)."""

    def __init__(self, shape: Polygon, inside, light: bool = False):
        self.shape, self.inside, self.mask = shape, inside, None
        self.light = light
        self.rings = None
        self.loose = None
        self.exact = None
        minx, miny, maxx, maxy = shape.bounds
        self.cell = max(0.5, max(maxx - minx, maxy - miny) / 220.0)
        self.minx, self.miny = minx, miny
        self.any_filled = False

    def _cell(self, pt):
        return (int(round((pt[0] - self.minx) / self.cell)) + 1,
                int(round((pt[1] - self.miny) / self.cell)) + 1)

    def _build(self):
        maxx, maxy = self.shape.bounds[2], self.shape.bounds[3]
        w = int((maxx - self.minx) / self.cell) + 3
        h = int((maxy - self.miny) / self.cell) + 3

        def px(coords):
            a = np.asarray(coords)[:, :2]
            return np.round(np.column_stack([(a[:, 0] - self.minx) / self.cell + 1,
                                             (a[:, 1] - self.miny) / self.cell + 1])).astype(np.int32)

        self.mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(self.mask, [px(self.shape.exterior.coords)], 1)
        for ring in self.shape.interiors:
            cv2.fillPoly(self.mask, [px(ring.coords)], 0)
        self.filled = np.zeros((h, w), np.uint8)
        # cells within about two cells of the shape's edge: thread laid there is the least
        # noticeable place to cross ground that is already filled
        inner = cv2.erode(self.mask, np.ones((3, 3), np.uint8), iterations=2)
        self.edge = (self.mask == 1) & (inner == 0)

    def along_edge(self, p, q, tol: float = 1.2):
        """The way from p to q along the shape's own outline (the shorter way round), when
        both lie on the same outline. Thread that must cross finished stitching is least
        seen there: right on the edge, under the outline or border when there is one
        ("travel on edge"), and it is a clean line instead of a path across the rows."""
        if self.rings is None:
            self.rings = [LineString(r.coords) for r in (self.shape.exterior, *self.shape.interiors)
                          if r.length > 1.0]
        P, Q = Point(p), Point(q)
        ring = min(self.rings, key=lambda r: r.distance(P), default=None)
        if ring is None or ring.distance(P) > tol or ring.distance(Q) > tol:
            return None
        a, b, total = ring.project(P), ring.project(Q), ring.length
        fwd = (b - a) % total
        if min(fwd, total - fwd) > 3.0 * dist(p, q) + 25.0:
            return None                              # too far round: not worth it

        def walk(s0, s1):                            # forward along the ring, with wrap
            if s1 >= s0:
                return list(substring(ring, s0, s1).coords)
            return list(substring(ring, s0, total).coords) + list(substring(ring, 0.0, s1).coords)
        pts = walk(a, b) if fwd <= total - fwd else walk(b, a)[::-1]
        return [p] + [(float(x), float(y)) for x, y in pts] + [q]

    def pulled_in(self, path) -> list:
        """The same route with any point that strayed just outside the shape (grid cells are
        coarse) moved onto its edge: travel stitches never show past the outline."""
        if self.exact is None:
            self.exact = prep(self.shape)
        out = [path[0]]
        for pt in path[1:-1]:
            if not self.exact.contains(Point(pt)):
                on = self.shape.exterior.interpolate(self.shape.exterior.project(Point(pt)))
                for ring in self.shape.interiors:
                    c = ring.interpolate(ring.project(Point(pt)))
                    if c.distance(Point(pt)) < on.distance(Point(pt)):
                        on = c
                pt = (on.x, on.y)
            out.append(pt)
        out.append(path[-1])
        return out

    def stays_in(self, path) -> bool:
        """Does every hop of a route stay on the shape (give or take a grid cell)? The grid
        is coarse, so a route can cut across a narrow notch between two parts of a shape,
        which would leave thread lying on open fabric."""
        if self.loose is None:
            self.loose = prep(self.shape.buffer(0.6 * self.cell))
        return all(self.loose.contains(LineString([a, b]))
                   for a, b in zip(path, path[1:]) if dist(a, b) > 1e-6)

    def mark_filled(self, section: list) -> None:
        """Remember that these rows (in this router's own frame) are now covered in thread."""
        if self.mask is None:
            self._build()
        h, w = self.mask.shape
        for _i, y, a, b in section:
            cy = int(round((y - self.miny) / self.cell)) + 1
            ca = int(round((a - self.minx) / self.cell)) + 1
            cb = int(round((b - self.minx) / self.cell)) + 1
            if 0 <= cy < h:
                self.filled[cy, max(0, ca):min(w, cb + 1)] = 1
        self.any_filled = True

    def __call__(self, p, q):
        if self.mask is None:
            self._build()
        mask, filled, edge = self.mask, self.filled, self.edge
        h, w = mask.shape
        s, t = self._cell(p), self._cell(q)
        best = {s: 0.0}
        prev = {s: None}
        heap = [(0.0, s)]
        steps = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414))
        while heap:
            d, c = heapq.heappop(heap)
            if c == t:
                break
            if d > best.get(c, 1e18):
                continue
            for dx, dy, cost in steps:
                n = (c[0] + dx, c[1] + dy)
                if not (0 <= n[0] < w and 0 <= n[1] < h):
                    continue
                if not (mask[n[1], n[0]] or n == t):
                    continue
                if self.light:
                    weight = 1.0 if edge[n[1], n[0]] else 6.0
                elif not filled[n[1], n[0]]:
                    weight = 1.0                 # will be covered by later stitches
                else:
                    weight = 3.0 if edge[n[1], n[0]] else 9.0
                nd = d + cost * weight
                if nd < best.get(n, 1e18):
                    best[n] = nd
                    prev[n] = c
                    heapq.heappush(heap, (nd, n))
        if t not in prev:
            return None
        cells = []
        c = t
        while c is not None:
            cells.append(c)
            c = prev[c]
        cells.reverse()
        pts = [p] + [((cx - 1) * self.cell + self.minx, (cy - 1) * self.cell + self.miny)
                     for cx, cy in cells[1:-1]] + [q]
        if self.any_filled or self.light:
            # keep to the route that dodges filled ground; just smooth the grid steps
            return [(float(x), float(y)) for x, y in
                    LineString(pts).simplify(self.cell * 0.6).coords] if len(pts) > 2 else pts
        # nothing filled yet: pull the path taut wherever a straight hop stays inside
        out, i = [pts[0]], 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self.inside.contains(LineString([pts[i], pts[j]])):
                j -= 1
            out.append(pts[j])
            i = j
        return out


def _travel(p, q, inside, router: _Router, step: float):
    """Needle points to walk from p to q hidden inside the shape (not including p or q).
    Returns None when there is no hidden route, meaning the thread must jump."""
    d = dist(p, q)
    if d < 0.6:
        return []
    short = 1.5 if router.light else 3.0
    if ((not router.any_filled and not router.light) or d < short)             and inside.contains(LineString([p, q])):
        path = [p, q]
    else:
        path = router.along_edge(p, q) if (router.any_filled or router.light) else None
        if path is None:
            path = router(p, q)
            if path is not None and not router.stays_in(path):
                # the grid route cut across a notch of open fabric: go round by the edge
                path = router.along_edge(p, q)
        if path is None:
            return None
        path = router.pulled_in(path)
    out = []
    for a, b in zip(path, path[1:]):
        n = max(1, math.ceil(dist(a, b) / step))
        out.extend((a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n)
                   for k in range(1, n + 1))
    return out[:-1]


# "Fill spacing" (0.40-0.45 mm) is the digitizing-software number: the distance from one
# forward row to the next, with a return row lying between them. The thread lines of a
# solid fill are therefore half that apart, about 0.2 mm, the same thread per millimetre as
# a satin at 0.4. (Measured on finished production files with tools/study.py: rows 0.19 to
# 0.22 mm apart. Sewing rows a full "spacing" apart gives a half-density fill the fabric
# shows through.)
ROW_SHARE = 0.5

FILL_ANGLES = (18.0, 162.0)   # deep-dive rule 38: 15-20 deg off level (or its mirror), never
                              # 45 deg (the fabric stretches on the bias), never upright


def best_fill_angle(poly: Polygon, base: float, spacing: float) -> float:
    """The fill angle that cuts this shape into the fewest separate sections, so it needs the
    fewest jumps and travel runs (the Soft Sight patent), choosing only from angles that
    stay close to level (Erich Campbell: 15-20 deg off horizontal). `base` is preferred on
    ties, so neighbouring colors get different angles."""
    if poly.area < 80.0:
        # Small blocks (windows): rows along the longer side give clean, square edges;
        # slanted rows leave a staircase along the sides of a small rectangle.
        rect = poly.minimum_rotated_rectangle
        if rect.area > 0 and poly.area / rect.area > 0.85:
            c = list(rect.exterior.coords)
            e1 = (c[1][0] - c[0][0], c[1][1] - c[0][1])
            e2 = (c[2][0] - c[1][0], c[2][1] - c[1][1])
            e = e1 if math.hypot(*e1) >= math.hypot(*e2) else e2
            return math.degrees(math.atan2(e[1], e[0])) % 180.0
        return base
    best = None
    for k, angle in enumerate((base,) + tuple(a for a in FILL_ANGLES if a != base)):
        rot = affinity.rotate(poly, -angle, origin=(0, 0))
        n = len(_sections(_rows(rot, max(spacing * 4, 1.2))))
        if best is None or n < best[0]:
            best = (n, angle)
    return best[1]


def fill_segments(poly: Polygon, angle_deg: float, spacing: float, length: float,
                  start=None, travel_step: float = 2.5,
                  extend: float = 0.0, rng=None, jitter: float = 0.0,
                  ragged: float = 0.0, light: bool = False) -> list[list[tuple[float, float]]]:
    """Fill a polygon with rows at the given angle. Returns one or more thread segments.
    With `rng`: random needle placement (`jitter`) and row ends cut back at random by up to
    `ragged` mm, so the fill has a soft, uneven edge (foliage, fur, grass)."""
    rot = affinity.rotate(poly, -angle_deg, origin=(0, 0))
    rows = _rows(rot, spacing, extend)
    if rng is not None and ragged > 0:
        rows = [(i, y, [(a + rng.uniform(0, ragged), b - rng.uniform(0, ragged))
                        for a, b in pairs if b - a > 2 * ragged + MIN_ROW_MM])
                for i, y, pairs in rows]
    sections = _sections(rows)
    if not sections:
        return []
    inside = prep(rot.buffer(INSIDE_TOL_MM + extend))
    router = _Router(rot, inside, light)
    ca, sa = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    cur = None
    if start is not None:   # rotate the start point into the working frame
        cur = (start[0] * ca + start[1] * sa, -start[0] * sa + start[1] * ca)

    segments, seg = [], []
    todo = list(range(len(sections)))
    while todo:
        best = None
        for s in todo:
            for top in (True, False):
                for left in (True, False):
                    d = 0.0 if cur is None else dist(cur, _entry(sections[s], top, left))
                    if best is None or d < best[0]:
                        best = (d, s, top, left)
            if cur is None:
                break
        _, s, top, left = best
        todo.remove(s)
        pts = _section_points(sections[s], top, left, length, rng, jitter, inside)
        if seg:
            walk = _travel(seg[-1], pts[0], inside, router, travel_step)
            if walk is None:
                segments.append(seg)
                seg = []
            else:
                seg.extend(walk)
        seg.extend(pts)
        cur = seg[-1]
        if todo:
            router.mark_filled(sections[s])
    segments.append(seg)
    # rotate back
    return [[(x * ca - y * sa, x * sa + y * ca) for x, y in sg] for sg in segments if len(sg) > 1]


def _append(segments: list, new: list, cover, router: _Router, step: float = 2.5) -> None:
    """Add new segments after existing ones. Thread is carried across inside the shape (where
    it will be covered) whenever there is a way; otherwise a jump is left."""
    for sg in new:
        if len(sg) < 2:
            continue
        if segments:
            walk = _travel(segments[-1][-1], sg[0], cover, router, step)
            if walk is not None:
                segments[-1].extend(walk)
                segments[-1].extend(sg)
                continue
        segments.append(list(sg))


def edge_run(poly: Polygon, inset: float, rules: dict) -> list:
    """A run just inside the edge of a shape (around holes too). It anchors the edge so the
    fill's row ends make a clean outline."""
    out = []
    for part in polygons_of(poly.buffer(-inset)):
        for ring in [part.exterior, *part.interiors]:
            line = LineString(ring.coords)
            if line.length < 3.0:
                continue
            n = max(3, math.ceil(line.length / UNDERLAY_STITCH_MM))
            pts = [line.interpolate(line.length * k / n) for k in range(n + 1)]
            out.append([(p.x, p.y) for p in pts])
    return out


def _per_part(fn, obj: dict, *args) -> list | None:
    """A shape cut into pieces (an overlap trimmed out of it can leave a sliver beside it):
    sew each real piece and skip slivers. Returns None for a single polygon."""
    if obj["poly"].geom_type == "Polygon":
        return None
    out: list = []
    for part in polygons_of(obj["poly"]):
        if part.area >= 0.5:
            out += fn(dict(obj, poly=part), *args)
    return out


def stitch_fill(obj: dict, rules: dict, angle_deg: float) -> list:
    parts = _per_part(stitch_fill, obj, rules, angle_deg)
    if parts is not None:
        return parts
    poly = obj["poly"]
    extend = rules["pull_comp_mm"] * 0.5          # fill pulls in about half as much as satin
    cover = prep(poly.buffer(INSIDE_TOL_MM + extend))
    router = _Router(poly, cover)
    segments: list = []
    if rules.get("underlay"):
        inset = poly.buffer(-rules["underlay_inset_mm"])
        wide_enough = [p for p in polygons_of(inset)
                       if not p.buffer(-rules["underlay_min_width_mm"] / 2).is_empty]
        if wide_enough and rules.get("underlay_edge_run", True):
            _append(segments, edge_run(poly, rules["underlay_inset_mm"], rules), cover, router)
        if rules.get("underlay_spacing_mm"):
            for part in wide_enough:
                start = segments[-1][-1] if segments else None
                under = fill_segments(part, angle_deg + 90.0, rules["underlay_spacing_mm"],
                                      rules["fill_stitch_mm"], start)
                _append(segments, under, cover, router)
    start = segments[-1][-1] if segments else None
    top = fill_segments(poly, angle_deg, rules["fill_spacing_mm"] * ROW_SHARE,
                        rules["fill_stitch_mm"], start, extend=extend)
    _append(segments, top, cover, router)
    if obj.get("edge") and segments:
        _outline(segments, poly, rules)
    return segments


EDGE_STITCH_MM = 1.8


def _outline(segments: list, poly: Polygon, rules: dict) -> None:
    """A bean run around a filled shape, sewn straight after its fill. The outer outline
    starts at the point nearest the fill's last stitch (the fill ends on the edge, so no
    travel shows); each hole's outline is its own thread."""
    first = True
    for part in polygons_of(poly.buffer(-0.15)):
        for ring in [part.exterior, *part.interiors]:
            line = LineString(ring.coords)
            if line.length < 2.0:
                continue
            coords = list(line.coords)[:-1]
            if first:                                   # begin where the fill left off
                last = segments[-1][-1]
                at = line.project(Point(last))
                head = list(substring(line, at, line.length).coords)
                tail = list(substring(line, 0.0, at).coords)
                coords = head[:-1] + tail
            loop = LineString(coords + [coords[0]])
            pts = bean(run_points(loop, dict(rules, run_stitch_mm=EDGE_STITCH_MM)))
            if first and dist(segments[-1][-1], pts[0]) <= 1.0:
                segments[-1].extend(pts)
            else:
                segments.append(pts)
            first = False


TEXTURE = {"jitter": 0.25, "ragged": 0.6, "length": 4.0, "second_turn": 40.0}


def stitch_texture(obj: dict, rules: dict) -> list:
    """Foliage and other soft masses (deep-dive rule 17, Erich Campbell's tree recipe): two
    half-density layers at different angles, the top one with random needle placement
    (25 %), 4 mm stitches, and a jagged edge (row ends cut back at random by up to 0.6 mm)
    instead of a hard, single-angle silhouette. A light hidden underlay holds it flat."""
    parts = _per_part(stitch_texture, obj, rules)
    if parts is not None:
        return parts
    poly = obj["poly"]
    seed = int(abs(poly.centroid.x * 1000 + poly.centroid.y * 37)) % (2 ** 31)
    rng = np.random.default_rng(seed)
    angle = obj.get("angle", 70.0)
    # As dark as the drawing was, but never more than about half density: foliage that
    # covers the fabric completely sews as a hard, heavy blob (Campbell builds trees from
    # half-density layers so the fabric shows through). A light drawing gets an even more
    # open layer. "solid" textures (fur, a texture the user asked for) may use both layers.
    ink = float(min(obj.get("max_ink", 0.55), max(0.2, obj.get("ink", 1.0))))
    solid = rules["fill_spacing_mm"] * ROW_SHARE          # row gap of a solid fill
    spacing = solid * 2.5 if ink >= 0.6 else min(2.5, solid / ink)
    turns = (0.0, TEXTURE["second_turn"]) if ink >= 0.6 else (0.0,)
    cover = prep(poly.buffer(INSIDE_TOL_MM))
    light = ink < 0.6
    router = _Router(poly, cover, light)
    segments: list = []
    if rules.get("underlay") and ink >= 0.6:
        inner = poly.buffer(-0.8)
        for part in polygons_of(inner):
            if part.area > 4.0:
                start = segments[-1][-1] if segments else None
                _append(segments, fill_segments(part, angle + 90.0, 3.5, 4.0, start), cover, router)
    # Foliage is built from clumps, each facing a slightly different way (Jesse Elliott's
    # "random tatami clumps, mostly vertical"), not one even hatch across the whole tree.
    for clump in _clumps(poly, rng):
        tilt = rng.uniform(-22.0, 22.0)
        for k, turn in enumerate(turns):
            start = segments[-1][-1] if segments else None
            layer = fill_segments(clump, angle + tilt + turn, spacing, TEXTURE["length"], start,
                                  rng=rng, jitter=TEXTURE["jitter"] if (k or len(turns) == 1) else 0.1,
                                  ragged=TEXTURE["ragged"], light=light)
            _append(segments, layer, cover, router)
    return segments


def _clumps(poly: Polygon, rng, size: float = 10.0) -> list:
    """Split a mass into clumps about `size` mm across (Voronoi cells around scattered
    points), overlapping a little so no gaps show between them."""
    minx, miny, maxx, maxy = poly.bounds
    n = int(max(1, min(40, poly.area / (size * size))))
    if n <= 1:
        return [poly]
    pts = []
    tries = 0
    while len(pts) < n and tries < 50 * n:
        tries += 1
        x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
        if poly.contains(Point(x, y)):
            pts.append(Point(x, y))
    if len(pts) <= 1:
        return [poly]
    cells = voronoi_diagram(MultiPoint(pts), envelope=poly.envelope.buffer(1.0))
    out = []
    for cell in cells.geoms:
        piece = cell.buffer(0.3).intersection(poly)
        out.extend(q for q in polygons_of(piece) if q.area > 1.0)
    # sew the clumps bottom to top, left to right, so the thread works across the tree
    out.sort(key=lambda q: (-round(q.centroid.y / size), q.centroid.x))
    return out or [poly]


def knockdown(objects: list, rules: dict) -> dict | None:
    """For towels and other deep pile: a very open criss-cross of stitches under the whole
    design, sewn first in thread that matches the fabric, to flatten the loops so the design
    doesn't sink into them."""
    geoms = []
    for o in objects:
        if o["kind"] == "fill":
            geoms.append(o["poly"])
        elif o["kind"] == "satin":
            geoms.append(o["outline"])
        else:
            geoms.append(o["line"].buffer(0.8))
    if not geoms:
        return None
    area = unary_union(geoms).buffer(4.0).buffer(-1.5).simplify(0.4)
    segments: list = []
    spacing = rules.get("knockdown_spacing_mm", 1.5)
    for part in polygons_of(area):
        cover = prep(part.buffer(INSIDE_TOL_MM))
        router = _Router(part, cover)
        for angle in (45.0, 135.0):
            start = segments[-1][-1] if segments else None
            _append(segments, fill_segments(part, angle, spacing, 4.0, start), cover, router)
        _append(segments, edge_run(part, 0.3, rules), cover, router)
    if not segments:
        return None
    return {"kind": "knockdown", "color": -1, "segments": segments, "reversible": False,
            "area_mm2": float(area.area)}


def run(classified: dict, rules: dict, debug_dir: Path | None = None) -> dict:
    stitched = []
    for obj in classified["objects"]:
        if obj["kind"] == "run" and obj.get("route"):
            # line art: one thread out along every branch and back over the same holes
            widths = obj.get("satin_w")
            pts = route_points(obj["route"], run_points,
                               dict(rules, run_stitch_mm=obj.get("stitch_mm") or rules.get("line_stitch_mm", 2.0)),
                               bean if obj.get("bold") else None,
                               _outline_satin(obj["route"], widths, rules) if widths else None)
            segments = [pts] if len(pts) > 1 else []
            if widths:
                # the ground each line really covers: its own satin, or a thread's width for
                # a line left as running stitch (travel may only hide under real coverage)
                shape = unary_union([LineString(seg).buffer(max(0.2, w / 2.0))
                                     for seg, w in zip(obj["route"], widths) if len(seg) >= 2])
            else:
                shape = obj["line"].buffer(0.4)
        elif obj["kind"] == "run":
            step = obj.get("stitch_mm") or (rules["bean_stitch_mm"] if obj["bean"] else None)
            pts = run_points(obj["line"], dict(rules, run_stitch_mm=step) if step else rules)
            if obj["bean"]:
                pts = bean(pts)
            segments = [pts] if len(pts) > 1 else []
            shape = obj["line"].buffer(0.4)          # the ground the thread itself covers
        elif obj["kind"] == "satin":
            segments = stitch_satin(obj, rules)
            shape = obj["outline"]
        elif obj.get("texture"):
            segments = stitch_texture(obj, rules)
            shape = obj["poly"]
        else:
            # neighbouring colors get crossing angles so they don't all pull the same way;
            # a row of lettering shares one angle so the letters look alike
            angle = obj.get("angle")
            if angle is None:
                angle = best_fill_angle(obj["poly"], FILL_ANGLES[obj["color"] % 2],
                                        rules["fill_spacing_mm"])
            segments = stitch_fill(obj, dict(rules, fill_spacing_mm=rules["fill_spacing_mm"]
                                             * obj.get("spacing_scale", 1.0)), angle)
            shape = obj["poly"]
        if segments:
            stitched.append({"kind": obj["kind"], "color": obj["color"], "segments": segments,
                             "reversible": obj["kind"] == "run" and obj.get("seq") is None,
                             "shape": shape, "area_mm2": obj.get("area_mm2", 0.0),
                             "seq": obj.get("seq"), "punct": obj.get("punct", False),
                             # a line a thread or two wide hides nothing sewn under it
                             "thin": obj["kind"] == "run" and not obj.get("satin_w")})
    if rules.get("knockdown") and stitched:
        base = knockdown(classified["objects"], rules)
        if base:
            stitched.insert(0, base)

    if debug_dir is not None and stitched:
        allpts = np.array([p for o in stitched for sg in o["segments"] for p in sg])
        bounds = (allpts[:, 0].min(), allpts[:, 1].min(), allpts[:, 0].max(), allpts[:, 1].max())
        img, to_px = canvas_for(bounds, 12.0, color=(255, 255, 255))
        colors = {"fill": (200, 90, 30), "satin": (160, 60, 160), "run": (30, 30, 200),
                  "knockdown": (170, 170, 170)}
        for o in stitched:
            for sg in o["segments"]:
                px = to_px(sg).astype(np.int32)
                cv2.polylines(img, [px], False, colors[o["kind"]], 1, cv2.LINE_AA)
                for x, y in px:
                    cv2.circle(img, (int(x), int(y)), 1, (0, 0, 0), -1)
        save_debug(debug_dir, "07_stitch.png", img)

    return {"objects": stitched}
