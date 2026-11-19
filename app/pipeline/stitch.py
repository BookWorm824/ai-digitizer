"""Stage 7 — Stitch.

Turn each classified object into needle points (millimetres).

  * run   -> evenly spaced stitches along a line (shorter on tight curves), optionally tripled
  * fill  -> back-and-forth rows at an angle (a "tatami" fill), with needle points staggered
             row to row so they don't line up, on top of a light underlay sewn across them

This is our own code. Nothing here is copied from Ink/Stitch.

Every object comes back as a list of "segments". A segment is one unbroken run of thread;
between segments the machine has to jump.
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from shapely import affinity
from shapely.geometry import LineString, Polygon
from shapely.prepared import prep

from .common import canvas_for, dist, polygons_of, save_debug

STAGGER = (0.0, 0.5, 0.25, 0.75)   # where needle points fall along a row, row after row
MIN_ROW_MM = 0.3                   # fill rows shorter than this are skipped
INSIDE_TOL_MM = 0.15               # how far outside a shape a hidden travel stitch may stray


# ----------------------------------------------------------------------------- running stitch

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
        for step in (rules["run_stitch_mm"], rules["run_stitch_curve_mm"]):
            n = max(1, math.ceil(sub.length / step))
            pts = [sub.interpolate(sub.length * k / n) for k in range(n + 1)]
            # does any straight stitch stray from the curve it replaces?
            worst = 0.0
            for k in range(n):
                mid_on_path = sub.interpolate(sub.length * (k + 0.5) / n)
                mx = (pts[k].x + pts[k + 1].x) / 2
                my = (pts[k].y + pts[k + 1].y) / 2
                worst = max(worst, math.hypot(mid_on_path.x - mx, mid_on_path.y - my))
            if worst <= 0.25:
                break
        out.extend((p.x, p.y) for p in pts[1:])
    return out


def bean(points: list) -> list:
    """Triple run: forward, back, forward over every stitch, for a bolder line."""
    if len(points) < 2:
        return list(points)
    out = [points[0]]
    for i in range(1, len(points)):
        out += [points[i], points[i - 1], points[i]]
    return out


# ------------------------------------------------------------------------------------- fill

def _rows(rot: Polygon, spacing: float) -> list:
    """Cut a polygon into horizontal rows. Returns [(row_index, y, [(x_start, x_end), ...])].
    Row positions sit on a fixed grid so the texture is the same in every shape."""
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
            pairs = [(float(xs[j]), float(xs[j + 1])) for j in range(0, len(xs) - 1, 2)
                     if xs[j + 1] - xs[j] >= MIN_ROW_MM]
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


def _row_xs(a: float, b: float, i: int, length: float) -> list:
    """Needle positions along one row from a to b, staggered by row number."""
    off = STAGGER[i % len(STAGGER)] * length
    gap = max(0.8, 0.25 * length)        # keep clear of the ends so no stitch is tiny
    xs = [a]
    x = off + math.ceil((a + gap - off) / length) * length
    while x < b - gap:
        xs.append(x)
        x += length
    if len(xs) == 1 and b - a > length:      # no grid point landed in a longish row: halve it
        xs.append((a + b) / 2)
    xs.append(b)
    return xs


def _section_points(sec: list, top_first: bool, left_first: bool, length: float) -> list:
    rows = sec if top_first else sec[::-1]
    pts, left = [], left_first
    for i, y, a, b in rows:
        xs = _row_xs(a, b, i, length)
        if not left:
            xs = xs[::-1]
        pts.extend((x, y) for x in xs)
        left = not left
    return pts


def _entry(sec: list, top_first: bool, left_first: bool) -> tuple:
    i, y, a, b = sec[0] if top_first else sec[-1]
    return (a if left_first else b, y)


class _Router:
    """Finds a hidden path between two points that stays inside a shape, for when a straight
    line would cross a hole or leave the shape. Works on a coarse grid of the shape."""

    CELL = 0.5  # mm

    def __init__(self, shape: Polygon, inside):
        self.shape, self.inside, self.mask = shape, inside, None

    def _cell(self, pt):
        return (int(round((pt[0] - self.minx) / self.CELL)) + 1,
                int(round((pt[1] - self.miny) / self.CELL)) + 1)

    def _build(self):
        self.minx, self.miny, maxx, maxy = self.shape.bounds
        w = int((maxx - self.minx) / self.CELL) + 3
        h = int((maxy - self.miny) / self.CELL) + 3

        def px(coords):
            a = np.asarray(coords)[:, :2]
            return np.round(np.column_stack([(a[:, 0] - self.minx) / self.CELL + 1,
                                             (a[:, 1] - self.miny) / self.CELL + 1])).astype(np.int32)

        self.mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(self.mask, [px(self.shape.exterior.coords)], 1)
        for ring in self.shape.interiors:
            cv2.fillPoly(self.mask, [px(ring.coords)], 0)

    def __call__(self, p, q):
        if self.mask is None:
            self._build()
        mask = self.mask
        h, w = mask.shape
        s, t = self._cell(p), self._cell(q)
        prev = {s: None}
        queue = deque([s])
        while queue and t not in prev:
            c = queue.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                n = (c[0] + dx, c[1] + dy)
                if n in prev or not (0 <= n[0] < w and 0 <= n[1] < h):
                    continue
                if mask[n[1], n[0]] or n == t:
                    prev[n] = c
                    queue.append(n)
        if t not in prev:
            return None
        cells = []
        c = t
        while c is not None:
            cells.append(c)
            c = prev[c]
        cells.reverse()
        pts = [p] + [((cx - 1) * self.CELL + self.minx, (cy - 1) * self.CELL + self.miny)
                     for cx, cy in cells[1:-1]] + [q]
        # pull the path taut: skip ahead whenever a straight hop stays inside the shape
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
    if dist(p, q) < 0.6:
        return []
    if inside.contains(LineString([p, q])):
        path = [p, q]
    else:
        path = router(p, q)
        if path is None:
            return None
    out = []
    for a, b in zip(path, path[1:]):
        n = max(1, math.ceil(dist(a, b) / step))
        out.extend((a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n)
                   for k in range(1, n + 1))
    return out[:-1]


def fill_segments(poly: Polygon, angle_deg: float, spacing: float, length: float,
                  start=None, travel_step: float = 2.5) -> list[list[tuple[float, float]]]:
    """Fill a polygon with rows at the given angle. Returns one or more thread segments."""
    rot = affinity.rotate(poly, -angle_deg, origin=(0, 0))
    sections = _sections(_rows(rot, spacing))
    if not sections:
        return []
    inside = prep(rot.buffer(INSIDE_TOL_MM))
    router = _Router(rot, inside)
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
        pts = _section_points(sections[s], top, left, length)
        if seg:
            walk = _travel(seg[-1], pts[0], inside, router, travel_step)
            if walk is None:
                segments.append(seg)
                seg = []
            else:
                seg.extend(walk)
        seg.extend(pts)
        cur = seg[-1]
    segments.append(seg)
    # rotate back
    return [[(x * ca - y * sa, x * sa + y * ca) for x, y in sg] for sg in segments if len(sg) > 1]


def _append(segments: list, new: list, cover, router: _Router, step: float = 2.5) -> None:
    """Add new segments after existing ones. Thread is carried across inside the shape (where
    it will be covered) whenever there is a way; otherwise a jump is left."""
    for sg in new:
        if segments:
            walk = _travel(segments[-1][-1], sg[0], cover, router, step)
            if walk is not None:
                segments[-1].extend(walk)
                segments[-1].extend(sg)
                continue
        segments.append(list(sg))


def stitch_fill(obj: dict, rules: dict, angle_deg: float) -> list:
    poly = obj["poly"]
    cover = prep(poly.buffer(INSIDE_TOL_MM))
    router = _Router(poly, cover)
    segments: list = []
    if rules.get("underlay"):
        inset = poly.buffer(-rules["underlay_inset_mm"])
        for part in polygons_of(inset):
            # skip slivers: underlay only helps where the shape is reasonably wide
            if part.buffer(-rules["underlay_min_width_mm"] / 2).is_empty:
                continue
            start = segments[-1][-1] if segments else None
            under = fill_segments(part, angle_deg + 90.0, rules["underlay_spacing_mm"],
                                  rules["fill_stitch_mm"], start)
            _append(segments, under, cover, router)
    start = segments[-1][-1] if segments else None
    top = fill_segments(poly, angle_deg, rules["fill_spacing_mm"], rules["fill_stitch_mm"], start)
    _append(segments, top, cover, router)
    return segments


def run(classified: dict, rules: dict, debug_dir: Path | None = None) -> dict:
    stitched = []
    for obj in classified["objects"]:
        if obj["kind"] == "run":
            pts = run_points(obj["line"], rules)
            if obj["bean"]:
                pts = bean(pts)
            segments = [pts] if len(pts) > 1 else []
        else:
            # neighbouring colors get crossing angles so they don't all pull the same way
            angle = 45.0 if obj["color"] % 2 == 0 else 135.0
            segments = stitch_fill(obj, rules, angle)
        if segments:
            stitched.append({"kind": obj["kind"], "color": obj["color"], "segments": segments,
                             "reversible": obj["kind"] == "run",
                             "area_mm2": obj.get("area_mm2", 0.0)})

    if debug_dir is not None and stitched:
        allpts = np.array([p for o in stitched for sg in o["segments"] for p in sg])
        bounds = (allpts[:, 0].min(), allpts[:, 1].min(), allpts[:, 0].max(), allpts[:, 1].max())
        img, to_px = canvas_for(bounds, 12.0, color=(255, 255, 255))
        for o in stitched:
            col = (200, 90, 30) if o["kind"] == "fill" else (30, 30, 200)
            for sg in o["segments"]:
                px = to_px(sg).astype(np.int32)
                cv2.polylines(img, [px], False, col, 1, cv2.LINE_AA)
                for x, y in px:
                    cv2.circle(img, (int(x), int(y)), 1, (0, 0, 0), -1)
        save_debug(debug_dir, "07_stitch.png", img)

    return {"objects": stitched}
