"""Stage 8 — Sequence.

Put the stitched objects into sewing order and turn them into the final thread plan:

  * one block per color, big background areas first, small details and thin outlines last
  * inside a color: fills, then satin columns, then lines, always going to the nearest
    object next (lines may be sewn from either end)
  * hats ("cap mode"): work from the centre of the design outward, starting at the bottom,
    because that pushes the cap fabric evenly to the sides
  * towels: the flattening (knockdown) layer is its own block, sewn before everything
  * short hops are sewn straight through, but only where the connecting thread will be
    hidden: over this color's own stitches or under a color sewn later. A hop across open
    ground (a cut-out letter, say) gets a trim instead, as does any hop longer than
    `trim_jump_mm`
  * enforce the shortest and longest allowed stitch, and lock the thread with tiny tie
    stitches wherever it starts or ends so it can't unravel
"""

from __future__ import annotations

import math
from pathlib import Path

from shapely.geometry import LineString
from shapely.ops import unary_union
from shapely.strtree import STRtree

from .common import dist

ALWAYS_JOIN_MM = 1.0       # a hop this short is no more visible than an ordinary stitch
KIND_ORDER = ("fill", "satin", "run")
CAP_BAND_MM = 20.0


class _NewLetter(list):
    """The first stretch of thread of a letter. The hop to it is sewn through only where it
    is hidden: even a short visible hop joins two letters (a period becomes a comma)."""


def _clean_segment(pts: list, min_len: float, max_len: float) -> list:
    """Drop stitches shorter than min_len, split stitches longer than max_len."""
    if len(pts) < 2:
        return []
    kept = [pts[0]]
    for p in pts[1:-1]:
        if dist(p, kept[-1]) >= min_len:
            kept.append(p)
    last = pts[-1]
    if dist(last, kept[-1]) >= min_len or len(kept) == 1:
        kept.append(last)
    else:
        kept[-1] = last            # keep the true end point rather than a too-short stitch
        while len(kept) > 2 and dist(kept[-1], kept[-2]) < min_len:
            del kept[-2]           # ...and don't let that move create a short one before it
    if len(kept) < 2 or dist(kept[0], kept[-1]) < 1e-9 and len(kept) == 2:
        return []
    out = [kept[0]]
    for p in kept[1:]:
        a = out[-1]
        d = dist(a, p)
        if d > max_len:
            n = math.ceil(d / max_len)
            out.extend((a[0] + (p[0] - a[0]) * k / n, a[1] + (p[1] - a[1]) * k / n)
                       for k in range(1, n))
        out.append(p)
    return out


def _tie(pts: list, tie: float) -> list:
    """Add lock stitches at both ends: a tiny step forward and back."""
    def step(a, b):
        d = dist(a, b)
        if d < 1e-9:
            return (a[0] + tie, a[1])
        t = min(tie, d) / d
        return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)

    head = [pts[0], step(pts[0], pts[1])]
    tail = [step(pts[-1], pts[-2]), pts[-1]]
    return head + pts + tail


def _extent(o: dict) -> tuple:
    xs = [p[0] for sg in o["segments"] for p in sg]
    ys = [p[1] for sg in o["segments"] for p in sg]
    return min(xs), min(ys), max(xs), max(ys)


def _nearest_first(group: list, pos):
    """Objects of one kind, each time going to whichever starts nearest. Returns
    ([(thread segment, the object's shape)] in order, where the needle ends up)."""
    group = list(group)
    out = []
    while group:
        best = None
        for o in group:
            start, end = o["segments"][0][0], o["segments"][-1][-1]
            d = 0.0 if pos is None else dist(pos, start)
            if best is None or d < best[0]:
                best = (d, o, False)
            if o["reversible"] and pos is not None and dist(pos, end) < best[0]:
                best = (dist(pos, end), o, True)
        _, o, rev = best
        group.remove(o)
        segs = o["segments"]
        if rev:
            segs = [sg[::-1] for sg in segs[::-1]]
        out.extend((sg, o.get("shape")) for sg in segs)
        pos = segs[-1][-1]
    return out, pos


def run(stitched: dict, palette: list, rules: dict, order: list | None = None,
        debug_dir: Path | None = None) -> dict:
    base = [o for o in stitched["objects"] if o["kind"] == "knockdown"]
    objects = [o for o in stitched["objects"] if o["kind"] != "knockdown"]
    min_len, max_len = rules["min_stitch_mm"], rules["max_stitch_mm"]

    # ---- color order: as decided when the shapes were classified (biggest area first);
    #      colors that are only thin lines go last so outlines land on top
    info = {}
    for o in objects:
        d = info.setdefault(o["color"], {"area": 0.0, "solid": 0})
        d["area"] += o.get("area_mm2", 0.0)
        d["solid"] += o["kind"] != "run"
    rank = {c: i for i, c in enumerate(order or [])}
    colors = sorted(info, key=lambda c: (info[c]["solid"] == 0, rank.get(c, 0), -info[c]["area"]))

    # ---- groups: normally everything at once; for hats, bands from the centre outward
    if rules.get("cap_mode") and objects:
        allx = [v for o in objects for v in (_extent(o)[0], _extent(o)[2])]
        ally = [v for o in objects for v in (_extent(o)[1], _extent(o)[3])]
        cx = (min(allx) + max(allx)) / 2

        def band(o):
            x0, _, x1, _ = _extent(o)
            if x0 <= cx <= x1:                       # straddles the centre line
                return 0
            return int(min(abs(x0 - cx), abs(x1 - cx)) // CAP_BAND_MM) + 1

        keys = sorted({band(o) for o in objects})
        groups = [[o for o in objects if band(o) == k] for k in keys]
        pos = (cx, max(ally))                        # start at the bottom, in the middle
    else:
        groups = [objects]
        pos = None

    raw_blocks: list = []                            # [color, [segments]]
    for group in groups:
        for color in colors:
            segs: list = []
            for kind in KIND_ORDER:
                mine = [o for o in group if o["color"] == color and o["kind"] == kind]
                # lettering is sewn in its own order: letter by letter, left to right, each
                # letter's strokes as the font planned them (research rule 36)
                lettered = sorted((o for o in mine if o.get("seq") is not None),
                                  key=lambda o: o["seq"])
                last, last_punct = None, False
                for o in lettered:
                    first, *rest = o["segments"]
                    letter = o["seq"][:2]
                    # a short hop to the next letter may be sewn through, except next to
                    # punctuation (a period joined to its letter reads as a comma)
                    fresh = letter != last and (o.get("punct") or last_punct)
                    segs.append((_NewLetter(first) if fresh else first, o.get("shape")))
                    segs.extend((sg, o.get("shape")) for sg in rest)
                    if letter != last:
                        last_punct = bool(o.get("punct"))
                    last = letter
                    pos = o["segments"][-1][-1]
                part, pos = _nearest_first([o for o in mine if o.get("seq") is None], pos)
                segs.extend(part)
            if not segs:
                continue
            if raw_blocks and raw_blocks[-1][0] == color:
                raw_blocks[-1][1].extend(segs)
            else:
                raw_blocks.append([color, segs])

    blocks = []
    for o in base:
        raw_blocks.insert(0, [-1, [(sg, None) for sg in o["segments"]]])

    # A connecting thread is hidden only where stitching sewn AFTER it covers it. Thread laid
    # across stitching that is already sewn shows on top (a line across a finished satin).
    order_shapes, order_of = [], {}
    for n, (_color, raw_segments) in enumerate(raw_blocks):
        for j, (_sg, shape) in enumerate(raw_segments):
            order_of[(n, j)] = len(order_shapes)
            order_shapes.append(shape.buffer(0.3) if shape is not None else None)
    indexed = [(k, g) for k, g in enumerate(order_shapes) if g is not None]
    tree = STRtree([g for _, g in indexed]) if indexed else None

    def hidden_by_later(line, k):
        if tree is None:
            return False
        near = [indexed[i] for i in tree.query(line)]
        later = [g for kk, g in near if kk >= k]
        return bool(later) and unary_union(later).contains(line)

    for n, (color, raw_segments) in enumerate(raw_blocks):
        # join short hops where the thread is hidden, cut the rest
        joined = []
        for j, (sg, _shape) in enumerate(raw_segments):
            hop = dist(joined[-1][-1], sg[0]) if joined else None
            short_ok = hop is not None and hop <= ALWAYS_JOIN_MM and not isinstance(sg, _NewLetter)
            ok = hop is not None and hop <= rules["trim_jump_mm"] and (
                short_ok or hidden_by_later(LineString([joined[-1][-1], sg[0]]), order_of[(n, j)]))
            if ok:
                joined[-1] = joined[-1] + list(sg)
            else:
                joined.append(list(sg))
        final = []
        for sg in joined:
            sg = _clean_segment(sg, min_len, max_len)
            if len(sg) >= 2:
                final.append(_tie(sg, rules["tie_mm"]))
        if not final:
            continue
        if color == -1:
            blocks.append({"color": -1, "rgb": tuple(rules.get("knockdown_rgb", (205, 205, 205))),
                           "segments": final,
                           "label": "Flattening layer: thread that matches the fabric"})
        else:
            blocks.append({"color": color, "rgb": tuple(palette[color]), "segments": final})

    stitches = sum(len(sg) for b in blocks for sg in b["segments"])
    trims = sum(len(b["segments"]) - 1 for b in blocks)
    xs = [p[0] for b in blocks for sg in b["segments"] for p in sg]
    ys = [p[1] for b in blocks for sg in b["segments"] for p in sg]
    bounds = (min(xs), min(ys), max(xs), max(ys)) if xs else (0, 0, 0, 0)
    return {"blocks": blocks, "stitch_count": stitches, "trims": trims,
            "color_changes": max(0, len(blocks) - 1), "bounds": bounds}
