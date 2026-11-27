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

import cv2
import numpy as np
from skimage.graph import route_through_array

from shapely.geometry import LineString
from shapely.ops import unary_union
from shapely.strtree import STRtree

from .common import dist
from .stitch import Cut

ALWAYS_JOIN_MM = 1.0       # a hop this short is no more visible than an ordinary stitch
BLOCK_OPEN_MM = 0.4        # ... but beside small blocks only this much: the gap is the design
OTHER_COLOR_MARGIN_MM = 0.3  # thread under another color stays this far inside its edges
THIN_COVER_TRIM_MM = 0.25  # a thin line's shape (line + 0.4 mm) less this is what it hides
COVER_PAD_MM = 0.15        # stitching hides thread this far past its own edge (thread spread)
ROUTE_MAX_MM = 30.0        # hidden travel is looked for on hops up to this long
ROUTE_PAD_MM = 6.0         # ...within this much room around the straight line
LETTER_OPEN_SHARE = 0.18   # a letter connector's open part is at most this share of its height
ROUTE_CELL_MM = 0.3        # grid used to find it
ROUTE_OPEN_MM = 1.5        # at most this much of it may lie on open ground: the connector
                           # between two satin letters at their closest points (digitizing
                           # software trims above 2 mm). Only lettering gets this much.
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


def _late_outlines(objects: list, colors: list, any_order: bool = False) -> dict:
    """Outlines (line work) that surround solid stitching of a color sewn after their own:
    {id: object}. A ring of gold around navy numbers, when gold is sewn first because of
    other gold letters, would end up under the navy."""
    from shapely.geometry import Polygon          # noqa: PLC0415
    turn = {c: i for i, c in enumerate(colors)}
    solids = [o for o in objects if o["kind"] in ("fill", "satin") and o.get("shape") is not None]
    out = {}
    for o in objects:
        if o["kind"] != "run" or o.get("shape") is None or o.get("seq") is not None:
            continue
        geoms = getattr(o["shape"], "geoms", [o["shape"]])
        rings = [Polygon(g.exterior) for g in geoms if g.geom_type == "Polygon"]
        if not rings:
            continue
        inside = [q for q in solids if q["color"] != o["color"]
                  and (any_order or turn.get(q["color"], 0) > turn.get(o["color"], 0))
                  and any(r.contains(q["shape"].representative_point()) for r in rings)]
        if inside and sum(q["shape"].area for q in inside) >= 0.2 * sum(r.area for r in rings):
            out[id(o)] = o
    return out


def _centre_out(lettered: list, cx: float) -> list:
    """Letters of each word in cap order (Erich Campbell, Jeff Fuller): from the letter
    nearest the centre seam outward to the left, then from the centre outward to the right.
    Each letter keeps its own stroke order."""
    out = []
    by_word: dict = {}
    for o in lettered:
        by_word.setdefault(o["seq"][0], []).append(o)
    for word in by_word.values():
        letters: dict = {}
        for o in word:
            letters.setdefault(o["seq"][1], []).append(o)
        mid = {li: (lambda e: (e[0] + e[2]) / 2)(_extent_all(objs)) for li, objs in letters.items()}
        idx = sorted(letters)
        centre = min(idx, key=lambda li: abs(mid[li] - cx))
        order = [li for li in reversed(idx) if li <= centre] + [li for li in idx if li > centre]
        for li in order:
            out.extend(letters[li])
    return out


def _extent_all(objs: list) -> tuple:
    boxes = [_extent(o) for o in objs]
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


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
            if o.get("alt") and pos is not None and dist(pos, o["alt"][0][0]) < best[0]:
                best = (dist(pos, o["alt"][0][0]), o, "alt")   # a column from its other end
        _, o, rev = best
        group.remove(o)
        segs = o["segments"]
        if rev == "alt":
            segs = o["alt"]
        elif rev:
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
    # An outline sews after what it surrounds (letter first, outline second). When a design
    # has both gold letters outlined in navy and navy numbers outlined in gold, one turn per
    # color can't do both: such an outline gets a second turn of its color at the end.
    # (on caps parts are sewn band by band from the centre, so an outline spanning a word
    # can be sewn before letters of a color that comes earlier: there any surrounding
    # outline goes last)
    late = _late_outlines(objects, colors, any_order=bool(rules.get("cap_mode")))
    if late:
        objects = [o for o in objects if id(o) not in late]

    # ---- groups: normally everything at once; for hats, bands from the centre outward
    if rules.get("cap_mode") and objects:
        allx = [v for o in objects for v in (_extent(o)[0], _extent(o)[2])]
        ally = [v for o in objects for v in (_extent(o)[1], _extent(o)[3])]
        cx = (min(allx) + max(allx)) / 2

        # A word is one piece: banded letter by letter, the needle crossed the word back and
        # forth and cut at nearly every letter. It is banded by its whole extent and sewn
        # from its centre letter outward (below).
        words: dict = {}
        for o in objects:
            if o.get("seq") is not None:
                x0, _, x1, _ = _extent(o)
                w = words.setdefault(o["seq"][0], [x0, x1])
                w[0], w[1] = min(w[0], x0), max(w[1], x1)

        def band(o):
            if o.get("seq") is not None:
                x0, x1 = words[o["seq"][0]]
            else:
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
        cx = None

    # (Tried: on caps under 3 inches wide, each color sewn once, centre-out. Fewer color
    # changes, but heavy black outlines sewn after every orange part covered the thin roofs
    # and the orange travel peeked out beside them. Bands from the centre keep each part's
    # own layering, and color changes are cheap on a multi-needle machine.)
    passes = [(g, c) for g in groups for c in colors]
    raw_blocks: list = []                            # [color, [segments]]
    for group, color in passes:
        segs: list = []
        for kind in KIND_ORDER:
            mine = [o for o in group if o["color"] == color and o["kind"] == kind
                    and not o.get("sew_last")]
            # lettering is sewn in its own order: letter by letter, left to right, each
            # letter's strokes as the font planned them (research rule 36)
            lettered = sorted((o for o in mine if o.get("seq") is not None),
                              key=lambda o: o["seq"])
            if cx is not None and lettered:
                lettered = _centre_out(lettered, cx)
            last, last_punct = None, False
            for o in lettered:
                first, *rest = o["segments"]
                letter = o["seq"][:2]
                # a short hop to the next letter may be sewn through, except next to
                # punctuation (a period joined to its letter reads as a comma)
                # ... and thin running-stitch letters are never joined to each other: a
                # connecting stitch is as heavy as their own lines and changes how
                # they read (E_S_T). Satin letters are joined at their closest points.
                fresh = letter != last and (o.get("punct") or last_punct or o.get("thin"))
                segs.append((_NewLetter(first) if fresh else first, o.get("shape")))
                segs.extend((sg, o.get("shape")) for sg in rest)
                if letter != last:
                    last_punct = bool(o.get("punct"))
                last = letter
                pos = o["segments"][-1][-1]
            part, pos = _nearest_first([o for o in mine if o.get("seq") is None], pos)
            segs.extend(part)
        # rings and round borders: after everything of their color that crosses them, in
        # the same turn of the color (no extra color change)
        part, pos = _nearest_first([o for o in group if o["color"] == color
                                    and o.get("sew_last")], pos)
        segs.extend(part)
        if not segs:
            continue
        if raw_blocks and raw_blocks[-1][0] == color:
            raw_blocks[-1][1].extend(segs)
        else:
            raw_blocks.append([color, segs])
    if late:
        for color in colors:
            mine = [o for o in late.values() if o["color"] == color]
            if not mine:
                continue
            part, pos = _nearest_first(mine, pos)
            if raw_blocks and raw_blocks[-1][0] == color:
                raw_blocks[-1][1].extend(part)
            else:
                raw_blocks.append([color, part])

    blocks = []
    for o in base:
        raw_blocks.insert(0, [-1, [(sg, None) for sg in o["segments"]]])

    # A connecting thread is hidden only where stitching sewn AFTER it covers it. Thread laid
    # across stitching that is already sewn shows on top (a line across a finished satin).
    # Thread may also hide under a later color only if that color is not lighter: dark
    # thread under a light fill shows through.
    thin_shapes = {id(o["shape"]) for o in objects if o.get("thin") and o.get("shape") is not None}
    # (lettering kept as drawn counts too: its letters are joined like a font's, never cut
    # inside the word; tutorials: only gaps over 3 mm between letters are trimmed)
    letter_shapes = {id(o["shape"]) for o in objects
                     if o.get("shape") is not None and (
                         o.get("seq") is not None
                         or (o.get("text_row") is not None and o["kind"] in ("satin", "fill")))}
    block_shapes = {id(o["shape"]) for o in objects if o.get("block") and o.get("shape") is not None}
    text_h = {id(o["shape"]): o["text_h"] for o in objects
              if o.get("text_h") and o.get("shape") is not None}
    order_shapes, order_colors, order_thin, order_letter, order_of = [], [], [], [], {}
    order_block, order_inner, order_ids, order_h = [], [], [], []
    for n, (color_n, raw_segments) in enumerate(raw_blocks):
        for j, (_sg, shape) in enumerate(raw_segments):
            order_of[(n, j)] = len(order_shapes)
            # (a running-stitch line hides only what lies right along it: its shape is
            # the line grown 0.4 mm, so it is taken back to about a thread's width; a
            # walk beside a thin letter is on open fabric)
            pad = -THIN_COVER_TRIM_MM if id(shape) in thin_shapes else COVER_PAD_MM
            order_shapes.append(shape.buffer(pad) if shape is not None else None)
            # (another color hides thread only well inside its edges: travel along the edge
            # of a black outline peeks out beside it)
            order_inner.append(shape.buffer(-OTHER_COLOR_MARGIN_MM) if shape is not None else None)
            order_colors.append(color_n)
            order_thin.append(id(shape) in thin_shapes)
            order_letter.append(id(shape) in letter_shapes)
            order_h.append(text_h.get(id(shape)))
            order_block.append(id(shape) in block_shapes)
            order_ids.append(id(shape) if shape is not None else None)
    indexed = [(k, g) for k, g in enumerate(order_shapes) if g is not None]
    tree = STRtree([g for _, g in indexed]) if indexed else None

    def lum(c):
        if c < 0 or c >= len(palette):
            return 128.0
        r, g, b = palette[c]
        return 0.299 * r + 0.587 * g + 0.114 * b

    def covers(k, color, line, skip=None):
        if tree is None:
            return []
        near = [indexed[i] for i in tree.query(line)]
        # same color: any later stitching hides it (even a line sewn along the same path);
        # another color: only solid stitching that is not lighter, well inside its edges.
        # (`skip`: a part already partly sewn, whose outline says nothing of where its
        # remaining pieces lie)
        out = []
        for kk, g in near:
            if kk < k or (skip is not None and order_ids[kk] == skip):
                continue
            if order_colors[kk] == color:
                out.append(g)
            elif not order_thin[kk] and lum(order_colors[kk]) <= lum(color) + 20:
                inner = order_inner[kk]
                if inner is not None and not inner.is_empty:
                    out.append(inner)
        return out

    def hidden_by_later(line, k, color, skip=None):
        later = covers(k, color, line, skip)
        return bool(later) and unary_union(later).contains(line)

    def hidden_route(p, q, k, color, skip=None):
        """A travel path from p to q that stays under stitching sewn later (Ink/Stitch's
        "underpath" idea, our own code): the cheapest route on a fine grid where covered
        ground is cheap and open ground very dear. Accepted only if at most 1 mm of it is
        uncovered and it isn't a long detour."""
        straight = dist(p, q)
        if straight > ROUTE_MAX_MM:
            return None
        box = LineString([p, q]).buffer(ROUTE_PAD_MM, cap_style=3).envelope
        later = covers(k, color, box, skip)
        if not later:
            return None
        cover = unary_union(later).intersection(box)
        if cover.is_empty:
            return None
        minx, miny, maxx, maxy = box.bounds
        cell = ROUTE_CELL_MM
        w, h = int(math.ceil((maxx - minx) / cell)) + 1, int(math.ceil((maxy - miny) / cell)) + 1
        mask = np.zeros((h, w), np.uint8)
        for poly in getattr(cover, "geoms", [cover]):
            if poly.geom_type != "Polygon":
                continue
            ext = np.round((np.asarray(poly.exterior.coords)[:, :2] - [minx, miny]) / cell).astype(np.int32)
            cv2.fillPoly(mask, [ext], 1)
            for ring in poly.interiors:
                hole = np.round((np.asarray(ring.coords)[:, :2] - [minx, miny]) / cell).astype(np.int32)
                cv2.fillPoly(mask, [hole], 0)
        cost = np.where(mask > 0, 1.0, 60.0)
        a = (min(h - 1, max(0, int(round((p[1] - miny) / cell)))), min(w - 1, max(0, int(round((p[0] - minx) / cell)))))
        b = (min(h - 1, max(0, int(round((q[1] - miny) / cell)))), min(w - 1, max(0, int(round((q[0] - minx) / cell)))))
        try:
            path, _total = route_through_array(cost, a, b, fully_connected=True, geometric=True)
        except Exception:  # noqa: BLE001 - no route: trim instead
            return None
        path = np.asarray(path)
        open_cells = int(sum(1 for r, c in path if mask[r, c] == 0))
        # Only satin lettering may be reached by a connector longer than an ordinary short
        # hop: letters in a word are joined at their closest points, not trimmed. (Between
        # thin running-stitch letters a connector is as heavy as the letters' own lines
        # and reads as part of them.)
        allowed = ROUTE_OPEN_MM if (order_letter[k] and not order_thin[k]) else ALWAYS_JOIN_MM
        if order_letter[k] and order_h[k]:
            # (on small lettering the same connector is a bigger share of the letter: a
            # line along the baseline under 5 mm letters, a stroke across the top of an S)
            allowed = min(allowed, max(ALWAYS_JOIN_MM, LETTER_OPEN_SHARE * order_h[k]))
        if order_block[k] or (k > 0 and order_block[k - 1]):
            allowed = BLOCK_OPEN_MM
        if open_cells * cell > allowed:
            return None
        pts = [(minx + c * cell, miny + r * cell) for r, c in path]
        line = LineString([p] + pts + [q]).simplify(cell)
        if line.length > 2.5 * straight + 4.0:
            return None
        coords = list(line.coords)
        out = []
        for u, v in zip(coords, coords[1:]):            # travel stitches of about 2 mm
            n = max(1, math.ceil(dist(u, v) / 2.0))
            out.extend((u[0] + (v[0] - u[0]) * t / n, u[1] + (v[1] - u[1]) * t / n)
                       for t in range(1, n + 1))
        return out[:-1]

    for n, (color, raw_segments) in enumerate(raw_blocks):
        # join short hops where the thread is hidden, cut the rest
        joined = []
        for j, (sg, _shape) in enumerate(raw_segments):
            hop = dist(joined[-1][-1], sg[0]) if joined else None
            k = order_of[(n, j)]
            # a hop to or from a small block (a window) crosses the wall between windows: a
            # stitch there reads as a stem joining them, so it is cut unless hidden
            blocky = order_block[k] or (j > 0 and order_block[order_of[(n, j - 1)]])
            limit = BLOCK_OPEN_MM if blocky else ALWAYS_JOIN_MM
            short_ok = hop is not None and hop <= limit and not isinstance(sg, _NewLetter)
            # (a piece of a fill cut off from the rows before it: those rows are sewn, so
            # its own outline hides nothing)
            skip = order_ids[k] if isinstance(sg, Cut) else None
            ok = hop is not None and hop <= rules["trim_jump_mm"] and (
                short_ok or hidden_by_later(LineString([joined[-1][-1], sg[0]]), k, color, skip))
            if ok:
                joined[-1] = joined[-1] + list(sg)
                continue
            # (a thin running-stitch letter is never reached by travel: see _NewLetter)
            route = (hidden_route(joined[-1][-1], sg[0], k, color, skip)
                     if hop is not None and not (isinstance(sg, _NewLetter) and order_thin[k])
                     else None)
            if route is not None:
                joined[-1] = joined[-1] + route + list(sg)
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
