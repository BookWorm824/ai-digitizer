"""Stage 8 — Sequence.

Put the stitched objects into sewing order and turn them into the final thread plan:

  * one block per color, big background areas first, small details and thin outlines last
  * inside a color, always go to the nearest object next (lines may be sewn from either end)
  * short hops are sewn straight through; hops longer than `trim_jump_mm` get a trim
  * enforce the shortest and longest allowed stitch, and lock the thread with tiny tie
    stitches wherever it starts or ends so it can't unravel
"""

from __future__ import annotations

import math
from pathlib import Path

from .common import dist


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


def run(stitched: dict, palette: list, rules: dict, debug_dir: Path | None = None) -> dict:
    objects = stitched["objects"]
    min_len, max_len = rules["min_stitch_mm"], rules["max_stitch_mm"]

    # ---- color order: biggest filled area first; colors that are only thin lines go last
    info = {}
    for o in objects:
        d = info.setdefault(o["color"], {"area": 0.0, "fills": 0})
        d["area"] += o.get("area_mm2", 0.0)
        d["fills"] += o["kind"] == "fill"
    order = sorted(info, key=lambda c: (info[c]["fills"] == 0, -info[c]["area"]))

    blocks = []
    pos = None
    for color in order:
        todo = [o for o in objects if o["color"] == color]
        # fills before lines within a color, so outlines land on top
        raw_segments = []
        for kind in ("fill", "run"):
            group = [o for o in todo if o["kind"] == kind]
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
                raw_segments.extend(segs)
                pos = segs[-1][-1]

        # join short hops, cut long ones
        joined = []
        for sg in raw_segments:
            if joined and dist(joined[-1][-1], sg[0]) <= rules["trim_jump_mm"]:
                joined[-1] = joined[-1] + list(sg)
            else:
                joined.append(list(sg))
        final = []
        for sg in joined:
            sg = _clean_segment(sg, min_len, max_len)
            if len(sg) >= 2:
                final.append(_tie(sg, rules["tie_mm"]))
        if final:
            blocks.append({"color": color, "rgb": tuple(palette[color]), "segments": final})
            pos = final[-1][-1]

    stitches = sum(len(sg) for b in blocks for sg in b["segments"])
    trims = sum(len(b["segments"]) - 1 for b in blocks)
    xs = [p[0] for b in blocks for sg in b["segments"] for p in sg]
    ys = [p[1] for b in blocks for sg in b["segments"] for p in sg]
    bounds = (min(xs), min(ys), max(xs), max(ys)) if xs else (0, 0, 0, 0)
    return {"blocks": blocks, "stitch_count": stitches, "trims": trims,
            "color_changes": max(0, len(blocks) - 1), "bounds": bounds}
