"""Stage 9 — Validate.

Check the finished plan against the digitizing rules and the hoop, and boil it down to a
traffic light: green (ready), yellow (will sew, but read the notes), red (won't sew well).

Messages are for the person at the machine: plain words. Wherever the app can fix the
problem itself the message carries one-click fixes, each a label plus the setting to change:
    {"label": "Make it fit", "set": {"size_in": 7.25}}
    {"label": "Sew it lighter", "set": {"density": -1}}
"""

from __future__ import annotations

import math
from pathlib import Path

from .common import dist

MM_PER_INCH = 25.4


def pick_hoop(w_mm: float, h_mm: float, hoops: list, margin: float):
    """Smallest hoop the design fits in with the safety margin (either way round)."""
    for hoop in sorted(hoops, key=lambda h: h["w_mm"] * h["h_mm"]):
        hw, hh = hoop["w_mm"] - 2 * margin, hoop["h_mm"] - 2 * margin
        if (w_mm <= hw and h_mm <= hh) or (w_mm <= hh and h_mm <= hw):
            return hoop
    return None


def _quarter_down(inches: float) -> float:
    return math.floor(inches * 4 + 1e-6) / 4


def largest_size_in(w: float, h: float, hoops: list, margin: float) -> float:
    """The biggest "widest side" (inches, to a quarter inch) that still fits some hoop."""
    best = 0.0
    for hoop in hoops:
        hw, hh = hoop["w_mm"] - 2 * margin, hoop["h_mm"] - 2 * margin
        best = max(best, min(hw / w, hh / h), min(hh / w, hw / h))
    return _quarter_down(max(w, h) * best * 0.995 / MM_PER_INCH)


def run(plan: dict, stitched: dict, classified: dict, fabric: dict, hoops: list, rules: dict,
        hints: dict | None = None, debug_dir: Path | None = None) -> dict:
    hints = hints or {}
    msgs = []

    def add(level, text, fixes=None):
        msgs.append({"level": level, "text": text, "fixes": fixes or []})

    def size_fix(label, inches):
        return {"label": f"{label} ({inches:g} in)", "set": {"size_in": inches}}

    minx, miny, maxx, maxy = plan["bounds"]
    w, h = maxx - minx, maxy - miny
    widest_in = max(w, h) / MM_PER_INCH
    max_in = largest_size_in(w, h, hoops, rules["hoop_margin_mm"]) if w > 0 and h > 0 else 0.0
    density = int(hints.get("density", 0))

    if plan["stitch_count"] == 0:
        add("red", "There is nothing to stitch. Try a bigger size or a simpler picture.")

    # ---- stitch lengths
    too_long = too_short = 0
    for b in plan["blocks"]:
        for sg in b["segments"]:
            for a, c in zip(sg, sg[1:]):
                d = dist(a, c)
                if d > rules["max_stitch_mm"] + 0.01:
                    too_long += 1
                elif d < rules["min_stitch_mm"] - 0.01:
                    too_short += 1
    if too_long:
        add("red", f"{too_long} stitches are too long and could snag.",
            [size_fix("Make it smaller", _quarter_down(widest_in * 0.8))])
    if too_short:
        bigger = min(max_in, _quarter_down(widest_in * 1.25))
        add("yellow", f"{too_short} stitches are very short, and thread could break there.",
            [size_fix("Make it bigger", bigger)] if bigger > widest_in + 0.1 else [])

    # ---- hoop
    hoop = pick_hoop(w, h, hoops, rules["hoop_margin_mm"])
    if hoop is None and plan["stitch_count"]:
        biggest = max(hoops, key=lambda x: x["w_mm"] * x["h_mm"])
        add("red", f"This design is too big for the largest hoop ({biggest['name']}).",
            [size_fix("Make it fit", max_in)] if max_in > 0 else [])

    # ---- fabric-specific
    limit = fabric.get("max_height_mm")
    if limit and h > limit + 0.5:
        fit = _quarter_down(max(w, h) * (limit / h) * 0.995 / MM_PER_INCH)
        add("yellow", f"Hat designs sew best under {limit / MM_PER_INCH:.2f} in tall. "
                      f"This one is {h / MM_PER_INCH:.2f} in tall.",
            [size_fix("Make it fit the hat", fit)])
    if fabric.get("warn_large_fills"):
        big = [o for o in stitched["objects"]
               if o["kind"] == "fill" and o.get("area_mm2", 0) > rules["large_fill_mm2"]]
        if big:
            fixes = [size_fix("Make it smaller", _quarter_down(widest_in * 0.75))]
            if density > -2:
                fixes.append({"label": "Sew it lighter", "set": {"density": density - 1}})
            add("yellow", "This design has big solid areas, which can pucker on T-shirts.", fixes)

    # ---- detail and text
    if classified.get("dropped_details"):
        n = classified["dropped_details"]
        bigger = min(max_in, _quarter_down(widest_in * 1.25))
        add("yellow", f"{n} tiny detail{'s were' if n != 1 else ' was'} too small to sew and "
                      "left out.",
            [size_fix("Make it bigger", bigger)] if bigger > widest_in + 0.1 else [])
    pct = hints.get("smallest_text_pct")
    if hints.get("text") and pct:
        text_mm = h * pct / 100.0
        if text_mm < rules["min_text_mm"] - 0.2:
            need = min(max_in, math.ceil(widest_in * rules["min_text_mm"] / text_mm * 4) / 4)
            add("yellow", "This text is too small to sew cleanly "
                          f"(about {text_mm:.0f} mm tall; {rules['min_text_mm']:g} mm is the "
                          "smallest that stays readable).",
                [size_fix("Make it bigger", need)] if need > widest_in + 0.1 else [])

    # ---- size of the job
    design_colors = [b for b in plan["blocks"] if b["color"] != -1]
    if len(design_colors) > rules["max_colors"]:
        add("yellow", "This design uses a lot of thread colors. Fewer colors sew faster.")
    if plan["stitch_count"] > 60000:
        add("yellow", f"This is a big job: {plan['stitch_count']:,} stitches.",
            [size_fix("Make it smaller", _quarter_down(widest_in * 0.75))])

    status = "red" if any(m["level"] == "red" for m in msgs) else \
             "yellow" if msgs else "green"
    return {"status": status, "messages": msgs, "hoop": hoop,
            "width_mm": w, "height_mm": h}
