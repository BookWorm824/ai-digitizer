"""Stage 9 — Validate.

Check the finished plan against the digitizing rules and the hoop, and boil it down to a
traffic light: green (ready), yellow (will sew, but read the notes), red (won't sew well).

Messages are for the person at the machine: plain words, and what to do about it.
"""

from __future__ import annotations

from pathlib import Path

from .common import dist


def pick_hoop(w_mm: float, h_mm: float, hoops: list, margin: float):
    """Smallest hoop the design fits in with the safety margin (either way round)."""
    for hoop in sorted(hoops, key=lambda h: h["w_mm"] * h["h_mm"]):
        hw, hh = hoop["w_mm"] - 2 * margin, hoop["h_mm"] - 2 * margin
        if (w_mm <= hw and h_mm <= hh) or (w_mm <= hh and h_mm <= hw):
            return hoop
    return None


def run(plan: dict, stitched: dict, classified: dict, fabric: dict, hoops: list, rules: dict,
        debug_dir: Path | None = None) -> dict:
    msgs = []

    def add(level, text):
        msgs.append({"level": level, "text": text})

    minx, miny, maxx, maxy = plan["bounds"]
    w, h = maxx - minx, maxy - miny

    if plan["stitch_count"] == 0:
        add("red", "There is nothing to stitch. Try a bigger size or a simpler picture.")

    # stitch lengths
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
        add("red", f"{too_long} stitches are too long and could snag. Try a smaller size.")
    if too_short:
        add("yellow", f"{too_short} stitches are very short. Thread could break there; "
                      "a bigger size usually fixes it.")

    # hoop
    hoop = pick_hoop(w, h, hoops, rules["hoop_margin_mm"])
    if hoop is None:
        biggest = max(hoops, key=lambda x: x["w_mm"] * x["h_mm"])
        add("red", f"This design is too big for the largest hoop ({biggest['name']}). "
                   "Choose a smaller size.")

    # fabric-specific
    if fabric.get("max_height_mm") and h > fabric["max_height_mm"] + 0.5:
        add("yellow", f"Hat designs sew best under {fabric['max_height_mm'] / 25.4:.2f} in tall. "
                      "Choose the Hat front size.")
    if fabric.get("warn_large_fills"):
        big = [o for o in stitched["objects"]
               if o["kind"] == "fill" and o.get("area_mm2", 0) > rules["large_fill_mm2"]]
        if big:
            add("yellow", "This design has big solid areas, which can pucker on T-shirts. "
                          "A smaller size will sew flatter.")

    if classified.get("dropped_details"):
        n = classified["dropped_details"]
        add("yellow", f"{n} tiny detail{'s were' if n != 1 else ' was'} too small to sew and "
                      "left out. A bigger size keeps more detail.")
    if len(plan["blocks"]) > rules["max_colors"]:
        add("yellow", "This design uses a lot of thread colors. Fewer colors sew faster.")

    status = "red" if any(m["level"] == "red" for m in msgs) else \
             "yellow" if msgs else "green"
    return {"status": status, "messages": msgs, "hoop": hoop,
            "width_mm": w, "height_mm": h}
