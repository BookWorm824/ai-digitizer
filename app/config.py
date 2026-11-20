"""Settings, digitizing rules and preset loading.

Everything a later "Advanced" drawer will show lives in RULES or in the JSON preset files,
never as a magic number inside a pipeline stage.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

APP_NAME = "AIDigitizer"
APP_DIR = Path(__file__).resolve().parent
PRESET_DIR = APP_DIR / "presets"
WEB_DIR = APP_DIR.parent / "web"

MM_PER_INCH = 25.4

# Starting values from the "Digitizing rules" table in BRIEF.md. All lengths in millimetres.
RULES: dict = {
    # Shapes and stitch types (see reports/Embroidery digitizing rules.md for sources)
    "min_detail_mm": 1.0,         # smaller than this won't sew
    "satin_min_mm": 1.6,          # narrower lines are sewn as running stitch, not a skinny
                                  # satin (A&E: a 4-ply run is the alternative to thin satin;
                                  # Embroidery Legacy: satin not under 1.5-2 mm)
    "line_bold_mm": 0.8,          # lines at least this wide get a heavier 4-pass run
    "satin_max_mm": 7.0,          # widest satin on wearables before it snags (Wilcom ~7)
    "satin": True,                # narrow columns are sewn as satin
    "bean_line_mm": 0.3,          # visible detail lines get a bean (triple) run (Goodall)
    "bean_stitch_mm": 2.0,        # bean stitch length (sources: about 2 mm)
    "text_run_mm": 4.0,           # lettering shorter than this is sewn as run/bean letters
    "min_text_mm": 6.0,           # smallest cap height that reads cleanly in satin (Hatch, Melco)
    # Density and stitch lengths
    "fill_spacing_mm": 0.4,       # row spacing of a fill (fabric presets override)
    "fill_stitch_mm": 4.0,        # tatami stitch length (Wilcom 4; shorter ripples after washing)
    "run_stitch_mm": 2.5,
    "run_stitch_curve_mm": 1.5,   # on tight curves
    "max_stitch_mm": 7.0,         # longer stitches are split (Wilcom auto-split ~7 on wearables)
    "min_stitch_mm": 0.5,         # shorter stitches are dropped
    # Pull compensation, PER SIDE, as in Wilcom's table: cotton/drill 0.20, T-shirt 0.35,
    # fleece 0.40, lettering 0.2-0.3. Fabric presets override. Satin uses the full value
    # (less on very narrow columns); fill rows use 60 % of it, since fill pulls in less.
    "pull_comp_mm": 0.2,
    "overlap_mm": 0.4,            # how far a color reaches under the colors sewn after it
    "trim_jump_mm": 2.5,          # jumps longer than this get a trim (Bernina 2, Wilcom 3)
    "max_colors": 6,
    "sew_enclosed_background": "auto",
    "hoop_margin_mm": 5.0,
    # Underlay
    "underlay": True,             # master switch for underlay
    "underlay_level": "medium",   # none | light | medium | medium-heavy | heavy (see rules_for)
    "underlay_edge_run": True,    # a run just inside the edge of every fill
    "underlay_spacing_mm": 2.5,   # row spacing of the fill underlay (0 = no fill underlay)
    "underlay_inset_mm": 0.5,     # edge-run inset (sources: 0.2-0.6)
    "underlay_min_width_mm": 3.0,  # shapes narrower than this get no fill underlay
    "satin_center_run_mm": 1.5,   # satin at least this wide gets a centre run (narrower: none)
    "satin_edge_run_mm": 2.5,     # ...at least this wide gets edge runs instead
    "satin_zigzag_mm": 4.0,       # ...at least this wide gets edge runs plus a zig-zag
    "tie_mm": 0.7,                # length of the lock stitches at thread starts and ends
    "machine_spm": 650,           # stitches per minute, for the sew-time estimate
    "large_fill_mm2": 2000.0,     # "big solid area" warning threshold
    "cap_mode": False,            # hats: sew centre-out, bottom-up
    "knockdown": False,           # towels: flatten the pile under the design first
    "knockdown_spacing_mm": 1.5,
    "work_px_per_mm": 10.0,       # working resolution of the cleanup pass
    "work_max_px": 2400,
    "min_gap_mm": 0.5,            # unsewn gaps narrower than this close up when sewn: fill them
    "min_enclosed_mm2": 25.0,     # blank areas inside the design smaller than this (letter
                                  # counters) stay open fabric rather than being sewn
}


def data_dir() -> Path:
    """Where the app keeps the user's files (jobs, settings, log). Never inside the code folder,
    so an update can't wipe it."""
    override = os.environ.get("AIDIGITIZER_DATA")
    if override:
        base = Path(override)
    else:
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / APP_NAME
    base.mkdir(parents=True, exist_ok=True)
    return base


@lru_cache(maxsize=None)
def _load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fabrics() -> dict:
    """All fabric presets, keyed by id, in display order."""
    out = {}
    for p in sorted((PRESET_DIR / "fabrics").glob("*.json")):
        d = _load_json(str(p))
        out[d["id"]] = d
    return dict(sorted(out.items(), key=lambda kv: kv[1].get("order", 99)))


def hoops() -> list:
    return _load_json(str(PRESET_DIR / "hoops.json"))


def sizes() -> list:
    return _load_json(str(PRESET_DIR / "sizes.json"))


def machines() -> list:
    return _load_json(str(PRESET_DIR / "machines.json"))


# How much fill underlay each level means: (edge run?, fill-underlay row spacing in mm).
UNDERLAY_LEVELS = {
    "none": (False, 0.0),
    "light": (True, 0.0),            # edge run only: stable fabrics like denim
    "medium": (True, 3.5),           # deep-dive rule 35 (Campbell): perpendicular, 3-4 mm
    "medium-heavy": (True, 3.0),
    "heavy": (True, 2.5),            # stretchy knits and deep pile
}

# The "softer <-> denser" slider: how much further apart or closer the stitch rows sit.
DENSITY_STEPS = {-2: 1.25, -1: 1.12, 0: 1.0, 1: 0.92, 2: 0.85}


def rules_for(fabric_id: str, overrides: dict | None = None, density: int = 0) -> dict:
    """RULES with the fabric preset's values (and any explicit overrides) applied."""
    r = dict(RULES)
    fab = fabrics().get(fabric_id) or fabrics()["tshirt"]
    r.update(fab.get("rules", {}))
    if overrides:
        r.update({k: v for k, v in overrides.items() if v is not None})
    edge, spacing = UNDERLAY_LEVELS.get(r["underlay_level"], UNDERLAY_LEVELS["medium"])
    r["underlay"] = bool(r["underlay"]) and r["underlay_level"] != "none"
    r["underlay_edge_run"] = edge
    r["underlay_spacing_mm"] = spacing
    r["fill_spacing_mm"] = round(
        r["fill_spacing_mm"] * DENSITY_STEPS.get(int(density), 1.0), 3)
    return r
