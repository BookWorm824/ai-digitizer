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
    "min_detail_mm": 1.0,         # smaller than this won't sew
    "thin_line_mm": 1.5,          # narrower than this becomes a running stitch
    "bean_line_mm": 0.6,          # thin lines at least this wide get a triple (bean) run
    "fill_spacing_mm": 0.4,       # row spacing of a fill (fabric presets override)
    "fill_stitch_mm": 3.5,
    "run_stitch_mm": 2.5,
    "run_stitch_curve_mm": 1.5,   # on tight curves
    "max_stitch_mm": 7.0,         # longer stitches are split
    "min_stitch_mm": 0.5,         # shorter stitches are dropped
    "pull_comp_mm": 0.2,          # shapes are grown by this much (fabric presets override)
    "trim_jump_mm": 3.0,          # jumps longer than this get a trim
    "max_colors": 6,
    "min_text_mm": 5.0,           # smallest capital-letter height that stays readable
    "hoop_margin_mm": 5.0,
    "underlay": True,             # light fill underlay at 90 degrees to the top stitches
    "underlay_spacing_mm": 3.0,
    "underlay_inset_mm": 0.6,
    "underlay_min_width_mm": 3.0,  # shapes narrower than this get no underlay
    "tie_mm": 0.7,                # length of the lock stitches at thread starts and ends
    "machine_spm": 650,           # stitches per minute, for the sew-time estimate
    "large_fill_mm2": 2000.0,     # "big solid area" warning threshold
    "work_px_per_mm": 10.0,       # working resolution of the cleanup pass
    "work_max_px": 2400,
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


def rules_for(fabric_id: str, overrides: dict | None = None) -> dict:
    """RULES with the fabric preset's values (and any explicit overrides) applied."""
    r = dict(RULES)
    fab = fabrics().get(fabric_id) or fabrics()["tshirt"]
    r.update(fab.get("rules", {}))
    if overrides:
        r.update({k: v for k, v in overrides.items() if v is not None})
    return r
