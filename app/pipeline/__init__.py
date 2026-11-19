"""The image -> stitch file pipeline. One module per stage (see BRIEF.md, "Under the hood").

Phase 1 runs the local stages only: intake, clean, vectorize, classify, stitch, sequence,
validate, export. The two AI stages (analyze, optimize) arrive in Phase 2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .. import config
from . import classify, clean, export, intake, sequence, stitch, validate, vectorize
from .common import PipelineError, hex_of

__all__ = ["run_pipeline", "PipelineError"]

# What the user sees while each stage runs.
LABELS = {
    "intake": "Opening your picture…",
    "clean": "Cleaning up your art…",
    "vectorize": "Tracing the shapes…",
    "classify": "Choosing stitch types…",
    "stitch": "Laying stitches…",
    "sequence": "Planning the sewing order…",
    "validate": "Checking the design…",
    "export": "Saving your files…",
}


def run_pipeline(data: bytes, out_dir: Path, name: str = "design", *,
                 fit_w_mm: float = 88.9, fit_h_mm: float = 88.9, fabric_id: str = "tshirt",
                 max_colors: int | None = None, overrides: dict | None = None,
                 progress: Callable[[str], None] | None = None, debug: bool = True) -> dict:
    """Run every stage and return a JSON-able summary. Files land in out_dir."""
    out_dir = Path(out_dir)
    debug_dir = out_dir / "debug" if debug else None
    fabric = config.fabrics().get(fabric_id) or config.fabrics()["tshirt"]
    rules = config.rules_for(fabric["id"], overrides)
    if max_colors:
        rules["max_colors"] = int(max_colors)

    def say(stage):
        if progress:
            progress(LABELS[stage])

    say("intake")
    img = intake.run(data, debug_dir)
    say("clean")
    cleaned = clean.run(img, fit_w_mm, fit_h_mm, rules, debug_dir)
    say("vectorize")
    vec = vectorize.run(cleaned, fit_w_mm, fit_h_mm, rules, debug_dir)
    say("classify")
    classified = classify.run(vec, rules, debug_dir)
    say("stitch")
    stitched = stitch.run(classified, rules, debug_dir)
    if not stitched["objects"]:
        raise PipelineError(
            "Everything in this picture is too small to stitch at this size. Try a bigger size.")
    say("sequence")
    plan = sequence.run(stitched, cleaned["palette"], rules, debug_dir)
    say("validate")
    check = validate.run(plan, stitched, classified, fabric, config.hoops(), rules, debug_dir)
    say("export")
    exported = export.run(plan, out_dir, name, debug_dir)

    for fmt, info in exported["files"].items():
        if not info["ok"]:
            check["messages"].append({
                "level": "red",
                "text": f"The {fmt.upper()} file didn't pass its safety check, so don't sew it. "
                        "Please try again."})
            check["status"] = "red"

    w, h = check["width_mm"], check["height_mm"]
    return {
        "name": name,
        "stats": {
            "width_mm": round(w, 1), "height_mm": round(h, 1),
            "width_in": round(w / config.MM_PER_INCH, 2),
            "height_in": round(h / config.MM_PER_INCH, 2),
            "stitches": plan["stitch_count"], "colors": len(plan["blocks"]),
            "trims": plan["trims"],
            "sew_minutes": max(1, round(plan["stitch_count"] / rules["machine_spm"]
                                        + 0.5 * plan["color_changes"])),
        },
        "check": {"status": check["status"], "messages": check["messages"],
                  "hoop": check["hoop"]["name"] if check["hoop"] else None},
        "fabric": {"id": fabric["id"], "name": fabric["name"], "stabilizer": fabric["stabilizer"]},
        "colors": [{"hex": hex_of(b["rgb"]),
                    "stitches": sum(len(s) for s in b["segments"])} for b in plan["blocks"]],
        "blocks": [{"hex": hex_of(b["rgb"]),
                    "segments": [[round(v, 2) for p in sg for v in p] for sg in b["segments"]]}
                   for b in plan["blocks"]],
        "bounds": [round(v, 2) for v in plan["bounds"]],
        "files": {fmt: {"ok": info["ok"], "path": info["path"], "stitches": info["stitches"]}
                  for fmt, info in exported["files"].items()},
        "preview": exported["preview"],
    }
