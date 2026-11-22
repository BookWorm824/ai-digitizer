"""The image -> stitch file pipeline. One module per stage (see BRIEF.md, "Under the hood").

run_pipeline() runs the local stages: intake, clean, vectorize, classify, stitch, sequence,
validate, export. The two AI stages (analyze, optimize) run before it, from app/cleanup.py,
and hand it the picture the user chose.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .. import ai, config, threads
from shapely.ops import unary_union

from . import (aiplan, analyze, classify, clean, edits as edits_stage, export, intake, ith, simplify,
               lettering, optimize, sequence, stitch, validate, vectorize)
from .common import PipelineError, hex_of

__all__ = ["run_pipeline", "PipelineError"]

# What the user sees while each stage runs.
LABELS = {
    "intake": "Opening your picture…",
    "clean": "Cleaning up your art…",
    "vectorize": "Tracing the shapes…",
    "classify": "Choosing stitch types…",
    "plan": "ChatGPT is planning the stitches…",
    "review": "ChatGPT is checking the stitches…",
    "stitch": "Laying stitches…",
    "sequence": "Planning the sewing order…",
    "validate": "Checking the design…",
    "export": "Saving your files…",
}


def run_pipeline(data: bytes, out_dir: Path, name: str = "design", *,
                 fit_w_mm: float = 88.9, fit_h_mm: float = 88.9, fabric_id: str = "tshirt",
                 max_colors: int | None = None, overrides: dict | None = None,
                 density: int = 0, hints: dict | None = None, inside: str = "auto",
                 edits: list | None = None, project: dict | None = None,
                 letters: str = "auto", ai_assist: bool = False, ai_review: bool = True,
                 garment: str | None = None,
                 progress: Callable[[str], None] | None = None, debug: bool = True) -> dict:
    """Run every stage and return a JSON-able summary. Files land in out_dir."""
    out_dir = Path(out_dir)
    debug_dir = out_dir / "debug" if debug else None
    project = ith.normalize(project)
    if project:                 # an in-the-hoop project sets its own material and art size
        fabric_id = ith.PROJECTS[project["type"]]["fabric"]
        fit_w_mm, fit_h_mm = ith.decoration_fit(project)
    fabric = config.fabrics().get(fabric_id) or config.fabrics()["tshirt"]
    density = max(-2, min(2, int(density)))
    rules = config.rules_for(fabric["id"], overrides, density)
    if max_colors:
        rules["max_colors"] = int(max_colors)
    # Text too small for any stitch to make readable is left out by default (a professional
    # drops or moves it, deep-dive rule 25 / 60); a negative value means "keep everything".
    rules["drop_text_mm"] = rules.get("auto_drop_text_mm", 2.0)
    if (overrides or {}).get("drop_text_mm"):
        v = float(overrides["drop_text_mm"])
        rules["drop_text_mm"] = 0.0 if v < 0 else v
    if inside in ("sew", "open"):       # the user's own choice for blank areas inside the design
        rules["sew_enclosed_background"] = inside == "sew"
    if garment:                         # "#rrggbb": decides whether blank areas need thread
        try:
            v = int(garment.lstrip("#")[:6], 16)
            rules["garment_rgb"] = (v >> 16 & 255, v >> 8 & 255, v & 255)
        except ValueError:
            pass

    def say(stage):
        if progress:
            progress(LABELS[stage])

    say("intake")
    img = intake.run(data, debug_dir)
    if rules.get("auto_simplify", True):
        # fine pen drawings (window panes, stonework, leafy trees) are redrawn for their
        # stitched size first; art that is already bold enough passes through unchanged
        simple = simplify.run(img, fit_w_mm, fit_h_mm, debug_dir)
        if simple is not None:
            img = simple
            rules["drawn_line_mm"] = simple.get("line_mm", simplify.LINE_W_MM)  # its lines
    say("clean")
    cleaned = clean.run(img, fit_w_mm, fit_h_mm, rules, debug_dir)
    say("vectorize")
    vec = vectorize.run(cleaned, fit_w_mm, fit_h_mm, rules, debug_dir)
    vec["inside_color"] = cleaned.get("inside_color")
    say("classify")
    if letters not in ("auto", "retype", "trace"):
        letters = "auto"
    classified = classify.run(vec, rules, debug_dir, text=(hints or {}).get("text"),
                              letters=letters)
    inside_c = cleaned.get("inside_color")
    if (cleaned.get("squeezed") and inside_c is not None
            and not any(o["color"] == inside_c for o in classified["objects"])):
        # The slot kept for blank areas went unused (they were all the insides of letters,
        # left as open fabric), but it cost the design a color: clean up again without it.
        rules = dict(rules, max_colors=rules["max_colors"] + 1)
        cleaned = clean.run(img, fit_w_mm, fit_h_mm, rules, debug_dir)
        vec = vectorize.run(cleaned, fit_w_mm, fit_h_mm, rules, debug_dir)
        vec["inside_color"] = cleaned.get("inside_color")
        classified = classify.run(vec, rules, debug_dir, text=(hints or {}).get("text"),
                                  letters=letters)
        rules["max_colors"] -= 1

    # ChatGPT as the digitizer's judgment (aiplan.py): what each part is and how to sew it.
    ai_info = {"used": False, "reviewed": False, "notes": [], "problem": None}
    size_text = f"{max(fit_w_mm, fit_h_mm) / config.MM_PER_INCH:.1f} inches across"
    fabric_text = f"a {fabric['name'].lower()}"
    ai_plan = None
    if ai_assist:
        say("plan")
        try:
            ai_plan = aiplan.make_plan(vec, size_text, fabric_text)
        except ai.AIUnavailable as e:
            ai_info["problem"] = e.reason
        if ai_plan:
            if debug_dir is not None:
                debug_dir.mkdir(parents=True, exist_ok=True)
                (debug_dir / "06a_ai_plan_map.png").write_bytes(ai_plan["png"])
            ai_info["used"] = True
            ai_info["notes"] = list(ai_plan["notes"])
            say("classify")
            classified = classify.run(vec, rules, debug_dir, text=(hints or {}).get("text"),
                                      letters=letters, plan=ai_plan)

    palette0 = list(cleaned["palette"])
    bx0, by0, bx1, by1 = vec["bounds"]
    ref = max(bx1 - bx0, by1 - by0)         # what click positions are fractions of

    def build(classified):
        palette, applied = list(palette0), 0
        if edits:                           # the user's own changes, made on the preview
            classified, palette, applied = edits_stage.apply(classified, palette, edits, ref, rules)
            if not classified["objects"]:
                raise PipelineError("There is nothing left to stitch. Undo your last change.")
        say("stitch")
        stitched = stitch.run(classified, rules, debug_dir)
        if not stitched["objects"]:
            raise PipelineError(
                "Everything in this picture is too small to stitch at this size. Try a bigger size.")
        say("sequence")
        plan = sequence.run(stitched, palette, rules, classified["order"], debug_dir)
        return classified, palette, applied, stitched, plan

    classified, palette, applied, stitched, plan = build(classified)
    if ai_plan and ai_review:
        # one round of review: ChatGPT looks at the stitches and names what would sew badly
        say("review")
        try:
            verdict = aiplan.review(ai_plan, vec, plan, size_text, fabric_text)
        except ai.AIUnavailable as e:
            verdict = None
            ai_info["problem"] = ai_info["problem"] or e.reason
        if verdict:
            if debug_dir is not None:
                (debug_dir / "08a_ai_review_map.png").write_bytes(verdict["png"])
            ai_info["reviewed"] = True
            ai_info["notes"] = (verdict["notes"] or ai_info["notes"])[:3]
            if verdict["changed"]:
                say("classify")
                again = classify.run(vec, rules, debug_dir, text=(hints or {}).get("text"),
                                     letters=letters, plan=ai_plan)
                classified, palette, applied, stitched, plan = build(again)
    made = None
    if project:                 # wrap the construction steps around the decoration
        decoration = unary_union([o["shape"] for o in stitched["objects"] if o.get("shape") is not None])
        plan, made = ith.build(plan, decoration, project, rules)
    say("validate")
    check = validate.run(plan, stitched, classified, fabric, config.hoops(), rules,
                         dict(hints or {}, density=density), debug_dir)
    say("export")
    exported = export.run(plan, out_dir, name, debug_dir)
    dieline, extras = {}, {}
    if made:                    # the placement line alone, as its own file, for cutting to shape
        dieline = export.write_files(made["dieline_plan"], out_dir, f"{name} dieline")
        for key, extra in made["extras"].items():     # parts sewn as their own file (plush ears)
            extras[key] = {"title": extra["title"],
                           "files": export.write_files(extra["plan"], out_dir, f"{name} {key}")}
        for text in made["problems"]:
            check["messages"].append({"level": "red", "text": text, "fixes": []})
            check["status"] = "red"

    for fmt, info in exported["files"].items():
        if not info["ok"]:
            check["messages"].append({
                "level": "red",
                "text": f"The {fmt.upper()} file didn't pass its safety check, so don't sew it. "
                        "Please try again.", "fixes": []})
            check["status"] = "red"

    w, h = check["width_mm"], check["height_mm"]
    return {
        "name": name,
        "stats": {
            "width_mm": round(w, 1), "height_mm": round(h, 1),
            "width_in": round(w / config.MM_PER_INCH, 2),
            "height_in": round(h / config.MM_PER_INCH, 2),
            "stitches": plan["stitch_count"],
            "colors": len({b["color"] for b in plan["blocks"] if b["color"] >= 0}),
            "size_in": round(max(w, h) / config.MM_PER_INCH, 2),
            "satin_columns": sum(1 for o in stitched["objects"] if o["kind"] == "satin"),
            "trims": plan["trims"],
            "sew_minutes": max(1, round(plan["stitch_count"] / rules["machine_spm"]
                                        + 0.5 * plan["color_changes"])),
        },
        "check": {"status": check["status"], "messages": check["messages"],
                  "hoop": check["hoop"]["name"] if check["hoop"] else None},
        "density": density,
        "inside": cleaned["inside"],
        "drop_text_mm": rules.get("drop_text_mm", 0.0),
        "project": None if not made else {
            "type": made["type"], "name": made["name"], "shape": made["shape"],
            "size_in": made["size_in"], "body_mm": made["body_mm"],
            "materials": made["materials"], "notes": made["notes"],
            "steps": ith.steps_of(plan),
            "pieces": [{"name": p["name"],
                        "ring": [round(v, 2) for xy in p["shape"].exterior.coords for v in xy]}
                       for p in made["pieces"]],
            "dieline": {fmt: {"ok": info["ok"], "path": info["path"]}
                        for fmt, info in dieline.items()},
            "extras": {key: {"title": e["title"],
                             "files": {fmt: {"ok": info["ok"], "path": info["path"]}
                                       for fmt, info in e["files"].items()}}
                       for key, e in extras.items()},
        },
        "fabric": {"id": fabric["id"], "name": fabric["name"],
                   "stabilizer": (made or {}).get("stabilizer") or fabric["stabilizer"],
                   "notes": fabric.get("notes", [])},
        "colors": [{"hex": hex_of(b["rgb"]), "label": b.get("label"),
                    "threads": None if b.get("label") else threads.match_all(b["rgb"]),
                    "stitches": sum(len(s) for s in b["segments"])} for b in plan["blocks"]],
        "areas": edits_stage.areas(classified["objects"], palette, classified.get("rows")),
        "ai_digitize": ai_info,
        "lettering": {"rows": len(classified.get("rows", [])),
                      "retyped": sum(1 for r in classified.get("rows", []) if r.get("text")),
                      "mode": letters,
                      "styles": [{"id": k, "name": v["name"]} for k, v in lettering.STYLES.items()]},
        "ref": round(ref, 3),
        "edits_applied": applied,
        "blocks": [{"hex": hex_of(b["rgb"]),
                    "segments": [[round(v, 2) for p in sg for v in p] for sg in b["segments"]]}
                   for b in plan["blocks"]],
        "bounds": [round(v, 2) for v in plan["bounds"]],
        "files": {fmt: {"ok": info["ok"], "path": info["path"], "stitches": info["stitches"]}
                  for fmt, info in exported["files"].items()},
        "preview": exported["preview"],
    }
