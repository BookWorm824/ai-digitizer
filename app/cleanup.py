"""The "Clean up" step: turn an uploaded picture into a few embroidery-ready versions to
choose from.

Always makes the local version (no AI). When the AI is available it also makes the three
ChatGPT versions, checks their spelling, and can apply a plain-words change to any version.
Everything is kept as files in the job's folder so the next step can stitch whichever
version was chosen.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from . import ai, config, settings
from .pipeline import PipelineError, analyze, clean, intake, optimize
from .pipeline.common import hex_of

LOCAL = {"id": "local", "name": "No AI", "blurb": "Simple clean-up on this PC", "kind": "local"}


def _chips(png: bytes, fit: tuple, rules: dict) -> list[str]:
    """The thread colors a version would end up with."""
    try:
        cleaned = clean.run(intake.run(png), fit[0], fit[1], rules)
        return [hex_of(c) for c in cleaned["palette"]]
    except PipelineError:
        return []


def _public(v: dict) -> dict:
    return {k: v[k] for k in ("id", "name", "blurb", "kind", "colors", "text_warning")}


def _load_ctx(job_dir: Path) -> dict:
    return json.loads((job_dir / "ctx.json").read_text(encoding="utf-8"))


def _save_ctx(job_dir: Path, ctx: dict) -> None:
    (job_dir / "ctx.json").write_text(json.dumps(ctx, indent=2), encoding="utf-8")


def _pick_default(versions: list[dict]) -> str:
    """Pressing Next without choosing must still give a good file: the first AI version whose
    text is right, otherwise the local clean-up."""
    for v in versions:
        if v["kind"] == "ai" and not v["text_warning"]:
            return v["id"]
    return "local"


def run_cleanup(data: bytes, job_dir: Path, *, fit: tuple, fabric_id: str, use_ai: bool = True,
                progress: Callable[[str], None] | None = None) -> dict:
    say = progress or (lambda _label: None)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "upload.bin").write_bytes(data)
    fabric = config.fabrics().get(fabric_id) or config.fabrics()["tshirt"]
    rules = config.rules_for(fabric["id"])

    say("Opening your picture…")
    img = intake.run(data)
    src_png = analyze.png_of(img["rgb"])
    (job_dir / "source.png").write_bytes(src_png)

    problem = None
    if not use_ai:
        problem = "skipped"
    else:
        try:
            ai.provider()
        except ai.AIUnavailable as e:
            use_ai, problem = False, e.reason
    if use_ai:
        say("ChatGPT is looking at your picture…")
    info = analyze.run(img, use_ai=use_ai)

    say("Cleaning up your art…")
    cleaned = clean.run(img, fit[0], fit[1], rules)
    (job_dir / "local.png").write_bytes(analyze.png_of(clean.render(cleaned)))
    local = dict(LOCAL, colors=[hex_of(c) for c in cleaned["palette"]], text_warning=None)

    ctx = {
        "size_in": round(max(fit) / config.MM_PER_INCH, 2),
        "fabric": f"a {fabric['name'].lower()} ({fabric['material']})",
        "fabric_id": fabric["id"],
        "fit": list(fit),
        "max_colors": int(rules["max_colors"]),
        "min_text_mm": rules["min_text_mm"],
        "text": info["text"],
        "attempts": {},
        "tweaks": 0,
    }

    problem = problem or info["ai_problem"]
    versions = []
    if use_ai and problem is None:
        say("ChatGPT is drawing 3 versions. This takes a minute or two…")
        made, problem = optimize.make_versions(src_png, ctx)
        for v in made:
            (job_dir / f"{v['id']}.png").write_bytes(v["png"])
            versions.append({"id": v["id"], "name": v["name"], "blurb": v["blurb"], "kind": "ai",
                             "colors": _chips(v["png"], fit, rules),
                             "text_warning": v["text_warning"]})
    versions.append(local)
    ctx["versions"] = versions
    _save_ctx(job_dir, ctx)

    tip = None
    if info["ai"] and info["ai"].get("fine_detail_risk") == "high" and len(versions) > 1:
        tip = ("This picture has a lot of fine detail that thread can't show. "
               "Simplified or Bold will sew best.")
    return {
        "versions": [_public(v) for v in versions],
        "default": _pick_default(versions),
        "ai": {"used": len(versions) > 1, "problem": problem,
               "note": ai.REASONS.get(problem) if problem else None},
        "text": info["text"],
        "tip": tip,
        "spend": ai.status(),
    }


def _using_api() -> bool:
    try:
        return ai.provider() == "api"
    except ai.AIUnavailable:
        return False


def _version_png(job_dir: Path, version_id: str) -> bytes:
    name = "source.png" if version_id == "local" else f"{version_id}.png"
    return (job_dir / name).read_bytes()


def run_tweak(job_dir: Path, version_id: str, change: str) -> dict:
    """Apply the user's plain-words change to a version. Adds a new version. Raises
    ai.AIUnavailable when the AI can't be used."""
    ctx = _load_ctx(job_dir)
    fit = tuple(ctx["fit"])
    rules = config.rules_for(ctx["fabric_id"])
    out = optimize.tweak(_version_png(job_dir, version_id), change, ctx, attempt=ctx["tweaks"])
    ctx["tweaks"] += 1
    vid = f"change{ctx['tweaks']}"
    (job_dir / f"{vid}.png").write_bytes(out["png"])
    short = change.strip()
    v = {"id": vid, "name": "Your change", "blurb": short[:60] + ("…" if len(short) > 60 else ""),
         "kind": "ai", "colors": _chips(out["png"], fit, rules),
         "text_warning": out["text_warning"]}
    ctx["versions"].insert(0, v)
    _save_ctx(job_dir, ctx)
    return _public(v)


def run_retry(job_dir: Path, version_id: str) -> dict:
    """Draw one of the three standard versions again (for "AI changed your text — fix it?")."""
    ctx = _load_ctx(job_dir)
    variant = next((v for v in optimize.VARIANTS if v["id"] == version_id), None)
    if variant is None:
        raise PipelineError("That version can't be drawn again. Try typing a change instead.")
    attempt = ctx["attempts"].get(version_id, 0) + 1
    png = ai.redraw((job_dir / "source.png").read_bytes(),
                    optimize.build_prompt(variant, ctx)
                    + "\nBe extra careful with the text: copy every letter exactly.",
                    attempt=attempt)
    ctx["attempts"][version_id] = attempt
    (job_dir / f"{version_id}.png").write_bytes(png)
    fit = tuple(ctx["fit"])
    for v in ctx["versions"]:
        if v["id"] == version_id:
            v["colors"] = _chips(png, fit, config.rules_for(ctx["fabric_id"]))
            v["text_warning"] = optimize.check_text(png, ctx.get("text") or [])
            _save_ctx(job_dir, ctx)
            return _public(v)
    raise PipelineError("That version can't be drawn again. Try typing a change instead.")


def picture_to_stitch(job_dir: Path, version_id: str,
                      progress: Callable[[str], None] | None = None) -> bytes:
    """The picture bytes to digitize for the chosen version. An AI version gets its final
    high-quality polish here (kept only if it still matches what the user approved)."""
    ctx = _load_ctx(job_dir)
    version = next((v for v in ctx["versions"] if v["id"] == version_id), None)
    if version is None or version["kind"] == "local":
        return (job_dir / "upload.bin").read_bytes()
    png = (job_dir / f"{version_id}.png").read_bytes()
    # Only the pay-as-you-go path draws low-quality previews, so only it needs a polish.
    if settings.load()["final_polish"] and _using_api():
        if progress:
            progress("Polishing your art…")
        polished = optimize.polish(png, ctx)
        if polished is not None:
            (job_dir / f"final_{version_id}.png").write_bytes(polished)
            return polished
    return png
