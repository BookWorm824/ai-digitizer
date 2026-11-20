"""Stage 3 — Optimize.

ChatGPT redraws the picture as embroidery-ready art for the chosen size and fabric, in three
versions that differ only in their dials:

  Faithful    closest to the original
  Simplified  fewer colors and details
  Bold        thicker outlines, patch style

Also here: the plain-words "tweak" (a follow-up edit on a chosen version), the spelling check
that runs after every redraw, and the final high-quality polish of the version the user picks.
Whenever the AI can't be used the caller gets nothing back and carries on with the local
clean-up.
"""

from __future__ import annotations

import io
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from .. import ai

# The dials follow what professional digitizers ask of artwork before they stitch it
# (reports/Embroidery digitizing rules.md): lines at least 1.2 mm (narrower can't be satin),
# details at least 1.5-2 mm (Melco: anything under 2 mm needs changing), gaps at least 1 mm,
# busy line work turned into simple shapes, and small lettering without hairline serifs.
VARIANTS = [
    {"id": "faithful", "name": "Faithful", "blurb": "Closest to your picture",
     "colors": lambda n: n, "line_mm": 1.2, "detail_mm": 1.5,
     "extra": "Stay as close to the original as these rules allow."},
    {"id": "simplified", "name": "Simplified", "blurb": "Fewer colors and details",
     "colors": lambda n: max(2, min(n - 2, 4)), "line_mm": 1.5, "detail_mm": 2.0,
     "extra": "Simplify: merge small shapes together, leave out minor details, and use bold, "
              "even-stroke letters without thin serifs."},
    {"id": "bold", "name": "Bold", "blurb": "Thick outlines, like a patch",
     "colors": lambda n: max(2, min(n - 1, 5)), "line_mm": 2.0, "detail_mm": 2.0,
     "extra": "Patch style: a thick dark outline around every shape and around the whole "
              "design, and bold, even-stroke letters without thin serifs."},
]

RULES_TEXT = ("Rules: flat solid colors only, no gradients, shadows, textures or photo detail.\n"
              "Maximum {n_colors} colors. Every line and outline at least {min_line_mm:g} mm "
              "thick at final size; leave out outlines that would be thinner.\n"
              "Remove any detail smaller than {min_detail_mm:g} mm at final size. Turn fine "
              "hatching, crosshatching, leaf and grass scribbles and tiny repeated details "
              "(window panes, bricks) into a few simple solid shapes.\n"
              "Leave at least 1 mm of space between separate shapes and inside letters.\n"
              "{text_rule}\n"
              "Plain white background. Keep the subject, pose and layout recognizable. Center it.\n"
              "{size_rule}")


def _size_rule(size_in: float) -> str:
    """How much detail survives at this size (Erich Campbell's left-chest / cap / jacket-back
    versions; reports/Digitizing training and AI tools.md)."""
    if size_in <= 4.5:
        return ("This is a small embroidery (left chest or cap): draw buildings as their outline "
                "and a few big blocks, with windows as rows of plain solid rectangles, no bricks, "
                "no hatching, no texture lines. Draw trees and bushes as one to three simple "
                "solid rounded masses, never leaves. Drop shading, shadows and thin decorative "
                "lines. Make the most important element (the wordmark) big and bold.")
    if size_in <= 7.5:
        return ("Medium-size embroidery: simplify buildings to clear outlines, solid windows and "
                "a few lines; trees as a few solid masses with a little inner detail; no shading.")
    return ("Large embroidery (jacket back): keep the main details, but still no hairlines, "
            "hatching or gradients; trees as several solid masses.")


def _text_rule(text: list[str], min_text_mm: float) -> str:
    if not text:
        return "Do not add any text or lettering."
    quoted = " / ".join(f'"{t}"' for t in text)
    return (f"Keep text exactly as written: {quoted}, at least {min_text_mm:g} mm tall. "
            "Do not change, add or drop any letter.")


def build_prompt(variant: dict, ctx: dict) -> str:
    """The redraw prompt from BRIEF.md, filled in from the first screen's answers.
    ctx: size_in, fabric, max_colors, text (list of exact strings), min_text_mm."""
    rules = RULES_TEXT.format(
        n_colors=variant["colors"](ctx["max_colors"]),
        min_line_mm=variant["line_mm"], min_detail_mm=variant["detail_mm"],
        text_rule=_text_rule(ctx.get("text") or [], ctx.get("min_text_mm", 5.0)),
        size_rule=_size_rule(float(ctx.get("size_in", 4.0))))
    return (f"Redraw this image as embroidery-ready artwork for a {ctx['size_in']:g}-inch "
            f"design on {ctx['fabric']}.\n{rules}\n{variant['extra']}")


def tweak_prompt(change: str, ctx: dict) -> str:
    change = re.sub(r"\s+", " ", change).strip()[:300]
    base = VARIANTS[0]
    rules = RULES_TEXT.format(
        n_colors=ctx["max_colors"], min_line_mm=base["line_mm"], min_detail_mm=base["detail_mm"],
        text_rule=_text_rule(ctx.get("text") or [], ctx.get("min_text_mm", 5.0)),
        size_rule=_size_rule(float(ctx.get("size_in", 4.0))))
    return (f"This is embroidery-ready artwork for a {ctx['size_in']:g}-inch design on "
            f"{ctx['fabric']}. Make this change and nothing else: {change}\n"
            f"Keep everything else exactly as it is.\n{rules}")


def polish_prompt(ctx: dict) -> str:
    return ("Redraw this exact embroidery artwork cleanly at high quality. Do not change the "
            "design: same shapes, same colors, same layout, same text. Only make edges crisp "
            "and colors perfectly flat.\n"
            + _text_rule(ctx.get("text") or [], ctx.get("min_text_mm", 5.0))
            + "\nPlain white background.")


# ------------------------------------------------------------------------- spelling check

def _words(strings: list[str]) -> list[str]:
    return sorted(w for s in strings for w in re.sub(r"\s+", " ", s).strip().casefold().split(" ")
                  if w)


def check_text(png: bytes, expected: list[str]) -> dict | None:
    """Compare the text in a redrawn picture with the original's. Returns None when it
    matches (or can't be checked), else what was expected and what was found."""
    if not expected:
        return None
    try:
        found = ai.read_text(png)
    except ai.AIUnavailable:
        return None
    if _words(found) == _words(expected):
        return None
    return {"expected": expected, "found": found}


# ----------------------------------------------------------------------------------- redraws

def make_versions(src_png: bytes, ctx: dict, attempt: int = 0) -> tuple[list[dict], str | None]:
    """The three redrawn versions. Returns (versions, problem). Each version is
    {id, name, blurb, png, text_warning}. `problem` is an ai.REASONS key when one or more
    versions could not be made."""
    def one(variant):
        png = ai.redraw(src_png, build_prompt(variant, ctx), attempt=attempt)
        return {"id": variant["id"], "name": variant["name"], "blurb": variant["blurb"],
                "png": png, "text_warning": check_text(png, ctx.get("text") or [])}

    versions, problem = [], None
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(one, v) for v in VARIANTS]
        for f in futures:
            try:
                versions.append(f.result())
            except ai.AIUnavailable as e:
                problem = problem or e.reason
    return versions, problem


def tweak(version_png: bytes, change: str, ctx: dict, attempt: int = 0) -> dict:
    """The user's plain-words change, applied to the version they chose. Raises AIUnavailable."""
    png = ai.redraw(version_png, tweak_prompt(change, ctx), attempt=attempt)
    return {"png": png, "text_warning": check_text(png, ctx.get("text") or [])}


def _shape_mask(png: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(png)).convert("RGB").resize((128, 128), Image.BILINEAR)
    a = np.asarray(img).astype(np.int16)
    return (np.abs(a - 255).max(axis=2) > 40)


def similar(a_png: bytes, b_png: bytes, threshold: float = 0.75) -> bool:
    """Do two pictures show roughly the same design? (overlap of their non-white areas)"""
    a, b = _shape_mask(a_png), _shape_mask(b_png)
    union = (a | b).sum()
    return bool(union == 0 or (a & b).sum() / union >= threshold)


def polish(version_png: bytes, ctx: dict) -> bytes | None:
    """Final high-quality redraw of the chosen version. Returns None (keep the preview) if
    the AI can't be used, or if the polished picture drifted: different layout or text."""
    try:
        png = ai.redraw(version_png, polish_prompt(ctx), final=True)
    except ai.AIUnavailable:
        return None
    if not similar(version_png, png):
        return None
    if check_text(png, ctx.get("text") or []) is not None:
        return None
    return png
