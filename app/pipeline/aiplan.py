"""ChatGPT as the digitizer's judgment.

Auto-digitizers fail at interpretation, not at stitch generation: they can't tell a tree from
a wall or shading from a border, so they sew everything the same way (reports/Embroidery
digitizing deep dive.md: Campbell likes automatic stitch *generation* but not automatic
*interpretation*). This module asks ChatGPT for that interpretation, the way a professional
looks at a design before digitizing it, and hands the answer to our own stitch engine:

  1. plan    ChatGPT sees the design with every part numbered, plus each part's measured size
             at the final sewn size, and says for each part what it is and how a professional
             would sew it (running stitch, bean, satin, fill, textured fill, or leave out),
             where the foliage is, and a few plain notes for the user.
  2. review  after stitching, ChatGPT sees the stitch preview with the same numbers and names
             any part that would sew badly, with a better stitch. The engine applies that once.

Every answer is checked: unknown part numbers and impossible choices are ignored, and if the
AI can't be used the engine simply carries on with its own rules.
"""

from __future__ import annotations

import hashlib
import json

import cv2
import numpy as np
from shapely.geometry import box as shp_box
from shapely.ops import unary_union

from .. import ai
from .common import hex_of, polygons_of

MAX_PARTS = 45           # parts numbered on the map (the rest follow the engine's own rules)
MIN_PART_MM2 = 2.0
MAP_PX = 1000

STITCHES = ["auto", "run", "bean", "satin", "fill", "texture", "omit"]
ROLES = ["lettering", "line_art", "outline", "foliage", "solid_area", "border", "detail",
         "shading", "background"]

_PART = {
    "type": "object",
    "properties": {
        "id": {"type": "integer"},
        "role": {"type": "string", "enum": ROLES},
        "stitch": {"type": "string", "enum": STITCHES},
        "angle": {"type": "integer"},
    },
    "required": ["id", "role", "stitch", "angle"],
    "additionalProperties": False,
}
_BOX = {
    "type": "object",
    "properties": {k: {"type": "integer"} for k in ("x0", "y0", "x1", "y1")},
    "required": ["x0", "y0", "x1", "y1"],
    "additionalProperties": False,
}
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "parts": {"type": "array", "items": _PART},
        "foliage": {"type": "array", "items": _BOX},
        "notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["parts", "foliage", "notes"],
    "additionalProperties": False,
}
REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["good", "fixable", "poor"]},
        "changes": {"type": "array", "items": _PART},
        "foliage": {"type": "array", "items": _BOX},
        "notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["verdict", "changes", "foliage", "notes"],
    "additionalProperties": False,
}

RULES_TEXT = """How professionals choose stitches (measured at the final sewn size, 40 wt thread):
- Lines under 0.8 mm wide: running stitch ("run", sewn twice along the line). Lines 0.8-1.6 mm:
  "bean" (a bold triple run). Do not jump to satin or fill too soon: thin line art stays run.
- Satin only for column shapes (ribbons, borders, letter strokes, roof edges) 1.6-7 mm wide.
- "fill" for flat areas wider than about 7-10 mm (walls, roofs, big shapes).
- "texture" for foliage, trees, bushes, grass, fur and clouds: a soft, layered, random fill
  with a jagged edge, never a hard flat silhouette and never leaf by leaf.
- "omit" for shading, gradients, specks, tiny repeated details that cannot sew at this size,
  and anything that is really background. Keep the parts that make the design recognisable.
- Lettering is handled by the app's own lettering engine: give letters role "lettering" and
  stitch "auto". Never omit lettering.
- A part made of thin strokes (stroke under 1.6 mm, usually with many holes) is line art even
  if it looks dark at a glance: use "run" or "bean", never "fill".
- Use "auto" when the app's own rules are fine. Angle is the fill or texture direction in
  degrees (0 = level, 90 = upright); professionals use about 15-20 degrees off level for
  fills (never 45) and 45-90 for foliage. Use -1 to let the app choose."""

PLAN_PROMPT = """You are an expert machine-embroidery digitizer planning how to sew this design.
It will be embroidered {size} on {fabric}. The picture shows the design with every part
outlined and numbered. Rulers along the edges give positions in percent of the picture.

Measured size of each numbered part at the final sewn size:
{table}

{rules}

Answer:
- parts: for EVERY numbered part, its role and how to sew it.
- foliage: boxes (in percent, from the rulers) around every tree, bush or other foliage
  area, even when it is drawn as many thin lines. An empty list if there is none.
- notes: up to 3 short, plain sentences for the customer about what you decided and why
  (no jargon, no numbers of parts)."""

REVIEW_PROMPT = """You are an expert machine-embroidery digitizer checking a stitch file before it
is sewn. The picture is a realistic preview of the stitches ({size} on {fabric}), with the
design's parts outlined and numbered in red. Rulers give positions in percent.

Current plan for each numbered part:
{table}

{rules}

Look for parts that will sew badly or look wrong: wobbly or messy line work that should be a
different stitch, solid blobs that should be texture or omitted, foliage drawn as a web of
lines, satin on shapes that are not columns, details too small to sew. Answer:
- verdict: good, fixable or poor.
- changes: only the parts that should change, each with its new role and stitch (and angle,
  or -1). Leave lettering alone.
- foliage: boxes (percent) around foliage that is still not soft and textured; empty if none.
- notes: up to 3 short, plain sentences for the customer about how the design will sew
  (no jargon, and never mention part numbers)."""


# --------------------------------------------------------------------------------- parts

def parts_of(vec: dict) -> list[dict]:
    """The design's numbered parts: its biggest separate shapes, largest first."""
    out = []
    for layer in vec["layers"]:
        for p in layer["polys"]:
            if p.area >= MIN_PART_MM2:
                out.append({"color": layer["color"], "rgb": layer["rgb"], "poly": p})
    out.sort(key=lambda d: -d["poly"].area)
    out = out[:MAX_PARTS]
    for k, d in enumerate(out, start=1):
        d["id"] = k
    return out


def _measure(poly) -> str:
    minx, miny, maxx, maxy = poly.bounds
    stroke = 2.0 * poly.area / max(poly.length, 1e-9)
    return (f"{maxx - minx:.1f} x {maxy - miny:.1f} mm, stroke about {stroke:.1f} mm wide, "
            f"area {poly.area:.0f} mm2, {len(poly.interiors)} holes")


def _table(parts: list[dict], plan: dict | None = None) -> str:
    rows = []
    for d in parts:
        line = f"{d['id']}: color {hex_of(d['rgb'])}, {_measure(d['poly'])}"
        if plan is not None:
            dv = plan.get(d["id"], {})
            line += f"; now {dv.get('role', '?')} / {dv.get('stitch', 'auto')}"
        rows.append(line)
    return "\n".join(rows)


def _frame(bounds):
    minx, miny, maxx, maxy = bounds
    span = max(maxx - minx, maxy - miny, 1e-6)
    pad = 0.06 * span
    return minx - pad, miny - pad, maxx + pad, maxy + pad


def _to_px(frame, size):
    fx0, fy0, fx1, fy1 = frame
    s = size / max(fx1 - fx0, fy1 - fy0)

    def f(coords):
        a = np.asarray(coords)[:, :2]
        return np.round(np.column_stack([(a[:, 0] - fx0) * s, (a[:, 1] - fy0) * s])).astype(np.int32)
    w = int(round((fx1 - fx0) * s))
    h = int(round((fy1 - fy0) * s))
    return f, w, h


def _rulers(img: np.ndarray) -> None:
    h, w = img.shape[:2]
    for k in range(0, 101, 10):
        x, y = int(k / 100 * (w - 1)), int(k / 100 * (h - 1))
        cv2.line(img, (x, 0), (x, h - 1), (225, 225, 225), 1)
        cv2.line(img, (0, y), (w - 1, y), (225, 225, 225), 1)
        cv2.putText(img, str(k), (min(x + 2, w - 26), 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (120, 120, 120), 1, cv2.LINE_AA)
        cv2.putText(img, str(k), (2, min(max(y - 2, 26), h - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (120, 120, 120), 1, cv2.LINE_AA)


def _labels(img, parts, to_px, color=(0, 0, 230)):
    for d in parts:
        for p in polygons_of(d["poly"]):
            cv2.polylines(img, [to_px(p.exterior.coords)], True, color, 1, cv2.LINE_AA)
    for d in parts:
        pt = d["poly"].representative_point()
        x, y = to_px([(pt.x, pt.y)])[0]
        label = str(d["id"])
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x - 2, y - th - 3), (x + tw + 2, y + 3), (255, 255, 255), -1)
        cv2.rectangle(img, (x - 2, y - th - 3), (x + tw + 2, y + 3), color, 1)
        cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)


def _png(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bgr)
    return buf.tobytes()


def plan_map(vec: dict, parts: list[dict]) -> tuple[bytes, tuple]:
    """The design drawn in its own colors with every part numbered. Returns (PNG, frame)."""
    frame = _frame(vec["bounds"])
    to_px, w, h = _to_px(frame, MAP_PX)
    img = np.full((h, w, 3), 255, np.uint8)
    _rulers(img)
    for layer in vec["layers"]:
        bgr = tuple(int(v) for v in layer["rgb"][::-1])
        for p in layer["polys"]:
            rings = [to_px(p.exterior.coords)] + [to_px(r.coords) for r in p.interiors]
            cv2.fillPoly(img, rings, bgr, lineType=cv2.LINE_AA)
    _labels(img, parts, to_px)
    return _png(img), frame


def review_map(sewing: dict, vec: dict, parts: list[dict]) -> bytes:
    """The stitches themselves (thread-thick lines, in sewing order) with the same numbered
    parts on top, in the same frame as the plan map, for the review."""
    frame = _frame(vec["bounds"])
    to_px, w, h = _to_px(frame, MAP_PX)
    img = np.full((h, w, 3), 238, np.uint8)
    _rulers(img)
    s = MAP_PX / max(frame[2] - frame[0], frame[3] - frame[1])
    thick = max(1, int(round(0.4 * s)))
    for block in sewing["blocks"]:
        rgb = np.array(block["rgb"], float)
        base = tuple(int(v) for v in rgb[::-1])
        dark = tuple(int(v) for v in (rgb * 0.7)[::-1])
        for seg in block["segments"]:
            px = to_px(seg)
            cv2.polylines(img, [px], False, dark, thick + 1, cv2.LINE_AA)
            cv2.polylines(img, [px], False, base, thick, cv2.LINE_AA)
    _labels(img, parts, to_px)
    return _png(img)


# ------------------------------------------------------------------------------ answers

def _clean(answer: dict, parts: list[dict], key: str = "parts") -> dict:
    """Part id -> {role, stitch, angle}, keeping only sensible entries."""
    ids = {d["id"] for d in parts}
    out = {}
    for e in answer.get(key, []) or []:
        try:
            pid = int(e["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if pid not in ids:
            continue
        stitch = e.get("stitch") if e.get("stitch") in STITCHES else "auto"
        role = e.get("role") if e.get("role") in ROLES else "detail"
        angle = e.get("angle", -1)
        angle = int(angle) % 180 if isinstance(angle, (int, float)) and angle >= 0 else -1
        out[pid] = {"role": role, "stitch": stitch, "angle": angle}
    return out


def _boxes(answer: dict, frame) -> list:
    fx0, fy0, fx1, fy1 = frame
    out = []
    for b in answer.get("foliage", []) or []:
        try:
            x0, y0, x1, y1 = (max(0, min(100, int(b[k]))) for k in ("x0", "y0", "x1", "y1"))
        except (KeyError, TypeError, ValueError):
            continue
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        out.append(shp_box(fx0 + (fx1 - fx0) * x0 / 100, fy0 + (fy1 - fy0) * y0 / 100,
                           fx0 + (fx1 - fx0) * x1 / 100, fy0 + (fy1 - fy0) * y1 / 100))
    return out


def _notes(answer: dict) -> list[str]:
    return [str(n).strip()[:200] for n in (answer.get("notes") or []) if str(n).strip()][:3]


def make_plan(vec: dict, size_text: str, fabric_text: str) -> dict | None:
    """Ask ChatGPT how to sew each part. Returns the plan for classify, or None if there is
    nothing to plan. Raises ai.AIUnavailable when the AI can't be used."""
    parts = parts_of(vec)
    if not parts:
        return None
    png, frame = plan_map(vec, parts)
    prompt = PLAN_PROMPT.format(size=size_text, fabric=fabric_text, table=_table(parts),
                                rules=RULES_TEXT)
    answer = ai.ask_json("plan", png, prompt, "digitizing_plan", PLAN_SCHEMA)
    return {"parts": parts, "frame": frame, "by_id": _clean(answer, parts),
            "foliage": _boxes(answer, frame), "notes": _notes(answer), "png": png}


def review(plan: dict, vec: dict, sewing: dict, size_text: str, fabric_text: str) -> dict:
    """Ask ChatGPT to check the stitched result (`sewing` = the sequenced thread plan).
    Returns {changed, verdict, notes}; the plan is updated in place. Raises AIUnavailable."""
    parts = plan["parts"]
    png = review_map(sewing, vec, parts)
    # One review per plan: small tweaks afterwards (density, colors) reuse it, so they stay
    # quick and don't ask ChatGPT again.
    key = ai._cache_key("review_for_plan", hashlib.sha256(plan["png"]).hexdigest(), size_text,
                        fabric_text)
    path = ai._cache_path(key, "json")
    answer = None
    if path.exists():
        try:
            answer = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            answer = None
    if answer is None:
        prompt = REVIEW_PROMPT.format(size=size_text, fabric=fabric_text,
                                      table=_table(parts, plan["by_id"]), rules=RULES_TEXT)
        answer = ai.ask_json("review", png, prompt, "stitch_review", REVIEW_SCHEMA)
        path.write_text(json.dumps(answer), encoding="utf-8")
    changes = _clean(answer, parts, "changes")
    changed = False
    for pid, dv in changes.items():
        if plan["by_id"].get(pid, {}).get("role") == "lettering":
            continue
        if plan["by_id"].get(pid) != dv:
            plan["by_id"][pid] = dv
            changed = True
    # a foliage box from the review counts only if it covers somewhere the plan didn't
    for bx in _boxes(answer, plan["frame"]):
        known = [b for b in plan["foliage"] if b.intersects(bx)]
        if not known or bx.difference(unary_union(known)).area > 0.5 * bx.area:
            plan["foliage"] = list(plan["foliage"]) + [bx]
            changed = True
    return {"changed": changed, "verdict": answer.get("verdict", "fixable"),
            "notes": _notes(answer), "png": png}


def directive_for(plan: dict | None, color: int, poly):
    """The planned directive for a shape (matched to a numbered part of the same color), or
    None if the plan has nothing to say about it."""
    if not plan:
        return None
    pt = poly.representative_point()
    best = None
    for d in plan["parts"]:
        if d["color"] != color:
            continue
        if d["poly"].buffer(0.3).contains(pt):
            dv = plan["by_id"].get(d["id"])
            if dv is not None and (best is None or d["poly"].area < best[0]):
                best = (d["poly"].area, dv)
    return None if best is None else best[1]

