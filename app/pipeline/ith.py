"""In-the-hoop (ITH) projects: the machine builds a standalone item inside the hoop.

The user's digitized art becomes the decoration; this module wraps the construction steps
around it. Every project follows the same pattern (BRIEF.md, "How every ITH file works"):

    placement line -> tackdown -> decoration -> (backing) -> finishing outline

The key rule: machines only reliably stop at a color change, so EVERY construction step is its
own color block in the file, even when it could use the same thread. Each block carries the
plain-words instruction for what to do when the machine stops.

Project types built so far, in the brief's order: felt patch, snap-tab key fob, coaster,
ornament. Plush, zipper bags, pillows and freestanding lace come later, one at a time.
"""

from __future__ import annotations

import math

import numpy as np
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import substring, unary_union

from . import sequence, stitch
from .common import polygons_of

MM_PER_INCH = 25.4
SEAM_ALLOWANCE_MM = 6.0      # 1/4 inch outside the final seam (stitch-and-turn)
KEEP_OUT_MM = 6.0            # decoration stays this far inside the final seam
BORDER_MM = 3.2              # width of the satin finishing border
TAB_WIDTH_MM = 19.0          # snap tab: about 3/4 inch
TAB_LENGTH_MM = 45.0

# Preview colors for construction steps (the thread itself can be anything).
STEP_RGB = {"placement": (120, 150, 190), "tackdown": (150, 120, 190), "final": (60, 60, 60)}

PROJECTS = {
    "feltie": {
        "name": "Felt patch", "fabric": "felt", "margin": 5.0, "sizes_in": [2.5, 3.0, 3.5],
        "default_in": 3.0,
        "materials": ["Felt, a piece a little bigger than the patch",
                      "Tearaway stabilizer, hooped on its own",
                      "Small sharp scissors"],
    },
    "keyfob": {
        "name": "Snap-tab key fob", "fabric": "vinyl", "margin": 5.0, "sizes_in": [2.0, 2.5, 3.0],
        "default_in": 2.5,
        "materials": ["Two pieces of vinyl, faux leather or felt (front and back)",
                      "Tearaway stabilizer, hooped on its own", "Tape",
                      "A snap and snap pliers, and a key ring"],
    },
    "coaster": {
        "name": "Coaster", "fabric": "cotton", "margin": KEEP_OUT_MM, "sizes_in": [3.5, 4.0, 4.5],
        "default_in": 4.0,
        "materials": ["Two pieces of cotton fabric (front and back)",
                      "Tearaway stabilizer, hooped on its own", "Tape",
                      "A needle and thread to close the gap"],
    },
    "ornament": {
        "name": "Ornament", "fabric": "cotton", "margin": KEEP_OUT_MM, "sizes_in": [2.5, 3.0, 3.5],
        "default_in": 3.0,       # with its seam allowance this still fits a 4x4 hoop
        "materials": ["Two pieces of fabric (front and back)", "A short loop of ribbon",
                      "Tearaway stabilizer, hooped on its own", "Tape", "Stuffing, if you like",
                      "A needle and thread to close the gap"],
    },
}
SHAPES = [{"id": "circle", "name": "Circle"}, {"id": "square", "name": "Rounded square"},
          {"id": "follow", "name": "Follow my design"}]
TURNED = ("coaster", "ornament")      # stitch-and-turn projects


def normalize(project: dict | None) -> dict | None:
    """A clean project spec, or None if this isn't a project."""
    if not project or project.get("type") not in PROJECTS:
        return None
    spec = PROJECTS[project["type"]]
    shape = project.get("shape") if project.get("shape") in {s["id"] for s in SHAPES} else "circle"
    try:
        size = float(project.get("size_in") or spec["default_in"])
    except (TypeError, ValueError):
        size = spec["default_in"]
    return {"type": project["type"], "shape": shape, "size_in": min(7.0, max(1.5, size))}


def decoration_fit(project: dict) -> tuple[float, float]:
    """The box the art is digitized into, so it sits inside the project with room to spare."""
    spec = PROJECTS[project["type"]]
    inner = project["size_in"] * MM_PER_INCH - 2 * spec["margin"]
    if project["shape"] == "circle":
        inner /= math.sqrt(2)             # a square that fits inside the circle
    elif project["shape"] == "square":
        inner *= 0.94                     # stay off the rounded corners
    return max(15.0, inner), max(15.0, inner)


def body_shape(project: dict, decoration) -> Polygon:
    """The finished outline of the item (without tabs or seam allowance), centred on (0, 0)."""
    spec = PROJECTS[project["type"]]
    size = project["size_in"] * MM_PER_INCH
    if project["shape"] == "square":
        r = size * 0.12
        return box(-size / 2, -size / 2, size / 2, size / 2).buffer(-r).buffer(r, quad_segs=12)
    if project["shape"] == "follow" and decoration is not None and not decoration.is_empty:
        grown = decoration.buffer(spec["margin"] + 4.0, quad_segs=12).buffer(-4.0, quad_segs=12)
        parts = polygons_of(grown)
        if len(parts) != 1:
            parts = polygons_of(decoration.convex_hull.buffer(spec["margin"], quad_segs=12))
        return Polygon(max(parts, key=lambda p: p.area).exterior).simplify(0.2)
    return Point(0, 0).buffer(size / 2, quad_segs=32)


def _ring(shape: Polygon, start=None) -> LineString:
    """A shape's outline as a closed line, starting at the point nearest `start`."""
    line = LineString(shape.exterior.coords)
    if start is None:
        return line
    d = line.project(Point(start))
    pts = list(substring(line, d, line.length).coords) + list(substring(line, 0, d).coords)[1:]
    return LineString(pts)


def _run(shape: Polygon, rules: dict, triple: bool = False, start=None) -> list:
    pts = stitch.run_points(_ring(shape, start), rules)
    return stitch.bean(pts) if triple else pts


def _satin_border(shape: Polygon, rules: dict) -> list:
    line = _ring(shape)
    n = max(16, int(line.length / 0.25))
    center = np.array([line.interpolate(line.length * k / n).coords[0] for k in range(n + 1)])
    half = np.full(len(center), BORDER_MM / 2)
    segs = stitch.stitch_satin({"center": center, "half": half}, dict(rules, pull_comp_mm=0.0))
    return segs[0] if segs else []


def _seam_with_gap(shape: Polygon, rules: dict) -> tuple[list, float]:
    """The final seam of a stitch-and-turn item: triple run all round, except for a turning
    gap at the bottom. Returns (needle points, gap length in mm)."""
    line = LineString(shape.exterior.coords)
    minx, _, maxx, maxy = shape.bounds
    gap = min(75.0, max(38.0, line.length * 0.2))      # 1.5 to 3 inches, scaled to the item
    mid = line.project(Point((minx + maxx) / 2, maxy))  # bottom centre
    a, b = (mid + gap / 2) % line.length, (mid - gap / 2) % line.length
    if a < b:
        path = list(substring(line, a, b).coords)
    else:
        path = list(substring(line, a, line.length).coords) + list(substring(line, 0, b).coords)[1:]
    return stitch.bean(stitch.run_points(LineString(path), rules)), gap


def _block(n: int, kind: str, title: str, pts: list, rules: dict, before: str, after: str,
           thread: str) -> dict:
    seg = sequence._clean_segment(pts, rules["min_stitch_mm"], rules["max_stitch_mm"])
    return {"color": -10 - n, "rgb": STEP_RGB[kind], "segments": [sequence._tie(seg, rules["tie_mm"])],
            "label": f"{title} ({thread})",
            "step": {"title": title, "before": before, "after": after, "thread": thread}}


def build(plan: dict, decoration, project: dict, rules: dict) -> tuple[dict, dict]:
    """Wrap the construction steps around a finished decoration plan.
    Returns (the whole project's plan, details for the instructions and templates)."""
    kind = project["type"]
    spec = PROJECTS[kind]
    body = body_shape(project, decoration)
    top = (0.0, body.bounds[1])

    if kind == "keyfob":
        cx = (body.bounds[0] + body.bounds[2]) / 2
        tab = box(cx - TAB_WIDTH_MM / 2, body.bounds[1] - TAB_LENGTH_MM,
                  cx + TAB_WIDTH_MM / 2, body.bounds[1] + body.area ** 0.5 * 0.25)
        tab = tab.buffer(-4, quad_segs=8).buffer(4, quad_segs=8)
        outline = Polygon(unary_union([body, tab]).buffer(2).buffer(-2).exterior)
    else:
        outline = body

    deco = []
    for b in plan["blocks"]:
        deco.append(dict(b, step={"title": "Decoration", "before": "",
                                  "after": "", "thread": "the design's thread colors"}))
    pre, post, pieces, notes = [], [], [], []

    if kind == "feltie":
        tack = outline.buffer(-0.7)
        pre.append(_block(0, "placement", "Placement line", _run(outline, rules, start=top), rules,
                          "Hoop tearaway stabilizer on its own, with nothing on it.",
                          "Lay the felt over the stitched outline so it covers all of it. "
                          "Hold it flat with tape at the corners.", "any color"))
        pre.append(_block(1, "tackdown", "Tackdown", _run(tack, rules, start=top), rules, "",
                          "Trim the felt close to the stitching, all the way round. "
                          "Don't cut the stitches.", "any color"))
        post.append(_block(2, "final", "Satin border", _satin_border(outline.buffer(-0.3), rules),
                           rules, "", "Take it out of the hoop and gently tear the stabilizer away.",
                           "your border color"))
        pieces.append({"name": "Felt", "shape": outline.buffer(8)})
        notes.append("The border hides the cut edge of the felt, so trim neatly after the tackdown.")

    elif kind == "keyfob":
        tack = outline.buffer(-1.0)
        pre.append(_block(0, "placement", "Placement line", _run(outline, rules, start=top), rules,
                          "Hoop tearaway stabilizer on its own, with nothing on it.",
                          "Lay the front vinyl over the outline, right side up, covering all of "
                          "it. Tape the corners.", "any color"))
        pre.append(_block(1, "tackdown", "Tackdown", _run(tack, rules, start=top), rules, "",
                          "Leave the vinyl as it is. Don't trim yet.", "any color"))
        post.append(_block(2, "final", "Final outline", _run(outline, rules, triple=True, start=top),
                           rules,
                           "Take the hoop off the machine, but don't unhoop. Tape the backing "
                           "vinyl to the BACK of the hoop, right side facing out, covering the "
                           "whole outline. Put the hoop back on.",
                           "Unhoop and tear the stabilizer away. Cut about 3 mm outside the "
                           "stitching. Fold the tab over a key ring and set a snap through both "
                           "layers.", "a color you like on both sides"))
        pieces.append({"name": "Vinyl, front", "shape": outline.buffer(8)})
        pieces.append({"name": "Vinyl, back", "shape": outline.buffer(8)})
        notes.append("Use the same thread in the bobbin for the final outline: it shows on the back.")

    else:   # coaster, ornament: stitch and turn
        cut = outline.buffer(SEAM_ALLOWANCE_MM, quad_segs=12)
        tack = cut.buffer(-1.5)
        seam, gap = _seam_with_gap(outline, rules)
        pre.append(_block(0, "placement", "Placement line", _run(cut, rules, start=top), rules,
                          "Hoop tearaway stabilizer on its own, with nothing on it.",
                          "Lay the front fabric over the outline, right side up, covering all "
                          "of it.", "any color"))
        pre.append(_block(1, "tackdown", "Tackdown", _run(tack, rules, start=top), rules, "",
                          "Leave the fabric as it is.", "any color"))
        loop = ("Tape the ribbon loop at the top, with the loop pointing in toward the middle "
                "and its ends sticking out past the outline. Then lay" if kind == "ornament"
                else "Lay")
        post.append(_block(2, "final", "Final seam", seam, rules,
                           f"{loop} the back fabric on top, right side DOWN, covering "
                           "everything. Tape the corners.",
                           "Unhoop and tear the stabilizer away. Trim along the outer line "
                           "(a 6 mm seam). Clip the curves, turn it right side out through the "
                           f"gap at the bottom (about {gap / MM_PER_INCH:.1f} in), push out the "
                           "edges, press, "
                           + ("stuff it lightly if you like, " if kind == "ornament" else "")
                           + "and hand-sew the gap closed.", "a color that matches the fabric"))
        pieces.append({"name": "Fabric, front", "shape": cut.buffer(6)})
        pieces.append({"name": "Fabric, back", "shape": cut.buffer(6)})
        notes.append("The outer stitched line is your cutting line. The design stays "
                     f"{KEEP_OUT_MM:g} mm inside the seam so it isn't caught when you turn it.")

    blocks = pre + deco + post
    xs = [p[0] for b in blocks for sg in b["segments"] for p in sg]
    ys = [p[1] for b in blocks for sg in b["segments"] for p in sg]
    whole = {"blocks": blocks,
             "stitch_count": sum(len(sg) for b in blocks for sg in b["segments"]),
             "trims": sum(len(b["segments"]) - 1 for b in blocks),
             "color_changes": len(blocks) - 1,
             "bounds": (min(xs), min(ys), max(xs), max(ys))}
    line = [p for sg in pre[0]["segments"] for p in sg]
    dieline = {"blocks": pre[:1], "stitch_count": len(line), "trims": 0, "color_changes": 0,
               "bounds": (min(p[0] for p in line), min(p[1] for p in line),
                          max(p[0] for p in line), max(p[1] for p in line))}
    info = {"type": kind, "name": spec["name"], "shape": project["shape"],
            "size_in": project["size_in"], "materials": spec["materials"], "notes": notes,
            "pieces": pieces, "dieline_plan": dieline,
            "body_mm": [round(body.bounds[2] - body.bounds[0], 1),
                        round(body.bounds[3] - body.bounds[1], 1)]}
    return whole, info


def steps_of(plan: dict) -> list[dict]:
    """The numbered steps of a project, one per color block, for the instructions."""
    out, colors = [], 0
    for n, b in enumerate(plan["blocks"]):
        s = dict(b.get("step") or {})
        if s.get("title") == "Decoration":
            colors += 1
            s["title"] = f"Decoration, color {colors}"
        out.append({"number": n + 1, "title": s.get("title", "Stitch"),
                    "before": s.get("before", ""), "after": s.get("after", ""),
                    "thread": s.get("thread", "")})
    return out


def options() -> dict:
    return {"types": [{"id": k, "name": v["name"], "sizes_in": v["sizes_in"],
                       "default_in": v["default_in"]} for k, v in PROJECTS.items()],
            "shapes": SHAPES}
