"""In-the-hoop (ITH) projects: the machine builds a standalone item inside the hoop.

The user's digitized art becomes the decoration; this module wraps the construction steps
around it. Every project follows the same pattern (BRIEF.md, "How every ITH file works"):

    placement line -> tackdown -> decoration -> (backing) -> finishing outline

The key rule: machines only reliably stop at a color change, so EVERY construction step is its
own color block in the file, even when it could use the same thread. Each block carries the
plain-words instruction for what to do when the machine stops.

Project types, in the brief's build order: felt patch, snap-tab key fob, coaster, ornament,
plush toy, lined zipper bag, pillow (and pillow front panel), freestanding lace.
"""

from __future__ import annotations

import math

import cv2
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
ZIP_FROM_TOP_MM = 22.0       # zipper bag: how far below the top edge the zipper sits
LACE_SPACING_MM = 0.9        # freestanding lace: row spacing of each base layer
LACE_MAX_HOLE_MM = 1.8       # the biggest opening the lace base may have
PANEL_MAX_MM = 184.0         # pillow front panel: the biggest design an 8x8 hoop takes

# Preview colors for construction steps (the thread itself can be anything).
STEP_RGB = {"placement": (120, 150, 190), "tackdown": (150, 120, 190), "final": (60, 60, 60),
            "lace": (235, 225, 200)}

HOOPED = "Tearaway stabilizer, hooped on its own"
PROJECTS = {
    "feltie": {
        "name": "Felt patch", "fabric": "felt", "margin": 5.0, "sizes_in": [2.5, 3.0, 3.5],
        "default_in": 3.0, "shapes": ["circle", "square", "follow"],
        "materials": ["Felt, a piece a little bigger than the patch", HOOPED,
                      "Small sharp scissors"],
    },
    "keyfob": {
        "name": "Snap-tab key fob", "fabric": "vinyl", "margin": 5.0, "sizes_in": [2.0, 2.5, 3.0],
        "default_in": 2.5, "shapes": ["circle", "square", "follow"],
        "materials": ["Two pieces of vinyl, faux leather or felt (front and back)", HOOPED,
                      "Tape", "A snap and snap pliers, and a key ring"],
    },
    "coaster": {
        "name": "Coaster", "fabric": "cotton", "margin": KEEP_OUT_MM, "sizes_in": [3.5, 4.0, 4.5],
        "default_in": 4.0, "shapes": ["circle", "square", "follow"],
        "materials": ["Two pieces of cotton fabric (front and back)", HOOPED, "Tape",
                      "A needle and thread to close the gap"],
    },
    "ornament": {
        "name": "Ornament", "fabric": "cotton", "margin": KEEP_OUT_MM, "sizes_in": [2.5, 3.0, 3.5],
        "default_in": 3.0,       # with its seam allowance this still fits a 4x4 hoop
        "shapes": ["circle", "square", "follow"],
        "materials": ["Two pieces of fabric (front and back)", "A short loop of ribbon", HOOPED,
                      "Tape", "Stuffing, if you like", "A needle and thread to close the gap"],
    },
    "plush": {
        "name": "Plush toy", "fabric": "plushfabric", "margin": KEEP_OUT_MM,
        "sizes_in": [3.5, 4.0, 4.5], "default_in": 4.0, "shapes": ["follow", "circle", "square"],
        "materials": ["Two pieces of fleece, flannel or minky (front and back)",
                      "A scrap of the same fabric for the ears", HOOPED, "Tape", "Stuffing",
                      "Water-soluble topping if you use minky",
                      "A needle and thread to close the gap"],
    },
    "zipbag": {
        "name": "Lined zipper bag", "fabric": "cotton", "margin": 10.0, "sizes_in": [5.0, 6.0, 7.0],
        "default_in": 6.0, "shapes": ["rect"],
        "stabilizer": "No-show mesh cutaway, hooped on its own (not tearaway)",
        "materials": ["Outer fabric: one top piece, one bottom piece, one back piece",
                      "Lining fabric: the same three pieces",
                      "A nylon-coil zipper at least 2 in longer than the bag is wide",
                      "No-show mesh cutaway stabilizer, hooped on its own", "Tape"],
    },
    "pillow": {
        "name": "Small pillow", "fabric": "cotton", "margin": KEEP_OUT_MM + 6.0,
        "sizes_in": [5.0, 6.0, 7.0], "default_in": 6.0, "shapes": ["square"],
        "materials": ["Front fabric", "Two back pieces, each with one long edge hemmed", HOOPED,
                      "Tape", "Stuffing or a small pillow form"],
    },
    "pillowpanel": {
        "name": "Pillow front panel", "fabric": "cotton", "margin": 0.0,
        "sizes_in": [12.0, 14.0, 16.0, 18.0], "default_in": 16.0, "shapes": ["square"],
        "stabilizer": "Cutaway, hooped with the front fabric",
        "materials": ["Front fabric and backing fabric (sizes on the cutting sheet)",
                      "Cutaway stabilizer", "A sewing machine to finish the pillow",
                      "A pillow form"],
    },
    "lace": {
        "name": "Freestanding lace", "fabric": "lace", "margin": 5.0, "sizes_in": [2.5, 3.0, 3.5],
        "default_in": 3.0, "shapes": ["circle", "square", "follow"],
        "stabilizer": "Two layers of water-soluble (wash-away) stabilizer",
        "materials": ["Two layers of water-soluble (wash-away) stabilizer, hooped drum-tight",
                      "The same thread in the bobbin as on top", "A bowl of warm water"],
    },
}
SHAPES = [{"id": "circle", "name": "Circle"}, {"id": "square", "name": "Rounded square"},
          {"id": "follow", "name": "Follow my design"}, {"id": "rect", "name": "Rectangle"}]
TURNED = ("coaster", "ornament", "plush")      # stitch-and-turn, with a turning gap
MIN_IN, MAX_IN = 1.5, 18.0


def normalize(project: dict | None) -> dict | None:
    """A clean project spec, or None if this isn't a project."""
    if not project or project.get("type") not in PROJECTS:
        return None
    spec = PROJECTS[project["type"]]
    shape = project.get("shape") if project.get("shape") in spec["shapes"] else spec["shapes"][0]
    try:
        size = float(project.get("size_in") or spec["default_in"])
    except (TypeError, ValueError):
        size = spec["default_in"]
    top = MAX_IN if project["type"] == "pillowpanel" else 7.0
    return {"type": project["type"], "shape": shape, "size_in": min(top, max(MIN_IN, size))}


def _bag(project: dict):
    """Zipper bag geometry: (width, height, y of the zipper's centre line)."""
    w = project["size_in"] * MM_PER_INCH
    h = w * 0.72
    return w, h, -h / 2 + ZIP_FROM_TOP_MM


def decoration_fit(project: dict) -> tuple[float, float]:
    """The box the art is digitized into, so it sits inside the project with room to spare."""
    spec = PROJECTS[project["type"]]
    size = project["size_in"] * MM_PER_INCH
    if project["type"] == "zipbag":
        w, h, zc = _bag(project)
        return w - 2 * spec["margin"] - 4, (h / 2 - (zc + 8.0)) - spec["margin"] - 4
    if project["type"] == "pillowpanel":
        side = min(PANEL_MAX_MM, size * 0.55)
        return side, side
    inner = size - 2 * spec["margin"]
    if project["shape"] == "circle":
        inner /= math.sqrt(2)             # a square that fits inside the circle
    elif project["shape"] == "square":
        inner *= 0.94                     # stay off the rounded corners
    return max(15.0, inner), max(15.0, inner)


def body_shape(project: dict, decoration) -> Polygon:
    """The finished outline of the item (without tabs or seam allowance), centred on (0, 0)."""
    spec = PROJECTS[project["type"]]
    size = project["size_in"] * MM_PER_INCH
    if project["type"] == "zipbag":
        w, h, _ = _bag(project)
        return box(-w / 2, -h / 2, w / 2, h / 2).buffer(-6).buffer(6, quad_segs=8)
    if project["shape"] == "square":
        r = 4.0 if project["type"] == "pillow" else size * 0.12
        return box(-size / 2, -size / 2, size / 2, size / 2).buffer(-r).buffer(r, quad_segs=12)
    if project["shape"] == "follow" and decoration is not None and not decoration.is_empty:
        grown = decoration.buffer(spec["margin"] + 4.0, quad_segs=12).buffer(-4.0, quad_segs=12)
        parts = polygons_of(grown)
        if len(parts) != 1:
            parts = polygons_of(decoration.convex_hull.buffer(spec["margin"], quad_segs=12))
        return Polygon(max(parts, key=lambda p: p.area).exterior).simplify(0.2)
    return Point(0, 0).buffer(size / 2, quad_segs=32)


# ------------------------------------------------------------------------- stitch helpers

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


def _path(points: list, rules: dict, triple: bool = False) -> list:
    pts = stitch.run_points(LineString(points), rules)
    return stitch.bean(pts) if triple else pts


def _satin_border(shape: Polygon, rules: dict, width: float = BORDER_MM) -> list:
    line = _ring(shape)
    n = max(16, int(line.length / 0.25))
    center = np.array([line.interpolate(line.length * k / n).coords[0] for k in range(n + 1)])
    half = np.full(len(center), width / 2)
    segs = stitch.stitch_satin({"center": center, "half": half}, dict(rules, pull_comp_mm=0.0))
    return segs[0] if segs else []


def _seam_with_gap(shape: Polygon, rules: dict, roomy: bool = False) -> tuple[list, float]:
    """The final seam of a stitch-and-turn item: triple run all round, except for a turning
    gap at the bottom. Returns (needle points, gap length in mm)."""
    line = LineString(shape.exterior.coords)
    minx, _, maxx, maxy = shape.bounds
    gap = min(75.0, max(38.0, line.length * 0.2))      # 1.5 to 3 inches, scaled to the item
    if roomy:                                          # room to get stuffing in
        gap = min(90.0, max(50.0, line.length * 0.22))
    mid = line.project(Point((minx + maxx) / 2, maxy))  # bottom centre
    a, b = (mid + gap / 2) % line.length, (mid - gap / 2) % line.length
    if a < b:
        path = list(substring(line, a, b).coords)
    else:
        path = list(substring(line, a, line.length).coords) + list(substring(line, 0, b).coords)[1:]
    return stitch.bean(stitch.run_points(LineString(path), rules)), gap


def _block(n: int, kind: str, title: str, pts, rules: dict, before: str, after: str,
           thread: str) -> dict:
    """One construction step. `pts` is one run of needle points, or a list of runs."""
    runs = pts if pts and isinstance(pts[0], list) else [pts]
    segs = []
    for run in runs:
        seg = sequence._clean_segment(run, rules["min_stitch_mm"], rules["max_stitch_mm"])
        if len(seg) >= 2:
            segs.append(sequence._tie(seg, rules["tie_mm"]))
    return {"color": -10 - n, "rgb": STEP_RGB[kind], "segments": segs,
            "label": f"{title} ({thread})",
            "step": {"title": title, "before": before, "after": after, "thread": thread}}


def _shift(blocks: list, dx: float, dy: float) -> list:
    return [dict(b, segments=[[(x + dx, y + dy) for x, y in sg] for sg in b["segments"]])
            for b in blocks]


def _totals(blocks: list) -> dict:
    xs = [p[0] for b in blocks for sg in b["segments"] for p in sg]
    ys = [p[1] for b in blocks for sg in b["segments"] for p in sg]
    return {"blocks": blocks,
            "stitch_count": sum(len(sg) for b in blocks for sg in b["segments"]),
            "trims": sum(len(b["segments"]) - 1 for b in blocks),
            "color_changes": len(blocks) - 1,
            "bounds": (min(xs), min(ys), max(xs), max(ys))}


def lace_hole_mm(segments: list, body: Polygon) -> float:
    """The widest opening left in a lace base, in mm: lace with a gap falls apart when the
    stabilizer washes away."""
    ppm = 4.0
    minx, miny, maxx, maxy = body.bounds
    w, h = int((maxx - minx) * ppm) + 8, int((maxy - miny) * ppm) + 8

    def px(points):
        a = np.asarray(points, float)
        return np.round(np.column_stack([(a[:, 0] - minx) * ppm + 4,
                                         (a[:, 1] - miny) * ppm + 4])).astype(np.int32)

    thread = np.zeros((h, w), np.uint8)
    for sg in segments:
        cv2.polylines(thread, [px(sg)], False, 1, 2)
    inside = np.zeros((h, w), np.uint8)
    core = polygons_of(body.buffer(-1.5))
    if not core:
        return 0.0
    cv2.fillPoly(inside, [px(list(max(core, key=lambda p: p.area).exterior.coords))], 1)
    empty = ((inside == 1) & (thread == 0)).astype(np.uint8)
    if not empty.any():
        return 0.0
    return float(2.0 * cv2.distanceTransform(empty, cv2.DIST_L2, 3).max() / ppm)


def _ears(body: Polygon, rules: dict) -> dict:
    """A separate little file: two ears, sewn first, turned, and added to the plush later."""
    w = max(22.0, (body.bounds[2] - body.bounds[0]) * 0.28)
    h = w * 1.25
    r = w / 2
    lines, outlines = [], []
    for cx in (-(r + 9), r + 9):
        arc = [(cx + r * math.cos(t), -(h - r) + r * math.sin(t))
               for t in np.linspace(math.pi, 2 * math.pi, 24)]
        seam = [(cx - r, 0.0)] + arc + [(cx + r, 0.0)]          # open along the bottom
        lines.append(_path(seam, rules, triple=True))
        outlines.append(_path(seam + [(cx - r, 0.0)], rules))
    blocks = [
        _block(0, "placement", "Ear outlines", outlines, rules,
               "Hoop tearaway stabilizer on its own, with nothing on it.",
               "Fold the ear fabric in half, right sides together, and lay it over both outlines.",
               "any color"),
        _block(1, "final", "Ear seams", lines, rules, "",
               "Unhoop. Cut each ear out about 5 mm outside the stitching, leaving the straight "
               "bottom open. Turn them right side out.", "a color that matches the fabric"),
    ]
    piece = box(-(w + 30), -(h + 12), w + 30, 12)
    return {"plan": _totals(blocks), "piece": piece}


# ----------------------------------------------------------------------------------- build

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

    deco = [dict(b, step={"title": "Decoration", "before": "", "after": "",
                          "thread": "the design's thread colors"}) for b in plan["blocks"]]
    pre, post, pieces, notes, extras, problems = [], [], [], [], {}, []
    bare = "Hoop tearaway stabilizer on its own, with nothing on it."

    if kind == "feltie":
        tack = outline.buffer(-0.7)
        pre.append(_block(0, "placement", "Placement line", _run(outline, rules, start=top), rules,
                          bare,
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
                          bare,
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

    elif kind in TURNED:
        cut = outline.buffer(SEAM_ALLOWANCE_MM, quad_segs=12)
        tack = cut.buffer(-1.5)
        seam, gap = _seam_with_gap(outline, rules, roomy=kind == "plush")
        pre.append(_block(0, "placement", "Placement line", _run(cut, rules, start=top), rules,
                          bare,
                          "Lay the front fabric over the outline, right side up, covering all "
                          "of it." + (" If it is minky, lay water-soluble topping over it."
                                      if kind == "plush" else ""), "any color"))
        pre.append(_block(1, "tackdown", "Tackdown", _run(tack, rules, start=top), rules, "",
                          "Leave the fabric as it is.", "any color"))
        lead = {"ornament": "Tape the ribbon loop at the top, with the loop pointing in toward "
                            "the middle and its ends sticking out past the outline. Then lay",
                "plush": "Tape the two finished ears at the top, pointing in toward the middle, "
                         "with their open ends sticking out past the outline. Then lay"}.get(kind, "Lay")
        fill = {"ornament": "stuff it lightly if you like, ", "plush": "stuff it firmly, "}.get(kind, "")
        post.append(_block(2, "final", "Final seam", seam, rules,
                           f"{lead} the back fabric on top, right side DOWN, covering "
                           "everything. Tape the corners.",
                           "Unhoop and tear the stabilizer away. Trim along the outer line "
                           "(a 6 mm seam). Clip the curves, turn it right side out through the "
                           f"gap at the bottom (about {gap / MM_PER_INCH:.1f} in), push out the "
                           f"edges, press, {fill}and hand-sew the gap closed.",
                           "a color that matches the fabric"))
        pieces.append({"name": "Fabric, front", "shape": cut.buffer(6)})
        pieces.append({"name": "Fabric, back", "shape": cut.buffer(6)})
        notes.append("The outer stitched line is your cutting line. The design stays "
                     f"{KEEP_OUT_MM:g} mm inside the seam so it isn't caught when you turn it.")
        if kind == "plush":
            ears = _ears(body, rules)
            extras["ears"] = {"title": "Ears (sew this file first)", "plan": ears["plan"]}
            pieces.append({"name": "Ear fabric (fold in half)", "shape": ears["piece"]})
            notes.append("Sew the separate ears file first, so the ears are ready when the "
                         "main file asks for them. Leave them out if you don't want ears.")
            notes.append("Faces are embroidered, with no buttons or plastic eyes. Toys for "
                         "children under 3 that you sell need small-parts testing at a "
                         "CPSC-accepted lab.")

    elif kind == "zipbag":
        w, h, zc = _bag(project)
        x0, x1 = -w / 2 - 10, w / 2 + 10
        lower = (zc + 8.0 + h / 2) / 2                       # middle of the panel below the zipper
        deco = _shift(deco, 0.0, lower)
        zip_box = box(x0 - 2, zc - 11, x1 + 2, zc + 11)
        pre.append(_block(0, "placement", "Zipper placement",
                          [_run(zip_box, rules), _run(outline, rules, start=top)], rules,
                          "Hoop no-show mesh cutaway stabilizer on its own.",
                          "Tape the zipper, face up, inside the long box, with the teeth along "
                          "the middle. Keep the metal ends and the pull outside the stitched "
                          "outline.", "any color"))
        pre.append(_block(1, "tackdown", "Zipper tackdown",
                          [_path([(x0, zc - 6), (x1, zc - 6)], rules),
                           _path([(x1, zc + 6), (x0, zc + 6)], rules)], rules, "",
                          "Top panel: fold the top outer piece in half, wrong sides together, "
                          "and lay its folded edge along the top side of the teeth. Tape. Do "
                          "the same with the top lining piece on the BACK of the hoop.",
                          "any color"))
        pre.append(_block(2, "tackdown", "Top panel seam",
                          _path([(x0, zc - 4.5), (x1, zc - 4.5)], rules), rules, "",
                          "Bottom panel: fold the bottom outer piece in half and lay its folded "
                          "edge along the bottom side of the teeth. Tape. Do the same with the "
                          "bottom lining piece on the BACK of the hoop.", "a color that matches"))
        pre.append(_block(3, "tackdown", "Bottom panel seam",
                          _path([(x1, zc + 4.5), (x0, zc + 4.5)], rules), rules, "",
                          "Smooth the bottom panel flat. The design sews on it next.",
                          "a color that matches"))
        post.append(_block(4, "final", "Final seam", _run(outline, rules, triple=True, start=top),
                           rules,
                           "IMPORTANT: open the zipper halfway, or you won't be able to turn the "
                           "bag. Lay the back outer piece on top, right side DOWN. Tape the back "
                           "lining to the BACK of the hoop, right side facing out.",
                           "Unhoop and trim the stabilizer. Trim to a 6 mm seam and clip the "
                           "corners. Turn the bag right side out through the open zipper and "
                           "push out the corners.", "a color that matches the fabric"))
        side = w + 30
        for name, tall in (
                ("Top piece, outer and lining (cut 1 of each, fold in half)", 2 * (ZIP_FROM_TOP_MM + 12)),
                ("Bottom piece, outer and lining (cut 1 of each, fold in half)", 2 * (h - ZIP_FROM_TOP_MM + 12)),
                ("Back piece, outer and lining (cut 1 of each)", h + 30)):
            pieces.append({"name": name, "shape": box(-side / 2, -tall / 2, side / 2, tall / 2)})
        notes.append("Use a nylon-coil zipper: the needle can sew across nylon teeth, but not metal.")
        notes.append("Keep any batting out of the seam allowance, or the seams get too thick.")

    elif kind == "pillow":
        cut = outline.buffer(SEAM_ALLOWANCE_MM, quad_segs=12)
        tack = cut.buffer(-1.5)
        size = project["size_in"] * MM_PER_INCH
        pre.append(_block(0, "placement", "Placement line", _run(cut, rules, start=top), rules,
                          bare, "Lay the front fabric over the outline, right side up, covering "
                                "all of it.", "any color"))
        pre.append(_block(1, "tackdown", "Tackdown", _run(tack, rules, start=top), rules, "",
                          "Leave the fabric as it is.", "any color"))
        post.append(_block(2, "final", "Final seam", _run(outline, rules, triple=True, start=top),
                           rules,
                           "Lay the two back pieces on top, right sides DOWN, with their hemmed "
                           "edges overlapping across the middle by about 2 in. Tape them flat.",
                           "Unhoop and tear the stabilizer away. Trim to a 6 mm seam and clip "
                           "the corners. Turn it right side out through the overlap in the back "
                           "and fill it.", "a color that matches the fabric"))
        pieces.append({"name": "Front", "shape": cut.buffer(6)})
        pieces.append({"name": "Back piece (cut 2, hem one long edge)",
                       "shape": box(-(size + 24) / 2, -(size * 0.65 + 12) / 2,
                                    (size + 24) / 2, (size * 0.65 + 12) / 2)})
        notes.append("The seam goes all the way round: the opening is the overlap in the back.")

    elif kind == "pillowpanel":
        size = project["size_in"] * MM_PER_INCH
        dminx, dminy, dmaxx, dmaxy = plan["bounds"]
        frame = box(dminx - 8, dminy - 8, dmaxx + 8, dmaxy + 8)
        pre.append(_block(0, "placement", "Basting box", _run(frame, rules), rules,
                          "Hoop cutaway stabilizer with the front fabric on top, with the middle "
                          "of the fabric in the middle of the hoop.",
                          "Check the fabric is flat and the box is where you want the design.",
                          "any color"))
        front = size + MM_PER_INCH
        pieces.append({"name": "Front panel (embroider first, then trim to this size)",
                       "shape": box(-front / 2, -front / 2, front / 2, front / 2)})
        pieces.append({"name": "Back piece (cut 2, hem one long edge)",
                       "shape": box(-front / 2, -(size * 0.65 + MM_PER_INCH) / 2,
                                    front / 2, (size * 0.65 + MM_PER_INCH) / 2)})
        notes += [
            f"This is the embroidered front for a {project['size_in']:g} in pillow. The pillow "
            "itself is too big for a hoop, so it is finished on a sewing machine.",
            "After embroidering: pull out the basting box, trim the stabilizer, and trim the "
            "front to the size on the cutting sheet with the design in the middle.",
            "Hem one long edge of each back piece. Lay them on the front, right sides together, "
            "hemmed edges overlapping across the middle.",
            "Sew all the way round with a 1/2 in seam, clip the corners, turn through the "
            "overlap, and put in the pillow form.",
        ]

    elif kind == "lace":
        lace_rules = dict(rules, pull_comp_mm=0.0)
        base: list = []
        for angle in (45.0, 135.0):                 # two layers crossing make the net
            start = base[-1][-1] if base else None
            base += stitch.fill_segments(outline, angle, LACE_SPACING_MM, 2.5, start)
        base += stitch.edge_run(outline, 0.6, lace_rules)
        pre.append(_block(0, "lace", "Lace base", base, lace_rules,
                          "Hoop two layers of water-soluble (wash-away) stabilizer, drum-tight. "
                          "Wind a bobbin with the same thread as the top.",
                          "Nothing to do. The design sews on the lace next.", "your lace color"))
        post.append(_block(1, "final", "Satin frame",
                           _satin_border(outline.buffer(-0.5), lace_rules, 3.6), lace_rules, "",
                           "Unhoop and trim the stabilizer close. Rinse in warm water until it "
                           "has dissolved, then lay the lace flat to dry.", "your lace color"))
        hole = lace_hole_mm(pre[0]["segments"], outline)
        if hole > LACE_MAX_HOLE_MM:
            problems.append("The lace base has open spots, so it could fall apart when the "
                            "stabilizer washes away. Try a simpler shape or a bigger size.")
        notes.append("Lace has no fabric under it: the stitches hold each other up. Keep the "
                     "same thread top and bottom, and don't trim any threads inside the lace.")
        notes.append("To hang it, thread a ribbon through the satin frame after it has dried.")

    blocks = pre + deco + post
    whole = _totals(blocks)
    line = [p for sg in pre[0]["segments"] for p in sg]
    dieline = {"blocks": pre[:1], "stitch_count": len(line), "trims": len(pre[0]["segments"]) - 1,
               "color_changes": 0,
               "bounds": (min(p[0] for p in line), min(p[1] for p in line),
                          max(p[0] for p in line), max(p[1] for p in line))}
    info = {"type": kind, "name": spec["name"], "shape": project["shape"],
            "size_in": project["size_in"], "materials": spec["materials"], "notes": notes,
            "pieces": pieces, "dieline_plan": dieline, "extras": extras, "problems": problems,
            "stabilizer": spec.get("stabilizer"),
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
                       "default_in": v["default_in"], "shapes": v["shapes"]}
                      for k, v in PROJECTS.items()],
            "shapes": SHAPES}
