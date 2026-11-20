"""The user's own changes to a design, made by clicking on the preview:

  {"type": "delete",  "at": [u, v]}                 remove the area under the click
  {"type": "outline", "at": [u, v]}                 sew just its outline, not the inside
  {"type": "color",   "at": [u, v], "hex": "#..."}  sew that one area in another color
  {"type": "recolor", "from": "#...", "hex": "#..."}  change a thread color everywhere
                                                     (choosing a color already in the design
                                                     merges the two)

`at` is the click position measured from the middle of the design, as a fraction of the
design's size, so a change still lands on the same area after the design is resized.

Edits are applied in order to the classified objects, before any stitches are made.
"""

from __future__ import annotations

from shapely.geometry import LineString, Point

from .common import hex_of, polygons_of


def shape_of(obj: dict):
    """The ground an object covers, for deciding what a click landed on."""
    if obj["kind"] == "fill":
        return obj["poly"]
    if obj["kind"] == "satin":
        return obj["outline"]
    return obj["line"].buffer(0.8)


def _rgb(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        raise ValueError(hex_color)
    return tuple(int(h[k:k + 2], 16) for k in (0, 2, 4))


def find(objects: list, x: float, y: float):
    """Index of the object under a point: the smallest one there, so details win over the
    big area they sit on. None if the click hit nothing."""
    pt = Point(x, y)
    best = None
    for i, o in enumerate(objects):
        shape = shape_of(o)
        if shape.contains(pt) or shape.distance(pt) < 0.3:
            if best is None or shape.area < best[0]:
                best = (shape.area, i)
    return None if best is None else best[1]


def _color_index(palette: list, order: list, rgb: tuple, after: int) -> int:
    """Index of rgb in the palette, adding it (sewn right after color `after`) if new."""
    for i, c in enumerate(palette):
        if tuple(c) == tuple(rgb):
            return i
    palette.append(tuple(rgb))
    new = len(palette) - 1
    order.insert(order.index(after) + 1 if after in order else len(order), new)
    return new


def apply(classified: dict, palette: list, edits: list, ref: float) -> tuple[dict, list, int]:
    """Returns (classified with the edits applied, the palette, how many edits took effect).
    `ref` is the design's size in mm, which `at` positions are fractions of."""
    objects = list(classified["objects"])
    palette = [tuple(c) for c in palette]
    order = list(classified["order"])
    applied = 0
    for e in edits or []:
        try:
            kind = e["type"]
            if kind == "recolor":
                src, dst = _rgb(e["from"]), _rgb(e["hex"])
                idx = [i for i, c in enumerate(palette) if tuple(c) == src]
                if not idx:
                    continue
                i = idx[0]
                same = [j for j, c in enumerate(palette) if tuple(c) == dst and j != i]
                if same:                        # merge into the color that is already there
                    for o in objects:
                        if o["color"] == i:
                            o["color"] = same[0]
                else:
                    palette[i] = dst
                applied += 1
                continue
            n = find(objects, e["at"][0] * ref, e["at"][1] * ref)
            if n is None:
                continue
            obj = objects[n]
            if kind == "delete":
                del objects[n]
            elif kind == "color":
                objects[n] = dict(obj, color=_color_index(palette, order, _rgb(e["hex"]),
                                                          obj["color"]))
            elif kind == "outline":
                if obj["kind"] == "run":
                    continue
                rings = []
                for p in polygons_of(shape_of(obj)):
                    rings += [p.exterior, *p.interiors]
                lines = [{"kind": "run", "color": obj["color"], "line": LineString(r.coords),
                          "bean": True, "width_mm": 0.8} for r in rings if r.length >= 3.0]
                objects[n:n + 1] = lines
            else:
                continue
            applied += 1
        except (KeyError, ValueError, TypeError, IndexError):
            continue                            # a malformed edit is skipped, never fatal
    used = {o["color"] for o in objects}
    return (dict(classified, objects=objects, order=[c for c in order if c in used]),
            palette, applied)


def areas(objects: list, palette: list) -> list:
    """What the preview needs to know which area a click lands on: one outline per object."""
    out = []
    for o in objects:
        polys = polygons_of(shape_of(o))
        if not polys:
            continue
        p = max(polys, key=lambda q: q.area).simplify(0.2)
        ring = [round(v, 2) for xy in p.exterior.coords for v in xy]
        out.append({"kind": o["kind"], "hex": hex_of(palette[o["color"]]),
                    "area": round(float(p.area), 1), "ring": ring})
    return out
