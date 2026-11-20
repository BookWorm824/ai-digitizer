"""The user's own changes to a design, made by clicking on the preview:

  {"type": "delete",  "at": [u, v]}                 remove the area under the click
  {"type": "outline", "at": [u, v]}                 sew just its outline, not the inside
  {"type": "color",   "at": [u, v], "hex": "#..."}  sew that one area in another color
  {"type": "recolor", "from": "#...", "hex": "#..."}  change a thread color everywhere
                                                     (choosing a color already in the design
                                                     merges the two)
  {"type": "retype",  "at": [u, v], "text": "...", "style": "block"}
                                                    sew the clicked words in the embroidery
                                                    font instead (empty text removes them)
  {"type": "text",    "at": [u, v], "text": "...", "size": h, "style": "block",
                      "hex": "#..."}                add words, centred on the click, with a
                                                    cap height of h (a fraction of the size)

`at` is the click position measured from the middle of the design, as a fraction of the
design's size, so a change still lands on the same area after the design is resized.

Edits are applied in order to the classified objects, before any stitches are made.
"""

from __future__ import annotations

from shapely.geometry import LineString, Point

from . import lettering
from .common import hex_of, polygons_of

TEXT_GROUP = 10000          # sewing-order groups for added words, after retyped rows


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


def _style(e: dict) -> str:
    style = str(e.get("style") or "block")
    return style if style in lettering.STYLES else "block"


def _clean_text(value) -> str:
    return " ".join(str(value or "").split())[:80]


def apply(classified: dict, palette: list, edits: list, ref: float,
          rules: dict | None = None) -> tuple[dict, list, int]:
    """Returns (classified with the edits applied, the palette, how many edits took effect).
    `ref` is the design's size in mm, which `at` positions are fractions of. `rules` are
    needed to sew retyped or added words."""
    objects = list(classified["objects"])
    palette = [tuple(c) for c in palette]
    order = list(classified["order"])
    rows = {r["row"]: dict(r) for r in classified.get("rows", [])}
    rules = rules or {}
    applied = 0
    for n_edit, e in enumerate(edits or []):
        try:
            kind = e["type"]
            if kind == "text":
                text = _clean_text(e.get("text"))
                size = float(e["size"]) * ref
                if not lettering.letters_of(text) or not (1.0 <= size <= 200.0):
                    continue
                style = _style(e)
                pls = lettering.layout(text, size, float(e["at"][0]) * ref,
                                       float(e["at"][1]) * ref, style)
                color = _color_index(palette, order, _rgb(e["hex"]), order[-1] if order else -1)
                if color not in order:
                    order.append(color)
                objects.extend(lettering.objects(pls, color, rules, style,
                                                 lettering.text_style(size),
                                                 group=TEXT_GROUP + n_edit))
                applied += 1
                continue
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
            if kind == "retype":
                row = obj.get("text_row")
                if row is None or row not in rows:
                    continue
                info = rows[row]
                text = _clean_text(e.get("text"))
                style = _style(e)
                color = obj["color"]
                objects = [o for o in objects if o.get("text_row") != row]
                if text:
                    pls = lettering.fit_box(text, info["glyphs"], style,
                                            rules.get("satin_max_mm", 7.0))
                    if pls:
                        objects.extend(lettering.objects(pls, color, rules, style,
                                                         lettering.text_style(pls[0].H),
                                                         group=row, row=row))
                info["text"] = text or None
                applied += 1
                continue
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
    return (dict(classified, objects=objects, order=[c for c in order if c in used],
                 rows=list(rows.values())),
            palette, applied)


def areas(objects: list, palette: list, rows: list | None = None) -> list:
    """What the preview needs to know which area a click lands on: one outline per object.
    Lettering says so, with its words when they are known, so it can be retyped."""
    words = {r["row"]: r.get("text") for r in rows or []}
    out = []
    for o in objects:
        polys = polygons_of(shape_of(o))
        if not polys:
            continue
        p = max(polys, key=lambda q: q.area).simplify(0.2)
        ring = [round(v, 2) for xy in p.exterior.coords for v in xy]
        a = {"kind": o["kind"], "hex": hex_of(palette[o["color"]]),
             "area": round(float(p.area), 1), "ring": ring}
        if o.get("text_row") is not None:
            a["text"] = True
            a["words"] = words.get(o["text_row"]) or ""
        out.append(a)
    return out
