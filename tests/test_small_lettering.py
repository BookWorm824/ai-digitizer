"""Small lettering lessons (tutorial study: Amann micro lettering, Wilcom lettering
connections): thin running-stitch letters close together in a word are sewn as one thread,
never joined across a word space; the dot of a thin i is not reached by a walk over open
fabric beside the letter; letter connectors may cross at most a millimetre of open fabric."""

from __future__ import annotations

from app import config
from app.pipeline import lettering, sequence, stitch

RULES = config.rules_for("tshirt")


def _thin(text, H=2.6, x=0.0):
    pls = lettering.layout(text, H, x, 0.0, "block")
    return pls, lettering.objects(pls, 0, RULES, "block", lettering.text_style(H), group=0, row=0)


def test_close_thin_letters_are_one_thread():
    pls = lettering.layout("ce", 2.6, 0.0, 0.0, "block")
    pls[1].bx -= 0.5                        # set tight: the letters nearly touch
    objs = lettering.objects(pls, 0, RULES, "block", lettering.text_style(2.6), group=0, row=0)
    runs = [o for o in objs if o["kind"] == "run" and o["seq"][2] < 100]
    assert len(runs) == 1


def test_thin_letters_at_ordinary_spacing_stay_apart():
    # (joined, "EST" reads as E_S_T: only nearly touching letters are joined)
    pls, objs = _thin("EST", H=3.5)
    runs = [o for o in objs if o["kind"] == "run" and o["seq"][2] < 100]
    assert len(runs) == 3


def test_thin_letters_are_not_joined_across_a_word_space():
    pls, objs = _thin("c e")
    runs = [o for o in objs if o["kind"] == "run" and o["seq"][2] < 100]
    assert len(runs) == 2


def test_the_dot_of_a_thin_i_is_cut_not_walked_to_beside_the_letter():
    pls, objs = _thin("ic")
    st = stitch.run({"objects": objs, "order": [0]}, RULES)
    plan = sequence.run(st, [(60, 60, 60)], RULES, [0])
    i = pls[0]
    stem_x = i.bx + i.outer_width() / 2
    for b in plan["blocks"]:
        for sg in b["segments"]:
            ys = [y for _x, y in sg]
            top_dot = min(ys) < i.by - 1.05 * i.H          # reaches up to the dot
            low = max(ys) > i.by - 0.2 * i.H               # and down to the baseline
            if top_dot and low:
                # one thread holds both: it may only run straight up the stem
                assert all(abs(x - stem_x) < 0.5 for x, y in sg if y < i.by - 0.3 * i.H)


def test_thin_line_letters_set_as_a_line_are_evenly_spaced():
    from shapely.geometry import LineString
    from shapely.ops import unary_union
    pls = lettering.fit_line("FRESH BAKED", (0.0, 0.0, 30.0, 3.0), 3.0, 1.0, "block")
    edges = []
    for pl in pls:
        lines = [LineString(st["pts"]) for st in lettering.letter_strokes(pl, "block")
                 if not st["dot"]]
        b = unary_union(lines).bounds
        edges.append((b[0], b[2]))
    gaps = [b[0] - a[1] for a, b in zip(edges, edges[1:])]
    inside = [g for g in gaps if g < 1.5]          # (not the word space)
    assert max(inside) - min(inside) < 0.45


def test_alike_strokes_in_a_row_are_sewn_alike():
    from shapely.geometry import LineString
    from app.pipeline import classify
    def stroke(x, satin):
        ln = LineString([(x, 0), (x + 1, 5)])
        o = {"kind": "run", "color": 0, "line": ln, "route": [list(ln.coords)],
             "width_mm": 0.9, "bold": True}
        if satin:
            o["satin_w"] = [1.1]
        return o
    objs = [stroke(0, False), stroke(2, True), stroke(4, False), stroke(6, False)]
    classify._same_weight_rows(objs)
    assert not any(o.get("satin_w") for o in objs)


def test_a_row_whose_satin_widths_are_all_zero_is_left_as_runs():
    from shapely.geometry import LineString
    from app.pipeline import classify
    objs = []
    for x, sw in ((0, [0.0]), (2, [0.0]), (4, None)):
        ln = LineString([(x, 0), (x + 1, 5)])
        o = {"kind": "run", "color": 0, "line": ln, "route": [list(ln.coords)],
             "width_mm": 0.9, "bold": True}
        if sw is not None:
            o["satin_w"] = sw
        objs.append(o)
    classify._same_weight_rows(objs)
    assert all(not any(o.get("satin_w") or []) for o in objs)


def test_a_line_inside_another_is_not_part_of_its_row():
    from shapely.geometry import LineString, Point
    from app.pipeline import classify
    head = Point(0, 0).buffer(10, 32).exterior
    mouth = LineString([(-3, -2), (0, -3), (3, -2)])
    objs = [{"kind": "run", "color": 0, "line": LineString(head.coords), "route": [list(head.coords)],
             "width_mm": 0.8, "bold": True, "satin_w": [1.1]},
            {"kind": "run", "color": 0, "line": mouth, "route": [list(mouth.coords)],
             "width_mm": 0.7, "bold": True}]
    classify._same_weight_rows(objs)
    assert not objs[1].get("satin_w")


def _clean_picture(hole_w, hole_h, size_mm=20.0):
    import io
    import numpy as np
    from PIL import Image, ImageDraw
    from app.pipeline import clean, intake
    img = Image.new("RGB", (1000, 1000), "white")
    d = ImageDraw.Draw(img)
    d.ellipse((300, 300, 700, 700), fill=(20, 20, 20))                   # a bold ring letter
    d.ellipse((410, 410, 590, 590), fill="white")                        # its bowl,
    d.rectangle((560, 450, 1000, 550), fill="white")                     # open: a C
    d.ellipse((500 - hole_w // 2, 355 - hole_h // 2, 500 + hole_w // 2, 355 + hole_h // 2),
              fill="white")                                              # a slit in its stroke
    buf = io.BytesIO()
    img.save(buf, "PNG")
    out = clean.run(intake.run(buf.getvalue()), size_mm, size_mm, config.rules_for("hat"))
    lab = out["labels"]
    H, W = lab.shape
    return lab, np


def _holes(lab):
    import cv2
    import numpy as np
    from app.pipeline.common import BG
    n, comp, st, _ = cv2.connectedComponentsWithStats((lab == BG).astype(np.uint8), connectivity=4)
    H, W = lab.shape
    return sum(1 for k in range(1, n) if st[k, 0] > 0 and st[k, 1] > 0
               and st[k, 0] + st[k, 2] < W and st[k, 1] + st[k, 3] < H)


def test_a_letter_counter_that_closes_when_shrunk_is_opened_again():
    lab, _np = _clean_picture(54, 8)          # 1.1 x 0.16 mm at 20 mm: closes when shrunk
    assert _holes(lab) == 1


def test_a_tiny_speck_of_a_hole_is_not_opened():
    lab, _np = _clean_picture(4, 4)           # far under the size worth keeping
    assert _holes(lab) == 0


def test_strokes_tuck_a_little_under_the_stem_not_right_across_it():
    pls = lettering.layout("E", 10.0, 0.0, 0.0, "block")
    pl = pls[0]
    objs = lettering.objects(pls, 0, RULES, "block", lettering.text_style(10.0), group=0, row=0)
    cols = [o for o in objs if o["kind"] == "satin"]
    stem = min(cols, key=lambda o: o["outline"].bounds[2] - o["outline"].bounds[0])
    stem_left = stem["outline"].bounds[0]
    top, bottom = stem["outline"].bounds[1], stem["outline"].bounds[3]
    # the middle arm (the top and bottom ones meet the stem in a mitred corner)
    arms = [o for o in cols if o is not stem
            and o["outline"].bounds[1] > top + 1.0 and o["outline"].bounds[3] < bottom - 1.0]
    assert len(arms) == 1
    # it ends on the stem's centre line (plus a short tuck), not out at its far edge
    for o in arms:
        assert o["outline"].bounds[0] > stem_left + 0.25 * pl.w
    # ... while the letter's outline used for recognition and spacing is unchanged
    assert abs(lettering.outline_of(pl, "block").bounds[0] - stem_left) < 0.05
