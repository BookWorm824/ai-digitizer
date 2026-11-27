"""General lessons from varied test designs (tools/make_bench_designs.py): small grey text
keeps its dark color, a row's word gaps help tell which known line it is, and a letter
overlapping a thin ring stays a solid letter."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from shapely.geometry import box

from app import config
from app.pipeline import clean, intake, lettering


def test_small_grey_text_keeps_its_dark_color():
    # thin dark grey lettering beside a big blue block: shrunk, it reads pale, but it must be
    # sewn dark grey, not dropped as shading or matched to white
    img = Image.new("RGB", (1600, 800), "white")
    d = ImageDraw.Draw(img)
    d.rectangle((80, 120, 340, 380), fill=(30, 80, 160))
    try:
        f = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 64)
    except OSError:
        f = ImageFont.load_default()
    d.text((400, 270), "Repairs - Installs - 24/7 Service", font=f, fill=(60, 60, 60))
    import io
    buf = io.BytesIO()
    img.save(buf, "PNG")
    got = clean.run(intake.run(buf.getvalue()), 88.9, 88.9, config.rules_for("tshirt"))
    dark = [c for c in got["palette"] if max(c) < 120 and max(c) - min(c) < 40]
    assert dark, got["palette"]


def test_word_gaps_are_counted_but_a_missing_letter_is_not_a_space():
    def row(xs, w=1.6, h=3.0):
        return [box(x, 0, x + w, h) for x in xs]
    # three words
    assert lettering._word_count(row([0, 2, 4, 6, 9.5, 11.5, 13.5, 17, 19, 21])) == 3
    # one word with a letter missing from the row (a wide hole, not a space)
    assert lettering._word_count(row([0, 2, 4, 6, 8, 15, 17, 19])) == 1


def test_a_three_word_row_is_not_matched_to_a_two_word_line():
    pls = lettering.layout("FRESH BAKED DAILY", 2.0, 0, 0, "block", w=0.4)
    glyphs = [lettering.outline_of(pl, "block") for pl in pls]
    # two letters merged, as small traced lettering does
    glyphs = glyphs[:3] + [glyphs[3].union(glyphs[4]).convex_hull] + glyphs[5:]
    fit = lettering.fit_row(glyphs, ["Bella's Bakery", "FRESH BAKED DAILY"])
    assert fit and fit["text"].upper() == "FRESH BAKED DAILY"


def test_lone_specks_go_but_details_inside_a_ring_stay():
    from shapely.geometry import LineString, Point
    from app.pipeline import classify
    ring = Point(0, 0).buffer(3.0).exterior
    objs = [
        {"kind": "run", "color": 0, "line": LineString(ring.coords), "route": [list(ring.coords)]},
        # clock hands inside the ring, more than a millimetre from it
        {"kind": "run", "color": 0, "line": LineString([(0, 0), (0, 1.2)]), "route": [[(0, 0), (0, 1.2)]]},
        # a lone mark well away from everything (a scrap of dropped lettering)
        {"kind": "run", "color": 0, "line": LineString([(20, 20), (21, 20.5)]), "route": [[(20, 20), (21, 20.5)]]},
    ]
    assert classify._drop_specks(objs) == 1
    assert len(objs) == 2 and objs[1]["line"].bounds[1] == 0


def test_an_outline_is_sewn_after_the_letters_it_surrounds_on_a_cap():
    from shapely.geometry import LineString
    from app.pipeline import sequence, stitch
    rules = config.rules_for("hat")
    objs = []
    # gold letters spread across the cap, one navy outline around them all
    for x in (-25.0, -10.0, 5.0, 20.0):
        objs.append({"kind": "fill", "color": 1, "poly": box(x, 0, x + 6, 8), "area_mm2": 48.0})
    ring = [(-26, -1), (27, -1), (27, 9), (-26, 9), (-26, -1)]
    objs.append({"kind": "run", "color": 0, "line": LineString(ring), "route": [ring],
                 "width_mm": 0.8, "bean": False, "bold": True})
    # navy is first in the color order (it also has big navy fills elsewhere)
    objs.append({"kind": "fill", "color": 0, "poly": box(-5, 20, 25, 40), "area_mm2": 600.0})
    st = stitch.run({"objects": objs, "order": [0, 1]}, rules)
    plan = sequence.run(st, [(20, 36, 90), (240, 180, 30)], rules, [0, 1])
    last = plan["blocks"][-1]
    assert last["color"] == 0
    xs = [x for sg in last["segments"] for x, _y in sg]
    assert min(xs) < -20 and max(xs) > 20          # the outline itself is in the last block
