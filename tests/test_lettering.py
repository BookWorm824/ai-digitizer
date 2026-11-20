"""Tests for the lettering tool: the embroidery font, how its strokes become satin columns,
retyping traced words, and the "retype" / "add words" edits on the preview."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from app import config
from app.pipeline import lettering as L
from app.pipeline import run_pipeline

RULES = config.rules_for("tshirt")


def _font(size):
    for name in ("arialbd.ttf", "DejaVuSans-Bold.ttf", "timesbd.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    pytest.skip("no TrueType font on this computer to draw test lettering with")


def _picture(big="GOOD DAY", small="Since 1984 Knoxville", small_px=60):
    img = Image.new("RGB", (1200, 700), "white")
    d = ImageDraw.Draw(img)
    d.ellipse([450, 30, 750, 330], fill=(20, 60, 140))
    d.text((600, 450), big, font=_font(100), fill=(140, 20, 40), anchor="mm")
    d.text((600, 600), small, font=_font(small_px), fill=(20, 20, 20), anchor="mm")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _outline(text, H=10.0, style="block"):
    pls = L.layout(text, H, 0.0, 0.0, style)
    return pls, unary_union([L.outline_of(pl, style) for pl in pls])


# ---------------------------------------------------------------------------------- the font

def test_every_character_makes_clean_satin_columns():
    for ch in L.GLYPHS:
        pl = L.Placement(ch, 0.0, 20.0, 10.0, 1.4)
        strokes = L.letter_strokes(pl)
        assert strokes, ch
        for st in strokes:
            col = L.column(st["pts"], pl.w, st["cut0"], st["cut1"])
            assert col is not None, ch
            assert col["outline"].is_valid and col["outline"].area > 0.3, ch
            assert len(col["left"]) == len(col["right"]) == len(col["center"]), ch


def test_feet_are_cut_flat_on_the_baseline_and_cap_line():
    # an I is exactly as tall as the cap height, standing on the baseline (y grows down)
    pl = L.Placement("I", 0.0, 20.0, 10.0, 1.4)
    o = L.outline_of(pl)
    assert o.bounds[3] == pytest.approx(20.0, abs=0.06)
    assert o.bounds[1] == pytest.approx(10.0, abs=0.06)
    # an A's legs end on the baseline, flat, not as pointed tips poking below it
    a = L.outline_of(L.Placement("A", 0.0, 20.0, 10.0, 1.4))
    assert a.bounds[3] == pytest.approx(20.0, abs=0.06)
    assert a.bounds[1] == pytest.approx(10.0, abs=0.06)


def test_corners_are_covered():
    # the outside corner of an L and of an E's top arm are both under thread
    pl = L.Placement("L", 0.0, 20.0, 10.0, 1.4)
    o = L.outline_of(pl)
    x0, _, _, y1 = o.bounds
    assert o.buffer(0.05).contains(Point(x0 + 0.2, y1 - 0.2))
    e = L.outline_of(L.Placement("E", 0.0, 20.0, 10.0, 1.4))
    ex0, ey0, _, _ = e.bounds
    assert e.buffer(0.05).contains(Point(ex0 + 0.2, ey0 + 0.2))


def test_strokes_that_end_on_another_are_sewn_first():
    # the arms of an E end on its stem, so they go first and the stem covers their ends
    strokes = L.letter_strokes(L.Placement("E", 0.0, 20.0, 10.0, 1.4))
    stem = max(range(len(strokes)),
               key=lambda i: abs(strokes[i]["pts"][0][1] - strokes[i]["pts"][-1][1]))
    middle = min(range(len(strokes)), key=lambda i: abs(strokes[i]["pts"][0][1] - 15.0))
    assert middle < stem
    assert any(c and c[0] == "ext" for c in (strokes[middle]["cut0"], strokes[middle]["cut1"]))
    # ...and the stem of a T is sewn before its bar
    t = L.letter_strokes(L.Placement("T", 0.0, 20.0, 10.0, 1.4))
    assert abs(t[0]["pts"][0][0] - t[0]["pts"][-1][0]) < 0.01     # the upright first


def test_letters_do_not_touch_and_words_are_spaced():
    pls, _ = _outline("AVATAR WAVE", 10.0)
    outlines = [L.outline_of(pl) for pl in pls]
    for a, b in zip(outlines, outlines[1:]):
        assert a.distance(b) > 0.3 or a.bounds[2] < b.bounds[0] + 0.5
    widths = [o.bounds[2] - o.bounds[0] for o in outlines]
    assert all(w > 1.0 for w in widths)


def test_layout_is_centred_on_the_point():
    _, o = _outline("HELLO", 10.0)
    cx = (o.bounds[0] + o.bounds[2]) / 2
    cy = (o.bounds[1] + o.bounds[3]) / 2
    assert cx == pytest.approx(0.0, abs=0.3) and cy == pytest.approx(0.0, abs=0.3)
    assert o.bounds[3] - o.bounds[1] == pytest.approx(10.0, abs=0.2)


def test_stroke_width_follows_the_research_floors():
    assert L.width_for(20.0) == pytest.approx(2.8)            # 14 % of the cap height
    assert L.width_for(5.0) == pytest.approx(1.0)             # never under 1 mm satin
    assert L.width_for(20.0, "bold") > L.width_for(20.0)
    assert L.width_for(20.0, weight_mm=9.0) == pytest.approx(0.2 * 20.0)    # stays a letter


def test_varsity_serifs_only_on_bigger_letters():
    big = L.letter_strokes(L.Placement("I", 0.0, 20.0, 12.0, 2.0), "varsity")
    small = L.letter_strokes(L.Placement("I", 0.0, 20.0, 6.0, 1.2), "varsity")
    assert len(big) == 3 and len(small) == 1


def test_small_letters_become_bean_runs_and_big_ones_satin():
    small = L.objects(L.layout("Hi", 3.0, 0, 0), 0, RULES)
    big = L.objects(L.layout("Hi", 12.0, 0, 0), 0, RULES)
    assert small and all(o["kind"] == "run" for o in small)       # run lettering under 6 mm
    assert big and all(o["kind"] == "satin" for o in big)
    # every object knows its place in the sewing order: letter by letter, stroke by stroke
    seqs = [o["seq"] for o in big]
    assert seqs == sorted(seqs)


# ------------------------------------------------------------------- retyping traced words

def test_retyping_finds_which_words_a_traced_row_spells():
    from app.pipeline import classify, clean, intake, vectorize
    rules = config.rules_for("tshirt")
    img = intake.run(_picture())
    cl = clean.run(img, 101.6, 101.6, rules)
    vec = vectorize.run(cl, 101.6, 101.6, rules)
    vec["inside_color"] = cl.get("inside_color")
    c = classify.run(vec, rules, text=["GOOD DAY", "Since 1984 Knoxville"], letters="retype")
    found = sorted(r["text"] for r in c["rows"] if r["text"])
    assert found == ["GOOD DAY", "Since 1984 Knoxville"]


def test_auto_retypes_small_words_and_keeps_big_ones_as_drawn(tmp_path):
    hints = {"text": ["GOOD DAY", "Since 1984 Knoxville"]}
    auto = run_pipeline(_picture(), tmp_path / "a", "x", debug=False, hints=hints,
                        fit_w_mm=101.6, fit_h_mm=101.6)
    assert auto["lettering"]["rows"] == 2 and auto["lettering"]["retyped"] == 1
    trace = run_pipeline(_picture(), tmp_path / "t", "x", debug=False, hints=hints,
                         fit_w_mm=101.6, fit_h_mm=101.6, letters="trace")
    assert trace["lettering"]["retyped"] == 0
    words = [a for a in auto["areas"] if a.get("text")]
    assert any(a["words"] == "Since 1984 Knoxville" for a in words)


def test_no_known_words_means_nothing_is_retyped(tmp_path):
    r = run_pipeline(_picture(), tmp_path, "x", debug=False, fit_w_mm=101.6, fit_h_mm=101.6)
    assert r["lettering"]["rows"] >= 1 and r["lettering"]["retyped"] == 0
    assert any(a.get("text") and a["words"] == "" for a in r["areas"])


def test_wrong_words_are_not_forced_onto_a_row():
    glyphs = [L.outline_of(pl) for pl in L.layout("HELLO", 10.0, 0, 0)]
    assert L.fit_row(glyphs, ["XYZ 12345678 WWWWWWWWW"]) is None
    got = L.fit_row(glyphs, ["WORLD", "HELLO"])
    assert got and got["text"] == "HELLO"


# ------------------------------------------------------------------- the edits on the preview

def _word_area(r, words=None):
    cands = [a for a in r["areas"] if a.get("text") and (words is None or a["words"] == words)]
    a = max(cands, key=lambda a: a["area"])
    poly = Polygon(list(zip(a["ring"][0::2], a["ring"][1::2]))).buffer(0)
    edge = poly.exterior.interpolate(0.3, normalized=True)     # just inside the outer edge,
    c = poly.centroid                                           # so not in a letter's hole
    d = max(edge.distance(c), 1e-6)
    x, y = edge.x + (c.x - edge.x) * 0.3 / d, edge.y + (c.y - edge.y) * 0.3 / d
    return [x / r["ref"], y / r["ref"]]


def test_a_clicked_word_can_be_retyped_or_removed(tmp_path):
    base = run_pipeline(_picture(), tmp_path / "b", "x", debug=False,
                        fit_w_mm=101.6, fit_h_mm=101.6)
    at = _word_area(base)
    new = run_pipeline(_picture(), tmp_path / "n", "x", debug=False, fit_w_mm=101.6,
                       fit_h_mm=101.6, edits=[{"type": "retype", "at": at, "text": "HELLO THERE"}])
    assert new["edits_applied"] == 1
    assert any(a.get("words") == "HELLO THERE" for a in new["areas"])
    gone = run_pipeline(_picture(), tmp_path / "g", "x", debug=False, fit_w_mm=101.6,
                        fit_h_mm=101.6, edits=[{"type": "retype", "at": at, "text": ""}])
    assert gone["edits_applied"] == 1
    assert gone["stats"]["stitches"] < base["stats"]["stitches"]


def test_words_can_be_added_in_a_new_color(tmp_path):
    base = run_pipeline(_picture(), tmp_path / "b", "x", debug=False,
                        fit_w_mm=101.6, fit_h_mm=101.6)
    added = run_pipeline(_picture(), tmp_path / "a", "x", debug=False, fit_w_mm=101.6,
                         fit_h_mm=101.6,
                         edits=[{"type": "text", "text": "Team 7", "at": [0.0, -0.3],
                                 "size": 8.0 / base["ref"], "style": "bold", "hex": "#c8102e"}])
    assert added["edits_applied"] == 1
    assert added["stats"]["stitches"] > base["stats"]["stitches"]
    assert "#c8102e" in [c["hex"] for c in added["colors"]]
    # junk is skipped, never fatal
    junk = run_pipeline(_picture(), tmp_path / "j", "x", debug=False, fit_w_mm=101.6,
                        fit_h_mm=101.6, edits=[{"type": "text", "text": "", "at": [0, 0],
                                                "size": 0.1, "hex": "#000000"},
                                               {"type": "text", "text": "Hi", "at": [0, 0],
                                                "size": "big", "hex": "#000000"}])
    assert junk["edits_applied"] == 0


def test_lettering_is_sewn_left_to_right(tmp_path):
    r = run_pipeline(_picture(), tmp_path, "x", debug=False, fit_w_mm=101.6, fit_h_mm=101.6,
                     edits=[{"type": "text", "text": "ABCDEFG", "at": [0.0, -0.3],
                             "size": 0.1, "style": "block", "hex": "#00aa00"}])
    block = next(b for b in r["blocks"] if b["hex"] == "#00aa00")
    pts = np.array([p for seg in block["segments"] for p in zip(seg[0::2], seg[1::2])])
    xs = pts[:, 0]
    # it starts at the left end, finishes at the right end, and moves steadily rightward
    assert xs[0] < xs.min() + 0.1 * np.ptp(xs) and xs[-1] > xs.max() - 0.15 * np.ptp(xs)
    assert np.corrcoef(np.arange(len(xs)), xs)[0, 1] > 0.9


def test_solid_blobs_in_a_row_are_not_taken_for_words():
    # three window-like squares side by side are not the word "EST"
    from shapely.geometry import box
    windows = [box(x, 0, x + 2.0, 3.0) for x in (0, 3, 6)]
    assert L.fit_row(windows, ["EST. 1794"]) is None


def test_punctuation_is_kept_beside_its_letter():
    glyphs = [L.outline_of(pl) for pl in L.layout("EST 1794", 6.0, 0, 0)]
    pls = L.place_on_glyphs("EST. 1794,", glyphs)
    assert [pl.ch for pl in pls] == list("EST.1794,")
    dot = pls[3]
    assert pls[2].bx < dot.bx < pls[4].bx


def test_thin_original_lettering_stays_a_fine_line():
    # letters traced with 0.6 mm strokes are retyped as bean lines, not chunky 1 mm satin
    thin = [L.outline_of(pl) for pl in L.layout("HELLO", 7.0, 0, 0, w=0.6)]
    pls = L.fit_box("HELLO", thin)
    objs = L.objects(pls, 0, RULES)
    assert objs and all(o["kind"] == "run" for o in objs)
    bold = [L.outline_of(pl) for pl in L.layout("HELLO", 7.0, 0, 0, w=1.3)]
    assert all(o["kind"] == "satin" for o in L.objects(L.fit_box("HELLO", bold), 0, RULES))


def test_satin_letters_need_room_for_their_strokes():
    # under 4 mm letters are sewn as lines; from 4 mm a narrow satin, thin enough that the
    # letters stay open (both counters of a B still show)
    assert all(o["kind"] == "run" for o in L.objects(L.layout("AB", 3.5, 0, 0), 0, RULES))
    narrow = L.objects(L.layout("B", 4.5, 0, 0), 0, RULES)
    assert all(o["kind"] == "satin" and o["width_mm"] <= 0.95 for o in narrow)
    b = unary_union([o["outline"] for o in narrow])
    assert sum(len(g.interiors) for g in getattr(b, "geoms", [b])) == 2
    assert all(o["kind"] == "satin" for o in L.objects(L.layout("AB", 6.5, 0, 0), 0, RULES))


def test_letter_insides_do_not_cost_the_design_a_color(tmp_path):
    # the blank insides of big letters are left open, so asking again for the design's own
    # number of colors must not merge black into blue to make room for them
    first = run_pipeline(_picture(small_px=90), tmp_path / "a", "x", debug=False,
                         fit_w_mm=254, fit_h_mm=254)
    n = first["stats"]["colors"]
    again = run_pipeline(_picture(small_px=90), tmp_path / "b", "x", debug=False,
                         fit_w_mm=254, fit_h_mm=254, max_colors=n)
    assert [c["hex"] for c in again["colors"]] == [c["hex"] for c in first["colors"]]


def test_the_dot_of_an_i_stays_clear_of_its_stem():
    for w in (1.0, 2.4):                              # light and heavy strokes
        pl = L.Placement("i", 0.0, 20.0, 10.0, w)
        stem, dot = [L.column(s["pts"], w, s["cut0"], s["cut1"])["outline"]
                     for s in L.letter_strokes(pl)]
        assert stem.distance(dot) >= 0.5 * w


def test_mid_size_letters_are_a_narrow_satin_not_a_bean():
    # 4-6 mm letters: a narrow satin reads bolder than a bean, which sinks into textured fabric
    mid = L.objects(L.layout("Hi", 5.0, 0, 0), 0, RULES)
    assert mid and all(o["kind"] == "satin" for o in mid)
    assert all(0.8 <= o["width_mm"] <= 1.2 for o in mid)
    # the setting can turn it off again
    off = L.objects(L.layout("Hi", 5.0, 0, 0), 0, {**RULES, "text_narrow_satin_mm": 6.0})
    assert off and all(o["kind"] == "run" for o in off)


def test_very_large_letters_stay_satin_and_are_split_when_sewn():
    from app.pipeline import stitch
    pls = L.layout("I", 60.0, 0, 0, w=9.0)
    objs = L.objects(pls, 0, RULES)
    assert objs and all(o["kind"] == "satin" for o in objs)      # not turned into fill
    segs = stitch.stitch_satin(objs[0], RULES)
    pts = np.asarray(segs[0])
    longest = np.linalg.norm(np.diff(pts, axis=0), axis=1).max()
    assert longest <= RULES["satin_max_mm"]                         # no long loose stitches
