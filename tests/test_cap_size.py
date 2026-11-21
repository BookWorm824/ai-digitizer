"""Cap-size (2.25 inch) lessons: small lettering broken into pieces is still retyped, widely
spaced words keep their spacing, 1.5 mm windows survive as satin bars with no thread across
the walls between them, small solid shapes (roofs) stay solid, and mirrored lines match."""

from __future__ import annotations

from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union

from app import config
from app.pipeline import classify, lettering, sequence, stitch

RULES = config.rules_for("hat")


def _glyphs(text, H=3.0, x0=0.0):
    pls = lettering.layout(text, H, x0, 0.0, "block", w=0.6)
    return [lettering.outline_of(pl, "block") for pl in pls], pls


def test_a_piece_of_a_long_word_is_recognised():
    glyphs, _ = _glyphs("MICRO")
    assert lettering.fit_row(glyphs, ["MICROBIOLOGY"]) is None or \
        lettering.fit_row(glyphs, ["MICROBIOLOGY"])["text"] != "MICRO"
    fit = lettering.fit_row(glyphs, ["MICROBIOLOGY"], parts=True)
    assert fit and fit["text"] == "MICRO" and fit["part"]


def test_a_broken_word_is_set_whole_across_its_pieces():
    left, _ = _glyphs("MICRO", x0=0.0)
    right, _ = _glyphs("IOLOGY", x0=14.0)
    rows = [{"row": 0, "color": 0, "height": 3.0, "text": None, "glyphs": left},
            {"row": 1, "color": 0, "height": 3.0, "text": None, "glyphs": right}]
    fits = {0: lettering.fit_row(left, ["MICROBIOLOGY"], parts=True),
            1: lettering.fit_row(right, ["MICROBIOLOGY"], parts=True)}
    fits = {k: dict(v, style="block") for k, v in fits.items() if v}
    taken = classify._whole_lines(fits, rows, ["MICROBIOLOGY"])
    assert len(fits) == 1 and taken
    (fit,) = fits.values()
    assert fit["text"] == "MICROBIOLOGY" and len(fit["placements"]) == 12


def test_a_lone_piece_of_a_word_is_not_retyped():
    glyphs, _ = _glyphs("MICRO")
    rows = [{"row": 0, "color": 0, "height": 3.0, "text": None, "glyphs": glyphs}]
    fits = {0: dict(lettering.fit_row(glyphs, ["MICROBIOLOGY"], parts=True), style="block")}
    classify._whole_lines(fits, rows, ["MICROBIOLOGY"])
    assert fits == {}


def test_widely_spaced_lettering_keeps_its_spacing_not_stretched_letters():
    pls = lettering.fit_line("MICRO", (0.0, 0.0, 40.0, 3.0), 3.0, 0.6)
    assert max(pl.k for pl in pls) <= 1.25 + 1e-9
    xs = [pl.bx for pl in pls]
    assert xs[0] < 3.0 and xs[-1] + pls[-1].outer_width() > 37.0


def test_thin_traced_letters_still_match_their_word():
    # a 0.4 mm pen outline of the word covers little of the font's letters at full weight
    glyphs, pls = _glyphs("MICROBIOLOGY", H=3.0)
    thin = [g.buffer(-0.25) for g in glyphs]
    thin = [t if not t.is_empty else g for t, g in zip(thin, glyphs)]
    fit = lettering.fit_row(thin, ["TENNESSEE", "MICROBIOLOGY", "EST. 1794, KNOXVILLE"])
    assert fit and fit["text"] == "MICROBIOLOGY"


def _classify(polys, color_polys=None):
    layers = [{"color": 0, "rgb": (0, 0, 0), "polys": polys}]
    if color_polys:
        layers.append({"color": 1, "rgb": (250, 100, 0), "polys": color_polys})
    allp = unary_union(polys + (color_polys or []))
    return classify.run({"layers": layers, "bounds": allp.bounds}, RULES)


def test_small_windows_are_satin_bars_not_dropped():
    wins = [box(x, y, x + 1.5, y + 1.5) for x in (0.0, 2.2) for y in (0.0, 2.2, 4.4)]
    out = _classify(wins)
    assert out["dropped_details"] == 0
    assert [o["kind"] for o in out["objects"]] == ["satin"] * 6
    assert all(o.get("block") for o in out["objects"])


def test_no_thread_bridges_the_wall_between_windows():
    wins = [box(0.0, y, 1.5, y + 1.5) for y in (0.0, 2.2, 4.4)]
    st = stitch.run(_classify(wins), RULES)
    plan = sequence.run(st, [(0, 0, 0)], RULES)
    walls = [box(0.0, 1.5 + 0.1, 1.5, 2.2 - 0.1), box(0.0, 3.7 + 0.1, 1.5, 4.4 - 0.1)]
    for b in plan["blocks"]:
        for sg in b["segments"]:
            for p, q in zip(sg, sg[1:]):
                assert not any(LineString([p, q]).intersects(w) for w in walls)


def test_a_small_solid_roof_stays_solid():
    # an orange roof band a millimetre and a half deep beside a black line drawing
    drawing = box(-20, -10, 20, 10).difference(box(-19.4, -9.4, 19.4, 9.4))
    roof = Polygon([(2, 2), (10, 2), (9, 3.5), (3, 3.5)])
    out = _classify([drawing], [roof])
    kinds = [o["kind"] for o in out["objects"] if o["color"] == 1]
    assert kinds and all(k in ("satin", "fill") for k in kinds)


def test_hairline_holes_from_the_overlap_are_not_openings():
    p = box(0, 0, 8, 1.5).difference(box(1, 0.7, 7, 0.7005))
    assert len(p.interiors) == 1
    assert not classify._no_slivers(p).interiors


def test_mirrored_lines_are_sewn_alike():
    objs = [
        {"kind": "run", "color": 1, "line": LineString([(-14, 16), (-2, 16)]),
         "route": [[(-14, 16), (-2, 16)]], "width_mm": 1.0, "satin_w": [1.15]},
        {"kind": "run", "color": 1, "line": LineString([(2, 16), (14, 16)]),
         "route": [[(2, 16), (14, 16)]], "width_mm": 0.9},
    ]
    classify._match_mirror_lines(objs, (-20, 0, 20, 20), 1.1)
    assert bool(objs[0].get("satin_w")) == bool(objs[1].get("satin_w"))


def test_bold_lettering_sews_small_outline_letters_bold():
    pls = lettering.layout("EST", 3.5, 0, 0, "block")
    for pl in pls:
        pl.heavy = True
    objs = lettering.objects(pls, 0, RULES, "block", lettering.text_style(3.5), group=0, row=0)
    runs = [o for o in objs if o["kind"] == "run" and not o.get("punct")]
    assert runs and all(o["bold"] for o in runs)
    plain = lettering.objects(lettering.layout("EST", 3.5, 0, 0, "block"), 0, RULES, "block",
                              lettering.text_style(3.5), group=0, row=0)
    assert not any(o["bold"] for o in plain if o["kind"] == "run")


def test_a_word_on_a_cap_is_sewn_whole_from_its_centre_letter_outward():
    pls = lettering.layout("TENNESSEE", 7.0, 0.0, 0.0, "block")
    objs = lettering.objects(pls, 0, RULES, "block", lettering.text_style(7.0), group=0, row=0)
    st = stitch.run({"objects": objs, "order": [0]}, RULES)
    plan = sequence.run(st, [(250, 100, 0)], RULES, [0])
    # one color block, and few cuts: the word is not split across the centre bands (sewn
    # centre-out, each letter of the left half is still cut: they are planned left to right)
    assert len(plan["blocks"]) == 1
    assert plan["trims"] <= 5
    xs = [x for o in st["objects"] for sg in o["segments"] for x, _y in sg]
    letters = sequence._centre_out([o for o in st["objects"] if o.get("seq")],
                                   (min(xs) + max(xs)) / 2.0)
    firsts = []
    for o in letters:
        if not firsts or firsts[-1] != o["seq"][1]:
            firsts.append(o["seq"][1])
    assert firsts[0] == 4 and firsts[:5] == [4, 3, 2, 1, 0] and firsts[5:] == [5, 6, 7, 8]
