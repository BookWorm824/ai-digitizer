"""Rings and round borders (training session: monograms, rings and borders). A ring is its
own object, a true circle at a fixed width; a letter of the same color crossing it does not
break it or merge into it; a ring covered by another color stays broken there; a hairline
ring is a run, a wider one a satin no narrower than the narrowest satin."""

from __future__ import annotations

from shapely.geometry import Point, box
from shapely.ops import unary_union

from app.pipeline import rings
from app.pipeline.common import polygons_of


def _ring(r1, r2):
    return Point(0, 0).buffer(r2, 256).difference(Point(0, 0).buffer(r1, 256))


def _monogram():
    thick = _ring(18.0, 20.0)
    thin = _ring(16.6, 17.1)
    letter = box(8.0, -4.0, 19.0, 4.0)                 # a stroke reaching across both rings
    return list(polygons_of(unary_union([thick, thin, letter])))


def test_rings_are_found_whole_where_a_letter_of_their_color_crosses_them():
    found = rings.find_rings(_monogram())
    widths = sorted(round(r["r2"] - r["r1"], 1) for r in found)
    assert len(found) == 2 and widths[1] >= 1.8
    assert all(r["arcs"] is None for r in found)


def test_the_letter_is_cut_free_of_the_rings_and_tucked_under_them():
    polys = _monogram()
    found = rings.find_rings(polys)
    rest = rings.split(polys, found)
    assert rest
    letter = max(rest, key=lambda p: p.area)
    assert letter.bounds[0] < 9.0                       # the letter itself is kept
    assert not any(p.area > 5 and p.bounds[1] < -10 for p in rest)   # no ring left in it


def test_a_ring_covered_by_another_color_stays_broken():
    ring = _ring(18.0, 20.0).difference(box(-30, -3, 30, 3))    # a banner across it
    found = rings.find_rings(list(polygons_of(ring)))
    assert found and found[0]["arcs"] is not None and len(found[0]["arcs"]) == 2


def test_a_hairline_ring_is_a_run_and_a_wide_one_a_satin():
    found = rings.find_rings(_monogram())
    objs = rings.objects(found, 0, 1.0)
    kinds = sorted(o["kind"] for o in objs)
    assert "satin" in kinds
    for o in objs:
        if o["kind"] == "satin":
            assert o["width_mm"] >= 1.0 and o["sew_last"]


def test_a_filled_disc_is_not_a_ring():
    assert rings.find_rings([Point(0, 0).buffer(15, 256)]) == []


def test_a_monogram_in_a_close_font_is_retyped_letter_by_letter_centre_first():
    from shapely.ops import unary_union
    from app import config
    from app.pipeline import classify, lettering
    rules = config.rules_for("tshirt")
    # draw a monogram with the app's own block letters: J and R about 60 % of the M
    pls = classify._monogram_place(list("JMR"), 1, 30.0, 18.0, (0.0, 0.0, 60.0, 30.0), 3.0,
                                   "block", "middle", rules)
    art = [lettering.outline_of(pl, "block") for pl in pls]
    shapes = {0: list(polygons_of(unary_union(art)))}
    objs, keys = classify._monogram(shapes, set(), ["JMR"], [0], rules)
    assert objs and len(keys) == len(shapes[0])
    first_letter = objs[0]["seq"][1]
    assert first_letter == 0                       # the centre letter, M, is sewn first
    assert min(o["seq"][1] for o in objs) == 0 and max(o["seq"][1] for o in objs) == 2


def test_a_monogram_in_a_different_font_is_left_as_drawn():
    from app import config
    from app.pipeline import classify
    blobs = [Point(5, 9).buffer(5), Point(27, 9).buffer(14), Point(49, 9).buffer(5)]  # not letters
    objs, keys = classify._monogram({0: blobs}, set(), ["JMR"], [0], config.rules_for("tshirt"))
    assert objs == [] and not keys
