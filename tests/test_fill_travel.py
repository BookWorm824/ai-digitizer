"""Thread walks inside a fill never lie across its own finished rows (they show on top as a
line): the walk is cut instead, sections are chosen so the thread reaches them over unsewn
ground, and a fill whose usual angle leaves cuts tries level rows (a band of knocked-out
text). Also: big circles are sewn as true circles."""

from __future__ import annotations

import math

from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from app import config
from app.pipeline import linenet, sequence, stitch

RULES = config.rules_for("tshirt")


def _star(cx, cy, r_out, r_in):
    pts = []
    for k in range(10):
        r = r_out if k % 2 == 0 else r_in
        a = math.radians(90 + 36 * k)
        pts.append((cx + r * math.cos(a), cy - r * math.sin(a)))
    return Polygon(pts)


def _disc_with_star():
    return Point(0, 0).buffer(30, 64).difference(_star(0, -5, 12, 5))


def _walks_over_finished_rows(poly, angle, segments, spacing):
    """mm of thread laid across rows already sewn (steps not along the row direction)."""
    sewn = []
    total = 0.0
    for sg in segments:
        for a, b in zip(sg, sg[1:]):
            t = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180.0
            off = min(abs(t - angle), 180.0 - abs(t - angle))
            if off > 25 and math.dist(a, b) > 1.2:
                seg = LineString([a, b])
                if any(seg.distance(LineString(r)) < 0.3 for r in sewn[:-40]):
                    total += seg.length
            sewn.append((a, b))
    return total


def test_no_walk_across_the_finished_rows_of_a_fill_with_a_star_hole():
    poly = _disc_with_star()
    segs = stitch.fill_segments(poly, 18.0, 0.22, 3.5)
    assert _walks_over_finished_rows(poly, 18.0, segs, 0.22) < 3.0


def test_the_edge_of_a_hole_is_not_a_travel_path():
    poly = _disc_with_star()
    router = stitch._Router(poly, None)
    tip = _star(0, -5, 12, 5).exterior.coords
    assert router.along_edge(tip[0], tip[4]) is None


def test_a_band_of_knocked_out_text_tries_level_rows_and_needs_few_cuts():
    band = box(-45, -8, 45, 8)
    for x in (-30, -12, 6, 24):          # four block letters knocked out of the band
        band = band.difference(box(x, -5, x + 9, 5).difference(box(x + 3, -2, x + 9, 2)))
    obj = {"kind": "fill", "color": 0, "poly": band}
    segs = stitch.stitch_fill(obj, RULES, 18.0)
    cut_pieces = [sg for sg in segs if isinstance(sg, stitch.Cut)]
    assert len(cut_pieces) <= 1


def test_a_cut_piece_of_a_fill_is_not_rejoined_across_its_own_rows():
    poly = box(0, 0, 40, 20)
    a = [(0.0, 0.0), (40.0, 0.0), (40.0, 10.0)]
    b = stitch.Cut([(0.0, 18.0), (40.0, 18.0)])
    st = {"objects": [{"kind": "fill", "color": 0, "segments": [a, b], "reversible": False,
                       "shape": poly, "area_mm2": poly.area, "seq": None, "punct": False,
                       "text_row": None, "alt": None, "block": False, "thin": False}]}
    plan = sequence.run(st, [(20, 40, 160)], RULES)
    assert plan["trims"] == 1


def test_a_big_wobbly_circle_is_sewn_as_a_true_circle():
    pts = [(40 * math.cos(t) + 0.15 * math.sin(7 * t), 40 * math.sin(t))
           for t in [k * math.pi / 60 for k in range(61)]]
    arc = linenet._circle_arc(pts)
    assert arc is not None
    assert max(abs(math.hypot(x, y) - 40) for x, y in arc) < 0.2


def _band_with_letter(stroke):
    letter = box(0, 0, stroke, 8).union(box(-3, 8 - stroke, 3 + stroke, 8))    # a T
    band = box(-20, -4, 20, 12)
    return band.difference(letter), letter


def test_small_thin_letters_are_sewn_on_top_of_an_unbroken_fill():
    from app.pipeline import classify
    band, letter = _band_with_letter(1.0)
    layers = [{"color": 0, "polys": [band]}, {"color": 1, "polys": [letter]}]
    out = classify.sew_on_top({0: [band], 1: [letter]}, layers, [0, 1])
    assert len(out[0]) == 1 and not out[0][0].interiors


def test_big_letters_and_letters_showing_the_fabric_stay_knocked_out():
    from app.pipeline import classify
    band, letter = _band_with_letter(3.0)                   # strokes well over a millimetre
    layers = [{"color": 0, "polys": [band]}, {"color": 1, "polys": [letter]}]
    out = classify.sew_on_top({0: [band], 1: [letter]}, layers, [0, 1])
    assert out[0][0].interiors
    band, letter = _band_with_letter(1.0)                   # no thread in the hole: fabric
    layers = [{"color": 0, "polys": [band]}]
    out = classify.sew_on_top({0: [band]}, layers, [0])
    assert out[0][0].interiors


def _c_ring(r_in=3.5, r_out=7.5, cut_y=-4.0):
    ring = Point(0, 0).buffer(r_out, 64).difference(Point(0, 0).buffer(r_in, 64))
    return ring.difference(box(-20, -20, 20, cut_y).union(
        Polygon([(-20, cut_y), (20, cut_y + 3), (20, -20), (-20, -20)])))


def test_a_narrow_ring_is_one_satin_column_not_pieces_meeting_in_fans():
    from app.pipeline import classify
    cols = classify.columns(_c_ring())
    assert len(cols) == 1


def test_a_corner_still_splits_a_column():
    from app.pipeline import classify
    ell = box(0, 0, 3, 20).union(box(0, 0, 20, 3))
    assert len(classify.columns(ell)) == 2


def test_what_satin_columns_miss_at_a_slanted_end_is_filled():
    from app.pipeline import classify
    cols = classify.try_satin(_c_ring(), config.rules_for("tshirt"))
    assert cols
    cover = cols[0]["outline"]
    for c in cols:
        cover = cover.union(c["outline"])
    for c in cols:
        for patch in c.get("patches", ()):
            cover = cover.union(patch)
    assert _c_ring().difference(cover.buffer(0.05)).area < 0.8


def test_no_pull_comp_on_a_satin_edge_under_a_later_outline():
    col = {"kind": "satin", "color": 0, "center": [(0, 0), (10, 0)], "half": [1.5, 1.5],
           "left": [(0, 1.5), (10, 1.5)], "right": [(0, -1.5), (10, -1.5)]}
    full = stitch.stitch_satin(col, RULES)
    covered = stitch.stitch_satin(dict(col, comp_left=0.0), RULES)
    top = lambda segs: max(y for sg in segs for _x, y in sg)
    assert top(covered) < top(full) and abs(top(covered) - 1.5) < 0.05


def test_a_solid_badge_with_a_shape_cut_out_is_filled_not_a_satin_ring():
    from app.pipeline import classify
    badge = box(0, 0, 9, 9).buffer(1.0).buffer(-1.0)
    badge = badge.difference(Polygon([(2.0, 2.0), (7.0, 2.0), (4.5, 7.0)]))
    assert classify.try_satin(badge, config.rules_for("hat")) is None


def test_an_even_ring_is_still_satin():
    from app.pipeline import classify
    ring = Point(0, 0).buffer(8, 64).difference(Point(0, 0).buffer(6, 64))
    assert classify.try_satin(ring, config.rules_for("hat"))


def test_short_detail_lines_of_a_redrawn_drawing_are_runs_and_structure_is_satin():
    from app.pipeline import classify
    route = [[(0.0, 0.0), (10.0, 0.0)], [(10.0, 0.0), (10.0, 1.5)], [(10.0, 1.5), (20.0, 1.5)]]
    piece = unary_union([LineString(seg).buffer(0.5) for seg in route])
    redrawn = classify._web_weights(route, piece, 1.1, 0.8, None, 1.0)
    assert redrawn[0] > 0 and redrawn[2] > 0 and redrawn[1] == 0.0   # the 1.5 mm bar is a run
    rebuilt = classify._web_weights(route, piece, 1.1, 0.8, None, 0.85)
    assert rebuilt[1] > 0                                             # a rebuilt one keeps it


def test_at_a_joint_one_line_goes_through_and_the_others_end_under_its_edge():
    import math as _m
    route = [[(-10.0, 0.0), (0.0, 0.0)], [(0.0, 0.0), (10.0, 0.0)], [(0.0, 0.0), (0.0, 10.0)]]
    widths = [1.2, 1.2, 1.2]
    satin = stitch._outline_satin(route, widths, RULES)
    stem = satin(2, route[2])
    bar = satin(0, route[0])
    assert stem and bar
    # the stem stops just under the bar's edge (0.6 mm half width less a 0.3 mm tuck)...
    assert min(abs(y) for _x, y in stem) > 0.2
    # ... while the bar runs right through the joint
    assert max(x for x, _y in bar) >= 0.0


def test_a_thin_outline_round_a_colored_shape_is_satin_and_covers_its_edge():
    from app.pipeline import classify
    letter = box(0, 0, 10, 12)
    outline = box(-0.65, -0.65, 10.65, 12.65).difference(box(0, 0, 10, 12))    # 0.65 mm
    layers = [{"color": 0, "rgb": (240, 180, 30), "polys": [letter]},
              {"color": 1, "rgb": (20, 36, 90), "polys": [outline]}]
    out = classify.run({"layers": layers, "bounds": outline.bounds}, config.rules_for("hat"))
    navy = [o for o in out["objects"] if o["color"] == 1]
    assert navy and any(o["kind"] == "satin" or o.get("satin_w") for o in navy)


def test_a_thin_free_line_stays_a_run():
    from app.pipeline import classify
    ring = box(-0.65, -0.65, 10.65, 12.65).difference(box(0, 0, 10, 12))    # nothing inside
    layers = [{"color": 1, "rgb": (20, 36, 90), "polys": [ring]}]
    out = classify.run({"layers": layers, "bounds": ring.bounds}, config.rules_for("hat"))
    assert out["objects"] and all(o["kind"] == "run" and not any(o.get("satin_w") or [])
                                  for o in out["objects"])


def test_an_inner_line_inside_another_colors_shape_keeps_its_light_weight():
    from app.pipeline import classify
    face = Point(0, 0).buffer(20, 64)
    muzzle = Point(0, -6).buffer(6, 64)
    line = Point(0, -6).buffer(6.6, 64).difference(muzzle)                 # 0.6 mm, in the face
    layers = [{"color": 0, "rgb": (140, 82, 40), "polys": [face.difference(muzzle.buffer(0.6))]},
              {"color": 1, "rgb": (226, 182, 130), "polys": [muzzle]},
              {"color": 2, "rgb": (20, 20, 20), "polys": [line]}]
    out = classify.run({"layers": layers, "bounds": face.bounds}, config.rules_for("hat"))
    black = [o for o in out["objects"] if o["color"] == 2]
    assert black and not any(o["kind"] == "satin" or any(o.get("satin_w") or []) for o in black)
