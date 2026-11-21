"""Tests for the deep-dive rules (reports/Embroidery digitizing deep dive.md): clean geometry,
line art as continuous two-pass tours, running stitch preferred for thin lines, halo removal,
fill angles and travel that is only hidden under later stitching."""

from __future__ import annotations

import io
import math

import numpy as np
import pytest
from PIL import Image, ImageDraw
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union

from app import config
from app.pipeline import geometry, lettering, linenet, run_pipeline, stitch
from app.pipeline import classify

RULES = config.rules_for("tshirt")


def _jagged_rect(w=20.0, h=10.0, noise=0.08, seed=1):
    """A rectangle outline with pixel-like noise, the way a trace comes back."""
    rng = np.random.default_rng(seed)
    pts = []
    for (x0, y0), (x1, y1) in zip([(0, 0), (w, 0), (w, h), (0, h)], [(w, 0), (w, h), (0, h), (0, 0)]):
        n = 40
        for k in range(n):
            t = k / n
            pts.append((x0 + (x1 - x0) * t + rng.uniform(-noise, noise),
                        y0 + (y1 - y0) * t + rng.uniform(-noise, noise)))
    return Polygon(pts)


# -------------------------------------------------------------------------------- geometry

def test_a_wobbly_traced_rectangle_comes_back_straight_and_square():
    clean = geometry.clean_polygons([_jagged_rect()])[0]
    coords = list(clean.exterior.coords)[:-1]
    assert len(coords) == 4                                    # four corners, nothing else
    for (x0, y0), (x1, y1) in zip(coords, coords[1:] + coords[:1]):
        assert min(abs(x1 - x0), abs(y1 - y0)) < 1e-6          # every side exactly level/upright
    assert clean.area == pytest.approx(200.0, rel=0.03)


def test_curves_stay_smooth_with_few_points():
    circle = Polygon([(10 * math.cos(t), 10 * math.sin(t))
                      for t in np.linspace(0, 2 * math.pi, 400, endpoint=False)])
    clean = geometry.clean_polygons([circle])[0]
    assert clean.area == pytest.approx(circle.area, rel=0.02)
    assert len(clean.exterior.coords) < 120                    # far fewer points than the trace


# ------------------------------------------------------------------------------ line networks

def _grid_paths():
    """A window-like grid of centre lines, split at its junctions as a skeleton is, with
    slightly crooked, noisy edges."""
    rng = np.random.default_rng(3)
    paths = []
    for y in (0.0, 5.0, 10.0):
        for x in (0.0, 5.0):
            paths.append(np.array([(x, y), (x + 2.5, y + rng.uniform(-0.05, 0.05)), (x + 5.0, y)]))
    for x in (0.0, 5.0, 10.0):
        for y in (0.0, 5.0):
            paths.append(np.array([(x, y), (x + rng.uniform(-0.05, 0.05), y + 2.5), (x, y + 5.0)]))
    return paths


def test_line_art_is_one_thread_with_every_line_sewn_twice():
    routes = linenet.network(_grid_paths(), [0.0, 90.0])
    assert len(routes) == 1                                    # one continuous thread, no trims
    route = routes[0]
    total = linenet.length(route)
    drawn = unary_union([LineString(p) for p in route]).length
    assert total == pytest.approx(2 * drawn, rel=0.01)         # two passes everywhere
    assert np.allclose(route[0][0], route[-1][-1])             # ends where it began


def test_line_art_lines_are_straightened_and_lined_up():
    routes = linenet.network(_grid_paths(), [0.0, 90.0])
    for poly in routes[0]:
        (x0, y0), (x1, y1) = poly[0], poly[-1]
        assert min(abs(x1 - x0), abs(y1 - y0)) < 1e-6          # dead level or upright


def test_the_return_pass_uses_the_same_needle_holes():
    routes = linenet.network([np.array([(0.0, 0.0), (12.0, 0.0)])], [0.0, 90.0])
    pts = linenet.route_points(routes[0], stitch.run_points, RULES)
    half = len(pts) // 2
    there, back = pts[:half + 1], pts[half:]
    assert np.allclose(np.array(there), np.array(back[::-1]))


def test_loose_stubs_go_but_long_lines_made_of_short_pieces_stay():
    pieces = [np.array([(x, 0.0), (x + 0.5, 0.0)]) for x in np.arange(0, 10, 0.5)]
    stub = [np.array([(5.0, 0.0), (5.0, 0.4)])]                # a 0.4 mm whisker
    routes = linenet.network(pieces + stub, [0.0, 90.0])
    drawn = unary_union([LineString(p) for p in routes[0]])
    assert drawn.length == pytest.approx(10.0, abs=0.3)        # the line survives whole
    assert drawn.bounds[3] < 0.1                               # the whisker is gone


# ---------------------------------------------------------------------------- running stitch

def test_thin_lines_are_running_stitch_not_skinny_satin():
    kinds = [o["kind"] for o in classify.run(
        {"layers": [{"color": 0, "rgb": (0, 0, 0), "polys": [box(0, 0, 30, 1.3)]}],
         "bounds": (0, 0, 30, 1.3)}, RULES)["objects"]]
    assert kinds == ["run"]


def _bar(width):
    return {"layers": [{"color": 0, "rgb": (0, 0, 0), "polys": [box(0, 0, 30, width)]}],
            "bounds": (0, 0, 30, width)}


def test_bold_lines_are_a_narrow_satin_over_their_own_run():
    # production files use satin from about 1 mm and almost no stacked running stitch
    objs = classify.run(_bar(1.2), RULES)["objects"]
    assert objs[0]["kind"] == "run" and objs[0]["satin_w"]
    pts = np.asarray(stitch.run({"objects": objs, "order": [0]}, RULES)["objects"][0]["segments"][0])
    d = np.diff(pts, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    turn = np.abs((np.diff(ang) + math.pi) % (2 * math.pi) - math.pi)
    zig = turn > math.radians(140)                       # the satin on the way back
    assert zig.sum() > 100
    across = np.abs(d[1:][zig, 1])
    assert 1.0 <= np.median(across) <= 1.7               # about the drawn width
    # one thread: out as a run (the underlay), back as the satin, ending where it began
    assert math.dist(pts[0], pts[-1]) < 2.0


def test_thin_lines_stay_running_stitch_and_pile_fabric_gets_four_passes():
    thin = classify.run(_bar(0.6), RULES)["objects"]
    assert thin[0]["kind"] == "run" and not thin[0].get("satin_w")
    towel = config.rules_for("towel")
    objs = classify.run(_bar(1.2), towel)["objects"]
    assert objs[0]["bold"] and not objs[0].get("satin_w")
    sewn = stitch.run({"objects": objs, "order": [0]}, towel)["objects"]
    pts = next(o for o in sewn if o["kind"] == "run")["segments"][0]
    travelled = sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))
    assert travelled == pytest.approx(4 * 30, rel=0.1)          # out once, back as a bean


def test_a_web_of_outlines_has_one_line_weight_and_crowded_lines_stay_thin():
    from app.pipeline.classify import _web_weights
    from shapely.geometry import LineString as LS
    walls = [np.array([(0, 0), (40, 0)], float), np.array([(40, 0), (40, 30)], float),
             np.array([(40, 30), (0, 30)], float), np.array([(0, 30), (0, 0)], float)]
    inner = [np.array([(18, 0), (18, 8)], float), np.array([(19.3, 0), (19.3, 8)], float)]
    route = walls + inner
    piece = unary_union([LS(w).buffer(0.6, cap_style=2) for w in walls]
                        + [LS(i).buffer(0.55, cap_style=2) for i in inner])
    w = _web_weights(route, piece, 1.1, 0.8)
    assert len({round(x, 3) for x in w[:4]}) == 1 and w[0] >= 1.1     # one weight for the walls
    assert min(w[4:]) == 0.0                                          # two lines 1.3 mm apart: one thins


def test_running_stitches_never_go_shorter_than_about_one_and_a_quarter_mm():
    wiggle = LineString([(x, 3.0 * math.sin(x * 0.4)) for x in np.linspace(0, 30, 300)])
    pts = stitch.run_points(wiggle, RULES)
    steps = [math.dist(a, b) for a, b in zip(pts, pts[1:])]
    assert min(steps[:-1]) >= 1.2


# ---------------------------------------------------------------------- clean-up and lettering

def test_soft_grey_edges_on_a_see_through_picture_are_not_a_thread_color(tmp_path):
    img = Image.new("RGBA", (900, 600), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for k in range(8):                                         # black lines with grey halos
        y = 60 + k * 60
        d.line([(60, y), (840, y)], fill=(170, 170, 170, 255), width=14)
        d.line([(60, y), (840, y)], fill=(10, 10, 10, 255), width=8)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    r = run_pipeline(buf.getvalue(), tmp_path, "x", debug=False)
    assert [c["hex"] for c in r["colors"]] == ["#0a0a0a"] or len(r["colors"]) == 1


def test_serif_lettering_is_recognised():
    serif_i = box(-0.9, 0, 0.9, 10).union(box(-2.5, 0, 2.5, 1.0)).union(box(-2.5, 9.0, 2.5, 10))
    sans_i = box(-0.9, 0, 0.9, 10)
    assert lettering.has_serifs([serif_i] * 3)
    assert not lettering.has_serifs([sans_i] * 3)


def test_fill_angles_stay_close_to_level():
    for poly in (box(0, 0, 40, 30), box(0, 0, 10, 60)):
        a = stitch.best_fill_angle(poly, stitch.FILL_ANGLES[0], 0.4)
        assert a in stitch.FILL_ANGLES                          # 18 or 162 deg, never 45


def test_split_long_staggers_needle_points_across_a_wide_satin():
    from app.pipeline.stitch import split_long
    # four crossings of a 10 mm column
    pts = [(0, 0), (10, 0.4), (0, 0.8), (10, 1.2), (0, 1.6)]
    out = split_long(pts, 7.0)
    seg = np.linalg.norm(np.diff(np.asarray(out), axis=0), axis=1)
    assert seg.max() <= 7.0
    mids = [x for x, _y in out if 0 < x < 10]
    assert len(set(round(m, 2) for m in mids)) >= 2               # staggered, not one line


def test_light_texture_travel_keeps_to_the_edge():
    from shapely.geometry import box
    from shapely.prepared import prep
    from app.pipeline.stitch import _Router, _travel
    sq = box(0, 0, 30, 30)
    r = _Router(sq, prep(sq.buffer(0.15)), light=True)
    walk = _travel((5, 15), (25, 15), prep(sq.buffer(0.15)), r, 2.0)
    assert walk                                                    # found a hidden way
    # it goes round by the edge, not straight across the open middle
    assert min(min(x, 30 - x, y, 30 - y) for x, y in walk) < 1.5
    assert not any(10 < x < 20 and 10 < y < 20 for x, y in walk)


# ------------------------------------------------- round three: what the close-ups showed

def test_a_nearly_symmetric_drawing_is_made_exactly_symmetric():
    rng = np.random.default_rng(1)
    left = np.array([[-10, 0], [-10, 8], [-6, 8], [-6, 3], [-3, 3], [-3, 12], [-8, 12]], float)
    right = left * [-1, 1]
    axis = np.array([[0, 0], [0, 12]], float)
    nodes = np.vstack([left + rng.uniform(-0.25, 0.25, left.shape), right, axis + [0.2, 0]]) + [40, 5]
    out, twin, c = linenet.symmetrize(nodes)
    assert twin is not None and abs(c - 40) < 0.3
    n = len(left)
    for i in range(n):                                     # each pair mirrors exactly
        assert abs((out[i, 0] - c) + (out[n + i, 0] - c)) < 1e-6
        assert abs(out[i, 1] - out[n + i, 1]) < 1e-6
    assert np.allclose(out[2 * n:, 0], c)                  # and the middle sits on the axis
    # a lopsided drawing is left alone
    lop = np.vstack([left, left + [3, 20], axis])
    same, twin2, _ = linenet.symmetrize(lop)
    assert twin2 is None and np.allclose(same, lop)


def test_short_arcs_of_a_small_circle_stay_curved():
    t = np.linspace(0, math.pi / 2, 40)
    arc = np.column_stack([2.5 * np.cos(t), 2.5 * np.sin(t)])     # a quarter of a 5 mm circle
    pieces = geometry.fit(arc, False, tol=0.3)
    assert any(p["kind"] == "curve" for p in pieces)
    wall = np.column_stack([np.linspace(0, 30, 200), 0.1 * np.sin(np.linspace(0, 20, 200))])
    assert [p["kind"] for p in geometry.fit(wall, False, tol=0.3)] == ["line"]


def test_small_circles_are_sewn_with_stitches_short_enough_to_look_round():
    circle = LineString([(2.5 * math.cos(a), 2.5 * math.sin(a))
                         for a in np.linspace(0, 2 * math.pi, 90)])
    pts = np.asarray(stitch.run_points(circle, dict(RULES, run_stitch_mm=2.0)))
    d = np.diff(pts, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    turn = np.abs((np.diff(ang) + math.pi) % (2 * math.pi) - math.pi)
    assert math.degrees(turn.max()) <= 31
    assert np.linalg.norm(d, axis=1).min() >= 0.95


def test_fill_rows_never_link_across_open_fabric():
    # an N: widely spaced rows straddle the notch between the stem and the diagonal
    n = Polygon([(0, 0), (4, 0), (10, 9), (10, 0), (14, 0), (14, 14), (10, 14), (4, 5),
                 (4, 14), (0, 14)])
    for spacing in (2.5, 0.45):
        for angle in (18.0, 108.0, 162.0):
            for seg in stitch.fill_segments(n, angle, spacing, 4.0):
                for a, b in zip(seg, seg[1:]):
                    if math.dist(a, b) > 1.0:
                        assert n.buffer(0.35).contains(LineString([a, b])), (spacing, angle, a, b)


def test_a_small_round_dot_is_one_satin_column():
    from shapely.geometry import Point
    from app.pipeline.classify import _dot
    dot = _dot(Point(5, 5).buffer(1.5), 7.0)
    assert dot is not None and abs(float(dot["half"].max()) - 1.5) < 0.1
    assert _dot(box(0, 0, 3, 3).buffer(0), 7.0) is None           # a square is not a dot
    assert _dot(Point(0, 0).buffer(6.0), 7.0) is None             # too big for one satin


def _house(garment):
    img = Image.new("RGB", (500, 400), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([60, 120, 440, 340], outline="black", width=14)
    d.line([250, 120, 250, 340], fill="black", width=14)
    d.rectangle([100, 170, 190, 290], fill="black")
    d.rectangle([310, 170, 400, 290], fill="black")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue(), garment


def test_blank_areas_are_sewn_only_when_the_garment_is_another_color(tmp_path):
    art, _ = _house(None)
    kw = dict(fit_w_mm=150, fit_h_mm=150, debug=False)
    on_white = run_pipeline(art, tmp_path / "w", "w", garment="#eeece6", **kw)
    on_navy = run_pipeline(art, tmp_path / "n", "n", garment="#1f2a44", **kw)
    assert on_white["stats"]["colors"] == 1 and not on_white["inside"]["sewn"]
    assert on_navy["stats"]["colors"] == 2 and on_navy["inside"]["sewn"]
    # the white is the ground: sewn first, as one piece under the middle wall
    first = on_navy["blocks"][0]
    assert first["hex"].lower() in ("#ffffff", "#fefefe", "#fdfdfd") or sum(first["rgb"]) > 700
    assert len(first["segments"]) <= 2


def test_color_around_knocked_out_letters_is_one_fill_not_satin_scraps(tmp_path):
    img = Image.new("RGB", (700, 300), "white")
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([20, 60, 680, 240], radius=25, fill=(200, 40, 50))
    for k, x in enumerate((250, 330, 410)):               # three blocky "letters" cut out
        d.rectangle([x, 100, x + 50, 200], fill="white")
        d.rectangle([x + 18, 120 + 20 * k, x + 50, 150 + 10 * k], fill=(200, 40, 50))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    r = run_pipeline(buf.getvalue(), tmp_path, "banner", fit_w_mm=90, fit_h_mm=90, debug=False,
                     garment="#eeece6")
    assert r["stats"]["satin_columns"] == 0
    assert r["stats"]["colors"] == 1
