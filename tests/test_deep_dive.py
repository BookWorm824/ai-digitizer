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


def test_bold_lines_get_four_passes():
    objs = classify.run({"layers": [{"color": 0, "rgb": (0, 0, 0), "polys": [box(0, 0, 30, 1.2)]}],
                         "bounds": (0, 0, 30, 1.2)}, RULES)["objects"]
    assert objs[0]["bold"]
    pts = stitch.run({"objects": objs, "order": [0]}, RULES)["objects"][0]["segments"][0]
    travelled = sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))
    assert travelled == pytest.approx(4 * 30, rel=0.1)          # out once, back as a bean


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
