"""Tests for every pipeline stage, plus the whole thing end to end.

Run:  .venv\\Scripts\\python.exe -m pytest
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from shapely.geometry import LineString, Point, Polygon, box

import make_images
from app import config
from app.pipeline import (PipelineError, classify, clean, export, intake, run_pipeline, sequence,
                          stitch, validate, vectorize)
from app.pipeline.common import BG, dist

IMAGES = Path(__file__).parent / "images"
RULES = config.rules_for("tshirt")
FIT = 88.9
ALL_IMAGES = ["logo_badge.png", "logo_badge_transparent.png", "logo_blocks.png",
              "logo_blocks.jpg", "logo_lines.png"]


@pytest.fixture(scope="session", autouse=True)
def _images():
    if not all((IMAGES / n).exists() for n in ALL_IMAGES):
        make_images.main()


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _near(rgb, target, tol=14):
    return max(abs(a - b) for a, b in zip(rgb, target)) <= tol


def _stitch_lengths(segments):
    return [dist(a, b) for sg in segments for a, b in zip(sg, sg[1:])]


# ------------------------------------------------------------------------------ 1. intake

def test_intake_rejects_things_that_are_not_pictures():
    with pytest.raises(PipelineError):
        intake.run(b"this is not a picture")


def test_intake_rejects_blank_picture():
    with pytest.raises(PipelineError):
        intake.run(_png(Image.new("RGB", (300, 300), "white")))


def test_intake_crops_to_design_and_upscales_small_pictures():
    img = Image.new("RGB", (400, 400), "white")
    img.paste((200, 30, 30), (150, 180, 250, 220))        # small red bar in the middle
    out = intake.run(_png(img))
    h, w = out["rgb"].shape[:2]
    assert out["alpha"] is None and out["bg_rgb"] is not None
    assert w > h * 2                                       # cropped to the wide bar
    assert max(h, w) < 800                                 # and not the full upscaled frame


def test_intake_keeps_transparency():
    out = intake.run((IMAGES / "logo_badge_transparent.png").read_bytes())
    assert out["alpha"] is not None and (out["alpha"] < 128).any()


# ------------------------------------------------------------------------------- 4. clean

@pytest.mark.parametrize("name", ["logo_blocks.png", "logo_blocks.jpg"])
def test_clean_finds_the_real_colors(name):
    c = clean.run(intake.run((IMAGES / name).read_bytes()), FIT, FIT, RULES)
    assert c["has_background"]
    assert len(c["palette"]) == 4                          # no extra edge-blend colors
    for target in (make_images.NAVY, make_images.GOLD, make_images.RED, make_images.WHITE):
        assert any(_near(p, target) for p in c["palette"])
    assert c["labels"][2, 2] == BG                         # the white surround is not sewn
    h, w = c["labels"].shape
    assert c["labels"][h // 2, w // 2] != BG               # the white disc inside the ring is
    assert c["inside"] == {"present": True, "sewn": True}


def test_line_art_keeps_its_insides_open():
    """A circle drawn in outline is not a solid white disc."""
    c = clean.run(intake.run((IMAGES / "logo_lines.png").read_bytes()), FIT, FIT, RULES)
    assert len(c["palette"]) == 1 and c["inside"] == {"present": True, "sewn": False}
    forced = clean.run(intake.run((IMAGES / "logo_lines.png").read_bytes()), FIT, FIT,
                       dict(RULES, sew_enclosed_background=True))
    assert len(forced["palette"]) == 2                      # but the user can ask for it


def test_enclosed_background_can_be_left_unsewn():
    rules = dict(RULES, sew_enclosed_background=False)
    c = clean.run(intake.run((IMAGES / "logo_blocks.png").read_bytes()), FIT, FIT, rules)
    h, w = c["labels"].shape
    assert len(c["palette"]) == 3 and c["labels"][h // 2, w // 2] == BG


def test_clean_respects_max_colors():
    rules = dict(RULES, max_colors=2)
    c = clean.run(intake.run((IMAGES / "logo_blocks.png").read_bytes()), FIT, FIT, rules)
    assert len(c["palette"]) <= 2


def test_clean_removes_specks():
    img = Image.new("RGB", (900, 900), "white")
    img.paste((20, 20, 120), (200, 200, 700, 700))
    img.paste((255, 255, 255), (450, 450, 453, 453))       # 3 px pinhole: far under 1 mm
    c = clean.run(intake.run(_png(img)), FIT, FIT, RULES)
    v = vectorize.run(c, FIT, FIT, RULES)
    polys = [p for layer in v["layers"] for p in layer["polys"]]
    assert len(polys) == 1 and len(polys[0].interiors) == 0


# --------------------------------------------------------------------------- 5. vectorize

def test_vectorize_scales_to_the_requested_size():
    c = clean.run(intake.run((IMAGES / "logo_blocks.png").read_bytes()), FIT, FIT, RULES)
    v = vectorize.run(c, FIT, FIT, RULES)
    minx, miny, maxx, maxy = v["bounds"]
    assert maxx - minx == pytest.approx(FIT - RULES["pull_comp_mm"], abs=0.05)
    assert maxy - miny <= FIT
    assert (minx + maxx) / 2 == pytest.approx(0, abs=0.01)   # centred
    ring = [p for layer in v["layers"] for p in layer["polys"] if p.interiors]
    assert ring, "the red ring should be traced with its hole"


# ---------------------------------------------------------------------------- 6. classify

def _vec(*polys):
    return {"layers": [{"color": 0, "rgb": (0, 0, 0), "polys": list(polys)}]}


def test_wide_shape_is_a_fill():
    objs = classify.run(_vec(box(0, 0, 20, 20)), RULES)["objects"]
    assert [o["kind"] for o in objs] == ["fill"]


def test_thin_line_is_a_run_along_its_middle():
    objs = classify.run(_vec(box(0, 0, 30, 0.7)), RULES)["objects"]
    assert [o["kind"] for o in objs] == ["run"]
    line = objs[0]["line"]
    assert line.length == pytest.approx(29.3, abs=1.5)
    assert all(abs(y - 0.35) < 0.2 for _, y in line.coords)
    assert objs[0]["bean"]                                 # 0.7 mm wide: bold enough for triple


def test_shape_with_a_thin_tail_is_split():
    shape = box(0, 0, 15, 15).union(box(15, 7, 40, 7.7))   # square with a 0.7 mm tail
    kinds = sorted(o["kind"] for o in classify.run(_vec(shape), RULES)["objects"])
    assert kinds == ["fill", "run"]


@pytest.mark.parametrize("width", [0.7, 0.9, 1.0, 1.15, 1.5])
def test_line_near_the_width_limit_is_not_chopped_up(width):
    """A ring whose width sits around the thin-line limit is sewn one way all the way round."""
    ring = Point(0, 0).buffer(30).difference(Point(0, 0).buffer(30 - width))
    objs = classify.run(_vec(ring), RULES)["objects"]
    assert 1 <= len(objs) <= 2 and len({o["kind"] for o in objs}) == 1
    assert objs[0]["kind"] == ("run" if width < 0.85 else "satin")


def test_too_small_details_are_dropped_and_counted():
    out = classify.run(_vec(box(0, 0, 0.5, 0.5)), RULES)
    assert out["objects"] == [] and out["dropped_details"] == 1


def test_line_art_becomes_runs_only():
    c = clean.run(intake.run((IMAGES / "logo_lines.png").read_bytes()), FIT, FIT, RULES)
    objs = classify.run(vectorize.run(c, FIT, FIT, RULES), RULES)["objects"]
    assert objs and all(o["kind"] in ("run", "satin") for o in objs)   # lines, never fills


# ------------------------------------------------------------------------------ 7. stitch

def test_running_stitch_spacing_and_ends():
    line = LineString([(0, 0), (20, 0), (20, 10)])
    pts = stitch.run_points(line, RULES)
    assert pts[0] == (0, 0) and pts[-1] == (20, 10)
    assert (20.0, 0.0) in [(round(x, 6), round(y, 6)) for x, y in pts]   # corner kept
    assert max(_stitch_lengths([pts])) <= RULES["run_stitch_mm"] + 1e-6


def test_running_stitch_shortens_on_tight_curves():
    circle = LineString(Point(0, 0).buffer(2, quad_segs=32).exterior.coords)
    lengths = _stitch_lengths([stitch.run_points(circle, RULES)])
    assert max(lengths) <= RULES["run_stitch_curve_mm"] + 1e-6


def test_bean_stitch_goes_over_each_stitch_three_times():
    pts = [(0, 0), (2, 0), (4, 0)]
    assert stitch.bean(pts) == [(0, 0), (2, 0), (0, 0), (2, 0), (4, 0), (2, 0), (4, 0)]


@pytest.mark.parametrize("angle", [0, 45, 135])
def test_fill_stays_inside_and_covers_the_shape(angle):
    poly = box(0, 0, 30, 20)
    segs = stitch.fill_segments(poly, angle, 0.4, 3.5)
    assert len(segs) == 1                                  # a plain box needs no jumps
    near = poly.buffer(0.05)
    assert all(near.contains(Point(p)) for p in segs[0])
    assert max(_stitch_lengths(segs)) <= 3.5 + 0.9
    # rows every 0.4 mm => thread laid down is about area / spacing
    thread = sum(_stitch_lengths(segs))
    assert thread == pytest.approx(poly.area / 0.4, rel=0.12)


def test_fill_never_sews_across_a_hole():
    poly = box(0, 0, 40, 40).difference(Point(20, 20).buffer(8))
    hole_core = Point(20, 20).buffer(7)
    for sg in stitch.fill_segments(poly, 45, 0.4, 3.5):
        for a, b in zip(sg, sg[1:]):
            assert not LineString([a, b]).intersects(hole_core)


def test_fill_object_gets_underlay_then_top():
    obj = {"kind": "fill", "color": 0, "poly": box(0, 0, 25, 25), "area_mm2": 625.0}
    with_under = stitch.stitch_fill(obj, RULES, 45.0)
    without = stitch.stitch_fill(obj, dict(RULES, underlay=False), 45.0)
    n = lambda segs: sum(len(s) for s in segs)             # noqa: E731
    assert n(with_under) > n(without)
    assert len(with_under) == 1                            # joined: no jump between the layers


# ---------------------------------------------------------------------------- 8. sequence

def _plan_for(objects, palette=((0, 0, 0), (255, 0, 0))):
    stitched = stitch.run({"objects": objects}, RULES)
    return stitched, sequence.run(stitched, list(palette), RULES)


def test_sequence_enforces_stitch_length_limits_and_ties_off():
    _, plan = _plan_for([{"kind": "fill", "color": 0, "poly": box(0, 0, 30, 30), "area_mm2": 900.0}])
    lengths = _stitch_lengths(plan["blocks"][0]["segments"])
    assert min(lengths) >= RULES["min_stitch_mm"] - 1e-6
    assert max(lengths) <= RULES["max_stitch_mm"] + 1e-6
    seg = plan["blocks"][0]["segments"][0]
    assert seg[0] == seg[2] and seg[-1] == seg[-3]         # lock stitches at both ends


def test_sequence_orders_big_fills_first_and_lines_last():
    objs = [
        {"kind": "run", "color": 0, "line": LineString([(0, 0), (30, 0)]), "bean": False},
        {"kind": "fill", "color": 1, "poly": box(0, 5, 30, 30), "area_mm2": 750.0},
    ]
    _, plan = _plan_for(objs)
    assert [b["color"] for b in plan["blocks"]] == [1, 0]
    assert plan["color_changes"] == 1


def test_sequence_trims_long_jumps_only():
    objs = [
        {"kind": "run", "color": 0, "line": LineString([(0, 0), (10, 0)]), "bean": False},
        {"kind": "run", "color": 0, "line": LineString([(11, 0), (20, 0)]), "bean": False},   # 1 mm hop
        {"kind": "run", "color": 0, "line": LineString([(40, 0), (50, 0)]), "bean": False},   # 20 mm hop
    ]
    _, plan = _plan_for(objs)
    assert plan["trims"] == 1


# ---------------------------------------------------------------------------- 9. validate

def test_hoop_choice():
    hoops = config.hoops()
    assert validate.pick_hoop(88, 80, hoops, 5)["id"] == "4x4"
    assert validate.pick_hoop(160, 100, hoops, 5)["id"] == "5x7"     # fits turned sideways
    assert validate.pick_hoop(254, 254, hoops, 5) is None


def test_validator_flags_a_design_too_big_for_any_hoop(tmp_path):
    r = run_pipeline((IMAGES / "logo_blocks.png").read_bytes(), tmp_path, "big",
                     fit_w_mm=320, fit_h_mm=320, debug=False)
    assert r["check"]["status"] == "red" and r["check"]["hoop"] is None


# ------------------------------------------------------------------------------ 10. export

def test_export_files_read_back_with_the_same_stitches(tmp_path):
    stitched, plan = _plan_for([
        {"kind": "fill", "color": 0, "poly": box(0, 0, 30, 30), "area_mm2": 900.0},
        {"kind": "fill", "color": 1, "poly": box(40, 0, 60, 20), "area_mm2": 400.0},
    ])
    files = export.write_files(plan, tmp_path, "two_boxes")
    assert set(files) == {"pes", "dst"}
    for info in files.values():
        assert info["ok"]
        assert info["stitches"] == plan["stitch_count"]
        assert Path(info["path"]).stat().st_size > 100


# ------------------------------------------------------------------------------ end to end

@pytest.mark.parametrize("name", ALL_IMAGES)
def test_every_test_image_makes_a_good_file_with_defaults(name, tmp_path):
    r = run_pipeline((IMAGES / name).read_bytes(), tmp_path, Path(name).stem)
    assert r["check"]["status"] != "red"
    assert r["check"]["hoop"] == "4×4"
    assert r["stats"]["width_mm"] <= FIT + 0.1 and r["stats"]["height_mm"] <= FIT + 0.1
    assert 1 <= r["stats"]["colors"] <= RULES["max_colors"]
    assert r["stats"]["stitches"] > 300
    for fmt in ("pes", "dst"):
        assert r["files"][fmt]["ok"]
        assert r["files"][fmt]["stitches"] == r["stats"]["stitches"]
    for dbg in ("01_intake", "04_clean", "05_vectorize", "06_classify", "07_stitch", "10_preview"):
        assert (tmp_path / "debug" / f"{dbg}.png").exists()


def test_same_picture_gives_the_same_stitches_every_time(tmp_path):
    data = (IMAGES / "logo_badge.png").read_bytes()
    a = run_pipeline(data, tmp_path / "a", "x", debug=False)
    b = run_pipeline(data, tmp_path / "b", "x", debug=False)
    assert a["blocks"] == b["blocks"]


def test_works_at_other_sizes_and_fabrics(tmp_path):
    data = (IMAGES / "logo_badge.png").read_bytes()
    hat = run_pipeline(data, tmp_path / "hat", "hat", fit_w_mm=101.6, fit_h_mm=57.15,
                       fabric_id="hat", debug=False)
    assert hat["stats"]["height_mm"] <= 57.2 and hat["check"]["status"] != "red"
    small = run_pipeline(data, tmp_path / "small", "small", fit_w_mm=40, fit_h_mm=40, debug=False)
    assert small["files"]["pes"]["ok"] and small["stats"]["width_mm"] <= 40.1
