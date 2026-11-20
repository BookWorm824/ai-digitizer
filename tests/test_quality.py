"""Tests for Phase 3, stitch quality: satin columns, underlay, pull compensation, color
overlap, fabric presets, hat order, towel knockdown, and the checks with one-click fixes."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union

from app import config
from app.api.server import app
from app.pipeline import classify, run_pipeline, sequence, stitch, validate
from app.pipeline.common import dist

IMAGES = Path(__file__).parent / "images"
RULES = config.rules_for("tshirt")


def _vec(*colors):
    """Layers from lists of polygons, one list per color."""
    return {"layers": [{"color": i, "rgb": (i * 60, 0, 0), "polys": list(polys)}
                       for i, polys in enumerate(colors)]}


def _kinds(*polys, rules=RULES):
    return sorted(o["kind"] for o in classify.run(_vec(polys), rules)["objects"])


def _badge():
    return (IMAGES / "logo_badge.png").read_bytes()


# ---------------------------------------------------------------- which stitch a shape gets

def test_a_narrow_bar_becomes_one_satin_column():
    objs = classify.run(_vec([box(0, 0, 30, 3)]), RULES)["objects"]
    assert [o["kind"] for o in objs] == ["satin"]
    col = objs[0]
    assert col["width_mm"] == pytest.approx(3.0, abs=0.35)
    length = float(np.linalg.norm(np.diff(col["center"], axis=0), axis=1).sum())
    assert length == pytest.approx(30, abs=1.5)            # carried right to both ends
    assert col["outline"].intersection(box(0, 0, 30, 3)).area > 0.9 * 90


@pytest.mark.parametrize("shape, kinds", [
    (box(0, 0, 30, 30), ["fill"]),                                    # wide: fill
    (Point(0, 0).buffer(3), ["fill"]),                                # a dot is not a column
    (box(0, 0, 30, 0.7), ["run"]),                                    # thin: running stitch
    (box(0, 0, 30, 1.0), ["satin"]),                                  # a 1 mm line is satin
    (Point(0, 0).buffer(20).difference(Point(0, 0).buffer(17)), ["satin"]),   # 3 mm ring
])
def test_width_decides_the_stitch_type(shape, kinds):
    assert _kinds(shape) == kinds


def test_a_lumpy_shape_is_filled_not_forced_into_a_column():
    dumbbell = Point(0, 0).buffer(3).union(Point(26, 0).buffer(3)).union(box(0, -0.9, 26, 0.9))
    assert "satin" not in _kinds(dumbbell)                            # two blobs on a stick


def test_letter_strokes_become_several_columns():
    t_shape = box(0, 0, 14, 2.5).union(box(5.75, 0, 8.25, 16))        # a capital T
    objs = classify.run(_vec([t_shape]), RULES)["objects"]
    assert {o["kind"] for o in objs} == {"satin"} and len(objs) >= 2
    cover = unary_union([o["outline"] for o in objs])
    assert cover.intersection(t_shape).area > 0.85 * t_shape.area


def test_satin_can_be_switched_off():
    assert _kinds(box(0, 0, 30, 3), rules=dict(RULES, satin=False)) == ["fill"]


# ------------------------------------------------------------------------- satin stitching

def _satin(width, rules=RULES):
    obj = classify.run(_vec([box(0, 0, 30, width)]), rules)["objects"][0]
    return obj, stitch.stitch_satin(obj, rules)[0]


def test_satin_top_zigzags_edge_to_edge_with_pull_compensation():
    obj, pts = _satin(3.0, dict(RULES, underlay=False))
    ys = np.array([p[1] for p in pts])
    assert np.all(np.sign(ys[1:] - 1.5) != np.sign(ys[:-1] - 1.5))    # alternates sides
    reach = ys.max() - ys.min()
    assert reach == pytest.approx(3.0 + RULES["pull_comp_mm"], abs=0.3)
    # peak-to-peak spacing along the column is the fill spacing
    top = [p for p in pts if p[1] > 1.5]
    gaps = np.diff(sorted(p[0] for p in top))
    assert np.median(gaps) == pytest.approx(RULES["fill_spacing_mm"], abs=0.05)


@pytest.mark.parametrize("width, expect", [(1.9, "none"), (3.0, "centre run"), (5.5, "edge run")])
def test_satin_underlay_follows_the_column_width(width, expect):
    obj, pts = _satin(width)
    _, bare = _satin(width, dict(RULES, underlay=False))
    extra = len(pts) - len(bare)
    if expect == "none":
        assert extra == 0
        return
    first = np.array(pts[:10])
    off_centre = np.abs(first[:, 1] - width / 2)
    if expect == "centre run":
        assert extra > 5 and off_centre.max() < 0.4               # starts down the middle
    else:
        assert extra > 30 and off_centre.min() > width * 0.2      # starts along an edge


# ------------------------------------------------------------- fills: underlay, pull, overlap

def test_fill_rows_are_lengthened_for_pull_compensation():
    poly = box(0, 0, 30, 20)
    plain = stitch.fill_segments(poly, 0, 0.4, 3.5)[0]
    longer = stitch.fill_segments(poly, 0, 0.4, 3.5, extend=0.2)[0]
    assert min(p[0] for p in plain) == pytest.approx(0, abs=1e-6)
    assert min(p[0] for p in longer) == pytest.approx(-0.2, abs=1e-6)
    assert max(p[0] for p in longer) == pytest.approx(30.2, abs=1e-6)
    assert min(p[1] for p in longer) >= 0                           # only along the stitches


def test_fill_gets_an_edge_run_then_underlay_then_top():
    obj = {"kind": "fill", "color": 0, "poly": box(0, 0, 30, 30), "area_mm2": 900.0}
    seg = stitch.stitch_fill(obj, RULES, 0.0)
    assert len(seg) == 1
    first = seg[0][:20]
    inset = RULES["underlay_inset_mm"]
    ring = box(inset, inset, 30 - inset, 30 - inset).exterior
    assert all(ring.distance(Point(p)) < 0.05 for p in first)       # starts with the edge run
    bare = stitch.stitch_fill(obj, dict(RULES, underlay=False), 0.0)
    assert len(seg[0]) > len(bare[0]) + 50


@pytest.mark.parametrize("fabric, edge, spacing", [
    ("tshirt", True, 2.0), ("hoodie", True, 2.5), ("polo", True, 3.0), ("denim", True, 0.0)])
def test_fabric_sets_how_much_underlay(fabric, edge, spacing):
    r = config.rules_for(fabric)
    assert r["underlay_edge_run"] is edge and r["underlay_spacing_mm"] == spacing


def test_a_color_reaches_under_the_color_sewn_after_it():
    big, small = box(0, 0, 40, 40), box(40, 10, 50, 30)             # touching along x = 40
    layers = _vec([big], [small])["layers"]
    order = classify.sewing_order(layers)
    assert order == [0, 1]                                          # biggest area first
    shapes = classify.underlap(layers, order, 0.4)
    assert shapes[0][0].bounds[2] == pytest.approx(40.4, abs=0.02)  # under the later color
    assert shapes[0][0].bounds[1] == pytest.approx(0, abs=1e-6)     # nowhere else
    assert shapes[1][0].equals(small)                               # the top color is untouched


def test_travel_inside_a_shape_prefers_ground_not_yet_filled():
    poly = box(0, 0, 40, 40).difference(Point(20, 20).buffer(9))    # square with a hole
    inside = stitch.prep(poly.buffer(0.15))
    router = stitch._Router(poly, inside)
    rows = [(i, 0.5 + i * 0.5, 0.0, 40.0) for i in range(18)]        # the top band is "sewn"
    router.mark_filled(rows)
    path = router((2.0, 2.0), (38.0, 2.0))                           # both ends in filled ground
    filled_band = box(0, 0, 40, 9)
    line = LineString(path)
    assert line.length > 0 and poly.buffer(0.6).contains(line)
    # with unfilled ground far away the route may cross the band, but it hugs the edge
    edge_zone = poly.exterior.buffer(1.6)
    assert line.intersection(filled_band).difference(edge_zone).length < 0.35 * line.length


# ------------------------------------------------------------------------- softer / denser

def test_softer_uses_fewer_stitches_and_denser_uses_more(tmp_path):
    counts = {}
    for d in (-2, 0, 2):
        r = run_pipeline(_badge(), tmp_path / str(d), "x", density=d, debug=False)
        counts[d] = r["stats"]["stitches"]
        assert r["density"] == d and r["files"]["pes"]["ok"]
    assert counts[-2] < counts[0] < counts[2]


# --------------------------------------------------------------------------- hats, towels

def _plan(objects, rules, order=None):
    stitched = stitch.run({"objects": objects}, rules)
    return sequence.run(stitched, [(0, 0, 0), (200, 0, 0)], rules, order)


def test_hats_are_sewn_from_the_centre_outward():
    rules = config.rules_for("hat")
    assert rules["cap_mode"]
    boxes = [{"kind": "fill", "color": 0, "poly": box(x, 0, x + 8, 8), "area_mm2": 64.0}
             for x in (-44, -4, 36)]                                 # left, centre, right
    plan = _plan(boxes, rules)
    xs = [np.mean([p[0] for p in sg]) for sg in plan["blocks"][0]["segments"]]
    assert abs(xs[0]) < 5                                            # the centre one first
    flat = _plan(boxes, config.rules_for("tshirt"))
    first = np.mean([p[0] for p in flat["blocks"][0]["segments"][0]])
    assert abs(first) > 20                                           # normally it isn't


def test_towels_get_a_flattening_layer_first(tmp_path):
    r = run_pipeline((IMAGES / "logo_blocks.png").read_bytes(), tmp_path, "t",
                     fabric_id="towel", debug=False)
    assert r["colors"][0]["label"].startswith("Flattening layer")
    assert r["stats"]["colors"] == 4                                 # not counted as a color
    assert all(c["label"] is None for c in r["colors"][1:])
    assert any("matches the towel" in n for n in r["fabric"]["notes"])
    plain = run_pipeline((IMAGES / "logo_blocks.png").read_bytes(), tmp_path / "p", "t",
                         fabric_id="hoodie", debug=False)
    assert plain["colors"][0]["label"] is None


@pytest.mark.parametrize("fabric", sorted(config.fabrics()))
def test_every_fabric_preset_makes_a_good_file(fabric, tmp_path):
    hat = config.fabrics()[fabric].get("max_height_mm")
    r = run_pipeline(_badge(), tmp_path, "x", fabric_id=fabric, debug=False,
                     fit_w_mm=101.6 if hat else 88.9, fit_h_mm=57.15 if hat else 88.9)
    assert r["check"]["status"] != "red"
    assert r["files"]["pes"]["ok"] and r["files"]["dst"]["ok"]
    assert r["fabric"]["stabilizer"]


def test_more_fabrics_are_marked_for_the_extra_list():
    fabrics = TestClient(app).get("/api/options").json()["fabrics"]
    main = [f["id"] for f in fabrics if not f["more"]]
    assert main == ["tshirt", "hoodie", "polo", "hat", "towel", "denim"]
    assert {"fleece", "leather", "neoprene", "delicate", "beanie", "softhat"} <= {
        f["id"] for f in fabrics if f["more"]}


# ------------------------------------------------------------------ checks with one-click fixes

def _fix(result, label_start):
    for m in result["check"]["messages"]:
        for f in m["fixes"]:
            if f["label"].startswith(label_start):
                return f["set"]
    raise AssertionError(f"no fix starting with {label_start!r}: {result['check']['messages']}")


def test_too_big_for_any_hoop_offers_make_it_fit_and_the_fix_works(tmp_path):
    data = (IMAGES / "logo_blocks.png").read_bytes()
    big = run_pipeline(data, tmp_path / "a", "x", fit_w_mm=320, fit_h_mm=320, debug=False)
    assert big["check"]["status"] == "red"
    size = _fix(big, "Make it fit")["size_in"]
    side = size * 25.4
    fixed = run_pipeline(data, tmp_path / "b", "x", fit_w_mm=side, fit_h_mm=side, debug=False)
    assert fixed["check"]["hoop"] is not None
    assert not any(m["level"] == "red" for m in fixed["check"]["messages"])


def test_tall_hat_design_offers_a_size_that_fits_the_hat(tmp_path):
    tall = run_pipeline(_badge(), tmp_path / "a", "x", fabric_id="hat", debug=False)
    size = _fix(tall, "Make it fit the hat")["size_in"]
    side = size * 25.4
    fixed = run_pipeline(_badge(), tmp_path / "b", "x", fabric_id="hat", debug=False,
                         fit_w_mm=side, fit_h_mm=side)
    assert fixed["stats"]["height_mm"] <= 57.5
    assert not any("Hat designs" in m["text"] for m in fixed["check"]["messages"])


def test_big_solid_areas_on_a_tshirt_offer_two_fixes(tmp_path):
    r = run_pipeline(_badge(), tmp_path, "x", debug=False)
    assert _fix(r, "Make it smaller")["size_in"] < r["stats"]["size_in"]
    assert _fix(r, "Sew it lighter") == {"density": -1}


def test_small_text_offers_make_it_bigger(tmp_path):
    hints = {"text": ["TEST"], "smallest_text_pct": 4}               # 4% of ~79 mm = 3 mm
    r = run_pipeline(_badge(), tmp_path, "x", hints=hints, debug=False)
    msg = next(m for m in r["check"]["messages"] if "text is too small" in m["text"])
    assert msg["fixes"][0]["set"]["size_in"] > r["stats"]["size_in"]
    ok = run_pipeline(_badge(), tmp_path / "ok", "x", debug=False,
                      hints={"text": ["TEST"], "smallest_text_pct": 12})
    assert not any("text is too small" in m["text"] for m in ok["check"]["messages"])


def test_largest_size_respects_the_hoop_margin():
    hoops = config.hoops()
    size = validate.largest_size_in(100, 100, hoops, 5)
    assert size == 7.25                                              # 8x8 hoop, 190 mm inside
    assert validate.pick_hoop(size * 25.4, size * 25.4, hoops, 5) is not None


# ----------------------------------------------------------------------------- through the API

def _wait(client, url):
    for _ in range(400):
        job = client.get(url).json()
        if job["status"] != "working":
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_size_and_density_can_be_changed_on_the_stitch_screen():
    c = TestClient(app)
    cid = c.post("/api/cleanup", files={"file": ("logo.png", _badge(), "image/png")}).json()["id"]
    _wait(c, f"/api/cleanup/{cid}")
    base = _wait(c, "/api/jobs/" + c.post(
        "/api/jobs", data={"cleanup_id": cid, "version": "local"}).json()["id"])["result"]
    small = _wait(c, "/api/jobs/" + c.post(
        "/api/jobs", data={"cleanup_id": cid, "version": "local", "size_in": 2.5,
                           "density": -1}).json()["id"])["result"]
    assert base["stats"]["size_in"] == pytest.approx(3.5, abs=0.02)
    assert small["stats"]["size_in"] == pytest.approx(2.5, abs=0.02) and small["density"] == -1
    assert small["stats"]["stitches"] < base["stats"]["stitches"]
    assert all("fixes" in m for m in base["check"]["messages"])


# ---------------------------------------------------------------------------- overall quality

def test_real_patch_style_art_gets_a_satin_border(tmp_path):
    """A badge with a thick outline all round: the outline must come out as satin, and every
    stitch must still obey the length rules."""
    from PIL import Image, ImageDraw
    import io
    img = Image.new("RGB", (1200, 1200), "white")
    d = ImageDraw.Draw(img)
    d.ellipse([150, 150, 1050, 1050], fill=(10, 10, 10))
    d.ellipse([190, 190, 1010, 1010], fill=(20, 40, 110))            # outline about 3 mm at 3.5 in
    buf = io.BytesIO()
    img.save(buf, "PNG")
    r = run_pipeline(buf.getvalue(), tmp_path, "patch", debug=False)
    assert r["stats"]["satin_columns"] >= 1
    assert r["files"]["pes"]["ok"] and r["check"]["status"] != "red"
    lengths = [dist((s[i], s[i + 1]), (s[i + 2], s[i + 3]))
               for b in r["blocks"] for s in b["segments"] for i in range(0, len(s) - 3, 2)]
    assert max(lengths) <= RULES["max_stitch_mm"] + 0.02
    assert min(lengths) >= RULES["min_stitch_mm"] - 0.02


def test_chatgpt_patch_art_sews_with_satin_and_no_short_stitches(tmp_path):
    """A real ChatGPT "Bold" redraw of the sample badge, kept as a test picture."""
    r = run_pipeline((IMAGES / "ai_bold_badge.png").read_bytes(), tmp_path, "bold", debug=False)
    assert r["stats"]["satin_columns"] >= 3 and r["stats"]["colors"] == 4   # white is sewn
    assert r["files"]["pes"]["ok"] and r["files"]["dst"]["ok"]
    assert not any("short" in m["text"] or "too long" in m["text"] for m in r["check"]["messages"])


def test_outline_colors_are_sewn_after_the_areas_they_surround():
    inner = Point(0, 0).buffer(15)
    outline = Point(0, 0).buffer(30).difference(Point(0, 0).buffer(27))   # bigger area, but narrow
    layers = _vec([outline], [inner])["layers"]
    assert outline.area < inner.area * 1.2 and classify.sewing_order(layers) == [1, 0]
    thick = Point(0, 0).buffer(60).difference(Point(0, 0).buffer(56))     # far more area, still narrow
    assert classify.sewing_order(_vec([thick], [inner])["layers"]) == [1, 0]


def test_a_color_never_shows_through_a_thin_outline():
    area = box(0, 0, 30, 30)
    outline = box(30, 0, 31, 30)                                    # a 1 mm line beside it
    layers = _vec([area], [outline])["layers"]
    shapes = classify.underlap(layers, [0, 1], 0.4)
    reach = shapes[0][0].bounds[2] - 30
    assert 0.1 < reach <= 0.5                                       # under it, not out the far side
    hair = box(30, 0, 30.4, 30)                                     # a 0.4 mm line: no room at all
    shapes = classify.underlap(_vec([area], [hair])["layers"], [0, 1], 0.4)
    assert shapes[0][0].bounds[2] == pytest.approx(30, abs=0.02)


def test_thread_never_crosses_open_ground_between_shapes():
    """Two areas of one color with a 2 mm gap of bare fabric between them (like the two
    sides of a cut-out letter): the thread must be trimmed, not dragged across the gap."""
    left = {"kind": "fill", "color": 0, "poly": box(0, 0, 10, 10), "area_mm2": 100.0}
    right = {"kind": "fill", "color": 0, "poly": box(12, 0, 22, 10), "area_mm2": 100.0}
    plan = _plan([left, right], RULES)
    gap = box(10.5, -1, 11.5, 11)
    for sg in plan["blocks"][0]["segments"]:
        for a, b in zip(sg, sg[1:]):
            assert not LineString([a, b]).crosses(gap)
    assert plan["trims"] == 1


def _sewn(color, pts, shape):
    return {"kind": "fill", "color": color, "segments": [pts], "reversible": False,
            "shape": shape, "area_mm2": float(shape.area)}


def test_thread_may_cross_a_gap_only_if_a_later_color_covers_it():
    left = _sewn(0, [(0, 0), (5, 0), (10, 0)], box(0, -1, 10, 1))
    right = _sewn(0, [(12, 0), (17, 0), (22, 0)], box(12, -1, 22, 1))       # 2 mm gap between
    palette = [(0, 0, 0), (200, 0, 0)]
    bare = sequence.run({"objects": [left, right]}, palette, RULES, [0])
    assert len(bare["blocks"][0]["segments"]) == 2 and bare["trims"] == 1   # open ground: trim
    patch = _sewn(1, [(9, -1), (13, -1), (13, 1), (9, 1)], box(9, -1, 13, 1))
    covered = sequence.run({"objects": [left, right, patch]}, palette, RULES, [0, 1])
    assert [b["color"] for b in covered["blocks"]] == [0, 1]
    assert len(covered["blocks"][0]["segments"]) == 1                       # hidden later: join


def test_white_lettering_inside_a_design_is_sewn_not_left_as_holes(tmp_path):
    """White letters on a red banner, on a white page: the letters are part of the design."""
    from PIL import Image, ImageDraw, ImageFont
    import io
    img = Image.new("RGB", (1200, 600), "white")
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([60, 120, 1140, 480], radius=40, fill=(200, 20, 30))
    try:
        font = ImageFont.truetype("arialbd.ttf", 260)
    except OSError:
        pytest.skip("needs the Arial Bold font")
    d.text((600, 300), "TEST", font=font, fill="white", anchor="mm")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    r = run_pipeline(buf.getvalue(), tmp_path, "t", debug=False)
    hexes = [c["hex"] for c in r["colors"]]
    assert len(hexes) == 2 and any(h.startswith("#f") or h == "#ffffff" for h in hexes)
    assert r["stats"]["satin_columns"] >= 4                     # the letter strokes
    assert r["files"]["pes"]["ok"]


def test_satin_stays_solid_on_the_outside_of_a_tight_curve():
    ring = Point(0, 0).buffer(4.5).difference(Point(0, 0).buffer(1.5))   # 3 mm wide, tight
    obj = classify.run(_vec([ring]), RULES)["objects"][0]
    assert obj["kind"] == "satin"
    pts = stitch.stitch_satin(obj, dict(RULES, underlay=False))[0]
    outer = sorted((np.arctan2(y, x) for x, y in pts if np.hypot(x, y) > 3.8))
    gaps = np.diff(outer) * 4.5                                # arc length between stitch ends
    assert np.median(gaps) <= RULES["fill_spacing_mm"] * 1.15  # no fanning open
    reach = [np.hypot(x, y) for x, y in pts if np.hypot(x, y) < 3.0]
    assert max(reach) - min(reach) > 0.4                       # some stitches stop short inside
