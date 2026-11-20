"""Tests for ChatGPT helping to digitize (aiplan.py), with a stand-in for ChatGPT: the plan
is followed part by part, foliage becomes a soft texture, the review can change a part, and
nothing breaks when the AI can't be used."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw

from app import config
from app.pipeline import aiplan, run_pipeline, stitch, vectorize, clean, intake

RULES = config.rules_for("tshirt")


def _picture():
    """A green 'tree' drawn as a dense scribble web, a blue block and a black line."""
    img = Image.new("RGB", (1000, 700), "white")
    d = ImageDraw.Draw(img)
    rng = np.random.default_rng(4)
    for _ in range(160):
        x, y = rng.uniform(80, 380), rng.uniform(80, 420)
        d.line([(x, y), (x + rng.uniform(-40, 40), y + rng.uniform(-40, 40))], fill=(20, 110, 40), width=5)
    d.rectangle([560, 120, 900, 420], fill=(30, 60, 160))
    d.line([(80, 600), (900, 600)], fill=(10, 10, 10), width=6)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _vec():
    rules = config.rules_for("tshirt")
    cl = clean.run(intake.run(_picture()), 150, 150, rules)
    return vectorize.run(cl, 150, 150, rules)


def _part_of(vec, rgb_pick):
    parts = aiplan.parts_of(vec)
    return next(p for p in parts if rgb_pick(p["rgb"]))


def test_parts_are_numbered_and_drawn_on_a_map():
    vec = _vec()
    parts = aiplan.parts_of(vec)
    assert 2 <= len(parts) <= aiplan.MAX_PARTS
    assert [p["id"] for p in parts] == list(range(1, len(parts) + 1))
    png, _frame = aiplan.plan_map(vec, parts)
    assert Image.open(io.BytesIO(png)).size[0] == aiplan.MAP_PX


def test_the_plan_is_followed_and_its_notes_are_shown(tmp_path, fake_ai):
    vec = _vec()
    blue = _part_of(vec, lambda c: c[2] > 120 and c[0] < 100)
    fake = fake_ai()
    fake.plan = {"parts": [{"id": blue["id"], "role": "shading", "stitch": "omit", "angle": -1}],
                 "foliage": [], "notes": ["The blue box is only shading, so it is left out."]}
    without = run_pipeline(_picture(), tmp_path / "a", "x", fit_w_mm=150, fit_h_mm=150, debug=False)
    with_ai = run_pipeline(_picture(), tmp_path / "b", "x", fit_w_mm=150, fit_h_mm=150, debug=False,
                           ai_assist=True)
    assert with_ai["ai_digitize"]["used"]
    assert with_ai["ai_digitize"]["notes"] == ["The blue box is only shading, so it is left out."]
    assert with_ai["stats"]["colors"] == without["stats"]["colors"] - 1
    assert with_ai["stats"]["stitches"] < without["stats"]["stitches"]


def test_foliage_becomes_one_soft_textured_mass(tmp_path, fake_ai):
    fake = fake_ai()
    fake.plan = {"parts": [], "notes": [],
                 "foliage": [{"x0": 0, "y0": 0, "x1": 45, "y1": 75}]}
    r = run_pipeline(_picture(), tmp_path, "x", fit_w_mm=150, fit_h_mm=150, debug=False,
                     ai_assist=True)
    plain = run_pipeline(_picture(), tmp_path / "plain", "x", fit_w_mm=150, fit_h_mm=150,
                         debug=False)
    runs = sum(1 for a in r["areas"] if a["kind"] == "run")
    before = sum(1 for a in plain["areas"] if a["kind"] == "run")
    # the scribbles are now one mass, not dozens of separate lines (a few ends that poke out
    # of the box, and the black ground line, stay lines)
    assert runs <= 8 < before
    assert any(a["kind"] == "fill" and a["area"] > 300 for a in r["areas"])


def test_texture_fill_has_a_soft_random_edge():
    from shapely.geometry import Point
    blob = Point(0, 0).buffer(12)
    segs = stitch.stitch_texture({"poly": blob, "angle": 70.0}, dict(RULES, underlay=False))
    pts = np.array([p for sg in segs for p in sg])
    r = np.hypot(pts[:, 0], pts[:, 1])
    ends = r[r > 9]
    assert ends.max() <= 12.05                    # never outside the shape
    assert np.std(ends[ends > 10.5]) > 0.1        # the edge is uneven, not a hard outline


def test_the_review_can_change_a_part(tmp_path, fake_ai):
    vec = _vec()
    blue = _part_of(vec, lambda c: c[2] > 120 and c[0] < 100)
    fake = fake_ai()
    fake.review = {"verdict": "fixable", "notes": ["The block looks better softened."],
                   "foliage": [],
                   "changes": [{"id": blue["id"], "role": "foliage", "stitch": "texture", "angle": 80}]}
    r = run_pipeline(_picture(), tmp_path, "x", fit_w_mm=150, fit_h_mm=150, debug=False,
                     ai_assist=True)
    assert r["ai_digitize"]["reviewed"]
    assert r["ai_digitize"]["notes"] == ["The block looks better softened."]
    assert fake.vision_calls == 2                  # one plan, one review


def test_nothing_breaks_when_chatgpt_cant_help(tmp_path):
    # no AI connected at all (the test setup pretends ChatGPT isn't installed)
    r = run_pipeline(_picture(), tmp_path, "x", fit_w_mm=150, fit_h_mm=150, debug=False,
                     ai_assist=True)
    assert not r["ai_digitize"]["used"] and r["ai_digitize"]["problem"]
    assert r["stats"]["stitches"] > 0


def test_nonsense_answers_are_ignored(tmp_path, fake_ai):
    fake = fake_ai()
    fake.plan = {"parts": [{"id": 999, "role": "x", "stitch": "explode", "angle": 45},
                           {"id": "one", "role": "fill", "stitch": "fill", "angle": 0}],
                 "foliage": [{"x0": 50, "y0": 50, "x1": 50, "y1": 50}], "notes": []}
    r = run_pipeline(_picture(), tmp_path, "x", fit_w_mm=150, fit_h_mm=150, debug=False,
                     ai_assist=True)
    assert r["stats"]["stitches"] > 0
