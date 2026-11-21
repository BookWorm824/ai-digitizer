"""What finished production stitch files taught us (tools/study.py; reports/Auto digitizing
engine deep research.md): fills at real density, satin outlines in one thread, lettering
without trims inside a letter, and the study tool that measures stitch files."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import box

from app import config
from app.pipeline import lettering as L
from app.pipeline import sequence, stitch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import study  # noqa: E402

RULES = config.rules_for("tshirt")


# ------------------------------------------------------------------------- fill density

def test_fill_rows_are_half_the_stated_spacing_apart():
    # "fill spacing 0.45 mm" is forward row to forward row with a return row between:
    # neighbouring rows of thread are about 0.22 mm apart (production files: 0.19-0.22)
    obj = {"kind": "fill", "color": 0, "poly": box(0, 0, 20, 20), "area_mm2": 400.0}
    segs = stitch.stitch_fill(obj, dict(RULES, underlay=False), 0.0)
    ys = sorted({round(y, 3) for sg in segs for _x, y in sg})
    gaps = np.diff(ys)
    gaps = gaps[gaps > 0.05]
    assert np.median(gaps) == pytest.approx(RULES["fill_spacing_mm"] / 2.0, rel=0.05)
    # about 800 or more stitches per square inch of solid fill, not 400
    stitches = sum(len(sg) for sg in segs)
    assert stitches / (400.0 / 645.16) > 700


# ---------------------------------------------------------------- lettering without trims

def _sew(objs):
    st = stitch.run({"objects": objs, "order": [0]}, RULES)
    return sequence.run(st, [(0, 0, 0)], RULES, [0])


@pytest.mark.parametrize("ch", list("ETNAHKMFLB"))
def test_a_satin_letter_is_one_thread(ch):
    pls = L.layout(ch, 12.0, 0, 0, "block")
    plan = _sew(L.objects(pls, 0, RULES, "block", L.text_style(12.0), group=0, row=0))
    # (an H's two stems share no hidden way across: one cut, where it was three)
    assert plan["trims"] <= (1 if ch == "H" else 0), f"{ch}: {plan['trims']} trims in one letter"


def test_a_satin_word_needs_few_trims():
    pls = L.layout("TENNESSEE", 12.5, 0, 0, "varsity")
    plan = _sew(L.objects(pls, 0, RULES, "varsity", L.text_style(12.5), group=0, row=0))
    assert plan["trims"] <= 3                      # it was one per stroke: more than twenty


def test_a_satin_stroke_ends_where_the_plan_says():
    col = L.column([(0, 0), (20, 0)], 3.0)
    for underlay in ("none", "center", "edge", "edge_zigzag"):
        for finish, end in (("far", (20, 0)), ("start", (0, 0))):
            pts = stitch.stitch_satin(dict(col, underlay=underlay, finish=finish), RULES)[0]
            assert math.dist(pts[-1], end) < 2.5, (underlay, finish)


def test_the_letter_plan_respects_which_stroke_tucks_under_which():
    # a T: the stem is sewn before the bar that covers its top
    for pl in L.layout("T", 12.0, 0, 0, "block"):
        strokes = [s for s in L.letter_strokes(pl, "block") if not s["dot"]]
        ys = [np.mean([p[1] for p in s["pts"]]) for s in strokes]
        spans = [abs(s["pts"][0][1] - s["pts"][-1][1]) for s in strokes]
        stem, bar = int(np.argmax(spans)), int(np.argmin(spans))
        assert stem < bar and ys[bar] < ys[stem]


# ------------------------------------------------------------------------ the study tool

def test_study_tells_satin_fill_run_and_bean_apart():
    # satin: a zig-zag 3 mm wide advancing 0.2 mm per stitch
    satin = np.array([(0.2 * k, 3.0 * (k % 2)) for k in range(200)], float)
    s = study.classify(satin)
    assert s["len"]["satin"] > 0.9 * sum(s["len"].values())
    assert np.median(s["satin_w"]) == pytest.approx(3.0, abs=0.1)
    assert np.median(s["satin_sp"]) == pytest.approx(0.2, abs=0.03)
    # fill: rows of four 4 mm stitches, 0.2 mm apart, back and forth
    rows = []
    for r in range(40):
        xs = [0, 4, 8, 12, 16] if r % 2 == 0 else [16, 12, 8, 4, 0]
        rows += [(x, 0.2 * r) for x in xs]
    f = study.classify(np.asarray(rows, float))
    assert f["len"]["fill"] > 0.8 * sum(f["len"].values())
    assert np.median(f["fill_sp"]) == pytest.approx(0.2, abs=0.05)
    # run: a gently curving line
    run = np.array([(2.0 * k, 3.0 * math.sin(k / 6.0)) for k in range(60)], float)
    r = study.classify(run)
    assert r["len"]["run"] > 0.9 * sum(r["len"].values())
    # bean: forward, back, forward over every 2 mm stitch
    pts = [(0.0, 0.0)]
    for k in range(1, 30):
        a, b = (2.0 * (k - 1), 0.0), (2.0 * k, 0.0)
        pts += [b, a, b]
    bn = study.classify(np.asarray(pts, float))
    assert bn["len"]["bean"] > 0.5 * sum(bn["len"].values())
    assert bn["len"]["satin"] < 0.1 * sum(bn["len"].values())


def test_our_own_fill_measures_like_a_production_fill():
    obj = {"kind": "fill", "color": 0, "poly": box(0, 0, 30, 20), "area_mm2": 600.0}
    segs = stitch.stitch_fill(obj, dict(RULES, underlay=False), 18.0)
    m = study.measure([np.asarray(sg, float) for sg in segs], {"trims": 0})
    assert 0.17 <= m["fill_spacing"]["p50"] <= 0.26
