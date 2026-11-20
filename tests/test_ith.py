"""Tests for Phase 5, in-the-hoop projects: felt patch, snap-tab key fob, coaster, ornament."""

from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path

import pyembroidery
import pytest
from fastapi.testclient import TestClient
from shapely.geometry import LineString, Point, Polygon

from app import config, ith_docs
from app.api.server import app
from app.pipeline import ith, run_pipeline
from app.pipeline.common import dist

IMAGES = Path(__file__).parent / "images"
TYPES = sorted(ith.PROJECTS)


def _art():
    return (IMAGES / "logo_blocks.png").read_bytes()


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    """Every project type built once from the test logo."""
    out = {}
    for kind in TYPES:
        out[kind] = run_pipeline(_art(), tmp_path_factory.mktemp(kind), "logo", debug=False,
                                 project={"type": kind, "shape": "circle"})
    return out


def _points(block):
    return [(s[i], s[i + 1]) for s in block["segments"] for i in range(0, len(s), 2)]


def _body(result):
    w, h = result["project"]["body_mm"]
    return Point(0, 0).buffer(w / 2, quad_segs=64)          # these tests use the circle shape


# ---------------------------------------------------------------------------- the spec

def test_unknown_or_missing_projects_are_not_projects():
    assert ith.normalize(None) is None and ith.normalize({"type": "spaceship"}) is None
    p = ith.normalize({"type": "coaster", "shape": "blob", "size_in": 99})
    assert p == {"type": "coaster", "shape": "circle", "size_in": 7.0}
    assert ith.normalize({"type": "feltie"})["size_in"] == ith.PROJECTS["feltie"]["default_in"]


def test_project_types_come_in_the_briefs_order():
    assert list(ith.PROJECTS) == ["feltie", "keyfob", "coaster", "ornament"]


# --------------------------------------------------------------------- how every file works

@pytest.mark.parametrize("kind", TYPES)
def test_steps_run_placement_tackdown_decoration_then_finish(made, kind):
    steps = [s["title"] for s in made[kind]["project"]["steps"]]
    assert steps[:2] == ["Placement line", "Tackdown"]
    assert all(t.startswith("Decoration, color") for t in steps[2:-1]) and len(steps) >= 4
    assert steps[-1] == {"feltie": "Satin border", "keyfob": "Final outline"}.get(kind, "Final seam")
    assert [s["number"] for s in made[kind]["project"]["steps"]] == list(range(1, len(steps) + 1))


@pytest.mark.parametrize("kind", TYPES)
def test_the_machine_stops_at_every_step(made, kind):
    """Machines only stop at a color change, so every step must be its own color block,
    and that must survive into the file."""
    r = made[kind]
    n_steps = len(r["project"]["steps"])
    assert len(r["blocks"]) == n_steps == len(r["colors"])
    for fmt in ("pes", "dst", "jef"):
        p = pyembroidery.read(r["files"][fmt]["path"])
        changes = sum(1 for s in p.stitches
                      if (s[2] & 0xFF) in (pyembroidery.COLOR_CHANGE, pyembroidery.STOP))
        assert changes == n_steps - 1, fmt
        assert r["files"][fmt]["ok"]


@pytest.mark.parametrize("kind", TYPES)
def test_every_step_says_what_to_do(made, kind):
    steps = made[kind]["project"]["steps"]
    assert "stabilizer" in steps[0]["before"].lower()           # hoop bare stabilizer first
    assert steps[0]["after"] and steps[-1]["after"]             # what to lay down; how to finish
    assert all(s["thread"] for s in steps)
    assert made[kind]["project"]["materials"]


@pytest.mark.parametrize("kind", TYPES)
def test_decoration_stays_well_inside_the_item(made, kind):
    r = made[kind]
    keep = _body(r).buffer(-(ith.PROJECTS[kind]["margin"] - 1.0))
    for block in r["blocks"][2:-1]:
        assert all(keep.contains(Point(p)) for p in _points(block))


@pytest.mark.parametrize("kind", TYPES)
def test_whole_project_fits_a_hoop_with_the_safety_margin(made, kind):
    r = made[kind]
    assert r["check"]["hoop"] is not None and r["check"]["status"] != "red"
    assert r["check"]["hoop"] == ("5×7" if kind in ("keyfob", "coaster") else "4×4")
    assert r["fabric"]["id"] == ith.PROJECTS[kind]["fabric"]
    assert not any("thread colors" in m["text"] for m in r["check"]["messages"])


# ------------------------------------------------------------------------- each project

def test_felt_patch_border_covers_the_tackdown_and_the_cut_edge(made):
    r = made["feltie"]
    tack = LineString(_points(r["blocks"][1]))
    border = _points(r["blocks"][-1])
    radii = [dist(p, (0, 0)) for p in border]
    edge = r["project"]["body_mm"][0] / 2
    assert min(radii) < edge - 1.2 and max(radii) > edge + 0.8        # spans the raw edge
    assert tack.distance(Point(0, 0)) < edge                          # tackdown is inside it
    assert len(border) > 400                                          # satin, not a run


def test_key_fob_has_a_tab_and_its_outline_is_sewn_three_times(made):
    r = made["keyfob"]
    w, h = r["stats"]["width_mm"], r["stats"]["height_mm"]
    assert h > w + ith.TAB_LENGTH_MM * 0.8                            # the tab sticks up
    place, final = _points(r["blocks"][0]), _points(r["blocks"][-1])
    assert min(p[1] for p in place) < -r["project"]["body_mm"][1] / 2 - 30
    assert len(final) > 2.5 * len(place)                              # triple run
    assert "BACK of the hoop" in r["project"]["steps"][-1]["before"]
    assert "snap" in r["project"]["steps"][-1]["after"]
    assert [p["name"] for p in r["project"]["pieces"]] == ["Vinyl, front", "Vinyl, back"]


@pytest.mark.parametrize("kind", ["coaster", "ornament"])
def test_stitch_and_turn_has_seam_allowance_and_a_turning_gap(made, kind):
    r = made[kind]
    seam = _points(r["blocks"][-1])
    body = _body(r)
    assert all(body.exterior.distance(Point(p)) < 0.6 for p in seam)  # on the finished outline
    start, end = seam[1], seam[-2]
    gap = dist(start, end)
    assert 30 <= gap <= 76                                            # 1.5 to 3 in, as a chord
    assert (start[1] + end[1]) / 2 > body.bounds[3] * 0.6             # at the bottom
    cut = LineString(_points(r["blocks"][0]))
    assert cut.distance(Point(0, 0)) == pytest.approx(
        r["project"]["body_mm"][0] / 2 + ith.SEAM_ALLOWANCE_MM, abs=1.0)
    assert "right side DOWN" in r["project"]["steps"][-1]["before"]
    assert ("ribbon" in r["project"]["steps"][-1]["before"]) == (kind == "ornament")


def test_follow_my_design_wraps_the_outline_round_the_art(tmp_path):
    r = run_pipeline((IMAGES / "logo_badge.png").read_bytes(), tmp_path, "b", debug=False,
                     project={"type": "feltie", "shape": "follow", "size_in": 3})
    w, h = r["project"]["body_mm"]
    assert abs(w - h) > 3                                             # not a circle: it follows
    place = Polygon(_points(r["blocks"][0]))
    for block in r["blocks"][2:-1]:
        assert all(place.buffer(0.5).contains(Point(p)) for p in _points(block))


def test_size_is_the_finished_item_and_too_big_is_flagged(tmp_path):
    small = run_pipeline(_art(), tmp_path / "s", "c", debug=False,
                         project={"type": "coaster", "shape": "square", "size_in": 3.5})
    assert small["project"]["body_mm"] == [pytest.approx(88.9, abs=0.3)] * 2
    big = run_pipeline(_art(), tmp_path / "b", "c", debug=False,
                       project={"type": "coaster", "shape": "square", "size_in": 7})
    assert big["project"]["body_mm"][0] > small["project"]["body_mm"][0]
    assert big["check"]["hoop"] in ("8×8", "8×12", None)


# -------------------------------------------------------------------------- extra outputs

@pytest.mark.parametrize("kind", TYPES)
def test_dieline_file_holds_only_the_placement_line(made, kind):
    r = made[kind]
    for fmt, d in r["project"]["dieline"].items():
        assert d["ok"], fmt
    p = pyembroidery.read(r["project"]["dieline"]["pes"]["path"])
    stitches = sum(1 for s in p.stitches if (s[2] & 0xFF) == pyembroidery.STITCH)
    assert stitches == sum(len(s) // 2 for s in r["blocks"][0]["segments"])


@pytest.mark.parametrize("kind", TYPES)
def test_instructions_and_templates_are_real_pdfs(made, kind):
    r = made[kind]
    doc = ith_docs.instructions_pdf(r)
    assert doc[:5] == b"%PDF-" and len(doc) > 20000                   # has the step pictures
    tpl = ith_docs.templates_pdf(r)
    pages = len([m for m in tpl.split(b"/Type /Page")[1:] if not m.startswith(b"s")])
    assert tpl[:5] == b"%PDF-" and pages == len(r["project"]["pieces"])
    for piece in r["project"]["pieces"]:
        xs = piece["ring"][0::2]
        assert max(xs) - min(xs) > r["project"]["body_mm"][0]         # bigger than the item


# ------------------------------------------------------------------------ through the API

def _wait(client, url):
    for _ in range(600):
        job = client.get(url).json()
        if job["status"] != "working":
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_options_offer_projects_and_keep_their_materials_off_the_garment_list():
    o = TestClient(app).get("/api/options").json()
    assert [t["id"] for t in o["projects"]["types"]] == ["feltie", "keyfob", "coaster", "ornament"]
    assert {s["id"] for s in o["projects"]["shapes"]} == {"circle", "square", "follow"}
    assert not {"felt", "vinyl", "cotton"} & {f["id"] for f in o["fabrics"]}
    assert all(config.fabrics()[i]["project"] for i in ("felt", "vinyl", "cotton"))


def test_project_flow_through_the_api_and_the_library():
    c = TestClient(app)
    spec = {"type": "keyfob", "shape": "circle", "size_in": 2.5}
    cid = c.post("/api/cleanup", files={"file": ("Dog Tag.png", _art(), "image/png")},
                 data={"project": json.dumps(spec)}).json()["id"]
    _wait(c, f"/api/cleanup/{cid}")
    jid = c.post("/api/jobs", data={"cleanup_id": cid, "version": "local"}).json()["id"]
    r = _wait(c, f"/api/jobs/{jid}")["result"]
    assert r["project"]["type"] == "keyfob" and r["fabric"]["id"] == "vinyl"
    assert "path" not in json.dumps(r["project"]["dieline"])          # no file paths to the screen

    assert c.get(f"/api/jobs/{jid}/instructions.pdf").content[:5] == b"%PDF-"
    assert c.get(f"/api/jobs/{jid}/templates.pdf").content[:5] == b"%PDF-"
    assert c.get(f"/api/jobs/{jid}/dieline/pes").content[:4] == b"#PES"
    names = set(zipfile.ZipFile(io.BytesIO(c.get(f"/api/jobs/{jid}/all.zip").content)).namelist())
    assert {"Dog Tag instructions.pdf", "Dog Tag cutting templates.pdf", "Dog Tag dieline.pes",
            "Dog Tag.pes", "Dog Tag worksheet.pdf"} <= names

    # the size control means the finished item for a project
    jid2 = c.post("/api/jobs", data={"cleanup_id": cid, "version": "local", "size_in": 3}).json()["id"]
    bigger = _wait(c, f"/api/jobs/{jid2}")["result"]
    assert bigger["project"]["size_in"] == 3 and bigger["project"]["body_mm"][0] > r["project"]["body_mm"][0]

    # saved and reopened, it is still a key fob
    card = c.post("/api/projects", data={"cleanup_id": cid, "job_id": jid, "settings_json": "{}"}).json()
    opened = c.post(f"/api/projects/{card['id']}/open").json()
    assert opened["project"] == spec
    jid3 = c.post("/api/jobs", data={"cleanup_id": opened["cleanup_id"], "version": "local"}).json()["id"]
    assert _wait(c, f"/api/jobs/{jid3}")["result"]["project"]["type"] == "keyfob"


def test_garment_designs_have_no_project_pages():
    c = TestClient(app)
    cid = c.post("/api/cleanup", files={"file": ("x.png", _art(), "image/png")}).json()["id"]
    _wait(c, f"/api/cleanup/{cid}")
    jid = c.post("/api/jobs", data={"cleanup_id": cid, "version": "local"}).json()["id"]
    assert _wait(c, f"/api/jobs/{jid}")["result"]["project"] is None
    assert c.get(f"/api/jobs/{jid}/instructions.pdf").status_code == 404
    assert c.get(f"/api/jobs/{jid}/templates.pdf").status_code == 404
