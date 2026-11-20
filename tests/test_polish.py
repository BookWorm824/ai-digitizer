"""Tests for Phase 4, friendly polish: thread matching, click-to-edit, the worksheet, the
garment mockup, every machine format, the zip, and the project library."""

from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from shapely.geometry import box

from app import config, projects, threads, worksheet
from app.api.server import app
from app.pipeline import PipelineError, edits, run_pipeline

IMAGES = Path(__file__).parent / "images"


def _badge():
    return (IMAGES / "logo_badge.png").read_bytes()


def _blocks():
    return (IMAGES / "logo_blocks.png").read_bytes()


@pytest.fixture(scope="module")
def plain(tmp_path_factory):
    """The blocks logo digitized once with no changes, to compare edits against."""
    return run_pipeline(_blocks(), tmp_path_factory.mktemp("plain"), "blocks", debug=False)


# ------------------------------------------------------------------------- thread matching

def test_thread_brands_are_listed_default_first():
    brands = threads.brands()
    assert brands[0]["id"] == threads.DEFAULT_BRAND
    assert {"isacord", "robison_anton", "brother"} <= {b["id"] for b in brands}
    assert all(b["count"] > 50 for b in brands)


def test_pure_black_and_white_match_the_obvious_threads():
    assert threads.match((0, 0, 0), "isacord")["number"] == "0020"
    assert threads.match((255, 255, 255), "isacord")["number"] == "0010"
    assert threads.match((0, 0, 0), "brother")["name"] == "Black"


def test_match_is_the_closest_color_in_the_chart():
    for rgb in ((26, 54, 110), (200, 40, 45), (240, 180, 40), (30, 140, 60)):
        m = threads.match(rgb, "isacord")
        chart = threads.chart("isacord")
        d = np.linalg.norm(chart["lab"] - threads._lab(np.array([rgb]))[0], axis=1)
        assert m["difference"] == pytest.approx(float(d.min()), abs=0.06)
        assert m["difference"] < 15 and m["label"] and m["hex"].startswith("#")


def test_every_design_color_gets_a_thread_in_every_brand(plain):
    for c in plain["colors"]:
        assert set(c["threads"]) == {b["id"] for b in threads.brands()}
        assert c["threads"]["robison_anton"]["name"]            # that chart has names
    assert threads.match((1, 2, 3), "no-such-brand")["brand"] == "Isacord"   # falls back


# ------------------------------------------------------------------------- click to edit

def _at(result, x_mm, y_mm):
    return [x_mm / result["ref"], y_mm / result["ref"]]


def test_design_reports_its_areas_for_clicking(plain):
    assert len(plain["areas"]) >= 4 and plain["ref"] > 60
    for a in plain["areas"]:
        assert a["kind"] in ("fill", "satin", "run") and len(a["ring"]) >= 8 and a["area"] > 0


def test_delete_removes_just_the_area_that_was_clicked(plain, tmp_path):
    # the white disc in the middle of the red ring
    r = run_pipeline(_blocks(), tmp_path, "x", debug=False,
                     edits=[{"type": "delete", "at": [0.0, 0.0]}])
    assert r["edits_applied"] == 1
    assert len(r["colors"]) == len(plain["colors"]) - 1          # white is gone
    assert r["stats"]["stitches"] < plain["stats"]["stitches"]
    assert len(r["areas"]) == len(plain["areas"]) - 1


def test_outline_only_swaps_a_fill_for_a_line(plain, tmp_path):
    r = run_pipeline(_blocks(), tmp_path, "x", debug=False,
                     edits=[{"type": "outline", "at": [0.0, 0.0]}])
    assert r["edits_applied"] == 1
    assert len(r["colors"]) == len(plain["colors"])              # still sewn, in its color
    assert r["stats"]["stitches"] < plain["stats"]["stitches"] - 200
    assert sum(a["kind"] == "run" for a in r["areas"]) > sum(a["kind"] == "run" for a in plain["areas"])


def test_one_area_can_be_given_its_own_color(plain, tmp_path):
    r = run_pipeline(_blocks(), tmp_path, "x", debug=False,
                     edits=[{"type": "color", "at": [0.0, 0.0], "hex": "#00aa55"}])
    hexes = [c["hex"] for c in r["colors"]]
    assert "#00aa55" in hexes and len(hexes) == len(plain["colors"])     # white became green
    assert r["files"]["pes"]["ok"]


def test_recolor_changes_a_thread_everywhere_and_can_merge_two(plain, tmp_path):
    first, second = plain["colors"][0]["hex"], plain["colors"][1]["hex"]
    r = run_pipeline(_blocks(), tmp_path / "a", "x", debug=False,
                     edits=[{"type": "recolor", "from": first, "hex": "#123456"}])
    assert r["colors"][0]["hex"] == "#123456" and len(r["colors"]) == len(plain["colors"])
    merged = run_pipeline(_blocks(), tmp_path / "b", "x", debug=False,
                          edits=[{"type": "recolor", "from": first, "hex": second}])
    assert len(merged["colors"]) == len(plain["colors"]) - 1
    assert merged["stats"]["colors"] == plain["stats"]["colors"] - 1


def test_an_edit_still_lands_on_the_same_area_after_resizing(tmp_path):
    e = [{"type": "color", "at": [0.0, 0.0], "hex": "#00aa55"}]
    for side in (50.0, 88.9, 150.0):
        r = run_pipeline(_blocks(), tmp_path / str(side), "x", debug=False, edits=e,
                         fit_w_mm=side, fit_h_mm=side)
        assert r["edits_applied"] == 1 and "#00aa55" in [c["hex"] for c in r["colors"]]


def test_bad_or_missed_edits_are_ignored_not_fatal(plain, tmp_path):
    junk = [{"type": "delete"}, {"type": "color", "at": [0, 0], "hex": "nope"}, "what",
            {"type": "delete", "at": [9, 9]}, {"type": "explode", "at": [0, 0]}]
    r = run_pipeline(_blocks(), tmp_path, "x", debug=False, edits=junk)
    assert r["edits_applied"] == 0 and r["stats"]["stitches"] == plain["stats"]["stitches"]


def test_deleting_everything_says_so_in_plain_words(tmp_path):
    from PIL import Image
    img = Image.new("RGB", (600, 600), "white")
    img.paste((20, 40, 110), (150, 150, 450, 450))               # one navy square, nothing else
    buf = io.BytesIO()
    img.save(buf, "PNG")
    with pytest.raises(PipelineError) as err:
        run_pipeline(buf.getvalue(), tmp_path, "x", debug=False,
                     edits=[{"type": "delete", "at": [0.0, 0.0]}])
    assert "Undo" in str(err.value)


def test_a_click_picks_the_smallest_thing_under_it():
    objs = [{"kind": "fill", "color": 0, "poly": box(0, 0, 40, 40)},
            {"kind": "fill", "color": 1, "poly": box(10, 10, 14, 14)}]
    assert edits.find(objs, 12, 12) == 1 and edits.find(objs, 30, 30) == 0
    assert edits.find(objs, 99, 99) is None


# --------------------------------------------------------------- files for the machine

def test_every_machine_in_the_list_gets_a_file_that_reads_back(plain):
    formats = {m["format"] for m in config.machines()}
    assert formats == {"pes", "dst", "jef", "exp", "vp3", "xxx"}
    for fmt in formats:
        assert plain["files"][fmt]["ok"], fmt
        assert plain["files"][fmt]["stitches"] == plain["stats"]["stitches"]


def test_worksheet_is_a_one_page_pdf(plain):
    pdf = worksheet.build_pdf(plain, Path(plain["preview"]), "isacord", "Brother / Baby Lock",
                              "blocks.pes")
    assert pdf[:5] == b"%PDF-" and len(pdf) > 5000
    assert len([m for m in pdf.split(b"/Type /Page")[1:] if not m.startswith(b"s")]) == 1


def test_worksheet_survives_a_missing_picture(plain, tmp_path):
    assert worksheet.build_pdf(plain, tmp_path / "gone.png")[:5] == b"%PDF-"


def test_mockup_puts_the_design_on_a_shirt_of_the_chosen_color(plain):
    png = worksheet.build_mockup(plain, "#23262d")
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    assert img is not None and img.shape[0] > 1000
    body = img[int(img.shape[0] * 0.8), img.shape[1] // 2]            # low on the shirt body
    assert tuple(int(v) for v in body) == (0x2d, 0x26, 0x23)          # BGR of #23262d
    corner = img[5, 5]
    assert int(corner.min()) > 200                                    # page, not garment
    chest = img[int(190 * 3) - 90:int(190 * 3) + 90, int(400 * 3) - 110:int(400 * 3) + 110]
    assert len(np.unique(chest.reshape(-1, 3), axis=0)) > 200         # the design is there


def test_things_that_are_not_shirts_get_a_fabric_swatch(tmp_path):
    r = run_pipeline(_blocks(), tmp_path, "t", fabric_id="towel", debug=False)
    img = cv2.imdecode(np.frombuffer(worksheet.build_mockup(r, "#8f1f24"), np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[0] == img.shape[1]
    assert tuple(int(v) for v in img[3, 3]) == (0x24, 0x1f, 0x8f)     # garment color to the edge


# ----------------------------------------------------------------------- through the API

def _wait(client, url):
    for _ in range(600):
        job = client.get(url).json()
        if job["status"] != "working":
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


@pytest.fixture
def stitched():
    c = TestClient(app)
    cid = c.post("/api/cleanup", files={"file": ("My Logo.png", _blocks(), "image/png")}).json()["id"]
    _wait(c, f"/api/cleanup/{cid}")
    jid = c.post("/api/jobs", data={"cleanup_id": cid, "version": "local"}).json()["id"]
    assert _wait(c, f"/api/jobs/{jid}")["status"] == "done"
    return c, cid, jid


def test_options_list_machines_and_thread_brands():
    o = TestClient(app).get("/api/options").json()
    assert [m["format"] for m in o["machines"]] == ["pes", "dst", "jef", "exp", "vp3", "xxx"]
    assert o["thread_brands"][0]["id"] == "isacord"


def test_worksheet_mockup_and_zip_download(stitched):
    c, _cid, jid = stitched
    pdf = c.get(f"/api/jobs/{jid}/worksheet.pdf?brand=robison_anton&machine=janome")
    assert pdf.status_code == 200 and pdf.content[:5] == b"%PDF-"
    assert "My Logo worksheet.pdf" in pdf.headers["content-disposition"]
    mock = c.get(f"/api/jobs/{jid}/mockup.png?color=1f2a44")
    assert mock.headers["content-type"] == "image/png" and mock.content[:4] == b"\x89PNG"
    z = zipfile.ZipFile(io.BytesIO(c.get(f"/api/jobs/{jid}/all.zip").content))
    names = set(z.namelist())
    assert {f"My Logo.{f}" for f in ("pes", "dst", "jef", "exp", "vp3", "xxx")} <= names
    assert {"My Logo worksheet.pdf", "My Logo preview.png", "My Logo mockup.png"} <= names
    assert z.read("My Logo.pes")[:4] == b"#PES"
    assert c.get("/api/jobs/nope/worksheet.pdf").status_code == 404


def test_edits_travel_through_the_api(stitched):
    c, cid, jid = stitched
    base = c.get(f"/api/jobs/{jid}").json()["result"]
    e = json.dumps([{"type": "delete", "at": [0.0, 0.0]}])
    jid2 = c.post("/api/jobs", data={"cleanup_id": cid, "version": "local", "edits": e}).json()["id"]
    changed = _wait(c, f"/api/jobs/{jid2}")["result"]
    assert changed["edits_applied"] == 1 and changed["stats"]["stitches"] < base["stats"]["stitches"]
    jid3 = c.post("/api/jobs", data={"cleanup_id": cid, "version": "local", "edits": "not json"}).json()["id"]
    assert _wait(c, f"/api/jobs/{jid3}")["result"]["edits_applied"] == 0


def test_projects_are_saved_listed_reopened_and_deleted(stitched, data_dir):
    c, cid, jid = stitched
    assert c.get("/api/projects").json()["projects"] == []
    chosen = {"version": "local", "colors": 3, "sizeIn": 2.5, "density": -1,
              "edits": [{"type": "delete", "at": [0.0, 0.0]}]}
    card = c.post("/api/projects", data={"cleanup_id": cid, "job_id": jid,
                                         "settings_json": json.dumps(chosen)}).json()
    assert card["name"] == "My Logo" and card["summary"]["stitches"] > 500
    listed = c.get("/api/projects").json()["projects"]
    assert [p["id"] for p in listed] == [card["id"]]
    assert c.get(f"/api/projects/{card['id']}/thumb.png").content[:4] == b"\x89PNG"
    assert (data_dir / "projects" / card["id"] / "files" / "upload.bin").exists()

    # saving again updates the same project instead of making a second one
    c.post("/api/projects", data={"cleanup_id": cid, "job_id": jid, "settings_json": "{}"})
    assert len(c.get("/api/projects").json()["projects"]) == 1
    c.post("/api/projects", data={"cleanup_id": cid, "job_id": jid, "settings_json": json.dumps(chosen)})

    opened = c.post(f"/api/projects/{card['id']}/open").json()
    assert opened["settings"] == chosen and opened["name"] == "My Logo"
    assert [v["id"] for v in opened["result"]["versions"]] == ["local"]
    jid2 = c.post("/api/jobs", data={"cleanup_id": opened["cleanup_id"], "version": "local",
                                     "size_in": 2.5, "edits": json.dumps(chosen["edits"])}).json()["id"]
    again = _wait(c, f"/api/jobs/{jid2}")["result"]
    assert again["stats"]["size_in"] == pytest.approx(2.5, abs=0.02) and again["edits_applied"] == 1
    # a design reopened from a project saves back to that project
    c.post("/api/projects", data={"cleanup_id": opened["cleanup_id"], "job_id": jid2, "settings_json": "{}"})
    assert len(c.get("/api/projects").json()["projects"]) == 1

    assert c.delete(f"/api/projects/{card['id']}").json() == {"ok": True}
    assert c.get("/api/projects").json()["projects"] == []
    assert c.post(f"/api/projects/{card['id']}/open").status_code == 404
    assert c.delete("/api/projects/..%2F..").status_code in (404, 405)


def test_project_ids_cannot_point_outside_the_library():
    for bad in ("../x", "abc", "ABCDEF123456", "0123456789abcdef"):
        with pytest.raises(KeyError):
            projects.thumb(bad)
