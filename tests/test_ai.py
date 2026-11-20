"""Tests for the ChatGPT stage (Phase 2). No real OpenAI calls are made: a stand-in client
answers instead, so these run offline and cost nothing."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import ai, cleanup, config, settings
from app.api.server import app
from app.pipeline import optimize

IMAGES = Path(__file__).parent / "images"
BADGE = (IMAGES / "logo_badge.png").read_bytes() if (IMAGES / "logo_badge.png").exists() else b""
FIT = (88.9, 88.9)
CTX = {"size_in": 3.5, "fabric": "a t-shirt (knit)", "max_colors": 6, "min_text_mm": 5.0,
       "text": ["TEST"]}


@pytest.fixture
def badge():
    return (IMAGES / "logo_badge.png").read_bytes()


# -------------------------------------------------------------------------------- settings

def test_key_is_saved_locally_and_never_shown(data_dir):
    settings.save({"openai_api_key": "sk-test-1234567890abcdefWXYZ"})
    assert json.loads((data_dir / "settings.json").read_text())["openai_api_key"].endswith("WXYZ")
    pub = settings.public()
    assert pub["has_key"] and pub["key_hint"] == "…WXYZ"
    assert "sk-test" not in json.dumps(pub) and "openai_api_key" not in pub
    assert "sk-test" not in json.dumps(ai.status())


def test_settings_file_lives_outside_the_code_folder(data_dir):
    settings.save({"daily_spend_cap_usd": 1.0})
    assert config.APP_DIR.parent not in (data_dir / "settings.json").resolve().parents


def test_model_names_come_from_config_and_can_be_overridden():
    assert settings.load()["models"]["redraw_final"] == settings.ai_defaults()["models"]["redraw_final"]
    settings.save({"models": {"redraw_final": "some-newer-model"}})
    s = settings.load()
    assert s["models"]["redraw_final"] == "some-newer-model"
    assert s["models"]["analyze"] == settings.ai_defaults()["models"]["analyze"]   # others kept


# ---------------------------------------------------------------------------------- prompt

def test_prompt_follows_the_brief_template():
    p = optimize.build_prompt(optimize.VARIANTS[0], CTX)
    assert p.startswith("Redraw this image as embroidery-ready artwork for a 3.5-inch design "
                        "on a t-shirt (knit).")
    for part in ("flat solid colors only", "Maximum 6 colors", "at least 1 mm thick",
                 'Keep text exactly as written: "TEST", at least 5 mm tall',
                 "Plain white background", "Center it"):
        assert part in p


def test_the_three_versions_differ_only_in_their_dials():
    faithful, simplified, bold = (optimize.build_prompt(v, CTX) for v in optimize.VARIANTS)
    assert "Maximum 4 colors" in simplified and "Maximum 5 colors" in bold
    assert "at least 2 mm thick" in bold and "Patch style" in bold
    assert len({faithful, simplified, bold}) == 3


def test_prompt_never_invents_text_when_the_picture_has_none():
    p = optimize.build_prompt(optimize.VARIANTS[0], dict(CTX, text=[]))
    assert "Do not add any text" in p and "Keep text exactly" not in p


# ----------------------------------------------------------------------------- AI wrapper

def test_no_key_means_ai_is_unavailable_with_a_plain_reason(badge):
    assert not ai.available()
    with pytest.raises(ai.AIUnavailable) as e:
        ai.analyze(badge)
    assert e.value.reason == "no_key" and "Settings" in e.value.message


def test_results_are_cached_so_nothing_is_paid_for_twice(fake_ai, badge):
    client = fake_ai()
    first = ai.redraw(badge, "make it flat")
    again = ai.redraw(badge, "make it flat")
    assert first == again and client.image_calls == 1
    ai.redraw(badge, "make it flat", attempt=1)            # "try again" asks for a fresh one
    assert client.image_calls == 2
    ai.analyze(badge), ai.analyze(badge)
    assert client.vision_calls == 1


def test_every_call_is_added_to_todays_spending(fake_ai, badge):
    fake_ai()
    assert ai.spent_today() == 0
    ai.redraw(badge, "x")
    # 200 text-in at $5 + 400 image-in at $8 + 300 image-out at $30, per million tokens
    assert ai.spent_today() == pytest.approx((200 * 5 + 400 * 8 + 300 * 30) / 1e6)
    before = ai.spent_today()
    ai.analyze(badge)
    assert ai.spent_today() > before


def test_daily_spending_limit_stops_calls(fake_ai, badge):
    client = fake_ai()
    settings.save({"daily_spend_cap_usd": 0.001})
    with pytest.raises(ai.AIUnavailable) as e:
        ai.redraw(badge, "x")
    assert e.value.reason == "cap" and client.image_calls == 0


def test_redraw_sends_what_the_docs_ask_for(fake_ai, badge):
    client = fake_ai()
    ai.redraw(badge, "x")
    ai.redraw(badge, "y", final=True)
    preview, final = client.image_kwargs
    defaults = settings.ai_defaults()
    assert preview["model"] == defaults["models"]["redraw_preview"] and preview["quality"] == "low"
    assert final["model"] == defaults["models"]["redraw_final"] and final["quality"] == "high"
    assert preview["size"] == "1024x1024" and preview["output_format"] == "png"
    name, data, mime = preview["image"][0]
    assert mime == "image/png" and data[:4] == b"\x89PNG"


@pytest.mark.parametrize("error, reason", [
    (type("AuthenticationError", (Exception,), {})("bad key"), "bad_key"),
    (type("APIConnectionError", (Exception,), {})("no internet"), "offline"),
    (type("RateLimitError", (Exception,), {})("You exceeded your current quota"), "quota"),
    (type("BadRequestError", (Exception,), {})("moderation_blocked"), "blocked"),
    (RuntimeError("anything else"), "error"),
])
def test_api_failures_become_plain_reasons(fake_ai, badge, error, reason):
    fake_ai(fail_with=error)
    with pytest.raises(ai.AIUnavailable) as e:
        ai.redraw(badge, "x")
    assert e.value.reason == reason
    assert "Traceback" not in e.value.message


# -------------------------------------------------------------------------------- clean up

def test_without_ai_the_cleanup_still_gives_a_version(tmp_path, badge):
    r = cleanup.run_cleanup(badge, tmp_path / "job", fit=FIT, fabric_id="tshirt")
    assert [v["id"] for v in r["versions"]] == ["local"] and r["default"] == "local"
    assert not r["ai"]["used"] and r["ai"]["problem"] == "no_key"
    assert len(r["versions"][0]["colors"]) == 3
    assert (tmp_path / "job" / "local.png").exists()


def test_with_ai_there_are_three_versions_plus_the_local_one(tmp_path, fake_ai, badge):
    client = fake_ai()
    r = cleanup.run_cleanup(badge, tmp_path / "job", fit=FIT, fabric_id="tshirt")
    assert [v["id"] for v in r["versions"]] == ["faithful", "simplified", "bold", "local"]
    assert r["ai"]["used"] and r["default"] == "faithful"
    assert client.image_calls == 3 and r["text"] == ["TEST"]
    assert all('"TEST"' in p and "3.5-inch" in p for p in client.prompts)
    assert all(v["text_warning"] is None for v in r["versions"])
    assert r["spend"]["spent_today_usd"] > 0


def test_changed_text_is_flagged_and_not_chosen_by_default(tmp_path, fake_ai, badge):
    fake_ai(redraw_text=["TSET"])
    r = cleanup.run_cleanup(badge, tmp_path / "job", fit=FIT, fabric_id="tshirt")
    warn = r["versions"][0]["text_warning"]
    assert warn == {"expected": ["TEST"], "found": ["TSET"]}
    assert r["default"] == "local"       # Next must never pick a version with wrong text


def test_spelling_check_ignores_case_and_spacing_only():
    assert optimize._words(["Morgan's  Corner", "SHOP"]) == optimize._words(["morgan's corner shop"])
    assert optimize._words(["TEST"]) != optimize._words(["TSET"])


def test_ai_failure_falls_back_to_the_local_version(tmp_path, fake_ai, badge):
    fake_ai(fail_with=type("APIConnectionError", (Exception,), {})("offline"))
    r = cleanup.run_cleanup(badge, tmp_path / "job", fit=FIT, fabric_id="tshirt")
    assert [v["id"] for v in r["versions"]] == ["local"]
    assert r["ai"]["problem"] == "offline" and "internet" in r["ai"]["note"]


def test_tweak_adds_a_version_using_the_users_words(tmp_path, fake_ai, badge):
    client = fake_ai()
    cleanup.run_cleanup(badge, tmp_path / "job", fit=FIT, fabric_id="tshirt")
    v = cleanup.run_tweak(tmp_path / "job", "faithful", "thicker outline")
    assert v["id"] == "change1" and v["name"] == "Your change"
    assert "thicker outline" in client.prompts[-1] and "nothing else" in client.prompts[-1]
    assert (tmp_path / "job" / "change1.png").exists()


def test_final_polish_is_used_only_if_it_still_matches(tmp_path, fake_ai, badge):
    client = fake_ai()
    job = tmp_path / "job"
    cleanup.run_cleanup(badge, job, fit=FIT, fabric_id="tshirt")
    data = cleanup.picture_to_stitch(job, "faithful")
    assert client.image_kwargs[-1]["quality"] == "high" and (job / "final_faithful.png").exists()
    assert data == (job / "final_faithful.png").read_bytes()
    # the local version is never sent to the AI
    calls = client.image_calls
    assert cleanup.picture_to_stitch(job, "local") == badge and client.image_calls == calls


def test_polish_that_drifts_is_thrown_away(fake_ai, badge):
    fake_ai()                                  # the stand-in always returns the "blocks" logo
    assert optimize.polish(badge, CTX) is None                # badge in, blocks out: too different
    assert optimize.similar(badge, badge)


# ------------------------------------------------------------------------- through the API

def _wait(client, url):
    for _ in range(200):
        job = client.get(url).json()
        if job["status"] != "working":
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_whole_flow_through_the_api_without_a_key(badge):
    c = TestClient(app)
    assert c.get("/api/settings").json()["has_key"] is False
    cid = c.post("/api/cleanup", files={"file": ("my logo.png", badge, "image/png")},
                 data={"size": "left_chest", "fabric": "tshirt"}).json()["id"]
    job = _wait(c, f"/api/cleanup/{cid}")
    assert job["status"] == "done" and job["result"]["default"] == "local"
    assert c.get(f"/api/cleanup/{cid}/image/local").headers["content-type"] == "image/png"
    assert c.get(f"/api/cleanup/{cid}/image/source").status_code == 200
    assert c.get(f"/api/cleanup/{cid}/image/..%2Fsettings").status_code == 404

    jid = c.post("/api/jobs", data={"cleanup_id": cid, "version": "local", "colors": 6}).json()["id"]
    done = _wait(c, f"/api/jobs/{jid}")
    assert done["status"] == "done" and done["result"]["files"]["pes"]["ok"]
    pes = c.get(f"/api/jobs/{jid}/download/pes")
    assert pes.status_code == 200 and pes.content[:4] == b"#PES"
    assert 'filename="my logo.pes"' in pes.headers["content-disposition"] \
        or "my%20logo.pes" in pes.headers["content-disposition"]


def test_whole_flow_through_the_api_with_ai_and_a_tweak(fake_ai, badge):
    fake_ai()
    c = TestClient(app)
    cid = c.post("/api/cleanup", files={"file": ("logo.png", badge, "image/png")}).json()["id"]
    job = _wait(c, f"/api/cleanup/{cid}")
    assert len(job["result"]["versions"]) == 4
    assert c.post(f"/api/cleanup/{cid}/tweak", data={"version": "bold", "change": "only 4 colors"}).status_code == 200
    job = _wait(c, f"/api/cleanup/{cid}")
    assert job["result"]["picked"] == "change1" and job["result"]["versions"][0]["id"] == "change1"
    jid = c.post("/api/jobs", data={"cleanup_id": cid, "version": "change1"}).json()["id"]
    assert _wait(c, f"/api/jobs/{jid}")["result"]["files"]["dst"]["ok"]


def test_settings_api_checks_the_key_and_the_limit():
    c = TestClient(app)
    bad = c.post("/api/settings", json={"api_key": "hello"})
    assert bad.status_code == 400 and "sk-" in bad.json()["error"]
    ok = c.post("/api/settings", json={"api_key": "sk-test-1234567890abcdefWXYZ", "daily_spend_cap_usd": 5})
    assert ok.json()["has_key"] and ok.json()["daily_spend_cap_usd"] == 5 and "sk-test" not in ok.text
    assert c.post("/api/settings", json={"daily_spend_cap_usd": 500}).status_code == 400
    assert c.post("/api/settings", json={"remove_key": True}).json()["has_key"] is False


def test_bad_uploads_get_plain_words_not_error_codes():
    c = TestClient(app)
    cid = c.post("/api/cleanup", files={"file": ("x.png", b"not a picture", "image/png")}).json()["id"]
    job = _wait(c, f"/api/cleanup/{cid}")
    assert job["status"] == "error" and "PNG" in job["error"] and "Traceback" not in job["error"]
    assert "SVG" in c.post("/api/cleanup", files={"file": ("x.svg", b"<svg/>", "image/svg+xml")}).json()["error"]
