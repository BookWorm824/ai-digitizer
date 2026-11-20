"""Tests for using the ChatGPT plan (through the Codex program) instead of an API key.

Codex itself is never run here: a stand-in for chatgpt._run answers the way Codex does."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from app import ai, chatgpt, cleanup, settings

IMAGES = Path(__file__).parent / "images"
FIT = (88.9, 88.9)


class FakeCodex:
    """Stands in for the Codex program. Records every command it is asked to run."""

    def __init__(self, signed_in=True, text=("TEST",), fail=None, timeout=False):
        self.signed_in, self.text, self.fail, self.timeout = signed_in, list(text), fail, timeout
        self.commands: list[list[str]] = []
        self.prompts: list[str] = []
        self.picture = (IMAGES / "logo_blocks.png").read_bytes()

    def __call__(self, args, stdin, timeout, cwd):
        self.commands.append(args)
        if args[1:3] == ["login", "status"]:
            return (0, "Logged in using ChatGPT", "") if self.signed_in else (1, "", "Not logged in")
        self.prompts.append(stdin)
        if self.timeout:
            raise subprocess.TimeoutExpired(args, timeout)
        if self.fail:
            return 1, json.dumps({"type": "error", "message": self.fail}), ""
        if "--output-schema" in args:
            schema = json.loads(Path(args[args.index("--output-schema") + 1]).read_text("utf8"))
            data = {"text": self.text}
            if "subject_type" in schema["properties"]:
                data = {"subject_type": "logo", "key_elements": ["badge"], "text": self.text,
                        "color_count": 3, "fine_detail_risk": "low"}
            Path(args[args.index("-o") + 1]).write_text(json.dumps(data), encoding="utf8")
        else:
            (Path(args[args.index("-C") + 1]) / "result.png").write_bytes(self.picture)
        return 0, '{"type":"turn.completed"}', ""

    def runs(self, kind):
        return [c for c in self.commands
                if c[1] == "exec" and ("--output-schema" in c) == (kind == "vision")]


@pytest.fixture
def codex(monkeypatch):
    def make(**kw):
        fake = FakeCodex(**kw)
        monkeypatch.setattr(chatgpt, "find", lambda: "codex")
        monkeypatch.setattr(chatgpt, "_run", fake)
        chatgpt.forget_status()
        return fake
    return make


@pytest.fixture
def badge():
    return (IMAGES / "logo_badge.png").read_bytes()


# ------------------------------------------------------------------- which way in is used

def test_chatgpt_plan_is_used_when_it_is_ready(codex):
    codex()
    assert ai.provider() == "chatgpt" and ai.available()
    st = ai.status()
    assert st["provider"] == "chatgpt" and st["chatgpt"] == "ready" and st["problem"] is None


def test_plan_wins_over_a_saved_key_unless_the_user_chooses_the_key(codex):
    codex()
    settings.save({"openai_api_key": "sk-test-1234567890abcdefWXYZ"})
    assert ai.provider() == "chatgpt"
    settings.save({"ai_provider": "api"})
    assert ai.provider() == "api"


def test_nothing_set_up_means_off_with_plain_words():
    assert not ai.available()
    with pytest.raises(ai.AIUnavailable) as e:
        ai.provider()
    assert e.value.reason == "off" and "Settings" in e.value.message
    settings.save({"ai_provider": "api"})
    with pytest.raises(ai.AIUnavailable) as e:
        ai.provider()
    assert e.value.reason == "no_key"


def test_signed_out_and_missing_are_told_apart(codex):
    codex(signed_in=False)
    with pytest.raises(ai.AIUnavailable) as e:
        ai.provider()
    assert e.value.reason == "chatgpt_signed_out" and "Sign in" in e.value.message
    assert ai.status()["chatgpt"] == "signed_out"


def test_signed_out_falls_back_to_a_saved_key(codex):
    codex(signed_in=False)
    settings.save({"openai_api_key": "sk-test-1234567890abcdefWXYZ"})
    assert ai.provider() == "api"


# ------------------------------------------------------------------------------ the calls

def test_looking_at_a_picture_runs_codex_read_only(codex, badge):
    fake = codex()
    found = ai.analyze(badge)
    assert found["text"] == ["TEST"] and found["subject_type"] == "logo"
    cmd = fake.runs("vision")[0]
    assert cmd[cmd.index("-s") + 1] == "read-only"
    assert "--ephemeral" in cmd and "--skip-git-repo-check" in cmd
    assert any(a.startswith("--image=") and a.endswith("picture.png") for a in cmd)
    assert "Don't run any commands" in fake.prompts[-1]


def test_redraw_asks_for_one_picture_and_returns_a_png(codex, badge):
    fake = codex()
    out = ai.redraw(badge, "Redraw this image as embroidery-ready artwork.")
    assert out[:4] == b"\x89PNG"
    cmd = fake.runs("redraw")[0]
    assert cmd[cmd.index("-s") + 1] == "workspace-write"
    assert "image generation tool" in fake.prompts[-1]
    assert "Redraw this image as embroidery-ready artwork." in fake.prompts[-1]
    assert "result.png" in fake.prompts[-1]


def test_plan_calls_cost_nothing_and_ignore_the_spending_limit(codex, badge):
    codex()
    settings.save({"daily_spend_cap_usd": 0})          # would block every pay-as-you-go call
    ai.redraw(badge, "x")
    ai.analyze(badge)
    assert ai.spent_today() == 0


def test_plan_results_are_cached(codex, badge):
    fake = codex()
    assert ai.redraw(badge, "x") == ai.redraw(badge, "x")
    assert len(fake.runs("redraw")) == 1
    ai.redraw(badge, "x", attempt=1)
    assert len(fake.runs("redraw")) == 2


def test_a_paid_key_in_the_environment_is_never_passed_to_codex(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-be-used")
    assert "OPENAI_API_KEY" not in chatgpt._env()


def test_temporary_files_are_cleaned_up(codex, badge):
    codex()
    ai.redraw(badge, "x")
    ai.analyze(badge)
    left = [p.name for p in chatgpt._base().iterdir()]
    assert left == ["empty"]


@pytest.mark.parametrize("kw, reason", [
    ({"fail": "You've hit your usage limit. Try again later."}, "chatgpt_limit"),
    ({"fail": "401 Unauthorized: please log in again"}, "chatgpt_signed_out"),
    ({"fail": "something unexpected"}, "error"),
    ({"timeout": True}, "slow"),
])
def test_codex_problems_become_plain_reasons(codex, badge, kw, reason):
    codex(**kw)
    with pytest.raises(ai.AIUnavailable) as e:
        ai.redraw(badge, "x")
    assert e.value.reason == reason and e.value.message


# -------------------------------------------------------------------------------- clean up

def test_cleanup_with_the_plan_makes_three_versions_and_no_polish(tmp_path, codex, badge):
    fake = codex()
    job = tmp_path / "job"
    r = cleanup.run_cleanup(badge, job, fit=FIT, fabric_id="tshirt")
    assert [v["id"] for v in r["versions"]] == ["faithful", "simplified", "bold", "local"]
    assert r["ai"]["used"] and r["default"] == "simplified" and r["spend"]["provider"] == "chatgpt"
    assert len(fake.runs("redraw")) == 3
    # the plan draws full quality the first time, so choosing a version costs no extra call
    assert cleanup.picture_to_stitch(job, "faithful") == (job / "faithful.png").read_bytes()
    assert len(fake.runs("redraw")) == 3 and not (job / "final_faithful.png").exists()


def test_skip_ai_makes_only_the_local_version_and_calls_nothing(tmp_path, codex, badge):
    fake = codex()
    r = cleanup.run_cleanup(badge, tmp_path / "job", fit=FIT, fabric_id="tshirt", use_ai=False)
    assert [v["id"] for v in r["versions"]] == ["local"]
    assert r["ai"]["problem"] == "skipped" and r["ai"]["note"] is None
    assert fake.runs("vision") == [] and fake.runs("redraw") == []


def test_plan_limit_falls_back_to_the_local_version(tmp_path, codex, badge):
    codex(fail="usage limit reached")
    r = cleanup.run_cleanup(badge, tmp_path / "job", fit=FIT, fabric_id="tshirt")
    assert [v["id"] for v in r["versions"]] == ["local"]
    assert r["ai"]["problem"] == "chatgpt_limit" and "allowance" in r["ai"]["note"]
