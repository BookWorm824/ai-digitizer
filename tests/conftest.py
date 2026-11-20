"""Shared test setup: every test gets its own empty data folder and no AI unless it asks."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import make_images
from app import ai

IMAGES = Path(__file__).parent / "images"


@pytest.fixture(scope="session", autouse=True)
def _images():
    if not (IMAGES / "logo_badge.png").exists():
        make_images.main()


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    """Settings, spending, cache and jobs all go to a throwaway folder."""
    d = tmp_path / "appdata"
    monkeypatch.setenv("AIDIGITIZER_DATA", str(d))
    ai.set_client_factory(None)
    yield d
    ai.set_client_factory(None)


class FakeOpenAI:
    """Stands in for the OpenAI client: answers like the real one, costs nothing, and counts
    how many calls were made."""

    def __init__(self, picture: bytes, text=("TEST",), redraw_text=None, fail_with=None):
        self.picture = picture
        self.text = list(text)
        self.redraw_text = list(text if redraw_text is None else redraw_text)
        self.fail_with = fail_with
        self.vision_calls = 0
        self.image_calls = 0
        self.prompts: list[str] = []
        self.image_kwargs: list[dict] = []
        self.responses = SimpleNamespace(create=self._responses_create)
        self.images = SimpleNamespace(edit=self._images_edit)

    def _responses_create(self, **kw):
        if self.fail_with:
            raise self.fail_with
        self.vision_calls += 1
        name = kw["text"]["format"]["name"]
        assert kw["text"]["format"]["strict"] is True
        assert kw["input"][0]["content"][1]["image_url"].startswith("data:image/png;base64,")
        if name == "embroidery_analysis":
            data = {"subject_type": "logo", "key_elements": ["badge", "star"], "text": self.text,
                    "color_count": 3, "fine_detail_risk": "low"}
        else:
            data = {"text": self.redraw_text}
        return SimpleNamespace(output_text=json.dumps(data),
                               usage=SimpleNamespace(input_tokens=1000, output_tokens=100))

    def _images_edit(self, **kw):
        if self.fail_with:
            raise self.fail_with
        self.image_calls += 1
        self.prompts.append(kw["prompt"])
        self.image_kwargs.append(kw)
        usage = SimpleNamespace(
            input_tokens=600, output_tokens=300,
            input_tokens_details=SimpleNamespace(image_tokens=400, text_tokens=200))
        return SimpleNamespace(
            data=[SimpleNamespace(b64_json=base64.b64encode(self.picture).decode())], usage=usage)


@pytest.fixture
def fake_ai():
    """Turn the AI on with a stand-in client. Returns the stand-in so tests can inspect it."""
    def make(**kw):
        client = FakeOpenAI((IMAGES / "logo_blocks.png").read_bytes(), **kw)
        ai.set_client_factory(lambda: client)
        return client
    return make
