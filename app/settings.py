"""The user's local settings: OpenAI key, model names, spending limit, favourites.

Saved as settings.json in the app's data folder on this PC (never in the code folder, never
on GitHub). The key is never sent back to the screen and never written to the log.
"""

from __future__ import annotations

import json
import threading

from . import config

_lock = threading.Lock()


def ai_defaults() -> dict:
    return config._load_json(str(config.PRESET_DIR / "ai.json"))


def _path():
    return config.data_dir() / "settings.json"


def load() -> dict:
    """Settings with defaults filled in."""
    d = ai_defaults()
    out = {
        "ai_provider": "auto",   # auto | chatgpt (the ChatGPT plan) | api (an OpenAI key)
        "openai_api_key": "",
        "models": dict(d["models"]),
        "quality": dict(d["quality"]),
        "final_polish": bool(d["final_polish"]),
        "daily_spend_cap_usd": float(d["daily_spend_cap_usd"]),
        "default_machine": "brother",
        "thread_brand": "Madeira Polyneon",
    }
    try:
        saved = json.loads(_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        saved = {}
    for k, v in saved.items():
        if k in ("models", "quality") and isinstance(v, dict):
            out[k].update({a: b for a, b in v.items() if isinstance(b, str) and b})
        elif k in out:
            out[k] = v
    return out


def save(changes: dict) -> dict:
    """Merge changes into the saved settings file and return the full settings."""
    with _lock:
        try:
            saved = json.loads(_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            saved = {}
        saved.update(changes)
        tmp = _path().with_suffix(".tmp")
        tmp.write_text(json.dumps(saved, indent=2), encoding="utf-8")
        tmp.replace(_path())
    return load()


def public() -> dict:
    """What the screens are allowed to see: everything except the key itself."""
    s = load()
    key = s.pop("openai_api_key", "") or ""
    s["has_key"] = bool(key)
    s["key_hint"] = ("…" + key[-4:]) if len(key) >= 12 else ""
    return s
