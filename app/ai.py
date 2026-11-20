"""Everything that talks to OpenAI lives here.

Three calls (BRIEF.md, "The ChatGPT stage"):
  analyze()    vision model -> facts about the picture, as JSON
  read_text()  vision model -> the text in a picture, for the spelling check
  redraw()     image edit   -> the picture redrawn as embroidery-ready art

Guardrails, also here so no caller can forget them:
  * no key, offline, bad key, out of credit, any error -> AIUnavailable with a plain-words
    reason; callers then fall back to the local clean-up, so the app always makes a file
  * every result is cached on disk, so the same request is never paid for twice
  * every call's cost is added to today's total, and nothing is sent once the daily
    spending limit would be passed
"""

from __future__ import annotations

import base64
import datetime
import hashlib
import io
import json
import threading
import traceback

from PIL import Image

from . import config, settings

REASONS = {
    "no_key": "AI clean-up is off. Add your OpenAI key in Settings to turn it on.",
    "cap": "You've reached today's AI spending limit, so this is the simple clean-up. "
           "You can raise the limit in Settings.",
    "offline": "We couldn't reach OpenAI. Check your internet connection. "
               "This is the simple clean-up for now.",
    "bad_key": "OpenAI didn't accept your key. Check it in Settings. "
               "This is the simple clean-up for now.",
    "quota": "Your OpenAI account is out of credit. Add credit at platform.openai.com. "
             "This is the simple clean-up for now.",
    "blocked": "OpenAI wouldn't redraw this picture, so this is the simple clean-up.",
    "error": "The AI clean-up didn't work this time, so this is the simple clean-up.",
}


class AIUnavailable(Exception):
    """The AI can't be used right now. `.reason` is a key of REASONS."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason if reason in REASONS else "error"

    @property
    def message(self) -> str:
        return REASONS[self.reason]


# ------------------------------------------------------------------------------- client

_client_factory = None       # tests put a stand-in here; None means the real OpenAI client


def set_client_factory(factory) -> None:
    global _client_factory
    _client_factory = factory


def _client():
    if _client_factory is not None:
        return _client_factory()
    key = settings.load().get("openai_api_key") or ""
    if not key:
        raise AIUnavailable("no_key")
    try:
        from openai import OpenAI
    except Exception:  # noqa: BLE001
        raise AIUnavailable("error")
    return OpenAI(api_key=key, timeout=180.0, max_retries=1)


def available() -> bool:
    return _client_factory is not None or bool(settings.load().get("openai_api_key"))


def _log(text: str) -> None:
    try:
        with open(config.data_dir() / "log.txt", "a", encoding="utf-8") as f:
            f.write(text + "\n")
    except OSError:
        pass


def _call(fn):
    """Run one API call, turning any failure into AIUnavailable with a plain reason."""
    try:
        return fn()
    except AIUnavailable:
        raise
    except Exception as e:  # noqa: BLE001
        name, text = type(e).__name__, str(e).lower()
        _log(f"OpenAI call failed: {name}\n{traceback.format_exc()}")
        if "Authentication" in name or "PermissionDenied" in name:
            raise AIUnavailable("bad_key")
        if "RateLimit" in name:
            raise AIUnavailable("quota" if ("quota" in text or "billing" in text) else "error")
        if "Connection" in name or "Timeout" in name:
            raise AIUnavailable("offline")
        if "moderation" in text or "safety" in text:
            raise AIUnavailable("blocked")
        raise AIUnavailable("error")


# -------------------------------------------------------------------------- money: spending

_money_lock = threading.Lock()


def _read(name: str) -> dict:
    try:
        return json.loads((config.data_dir() / name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write(name: str, data: dict) -> None:
    (config.data_dir() / name).write_text(json.dumps(data, indent=2), encoding="utf-8")


def _today() -> str:
    return datetime.date.today().isoformat()


def spent_today() -> float:
    return float(_read("spend.json").get(_today(), 0.0))


def estimate(kind: str) -> float:
    """Expected cost of one call: the average of real past calls, or a rough guess until
    there have been some."""
    stats = _read("ai_stats.json").get(kind)
    if stats and stats.get("n"):
        return stats["total"] / stats["n"]
    return float(settings.ai_defaults()["estimate_usd"].get(kind, 0.02))


def _record(kind: str, usd: float) -> None:
    with _money_lock:
        spend = _read("spend.json")
        spend[_today()] = round(float(spend.get(_today(), 0.0)) + usd, 6)
        _write("spend.json", spend)
        stats = _read("ai_stats.json")
        s = stats.setdefault(kind, {"n": 0, "total": 0.0})
        s["n"] += 1
        s["total"] = round(s["total"] + usd, 6)
        _write("ai_stats.json", stats)


def _check_cap(kind: str) -> None:
    cap = float(settings.load()["daily_spend_cap_usd"])
    if spent_today() + estimate(kind) > cap + 1e-9:
        raise AIUnavailable("cap")


def estimate_run(has_text: bool = True) -> float:
    """Rough cost of cleaning up one picture: 1 analysis + 3 previews + 1 final."""
    s = settings.load()
    pq, fq = s["quality"]["preview"], s["quality"]["final"]
    total = estimate("analyze") + 3 * estimate(f"redraw_{pq}")
    checks = 3
    if s["final_polish"]:
        total += estimate(f"redraw_{fq}")
        checks += 1
    if has_text:
        total += checks * estimate("read_text")
    return total


def status() -> dict:
    s = settings.public()
    return {"has_key": s["has_key"] or _client_factory is not None, "key_hint": s["key_hint"],
            "daily_spend_cap_usd": s["daily_spend_cap_usd"],
            "spent_today_usd": round(spent_today(), 4),
            "estimate_usd": round(estimate_run(), 3)}


def _num(obj, name: str) -> float:
    v = getattr(obj, name, None) if obj is not None else None
    return float(v) if isinstance(v, (int, float)) else 0.0


def _text_cost(model: str, usage) -> float | None:
    if usage is None:
        return None
    prices = settings.ai_defaults()["prices_per_million_tokens"]["text_models"]
    p = prices.get(model, prices["default"])
    return (_num(usage, "input_tokens") * p["in"] + _num(usage, "output_tokens") * p["out"]) / 1e6


def _image_cost(model: str, usage) -> float | None:
    if usage is None:
        return None
    prices = settings.ai_defaults()["prices_per_million_tokens"]["image_models"]
    p = prices.get(model, prices["default"])
    details = getattr(usage, "input_tokens_details", None)
    img_in, txt_in = _num(details, "image_tokens"), _num(details, "text_tokens")
    if details is None:
        img_in = _num(usage, "input_tokens")
    return (txt_in * p["text_in"] + img_in * p["image_in"]
            + _num(usage, "output_tokens") * p["image_out"]) / 1e6


# ---------------------------------------------------------------------------------- cache

def _cache_key(*parts) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part if isinstance(part, bytes) else json.dumps(part, sort_keys=True).encode())
        h.update(b"\x00")
    return h.hexdigest()[:40]


def _cache_path(key: str, ext: str):
    d = config.data_dir() / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.{ext}"


# ---------------------------------------------------------------------------------- calls

def _shrunk_png(png: bytes, max_side: int) -> bytes:
    """A smaller PNG copy, so we don't pay to send more pixels than the model needs."""
    img = Image.open(io.BytesIO(png)).convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _vision_json(kind: str, png: bytes, instruction: str, schema_name: str, schema: dict) -> dict:
    model = settings.load()["models"]["analyze"]
    small = _shrunk_png(png, 1024)
    key = _cache_key(kind, model, instruction, schema, hashlib.sha256(small).hexdigest())
    path = _cache_path(key, "json")
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            pass
    client = _client()
    _check_cap(kind)
    b64 = base64.b64encode(small).decode("ascii")
    resp = _call(lambda: client.responses.create(
        model=model,
        input=[{"role": "user", "content": [
            {"type": "input_text", "text": instruction},
            {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"},
        ]}],
        text={"format": {"type": "json_schema", "name": schema_name,
                         "schema": schema, "strict": True}},
    ))
    cost = _text_cost(model, getattr(resp, "usage", None))
    _record(kind, cost if cost is not None else estimate(kind))
    try:
        data = json.loads(resp.output_text)
    except (ValueError, TypeError, AttributeError):
        _log(f"OpenAI {kind}: reply was not JSON")
        raise AIUnavailable("error")
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "subject_type": {"type": "string", "enum": ["logo", "cartoon", "text", "photo", "other"]},
        "key_elements": {"type": "array", "items": {"type": "string"}},
        "text": {"type": "array", "items": {"type": "string"}},
        "color_count": {"type": "integer"},
        "fine_detail_risk": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["subject_type", "key_elements", "text", "color_count", "fine_detail_risk"],
    "additionalProperties": False,
}

TEXT_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "array", "items": {"type": "string"}}},
    "required": ["text"],
    "additionalProperties": False,
}

READ_TEXT_PROMPT = (
    "List every piece of text that appears in this picture, exactly as written, character "
    "for character, including capitals and punctuation. One entry per separate line or block "
    "of text, in reading order. If there is no text, return an empty list. Do not correct "
    "spelling and do not guess at text that is not there.")

ANALYZE_PROMPT = (
    "This picture is going to be turned into machine embroidery. Describe it:\n"
    "- subject_type: logo, cartoon, text (mostly lettering), photo, or other.\n"
    "- key_elements: the few things that must stay recognizable, most important first.\n"
    "- text: " + READ_TEXT_PROMPT + "\n"
    "- color_count: how many distinct flat colors the design really uses, not counting the "
    "background.\n"
    "- fine_detail_risk: low, medium or high: how much of the design is thin lines, tiny "
    "shapes, gradients or photo detail that thread cannot reproduce.")


def analyze(png: bytes) -> dict:
    return _vision_json("analyze", png, ANALYZE_PROMPT, "embroidery_analysis", ANALYSIS_SCHEMA)


def read_text(png: bytes) -> list[str]:
    data = _vision_json("read_text", png, READ_TEXT_PROMPT, "text_read", TEXT_SCHEMA)
    return [str(t) for t in data.get("text", [])]


def _edit_size(png: bytes) -> str:
    w, h = Image.open(io.BytesIO(png)).size
    if w / h > 1.25:
        return "1536x1024"
    if w / h < 0.8:
        return "1024x1536"
    return "1024x1024"


def redraw(png: bytes, prompt: str, *, final: bool = False, attempt: int = 0) -> bytes:
    """Image edit: returns the redrawn picture as PNG bytes. `attempt` > 0 asks for a fresh
    try instead of the cached result."""
    s = settings.load()
    model = s["models"]["redraw_final" if final else "redraw_preview"]
    quality = s["quality"]["final" if final else "preview"]
    kind = f"redraw_{quality}"
    small = _shrunk_png(png, 1536)
    size = _edit_size(small)
    key = _cache_key("redraw", model, quality, size, prompt, attempt,
                     hashlib.sha256(small).hexdigest())
    path = _cache_path(key, "png")
    if path.exists() and path.stat().st_size > 0:
        return path.read_bytes()
    client = _client()
    _check_cap(kind)
    resp = _call(lambda: client.images.edit(
        model=model,
        image=[("picture.png", small, "image/png")],
        prompt=prompt,
        size=size,
        quality=quality,
        n=1,
        background="opaque",
        output_format="png",
    ))
    cost = _image_cost(model, getattr(resp, "usage", None))
    _record(kind, cost if cost is not None else estimate(kind))
    try:
        out = base64.b64decode(resp.data[0].b64_json)
        Image.open(io.BytesIO(out)).verify()
    except Exception:  # noqa: BLE001
        _log("OpenAI redraw: reply did not contain a picture")
        raise AIUnavailable("error")
    path.write_bytes(out)
    return out
