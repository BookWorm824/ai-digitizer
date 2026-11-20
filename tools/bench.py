"""The digitizer's training bench: digitize a set of test designs at several sizes, score every
result, and keep the scores so each change can be measured against the last.

    .venv\\Scripts\\python.exe tools\\bench.py                 # all designs, all sizes
    .venv\\Scripts\\python.exe tools\\bench.py --no-ai         # measurable checks only
    .venv\\Scripts\\python.exe tools\\bench.py --set key=value  # try a different setting
    .venv\\Scripts\\python.exe tools\\bench.py --only tennessee --sizes 3.5

Designs live in the app's data folder (bench\\*.png, never committed: they may be customers'
logos). Each run writes bench\\runs\\<time>\\ with the stitch previews, the side-by-side
pictures and scores.json, and prints how it compares with the previous run.

Two kinds of score:
  checks   measured from the stitches (reports/Embroidery digitizing deep dive.md, usability
           rubric): trims per 1000 stitches, stitches that are too short or too long, density
           hot spots (too many needle holes in a 2 mm circle), satin columns that are too wide
           or too narrow, stitches per square inch.
  critique the local vision model (qwen3-vl through Ollama, as CleanView uses) looks at the
           art beside the stitch preview and scores it like a strict customer and digitizer.
"""

from __future__ import annotations

import argparse
import base64
import datetime
import io
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.pipeline import run_pipeline  # noqa: E402

SIZES = {2.25: "cap", 3.5: "left chest", 10.0: "jacket back"}
OLLAMA = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen3-vl:8b-instruct"

CRITIQUE_SCHEMA = {
    "type": "object",
    "properties": {
        "recognisable": {"type": "integer"},
        "clean_edges": {"type": "integer"},
        "text_readable": {"type": "integer"},
        "looks_professional": {"type": "integer"},
        "customer_accepts": {"type": "integer"},
        "problems": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["recognisable", "clean_edges", "text_readable", "looks_professional",
                 "customer_accepts", "problems"],
}
CRITIQUE_PROMPT = (
    "You are a strict professional embroidery digitizer and a demanding customer. LEFT: the "
    "customer's artwork. RIGHT: a simulation of the embroidered result at {size} inches "
    "({place}). Judge the RIGHT side as real embroidery. Score each 1-10 (10 = flawless, 5 = "
    "mediocre, 1 = unusable): recognisable (does it clearly read as the artwork), clean_edges "
    "(crisp shapes, straight lines straight, no ragged or wobbly edges), text_readable "
    "(lettering clear; 10 if there is no text), looks_professional (like a professional "
    "digitizer made it), customer_accepts (would a paying customer accept it). Then list up "
    "to 4 specific problems you can see, most important first. Be harsh and specific.")


def data_bench() -> Path:
    d = config.data_dir() / "bench"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------------------- checks

def checks(result: dict) -> dict:
    """Measurable quality from the finished stitches."""
    blocks = result["blocks"]
    segs = [np.asarray(list(zip(sg[0::2], sg[1::2]))) for b in blocks for sg in b["segments"]]
    stitches = sum(len(s) for s in segs)
    lengths = np.concatenate([np.linalg.norm(np.diff(s, axis=0), axis=1) for s in segs if len(s) > 1])
    # ties at the ends of every segment are deliberately short: ignore 3 stitches each end
    body = np.concatenate([np.linalg.norm(np.diff(s[3:-3], axis=0), axis=1)
                           for s in segs if len(s) > 8]) if segs else np.array([])
    short = int((body < 0.3).sum()) if len(body) else 0
    long_ = int((lengths > 7.0).sum()) if len(lengths) else 0
    # density hot spots: needle holes on a 0.5 mm grid, counted within 2 mm circles
    pts = np.vstack(segs) if segs else np.zeros((0, 2))
    hot = 0
    if len(pts):
        minx, miny = pts.min(axis=0)
        g = np.floor((pts - [minx, miny]) / 0.5).astype(int)
        grid = np.zeros(g.max(axis=0) + 5)
        np.add.at(grid, (g[:, 0], g[:, 1]), 1)
        from scipy.ndimage import uniform_filter  # noqa: PLC0415
        win = uniform_filter(grid, size=4, mode="constant") * 16      # 2 mm square
        hot = int((win > 90).sum())          # ~3+ full layers of 0.4 mm satin in 2 mm
    w, h = result["stats"]["width_mm"], result["stats"]["height_mm"]
    area_in2 = max(w * h / 645.16, 0.01)
    trims = result["stats"]["trims"]
    return {
        "stitches": stitches,
        "trims": trims,
        "trims_per_1000": round(1000 * trims / max(stitches, 1), 2),
        "colors": result["stats"]["colors"],
        "short_stitches": short,
        "long_stitches": long_,
        "hot_spots": hot,
        "stitches_per_in2": round(stitches / area_in2),
        "status": result["check"]["status"],
        "warnings": len(result["check"]["messages"]),
    }


def checks_score(c: dict) -> float:
    """0-10 from the checks (10 = no measurable problem)."""
    s = 10.0
    s -= min(3.0, max(0.0, c["trims_per_1000"] - 4.0) * 0.4)       # >4 trims per 1000
    s -= min(2.0, c["short_stitches"] / 50.0)
    s -= min(1.5, c["long_stitches"] / 10.0)
    s -= min(2.0, c["hot_spots"] / 40.0)
    if c["status"] == "red":
        s -= 1.5
    return round(max(0.0, s), 2)


# ----------------------------------------------------------------------------- critique

def side_by_side(art: bytes, preview_path: str) -> bytes:
    a = Image.open(io.BytesIO(art)).convert("RGBA")
    bg = Image.new("RGBA", a.size, (255, 255, 255, 255))
    bg.alpha_composite(a)
    a = bg.convert("RGB")
    b = Image.open(preview_path).convert("RGB")
    h = 700
    a = a.resize((max(1, int(a.width * h / a.height)), h))
    b = b.resize((max(1, int(b.width * h / b.height)), h))
    out = Image.new("RGB", (a.width + b.width + 30, h + 40), "white")
    out.paste(a, (0, 40))
    out.paste(b, (a.width + 30, 40))
    d = ImageDraw.Draw(out)
    d.text((10, 10), "ARTWORK", fill=(0, 0, 0))
    d.text((a.width + 40, 10), "EMBROIDERY SIMULATION", fill=(0, 0, 0))
    buf = io.BytesIO()
    out.save(buf, "PNG")
    return buf.getvalue()


def critique(png: bytes, size: float) -> dict | None:
    body = {"model": MODEL, "stream": False, "format": CRITIQUE_SCHEMA,
            "prompt": CRITIQUE_PROMPT.format(size=size, place=SIZES.get(size, "garment")),
            "images": [base64.b64encode(png).decode()],
            "options": {"temperature": 0.0, "num_predict": 400}}
    try:
        req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.loads(json.loads(r.read())["response"])
    except Exception as e:  # noqa: BLE001 - the bench still runs without the vision model
        print(f"   (vision model unavailable: {e})")
        return None
    keys = ["recognisable", "clean_edges", "text_readable", "looks_professional", "customer_accepts"]
    for k in keys:
        data[k] = int(max(1, min(10, data.get(k, 5))))
    data["score"] = round(sum(data[k] for k in keys) / len(keys), 2)
    return data


# ---------------------------------------------------------------------------------- run

def run(designs: list[Path], sizes: list[float], overrides: dict, use_ai: bool) -> dict:
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = data_bench() / "runs" / stamp
    out.mkdir(parents=True)
    results = {"time": stamp, "overrides": overrides, "items": []}
    for path in designs:
        art = path.read_bytes()
        for size in sizes:
            t = time.time()
            name = f"{path.stem}@{size:g}"
            hints = {}
            words = path.with_suffix(".txt")
            if words.exists():
                hints["text"] = [l.strip() for l in words.read_text(encoding="utf-8").splitlines() if l.strip()]
            r = run_pipeline(art, out / name, name, fit_w_mm=size * 25.4, fit_h_mm=size * 25.4,
                             fabric_id="hat" if size <= 2.5 else "tshirt", hints=hints,
                             overrides=overrides or None, debug=False)
            c = checks(r)
            item = {"design": path.stem, "size": size, "checks": c, "checks_score": checks_score(c),
                    "seconds": round(time.time() - t, 1)}
            if use_ai:
                sbs = side_by_side(art, r["preview"])
                (out / f"{name}_compare.png").write_bytes(sbs)
                item["critique"] = critique(sbs, size)
            results["items"].append(item)
            crit = item.get("critique") or {}
            print(f"{name:28s} checks {item['checks_score']:5.2f}  vision {crit.get('score', '-'):>5}  "
                  f"trims {c['trims']:4d} ({c['trims_per_1000']}/1000)  short {c['short_stitches']:4d}  "
                  f"hot {c['hot_spots']:4d}  {c['stitches']} st", flush=True)
            for p in (crit.get("problems") or [])[:3]:
                print(f"      - {p}")
    items = results["items"]
    results["mean_checks"] = round(float(np.mean([i["checks_score"] for i in items])), 2)
    vis = [i["critique"]["score"] for i in items if i.get("critique")]
    results["mean_vision"] = round(float(np.mean(vis)), 2) if vis else None
    (out / "scores.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    return results


def previous(before: str) -> dict | None:
    runs = sorted((data_bench() / "runs").glob("*/scores.json"))
    runs = [r for r in runs if r.parent.name < before]
    return json.loads(runs[-1].read_text(encoding="utf-8")) if runs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ai", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--sizes", default="")
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    designs = sorted(p for p in data_bench().glob("*.png") if a.only.lower() in p.stem.lower())
    if not designs:
        print(f"No designs: put test pictures (PNG) in {data_bench()}")
        return
    sizes = [float(s) for s in a.sizes.split(",")] if a.sizes else list(SIZES)
    overrides = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        try:
            overrides[k] = json.loads(v)
        except ValueError:
            overrides[k] = v
    res = run(designs, sizes, overrides, not a.no_ai)
    prev = previous(res["time"])
    print(f"\nmean checks {res['mean_checks']}  mean vision {res['mean_vision']}")
    if prev:
        print(f"previous    {prev['mean_checks']}  mean vision {prev.get('mean_vision')} "
              f"({prev['time']}, settings {prev.get('overrides') or 'default'})")


if __name__ == "__main__":
    main()
