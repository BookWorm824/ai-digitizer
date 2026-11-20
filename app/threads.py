"""Thread matching: find the real thread closest to each color in a design.

The charts are JSON files in app/threads/ (one per brand). "Closest" is measured in Lab color
space, which follows how different two colors look to the eye far better than raw RGB.

Screen colors are only ever approximate, so the worksheet tells the user to check the match
against a real thread card before buying.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

THREAD_DIR = Path(__file__).resolve().parent / "threads"
DEFAULT_BRAND = "isacord"


def _lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (0-255) to CIE Lab, for an N x 3 array."""
    c = rgb.astype(np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    return np.column_stack([116.0 * f[:, 1] - 16.0, 500.0 * (f[:, 0] - f[:, 1]),
                            200.0 * (f[:, 1] - f[:, 2])])


@lru_cache(maxsize=None)
def chart(brand_id: str) -> dict:
    """One brand's chart, with its colors already converted to Lab."""
    path = THREAD_DIR / f"{brand_id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["lab"] = _lab(np.array([t["rgb"] for t in data["threads"]]))
    return data


def brands() -> list[dict]:
    """The brands on offer: [{id, brand, line, count}], default first."""
    out = []
    for p in sorted(THREAD_DIR.glob("*.json")):
        c = chart(p.stem)
        out.append({"id": c["id"], "brand": c["brand"], "line": c["line"],
                    "count": len(c["threads"])})
    return sorted(out, key=lambda b: (b["id"] != DEFAULT_BRAND, b["brand"]))


def match(rgb, brand_id: str = DEFAULT_BRAND) -> dict:
    """The thread in this brand closest to an (r, g, b) color."""
    c = chart(brand_id if (THREAD_DIR / f"{brand_id}.json").exists() else DEFAULT_BRAND)
    d = np.linalg.norm(c["lab"] - _lab(np.array([rgb]))[0], axis=1)
    i = int(d.argmin())
    t = c["threads"][i]
    label = " ".join(x for x in (t["number"], t["name"]) if x)
    return {"brand": c["brand"], "number": t["number"], "name": t["name"], "label": label,
            "hex": "#%02x%02x%02x" % tuple(t["rgb"]), "difference": round(float(d[i]), 1)}


def match_all(rgb) -> dict:
    """The closest thread in every brand, keyed by brand id."""
    return {b["id"]: match(rgb, b["id"]) for b in brands()}
