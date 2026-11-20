"""Stage 10 — Export.

Write the machine files with pyembroidery, read each one back to prove it holds what we meant
to write, and draw the preview picture.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pyembroidery
from pyembroidery import COLOR_CHANGE, END, JUMP, STITCH, TRIM, EmbPattern, EmbThread

from .common import canvas_for, save_debug

UNITS_PER_MM = 10.0          # embroidery files count in tenths of a millimetre
FORMATS = ("pes", "dst", "jef", "exp", "vp3", "xxx")


def build_pattern(plan: dict) -> EmbPattern:
    pattern = EmbPattern()
    first = True
    last = (0.0, 0.0)
    for n, block in enumerate(plan["blocks"]):
        r, g, b = block["rgb"]
        thread = EmbThread()
        thread.color = (int(r) << 16) | (int(g) << 8) | int(b)
        thread.description = f"Color {n + 1}"
        pattern.add_thread(thread)
        for s, seg in enumerate(block["segments"]):
            x0, y0 = seg[0][0] * UNITS_PER_MM, seg[0][1] * UNITS_PER_MM
            if not first:
                pattern.add_stitch_absolute(TRIM, last[0], last[1])
                if s == 0:
                    pattern.add_stitch_absolute(COLOR_CHANGE, last[0], last[1])
            pattern.add_stitch_absolute(JUMP, x0, y0)
            for x, y in seg:
                last = (x * UNITS_PER_MM, y * UNITS_PER_MM)
                pattern.add_stitch_absolute(STITCH, last[0], last[1])
            first = False
    pattern.add_stitch_absolute(END, last[0], last[1])
    return pattern


def _read_back(path: Path) -> dict:
    """Open a written file again and measure what is really in it."""
    p = pyembroidery.read(str(path))
    if p is None:
        return {"stitches": 0, "width_mm": 0.0, "height_mm": 0.0}
    pts = [(s[0], s[1]) for s in p.stitches if (s[2] & 0xFF) == STITCH]
    if not pts:
        return {"stitches": 0, "width_mm": 0.0, "height_mm": 0.0}
    a = np.array(pts, dtype=float)
    return {"stitches": len(pts),
            "width_mm": float((a[:, 0].max() - a[:, 0].min()) / UNITS_PER_MM),
            "height_mm": float((a[:, 1].max() - a[:, 1].min()) / UNITS_PER_MM)}


def write_files(plan: dict, out_dir: Path, name: str, formats=FORMATS) -> dict:
    """Write one file per format. Each result says whether the read-back check passed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = build_pattern(plan)
    minx, miny, maxx, maxy = plan["bounds"]
    want_w, want_h = maxx - minx, maxy - miny
    results = {}
    for fmt in formats:
        path = out_dir / f"{name}.{fmt}"
        pyembroidery.write(pattern, str(path))
        back = _read_back(path)
        ok = (path.exists() and path.stat().st_size > 0
              and back["stitches"] == plan["stitch_count"]
              and abs(back["width_mm"] - want_w) <= 0.3
              and abs(back["height_mm"] - want_h) <= 0.3)
        results[fmt] = {"path": str(path), "ok": bool(ok), **back}
    return results


def render_preview(plan: dict, px_per_mm: float = 10.0, fabric_rgb=(238, 236, 230)) -> np.ndarray:
    """A simple thread-look picture (BGR) of the plan: every stitch drawn as a thread-thick
    line with a lighter highlight down the middle."""
    img, to_px = canvas_for(plan["bounds"], px_per_mm, pad_mm=4.0, color=fabric_rgb[::-1])
    thick = max(1, int(round(0.38 * px_per_mm)))
    for block in plan["blocks"]:
        rgb = np.array(block["rgb"], dtype=float)
        dark = tuple(int(v) for v in (rgb * 0.72)[::-1])
        base = tuple(int(v) for v in rgb[::-1])
        light = tuple(int(v) for v in (rgb + (255 - rgb) * 0.35)[::-1])
        for seg in block["segments"]:
            allpx = np.round(to_px(seg) * 8).astype(np.int32)    # 1/8 pixel precision
            # draw in sewing order, a few stitches at a time, so later thread covers earlier
            for i in range(0, len(allpx) - 1, 40):
                px = allpx[i:i + 41]
                cv2.polylines(img, [px], False, dark, thick + 1, cv2.LINE_AA, shift=3)
                cv2.polylines(img, [px], False, base, thick, cv2.LINE_AA, shift=3)
                if thick >= 3:
                    cv2.polylines(img, [px], False, light, 1, cv2.LINE_AA, shift=3)
    return img


def run(plan: dict, out_dir: Path, name: str, debug_dir: Path | None = None) -> dict:
    files = write_files(plan, out_dir, name)
    preview = render_preview(plan)
    ok, buf = cv2.imencode(".png", preview)
    preview_path = out_dir / f"{name}_preview.png"
    if ok:
        preview_path.write_bytes(buf.tobytes())
    save_debug(debug_dir, "10_preview.png", preview)
    return {"files": files, "preview": str(preview_path)}
