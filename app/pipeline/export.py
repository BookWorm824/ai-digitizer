"""Stage 10 — Export.

Write the machine files with pyembroidery, read each one back to prove it holds what we meant
to write, and draw the preview picture.
"""

from __future__ import annotations

import math
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


def _thread_light(dx: float, dy: float) -> float:
    """How bright a thread looks by the way it runs (light from the upper left)."""
    return 0.9 + 0.1 * math.cos(2 * math.atan2(dy, dx))   # same for both zig-zag slants


THREAD_MM = 0.32


def render_preview(plan: dict, px_per_mm: float = 10.0, fabric_rgb=(238, 236, 230)) -> np.ndarray:
    """A thread-look picture (BGR) of the plan. Every stitch is drawn as its own piece of
    thread, in sewing order: a soft shadow, the thread shaded by its direction (which is what
    makes satin shine and fills look woven), and a thin highlight. Drawn at twice the size
    and shrunk, for clean edges."""
    k = 2
    ppm = px_per_mm * k
    img, to_px = canvas_for(plan["bounds"], ppm, pad_mm=4.0, color=fabric_rgb[::-1])
    # 40-weight thread lies about 0.3 mm wide. Drawn any wider, a thin fill or a single run
    # looks more solid on screen than it will sew.
    lw = max(1, int(round(THREAD_MM * ppm)))
    core = max(1, int(round(0.85 * THREAD_MM * ppm)))
    sheen = max(1, int(round(0.1 * ppm)))
    off = max(1, int(round(0.12 * ppm)))
    # Two passes. First every stitch's shadow, onto the fabric; then the thread. A shadow
    # drawn stitch by stitch (or in runs of stitches) lands on the thread sewn just before
    # it, which streaks satin columns and draws dark lines across fills that aren't there.
    drawn = []
    for block in plan["blocks"]:
        rgb = np.array(block["rgb"], dtype=float)
        shade = tuple(int(v) for v in (rgb * 0.45 * 0.7 + np.array(fabric_rgb) * 0.3)[::-1])
        for seg in block["segments"]:
            px = np.round(to_px(seg)).astype(np.int32)
            pairs = [((int(a[0]), int(a[1])), (int(b[0]), int(b[1])))
                     for a, b in zip(px[:-1], px[1:])
                     if (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 >= (0.05 * ppm) ** 2]
            for pa, pb in pairs:
                cv2.line(img, (pa[0] + off, pa[1] + off), (pb[0] + off, pb[1] + off), shade,
                         lw, cv2.LINE_AA)
            drawn.append((rgb, pairs))
    for rgb, pairs in drawn:
        for pa, pb in pairs:
            lit = _thread_light(pb[0] - pa[0], pb[1] - pa[1])
            cv2.line(img, pa, pb, tuple(int(min(255, v)) for v in (rgb * 0.84 * lit)[::-1]),
                     lw, cv2.LINE_AA)
            cv2.line(img, pa, pb, tuple(int(min(255, v)) for v in (rgb * lit)[::-1]),
                     core, cv2.LINE_AA)
            hi = rgb * lit * 1.1 + (255 - rgb * lit * 1.1) * 0.3
            cv2.line(img, pa, pb, tuple(int(min(255, v)) for v in hi[::-1]), sheen,
                     cv2.LINE_AA)
    h, w = img.shape[:2]
    return cv2.resize(img, (w // k, h // k), interpolation=cv2.INTER_AREA)


def run(plan: dict, out_dir: Path, name: str, debug_dir: Path | None = None) -> dict:
    files = write_files(plan, out_dir, name)
    preview = render_preview(plan)
    ok, buf = cv2.imencode(".png", preview)
    preview_path = out_dir / f"{name}_preview.png"
    if ok:
        preview_path.write_bytes(buf.tobytes())
    save_debug(debug_dir, "10_preview.png", preview)
    return {"files": files, "preview": str(preview_path)}
