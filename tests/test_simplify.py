"""The no-AI simplifier (app/pipeline/simplify.py): fine pen drawings are redrawn for their
stitched size; bold art passes through untouched."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw

from app.pipeline import intake, simplify


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _fine_building(px_per_mm: float = 40.0) -> bytes:
    """A 60 mm wide building drawn with 0.15 mm pen lines: walls, a roof line, and rows of
    windows each drawn as a frame with a 3 x 3 grid of panes."""
    w, h = int(70 * px_per_mm), int(40 * px_per_mm)
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    lw = max(1, int(0.15 * px_per_mm))

    def mm(x, y):
        return (int(x * px_per_mm), int(y * px_per_mm))
    d.rectangle([mm(5, 8), mm(65, 35)], outline="black", width=lw)          # walls
    d.line([mm(5, 8), mm(35, 2), mm(65, 8)], fill="black", width=lw)       # roof
    for row in range(3):
        for col in range(8):
            x0, y0 = 8 + col * 7, 12 + row * 7.5
            d.rectangle([mm(x0, y0), mm(x0 + 3, y0 + 4)], outline="black", width=lw)
            for k in (1, 2):
                d.line([mm(x0 + k, y0), mm(x0 + k, y0 + 4)], fill="black", width=lw)
                d.line([mm(x0, y0 + k * 4 / 3), mm(x0 + 3, y0 + k * 4 / 3)], fill="black", width=lw)
    return _png(img)


def _bold_building(px_per_mm: float = 40.0) -> bytes:
    w, h = int(70 * px_per_mm), int(40 * px_per_mm)
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    lw = int(1.3 * px_per_mm)

    def mm(x, y):
        return (int(x * px_per_mm), int(y * px_per_mm))
    d.rectangle([mm(5, 8), mm(65, 35)], outline="black", width=lw)
    for col in range(8):
        x0 = 8 + col * 7
        d.rectangle([mm(x0, 14), mm(x0 + 3, 19)], fill="black")
    return _png(img)


def test_a_fine_pen_drawing_is_found_and_redrawn_with_solid_windows():
    img = intake.run(_fine_building())
    a = simplify.analyse(img, 70.0, 40.0)
    assert len(a["drawings"]) == 1
    out = simplify.run(img, 70.0, 40.0)
    assert out is not None and out["simplified"]
    dark = (out["rgb"].mean(axis=2) < 100).astype(np.uint8)
    import cv2
    n, _lab, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    ppm = a["ppm"]
    # the 24 windows (each 3 x 4 mm of panes) are now solid blocks: big dark components
    # roughly window-sized, and no more tiny pane cells inside them
    solid = [k for k in range(1, n)
             if 6.0 * ppm * ppm <= stats[k, cv2.CC_STAT_AREA] <= 16.0 * ppm * ppm]
    assert len(solid) >= 20


def test_bold_art_passes_through_unchanged():
    img = intake.run(_bold_building())
    assert simplify.run(img, 70.0, 40.0) is None


def test_lettering_is_never_simplified():
    img = Image.new("RGB", (2400, 600), "white")
    d = ImageDraw.Draw(img)
    for k in range(10):                     # thin letter-sized marks, each its own shape
        x = 100 + k * 220
        d.rectangle([x, 150, x + 120, 450], outline="black", width=4)
    out = simplify.run(intake.run(_png(img)), 60.0, 15.0)
    assert out is None


def test_a_small_line_of_text_is_not_taken_for_a_drawing():
    # 20 thin letters close together (they group into one 3 mm tall, 50 mm wide shape)
    img = Image.new("RGB", (2600, 300), "white")
    d = ImageDraw.Draw(img)
    for k in range(20):
        x = 50 + k * 125
        d.line([(x, 60), (x, 240)], fill="black", width=3)
        d.line([(x, 60), (x + 90, 60)], fill="black", width=3)
        d.line([(x, 150), (x + 70, 150)], fill="black", width=3)
        d.line([(x, 240), (x + 90, 240)], fill="black", width=3)
    a = simplify.analyse(intake.run(_png(img)), 52.0, 6.0)
    assert a["drawings"] == []


def test_a_window_grid_too_fine_for_thread_is_thinned_to_fit():
    # a 4 x 3 grid of 1 mm windows 0.6 mm apart (cap size): fewer, bigger blocks with at least
    # a millimetre of wall between them, spread over the same field (cartographic typification)
    import cv2
    from app.pipeline import simplify as sp
    ppm = 20.0
    body = np.ones((400, 400), bool)
    cands = [((10 + 1.6 * k) * ppm, (10 + 1.6 * r) * ppm, 1.0 * ppm, 1.0 * ppm)
             for r in range(3) for k in range(4)]
    out = sp._window_grid(cands, ppm, body)
    n, lab, st, _ = cv2.connectedComponentsWithStats(out.astype(np.uint8), connectivity=4)
    blocks = n - 1
    assert blocks == 6                       # 4 x 3 becomes 3 x 2
    xs = sorted({int(st[i, 0]) for i in range(1, n)})
    widths = [st[i, 2] / ppm for i in range(1, n)]
    assert min(widths) >= 0.95
    gaps = [(b - a) / ppm - max(widths) for a, b in zip(xs, xs[1:])]
    assert all(g >= 0.9 for g in gaps)
    # the field is kept: blocks still span the original grid's width
    assert (st[1:, 0].min() / ppm) <= 10.0 and ((st[1:, 0] + st[1:, 2]).max() / ppm) >= 14.3


def test_marks_that_line_up_with_nothing_are_not_windows():
    from app.pipeline import simplify as sp
    ppm = 20.0
    body = np.ones((200, 200), bool)
    cands = [(20 * ppm, 20 * ppm, ppm, ppm), (70 * ppm, 45 * ppm, ppm, ppm),
             (40 * ppm, 80 * ppm, ppm, ppm), (90 * ppm, 10 * ppm, ppm, ppm)]
    assert not sp._window_grid(cands, ppm, body).any()
