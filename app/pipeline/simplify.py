"""Stage 1b — Simplify fine line drawings for their stitched size (no AI needed).

A detailed pen drawing of a building (window panes, stonework, leafy trees) has lines a
fraction of a millimetre wide at left-chest size, far finer than thread. Traced as it is, it
sews as a tangle. A digitizer redraws such art for its size first; ChatGPT can do that
(the "Simplified" version), and this stage does it on this PC for when ChatGPT isn't used.

It follows cartographic generalization (reports/Auto digitizing engine deep research.md, the
building-simplifier rule): judge the drawing in millimetres at its final size, then

  * masses   where lines are packed closer than thread can separate them (a window drawn as
             a frame with panes, a scribbled tree), the area becomes one solid shape
  * lines    the long straight lines that carry the structure (walls, roof edges, floors) are
             kept and redrawn at a width thread can sew; short marks (stonework, hatching,
             pane bars, leaves) are dropped
  * outline  the whole drawing's silhouette is drawn as a line, so its shape survives

Only parts of the picture that are fine line drawings are touched: lettering, bold shapes
and colored areas are left as they are (lettering is retyped later). A colored area's own
hatching (lines drawn over a roof) is painted out in the area's color.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .common import save_debug

WORK_PPM = 20.0          # pixels per mm at the stitched size (0.05 mm per pixel)
FINE_STROKE_MM = 0.5     # a drawing whose typical line is thinner than this is too fine
GROUP_MM = 0.6           # ink closer than this belongs to one drawing
MIN_DRAWING_MM = 10.0    # only drawings at least this big are simplified (not letters)
MIN_JUNCTIONS = 40       # ... and only busy ones (a line drawing, not a simple shape)
TEXT_LINE_MM = 6.0       # a group shorter than this and 3 x as wide is a line of text
CELL_R_MM = 0.6          # a white cell must hold a 1.2 mm circle to stay open fabric
CELL_MIN_MM2 = 1.0       # ... and be at least this big; smaller cells are detail
SOLID_R_MM = 0.5         # ink (with its detail cells) 1 mm thick or more becomes solid
MIN_MASS_MM2 = 1.2        # smallest solid block (a window)
BIG_MASS_MM2 = 6.0        # solids this big are kept whatever their shape
SPUR_MM = 1.5            # marks hanging off the line work shorter than this are dropped
LINE_MIN_MM = 2.5        # loose line work shorter than this is dropped
LINE_W_MM = 1.0          # width structural lines are redrawn at (narrow satin)
SILHOUETTE_MM = 1.0      # closing that turns the drawing into one silhouette


def _disk(r_px: float) -> np.ndarray:
    k = max(1, int(round(r_px)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))


def _stroke_mm(mask: np.ndarray, ppm: float) -> float:
    """Typical stroke width of ink (mm): twice the distance to the edge, read along the
    skeleton (the middle of each stroke)."""
    from skimage.morphology import skeletonize   # noqa: PLC0415
    if mask.sum() < 20:
        return 0.0
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    sk = skeletonize(mask)
    vals = dt[sk]
    return float(2.0 * np.median(vals) / ppm) if len(vals) else 0.0


def _junctions(mask: np.ndarray) -> int:
    from skimage.morphology import skeletonize   # noqa: PLC0415
    sk = skeletonize(mask).astype(np.uint8)
    n = cv2.filter2D(sk, -1, np.ones((3, 3), np.float32)) - sk
    return int(((n >= 3) & (sk == 1)).sum())


def analyse(img: dict, fit_w_mm: float, fit_h_mm: float) -> dict:
    """Measure the picture at its stitched size. Returns {"ppm", "rgb", "ink", "colored",
    "drawings": [mask, ...]} where drawings are the fine, busy line drawings found."""
    rgb = img["rgb"]
    h, w = rgb.shape[:2]
    mm_per_px = min(fit_w_mm / w, fit_h_mm / h)
    ppm = min(WORK_PPM, 4096.0 / max(w * mm_per_px, h * mm_per_px))
    W, H = max(8, round(w * mm_per_px * ppm)), max(8, round(h * mm_per_px * ppm))
    small = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA if W < w else cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    val = hsv[:, :, 2].astype(np.float32) / 255.0
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    if img.get("alpha") is not None:
        a = cv2.resize(img["alpha"], (W, H), interpolation=cv2.INTER_AREA) >= 128
    else:
        a = np.ones((H, W), bool)
    colored = (sat > 0.35) & (val > 0.35) & a
    ink = (val < 0.55) & ~colored & a
    groups = cv2.dilate(ink.astype(np.uint8), _disk(GROUP_MM * ppm / 2.0))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(groups, connectivity=8)
    drawings = []
    for k in range(1, n):
        x, y, bw, bh = stats[k, :4]
        if max(bw, bh) / ppm < MIN_DRAWING_MM:
            continue
        if bh / ppm < TEXT_LINE_MM and bw >= 3 * bh:
            continue                     # a line of lettering: it is retyped, never redrawn
        region = lab == k
        part = ink & region
        if _stroke_mm(part, ppm) >= FINE_STROKE_MM:
            continue                     # bold lines or shapes: sewable as drawn
        # busy? count junctions of all the ink inside its outline (windows drawn as
        # islands inside a building's walls belong to it)
        whole = _filled_silhouette(part, colored & region, ppm) | region
        if _junctions(ink & whole) < MIN_JUNCTIONS:
            continue
        drawings.append(region)
    return {"ppm": ppm, "rgb": small, "ink": ink, "colored": colored, "alpha": a,
            "drawings": drawings}


def _thick(mask: np.ndarray, r_px: float) -> np.ndarray:
    """The parts of a mask at least 2 x r thick (morphological opening)."""
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, _disk(r_px)).astype(bool)


def _prune(sk: np.ndarray, length_px: int) -> np.ndarray:
    """Remove dangling skeleton branches shorter than length_px (a few passes of end-
    point peeling, then grow back what remains so long lines keep their ends)."""
    from skimage.morphology import skeletonize  # noqa: PLC0415
    k = np.ones((3, 3), np.float32)
    cur = sk.copy().astype(np.uint8)
    for _ in range(length_px):
        n = cv2.filter2D(cur, -1, k) - cur
        ends = (cur == 1) & (n <= 1)
        if not ends.any():
            break
        cur[ends] = 0
    # grow the survivors back along the original skeleton by the same length
    grown = cur.copy()
    for _ in range(length_px):
        grown = (cv2.dilate(grown, np.ones((3, 3), np.uint8)) & sk.astype(np.uint8))
    return skeletonize(grown.astype(bool))


def _filled_silhouette(part: np.ndarray, col: np.ndarray, ppm: float) -> np.ndarray:
    """The drawing's outline, filled: everything inside it (closing gaps up to 2 mm)."""
    inside = cv2.morphologyEx((part | col).astype(np.uint8), cv2.MORPH_CLOSE,
                              _disk(SILHOUETTE_MM * ppm))
    ff = inside.copy()
    h, w = ff.shape
    flood = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, flood, (0, 0), 1)
    return (inside | (ff == 0)).astype(bool)


def redraw(part: np.ndarray, col: np.ndarray, ppm: float) -> tuple:
    """Redraw one fine line drawing from its white cells. Returns (solid, lines, inside)
    masks: solid areas, structural lines (already at a sewable width) and the drawing's
    filled silhouette."""
    from skimage.morphology import skeletonize  # noqa: PLC0415
    inside = _filled_silhouette(part, col, ppm)
    # white cells inside the drawing; the ones too small or thin for thread are detail
    white = inside & ~part & ~col
    n, lab, stats, _ = cv2.connectedComponentsWithStats(white.astype(np.uint8), connectivity=4)
    roomy = _thick(white, CELL_R_MM * ppm)
    big = np.zeros(n, bool)
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < CELL_MIN_MM2 * ppm * ppm:
            continue
        x, y, bw, bh = stats[k, :4]
        sub = lab[y:y + bh, x:x + bw] == k
        big[k] = bool((roomy[y:y + bh, x:x + bw] & sub).any())
    open_space = big[lab] & white
    ink = inside & ~open_space & ~col                    # lines plus every too-small cell
    solid = _thick(ink, SOLID_R_MM * ppm)
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(solid.astype(np.uint8), connectivity=8)
    keep = np.zeros(n2, bool)
    for k in range(1, n2):
        area = st2[k, cv2.CC_STAT_AREA] / (ppm * ppm)
        if area >= BIG_MASS_MM2:
            keep[k] = True                               # a tree, a doorway, a big block
        elif area >= MIN_MASS_MM2:
            # a small solid is kept only if it is a block (a window): a lump on a line
            # (stonework packed against a wall) goes back to being line work
            ys, xs = np.nonzero(lab2 == k)
            pts = np.column_stack([xs, ys]).astype(np.float32)
            (_c, (rw, rh), _a) = cv2.minAreaRect(pts)
            keep[k] = st2[k, cv2.CC_STAT_AREA] >= 0.8 * max(rw * rh, 1.0)
    solid = keep[lab2]
    # what is left is line work: one centre line per wall, pruned of short marks
    rest = ink & ~cv2.dilate(solid.astype(np.uint8), _disk(0.15 * ppm)).astype(bool)
    sk = skeletonize(rest)
    sk = _prune(sk, int(round(SPUR_MM * ppm)))
    n3, lab3, st3, _ = cv2.connectedComponentsWithStats(sk.astype(np.uint8), connectivity=8)
    keep3 = np.zeros(n3, bool)
    keep3[1:] = st3[1:, cv2.CC_STAT_AREA] >= LINE_MIN_MM * ppm
    sk = keep3[lab3]
    lines = cv2.dilate(sk.astype(np.uint8), _disk(LINE_W_MM * ppm / 2.0)).astype(bool)
    return solid, lines, inside


def run(img: dict, fit_w_mm: float, fit_h_mm: float,
        debug_dir: Path | None = None) -> dict | None:
    """The picture with its fine line drawings redrawn for the stitched size, or None when
    nothing in it needs simplifying."""
    a = analyse(img, fit_w_mm, fit_h_mm)
    if not a["drawings"]:
        return None
    ppm, rgb, ink, colored = a["ppm"], a["rgb"].copy(), a["ink"], a["colored"]
    dark = rgb[ink]
    pen = tuple(int(v) for v in np.median(dark, axis=0)) if len(dark) else (0, 0, 0)
    bg = img.get("bg_rgb") or (255, 255, 255)
    out = rgb.copy()
    for region in a["drawings"]:
        region = _filled_silhouette(ink & region, colored & region, ppm) | region
        part = ink & region
        col = colored & region
        # colored areas inside the drawing (a roof): their own hatching is painted out, in
        # the area's own color (a blend would read as a second orange). Hatching is every
        # pixel inside the area that isn't its color: dark lines and their grey edges alike.
        if col.any():
            areas = cv2.morphologyEx(col.astype(np.uint8), cv2.MORPH_CLOSE, _disk(0.5 * ppm))
            n, lab = cv2.connectedComponents(areas, connectivity=8)
            for k in range(1, n):
                area = lab == k
                own = rgb[area & col]
                if len(own) < 20:
                    continue
                hatch = area & ~col
                out[hatch] = np.median(own, axis=0).astype(np.uint8)
                col = col | area
            part = part & ~col
        solid, lines, inside = redraw(part, col, ppm)
        clear = region & ~col & (inside | part)
        out[clear] = bg
        out[lines & ~col] = pen
        out[solid] = pen
    save_debug(debug_dir, "01b_simplified.png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    alpha = None
    if img.get("alpha") is not None:
        alpha = (a["alpha"].astype(np.uint8) * 255)
    return {"rgb": np.ascontiguousarray(out), "alpha": alpha, "bg_rgb": img.get("bg_rgb"),
            "simplified": True}
