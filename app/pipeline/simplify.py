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
REBUILD_LINE_MM = 0.85     # a rebuilt drawing's lines: bold running lines (out and back as a bean, as a cap
                           # digitizer's outline (1 mm satin lines read as lumpy rope here)
REBUILD_SOLID_SHARE = 0.85 # a drawing more than this solid once redrawn is rebuilt instead
                           # (the cap Original: 97 %; left chest 62 %, jacket back 69 %)
COVER_MM = 1.2             # rebuilding: ink coverage is judged over this much...
DENSE_COVER = 0.35         # ...and spots this densely inked become solid (windows, trees)
MIN_SPOT_MM2 = 0.8         # ...if at least this big
BLOCK_COVER_MM = 0.9       # windows are found at a finer scale (the walls between them
BLOCK_COVER = 0.45         # are thinner than COVER_MM), as block-shaped spots this dense
MIN_WINDOW_MM2 = 0.2       # the smallest window spot found
GRID_TOL_MM = 0.6          # window spots this near in x (or y) share a column (or row)
WINDOW_MIN_MM = 1.0        # a window block is at least this wide and tall...
WALL_MM = 1.0              # ...leaving at least this much wall open to its neighbours (caps)
WINDOW_BAR_MM = 1.2        # a facade one window high (or wide) becomes a bar this thick
FACADE_LINK_MM = 2.6       # windows this near in a row or column belong to one facade
WINDOW_CLEAR_MM = 0.35     # lines keep this far from window blocks...
WINDOW_LINE_SHARE = 0.25   # ...and a line this much beside windows is a floor line: dropped
SOLID_GAP_MM = 0.35        # lines keep this far from solid shapes (trees stay solid)
WINDOW_MAX_MM2 = 5.0       # dense spots up to this big are blocks (windows)...
MASS_COVER = 0.55          # ...bigger ones are solid only if this densely inked (trees)
STRUCT_MIN_MM = 2.0        # straight lines this long carry the structure and are kept
INNER_EDGE_MM = 2.0        # inner lines nearer the outline than this are left out
MIRROR_IOU = 0.85          # a drawing this symmetric gets matching halves
MERGE_MM = 0.8             # lines nearer each other than this are one line...
MERGE_SHARE = 0.6          # ...when this much of the shorter one is that near the other
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


def rebuild(part: np.ndarray, col: np.ndarray, ppm: float) -> tuple:
    """A drawing far too fine for its size (a pen-drawn building on a cap) rebuilt the way
    the tutorials teach rather than redrawn from its cells: the outline as one line, the long
    straight lines that carry the structure (tower edges, walls, floors) with lines nearer
    each other than thread can separate merged into one, densely inked spots (windows, trees)
    as solid shapes, and everything else left out. Returns (solid, lines, inside)."""
    inside = _filled_silhouette(part, col, ppm)
    body = inside & ~col
    w = max(1, int(round(REBUILD_LINE_MM * ppm)))
    # the outline: a band one line wide just inside the silhouette
    outline = body & ~cv2.erode(inside.astype(np.uint8), _disk(w)).astype(bool)
    # dense spots: judged over about a millimetre
    k = max(3, int(round(COVER_MM * ppm)) | 1)
    cover = cv2.blur(part.astype(np.float32), (k, k))
    dense = _thick((cover >= DENSE_COVER) & body, 0.4 * ppm)
    n, lab, st, _ = cv2.connectedComponentsWithStats(dense.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        area = st[i, cv2.CC_STAT_AREA] / (ppm * ppm)
        if area < MIN_SPOT_MM2:
            continue
        x, y, bw, bh = st[i, :4]
        mine = lab[y:y + bh, x:x + bw] == i
        if area <= WINDOW_MAX_MM2:
            # a small separate block (a window with its panes): solid if block-shaped
            keep[i] = mine.sum() >= 0.6 * bw * bh
        else:
            # a big area is solid only where it is really dark (a tree), not where it is
            # merely busy (a hatched tower: its structure is in the straight lines)
            keep[i] = float(cover[y:y + bh, x:x + bw][mine].mean()) >= MASS_COVER
    solid = keep[lab]
    # windows: small dark spots found at a finer scale (the walls between windows are
    # thinner than COVER_MM), then lined up on their grid (cartographic typification): each
    # becomes one clean block of the row's height and the column's width, at least a
    # millimetre, with the walls between them kept open. Spots that line up with no others
    # (marks along a wall) are not windows.
    kb = max(3, int(round(BLOCK_COVER_MM * ppm)) | 1)
    near_cover = cv2.blur(part.astype(np.float32), (kb, kb))
    blocks = _thick((near_cover >= BLOCK_COVER) & body & ~solid, 0.2 * ppm)
    n, lab, st, cen = cv2.connectedComponentsWithStats(blocks.astype(np.uint8), connectivity=4)
    cands = []
    for i in range(1, n):
        area = st[i, cv2.CC_STAT_AREA] / (ppm * ppm)
        x, y, bw, bh = st[i, :4]
        if MIN_WINDOW_MM2 <= area <= WINDOW_MAX_MM2 and max(bw, bh) <= 3.0 * min(bw, bh):
            cands.append((float(cen[i][0]), float(cen[i][1]), float(bw), float(bh)))
    masses = solid.copy()                 # trees and other big solids (not windows)
    windows = _window_grid(cands, ppm, body)
    mirror = _mirror_axis(inside)
    if mirror is not None:
        # a symmetric building has matching wings: windows found on one side are given to
        # the other (the drawing's two halves trace a little differently)
        # (one side's windows mirrored onto the other, the side that found more of them:
        # laid over each other, slightly offset windows fuse into double-width bars)
        cut = int(round(mirror))
        left, right = windows.copy(), windows.copy()
        left[:, cut:] = False
        right[:, :cut] = False
        side = left if left.sum() >= right.sum() else right
        windows = (side | _flip(side, mirror)) & body
    solid = solid | windows
    by_window = cv2.dilate(windows.astype(np.uint8), _disk(WINDOW_CLEAR_MM * ppm)).astype(bool)
    # structural lines: long straight segments of the drawing, longest first, each kept only
    # if it is not already said by a line kept within MERGE_MM
    lsd = cv2.createLineSegmentDetector()
    found = lsd.detect((part.astype(np.uint8) * 255))[0]
    segs = [] if found is None else [tuple(map(float, f[0])) for f in found]
    segs = [sg for sg in segs if np.hypot(sg[2] - sg[0], sg[3] - sg[1]) >= STRUCT_MIN_MM * ppm]
    segs.sort(key=lambda sg: -np.hypot(sg[2] - sg[0], sg[3] - sg[1]))
    near = max(1, int(round(MERGE_MM * ppm)))
    # an inner line closer than INNER_EDGE_MM to the outline only echoes it (tutorials: inner
    # lines only where they sit at least 2 mm from every parallel edge)
    zone = cv2.dilate(outline.astype(np.uint8), _disk(INNER_EDGE_MM * ppm))
    drawn = np.zeros(part.shape, np.uint8)
    for x0, y0, x1, y1 in segs:
        one = np.zeros(part.shape, np.uint8)
        cv2.line(one, (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))), 1, 1)
        on = one.astype(bool)
        if not on.any() or zone[on].mean() > MERGE_SHARE:
            continue
        if by_window[on].mean() > WINDOW_LINE_SHARE:
            continue                  # a floor or stonework line through the window rows
        drawn |= one
        zone |= cv2.dilate(one, _disk(near))
    lines = cv2.dilate(drawn, _disk(w / 2.0)).astype(bool) & inside
    # a gap between the lines and every solid shape: a tree joined to the outline would be
    # sewn as part of the line drawing (hollow), not as a solid
    apart = cv2.dilate(masses.astype(np.uint8), _disk(SOLID_GAP_MM * ppm)).astype(bool)
    apart |= by_window                # windows stay separate blocks, never part of a line
    return solid, (lines | outline) & ~col & ~apart, inside


def _mirror_axis(mask: np.ndarray) -> float | None:
    """The x of a vertical mirror axis if the shape is nearly symmetric about it, else None."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 100:
        return None
    best = None
    lo, hi = float(xs.min()), float(xs.max())
    for ax in np.linspace(lo + 0.4 * (hi - lo), lo + 0.6 * (hi - lo), 21):
        f = _flip(mask, ax)
        iou = (f & mask).sum() / max(1, (f | mask).sum())
        if best is None or iou > best[0]:
            best = (iou, float(ax))
    return best[1] if best and best[0] >= MIRROR_IOU else None


def _flip(mask: np.ndarray, ax: float) -> np.ndarray:
    """A mask mirrored about the vertical line x = ax."""
    h, w = mask.shape
    xs = np.arange(w)
    src = np.round(2 * ax - xs).astype(int)
    ok = (src >= 0) & (src < w)
    out = np.zeros_like(mask)
    out[:, ok] = mask[:, src[ok]]
    return out


def _clusters(vals: list, tol: float) -> list:
    """Group 1-D positions that lie within tol of their neighbours: [(centre, [indices])]."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    groups, cur = [], []
    for i in order:
        if cur and vals[i] - vals[cur[-1]] > tol:
            groups.append(cur)
            cur = []
        cur.append(i)
    if cur:
        groups.append(cur)
    return [(float(np.mean([vals[i] for i in g])), g) for g in groups]


def _window_grid(cands: list, ppm: float, body: np.ndarray) -> np.ndarray:
    """Window spots (cx, cy, w, h in pixels) as a clean window grid for the stitched size.
    Spots are lined up on rows and columns and grouped into facades; each facade's grid is
    then thinned to what thread can show (cartographic typification; Erich Campbell cuts a
    flag's 50 stars down to fewer at cap size): blocks at least WINDOW_MIN_MM with walls of
    at least WALL_MM, as many columns and rows as fit, spread evenly over the same field so
    the facade keeps its darkness. Spots that line up with nothing are marks, not windows."""
    out = np.zeros(body.shape, bool)
    if len(cands) < 4:
        return out
    tol = GRID_TOL_MM * ppm
    rows = _clusters([c[1] for c in cands], tol)
    cols = _clusters([c[0] for c in cands], tol)
    row_of = {i: r for r, (_y, g) in enumerate(rows) for i in g}
    col_of = {i: k for k, (_x, g) in enumerate(cols) for i in g}
    keep = [i for i in range(len(cands))
            if len(rows[row_of[i]][1]) >= 2 and len(cols[col_of[i]][1]) >= 2]
    # facades: windows linked to a neighbour in the same row or column
    parent = {i: i for i in keep}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    reach = FACADE_LINK_MM * ppm
    for ii, i in enumerate(keep):
        for j in keep[ii + 1:]:
            same_row, same_col = row_of[i] == row_of[j], col_of[i] == col_of[j]
            dx, dy = abs(cands[i][0] - cands[j][0]), abs(cands[i][1] - cands[j][1])
            if (same_row and dx <= reach) or (same_col and dy <= reach):
                parent[find(i)] = find(j)
    facades: dict = {}
    for i in keep:
        facades.setdefault(find(i), []).append(i)
    unit = ppm
    for members in facades.values():
        rs = sorted({row_of[i] for i in members}, key=lambda r: rows[r][0])
        ks = sorted({col_of[i] for i in members}, key=lambda k: cols[k][0])
        if len(rs) < 2 and len(ks) < 2:
            continue
        x0 = min(cands[i][0] - cands[i][2] / 2 for i in members)
        x1 = max(cands[i][0] + cands[i][2] / 2 for i in members)
        y0 = min(cands[i][1] - cands[i][3] / 2 for i in members)
        y1 = max(cands[i][1] + cands[i][3] / 2 for i in members)
        fw, fh = (x1 - x0) / unit, (y1 - y0) / unit           # the field, in mm

        def fit(n_have, span):
            # the most blocks of at least WINDOW_MIN_MM with walls of WALL_MM that fit
            n = int((span + WALL_MM) // (WINDOW_MIN_MM + WALL_MM))
            return max(1, min(n_have, n))
        nk, nr = fit(len(ks), fw), fit(len(rs), fh)
        if nk == 1 and nr == 1:
            continue
        bw = (fw - (nk - 1) * WALL_MM) / nk if nk > 1 else min(fw, WINDOW_BAR_MM)
        bh = (fh - (nr - 1) * WALL_MM) / nr if nr > 1 else min(fh, WINDOW_BAR_MM)
        # (a single column or row becomes one bar of the field's length)
        for r in range(nr):
            cy = y0 / unit + (bh / 2 if nr > 1 else fh / 2) + r * (bh + WALL_MM)
            for k in range(nk):
                cx = x0 / unit + (bw / 2 if nk > 1 else fw / 2) + k * (bw + WALL_MM)
                X0, X1 = int(round((cx - bw / 2) * unit)), int(round((cx + bw / 2) * unit))
                Y0, Y1 = int(round((cy - bh / 2) * unit)), int(round((cy + bh / 2) * unit))
                out[max(0, Y0):Y1, max(0, X0):X1] = True
    return out & body


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
    rebuilt = np.zeros(ink.shape, bool)
    drawn_new = np.zeros(ink.shape, bool)
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
        if solid.sum() > REBUILD_SOLID_SHARE * max(1, (inside & ~col).sum()):
            # most of it came out solid: far too fine for this size, so it is rebuilt
            solid, lines, inside = rebuild(part, col, ppm)
            rebuilt |= inside
            drawn_new |= solid | lines | col
        clear = region & ~col & (inside | part)
        out[clear] = bg
        out[lines & ~col] = pen
        out[solid] = pen
    save_debug(debug_dir, "01b_simplified.png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    alpha = None
    if img.get("alpha") is not None:
        # (a rebuilt drawing is drawn whole: its new lines must not be cut back to where the
        # artist's fine lines were; and only what it draws is part of the design)
        keep_a = (a["alpha"] & ~rebuilt) | drawn_new
        alpha = keep_a.astype(np.uint8) * 255
    return {"rgb": np.ascontiguousarray(out), "alpha": alpha, "bg_rgb": img.get("bg_rgb"),
            "simplified": True,
            "line_mm": REBUILD_LINE_MM if rebuilt.any() else LINE_W_MM}
