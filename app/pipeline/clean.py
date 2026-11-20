"""Stage 4 — Clean (local pass, no AI).

Reduce the picture to a few flat colors and tidy it so every remaining shape can be sewn:
  * k-means color reduction in Lab color space
  * merge near-identical colors, dissolve the in-between shades that anti-aliasing leaves
    along edges
  * find the background (transparent, or the flat color around the border) so it isn't sewn.
    Only the part that reaches the edge of the picture counts: an area of the same color
    shut inside the design (white lettering on a white page, say) is part of the design and
    is sewn in that color
  * smooth edges, remove specks and close pinholes smaller than the smallest sewable detail

Output is a "label map": one small number per pixel saying which color it is (BG = not sewn).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .common import BG, PipelineError, save_debug

MERGE_DELTA_E = 9.0     # colors closer than this are treated as the same color
TINY_CLUSTER = 0.002    # a color covering less than this share of the design is dropped


def _nearest(pix: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Index of the nearest center for every pixel (chunked to keep memory low)."""
    out = np.empty(len(pix), np.int32)
    for i in range(0, len(pix), 400_000):
        chunk = pix[i:i + 400_000]
        d = ((chunk[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        out[i:i + 400_000] = d.argmin(axis=1)
    return out


def _is_blend(c: np.ndarray, others: list[np.ndarray]) -> bool:
    """True if color c sits roughly on the line between two other colors (an edge blend)."""
    for i in range(len(others)):
        for j in range(i + 1, len(others)):
            a, b = others[i], others[j]
            ab = b - a
            n2 = float(ab @ ab)
            if n2 < 1e-6:
                continue
            t = float((c - a) @ ab) / n2
            if 0.1 < t < 0.9:
                off = float(np.linalg.norm(c - (a + t * ab)))
                if off < 0.25 * np.sqrt(n2) + 4.0:
                    return True
    return False


def _remove_specks(labels: np.ndarray, min_area_px: float) -> np.ndarray:
    """Give every too-small island (of any color, background included) to its nearest
    neighbour, so no unsewable specks or pinholes are left."""
    unassigned = np.zeros(labels.shape, bool)
    for c in np.unique(labels):
        n, cc, stats, _ = cv2.connectedComponentsWithStats(
            (labels == c).astype(np.uint8), connectivity=8)
        small = np.nonzero(stats[:, cv2.CC_STAT_AREA] < min_area_px)[0]
        small = small[small != 0]
        if len(small):
            unassigned |= np.isin(cc, small)
    if not unassigned.any() or unassigned.all():
        return labels
    _, lbl = cv2.distanceTransformWithLabels(
        unassigned.astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    lut = np.zeros(int(lbl.max()) + 1, labels.dtype)
    seeds = ~unassigned
    lut[lbl[seeds]] = labels[seeds]
    return lut[lbl]


def _smooth(labels: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth the edges between colors: blur each color's mask and keep the strongest."""
    best = np.full(labels.shape, -1.0, np.float32)
    out = labels.copy()
    for c in np.unique(labels):
        score = cv2.GaussianBlur((labels == c).astype(np.float32), (0, 0), sigma)
        win = score > best
        best[win] = score[win]
        out[win] = c
    return out


def run(intake: dict, fit_w_mm: float, fit_h_mm: float, rules: dict,
        debug_dir: Path | None = None) -> dict:
    rgb, alpha = intake["rgb"], intake["alpha"]
    h, w = rgb.shape[:2]
    max_colors = int(max(1, min(rules["max_colors"], 12)))

    # Work at about 0.1 mm per pixel of the finished design.
    mm_per_src_px = min(fit_w_mm / w, fit_h_mm / h)
    design_w, design_h = w * mm_per_src_px, h * mm_per_src_px
    ppm = min(rules["work_px_per_mm"], rules["work_max_px"] / max(design_w, design_h))
    W, H = max(8, round(design_w * ppm)), max(8, round(design_h * ppm))
    interp = cv2.INTER_AREA if W < w else cv2.INTER_CUBIC
    rgb_w = cv2.resize(rgb, (W, H), interpolation=interp)
    alpha_w = None if alpha is None else cv2.resize(alpha, (W, H), interpolation=cv2.INTER_LINEAR)
    rgb_w = cv2.medianBlur(rgb_w, 3)
    lab = cv2.cvtColor(rgb_w.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
    fg = np.ones((H, W), bool) if alpha_w is None else alpha_w >= 128

    pix = lab[fg]
    if len(pix) < 50:
        raise PipelineError(
            "This picture looks blank. Try a picture with a clear design on a plain background.")

    # --- k-means, with a few spare clusters to soak up edge blends -----------------------
    rng = np.random.default_rng(0)
    sample = pix if len(pix) <= 150_000 else pix[rng.choice(len(pix), 150_000, replace=False)]
    k = int(min(max_colors + 4, 12, len(sample)))
    cv2.setRNGSeed(0)  # same picture in, same colors out
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, _, centers = cv2.kmeans(np.ascontiguousarray(sample, np.float32), k, None, crit, 3,
                               cv2.KMEANS_PP_CENTERS)
    centers = centers.astype(np.float64)
    cmap = np.full((H, W), -1, np.int32)   # cluster id per pixel, -1 = transparent
    cmap[fg] = _nearest(pix, centers.astype(np.float32))
    alive = [c for c in range(k) if (cmap == c).any()]

    def area(c):
        return int((cmap == c).sum())

    def merge(a, b):
        """Fold cluster b into cluster a."""
        na, nb = area(a), area(b)
        centers[a] = (centers[a] * na + centers[b] * nb) / max(1, na + nb)
        cmap[cmap == b] = a
        alive.remove(b)

    def dissolve(c):
        """Hand each pixel of cluster c to the nearest remaining color."""
        others = [o for o in alive if o != c]
        idx = cmap == c
        near = _nearest(lab[idx], centers[others].astype(np.float32))
        cmap[idx] = np.array(others, np.int32)[near]
        alive.remove(c)

    def closest_pair(ids):
        best = None
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                d = float(np.linalg.norm(centers[ids[i]] - centers[ids[j]]))
                if best is None or d < best[0]:
                    best = (d, ids[i], ids[j])
        return best

    # 1. merge near-identical colors
    while len(alive) > 1:
        d, a, b = closest_pair(alive)
        if d >= MERGE_DELTA_E:
            break
        if area(a) < area(b):
            a, b = b, a
        merge(a, b)

    # 2. find the background: the flat color that surrounds the design
    bg_id = None
    if alpha_w is None:
        t = max(2, round(min(H, W) * 0.01))
        ring = np.concatenate([cmap[:t].ravel(), cmap[-t:].ravel(),
                               cmap[:, :t].ravel(), cmap[:, -t:].ravel()])
        ids, counts = np.unique(ring, return_counts=True)
        top = int(counts.argmax())
        if counts[top] / counts.sum() >= 0.6:
            bg_id = int(ids[top])

    # 3. drop colors that cover almost nothing
    total = int(fg.sum())
    for c in sorted(alive, key=area):
        if len(alive) > 1 and c != bg_id and area(c) < TINY_CLUSTER * total:
            dissolve(c)

    # 4. dissolve edge blends: thin clusters whose color is a mix of two other colors
    r = max(1, round(0.2 * ppm), round(1.5 * W / w))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    for c in sorted(alive, key=area):
        if len(alive) <= 2 or c == bg_id:
            continue
        mask = (cmap == c).astype(np.uint8)
        kept = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel).sum()
        if 1.0 - kept / max(1, mask.sum()) > 0.6 and \
                _is_blend(centers[c], [centers[o] for o in alive if o != c]):
            dissolve(c)

    found = {"present": False, "sewn": False}

    def enclosed_background():
        """Pixels of the background color that are shut inside the design, if they are to
        be sewn. Records in `found` whether there are any and what was decided."""
        if bg_id is None:
            return None
        m = (cmap == bg_id).astype(np.uint8)
        _, cc = cv2.connectedComponents(m, connectivity=4)
        outside = np.unique(np.concatenate([cc[0], cc[-1], cc[:, 0], cc[:, -1]]))
        inside = (m == 1) & ~np.isin(cc, outside[outside != 0])
        if inside.sum() < 4 * (rules["min_detail_mm"] * ppm) ** 2:
            return None
        found["present"] = True
        mode = rules.get("sew_enclosed_background", "auto")
        if mode == "auto":
            # Solid artwork (a badge, a logo): the blank areas inside it are meant to be that
            # color, so sew them. A line drawing: the blanks are just paper, leave them open.
            design = ((cmap != bg_id) & fg).astype(np.uint8)
            r_mm = max(1, round(1.0 * ppm))
            core = cv2.erode(design, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * r_mm + 1, 2 * r_mm + 1)))
            mode = core.sum() >= 0.2 * max(1, design.sum())
        found["sewn"] = bool(mode)
        return inside if mode else None

    # 5. still too many colors? merge the two closest until it fits
    room = max_colors - (1 if enclosed_background() is not None else 0)
    while True:
        sewn = [c for c in alive if c != bg_id]
        if len(sewn) <= max(1, room):
            break
        d, a, b = closest_pair(sewn)
        if area(a) < area(b):
            a, b = b, a
        merge(a, b)

    # --- label map ----------------------------------------------------------------------
    sewn = [c for c in alive if c != bg_id]
    if not sewn:
        raise PipelineError(
            "We could only find one flat color in this picture, so there is nothing to stitch. "
            "Try a picture where the design stands out from its background.")
    labels = np.full((H, W), BG, np.uint8)
    for i, c in enumerate(sewn):
        labels[cmap == c] = i
    inside = enclosed_background()
    if inside is not None:
        labels[inside] = len(sewn)          # its own color: the background color, but sewn
        sewn.append(bg_id)

    labels = _smooth(labels, max(0.8, 0.12 * ppm))
    min_area_px = 0.8 * (rules["min_detail_mm"] * ppm) ** 2
    for _ in range(2):
        labels = _remove_specks(labels, min_area_px)

    # Re-number colors by size, biggest first, and take each color from the real pixels.
    ids = [i for i in range(len(sewn)) if (labels == i).any()]
    if not ids:
        raise PipelineError(
            "Everything in this picture is too small to stitch at this size. Try a bigger size.")
    ids.sort(key=lambda i: -int((labels == i).sum()))
    out = np.full((H, W), BG, np.uint8)
    palette = []
    small = np.ones((3, 3), np.uint8)
    for new, old in enumerate(ids):
        m = labels == old
        out[m] = new
        core = cv2.erode(m.astype(np.uint8), small, iterations=2).astype(bool)
        src = core if core.sum() >= 50 else m
        palette.append(tuple(int(v) for v in np.median(rgb_w[src], axis=0)))

    result = {"labels": out, "palette": palette, "ppm": float(ppm),
              "has_background": bool(alpha_w is not None or bg_id is not None),
              "inside": dict(found)}
    if debug_dir is not None:
        dbg = render(result, checker=True)
        save_debug(debug_dir, "04_clean.png", dbg[:, :, ::-1])
    return result


def render(cleaned: dict, checker: bool = False) -> np.ndarray:
    """The cleaned-up art as an RGB picture: flat colors on white (or on a checkerboard,
    to show which parts will not be sewn)."""
    labels, palette = cleaned["labels"], cleaned["palette"]
    H, W = labels.shape
    img = np.full((H, W, 3), 255, np.uint8)
    if checker:
        yy, xx = np.indices((H, W))
        img[((yy // 12 + xx // 12) % 2 == 0)] = 235
    for i, col in enumerate(palette):
        img[labels == i] = col
    return img
