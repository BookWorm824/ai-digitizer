"""The one-page PDF worksheet that goes to the machine with the stitch file: a picture of the
design, its size, stitch count, sewing time, hoop, stabilizer, and the thread order with real
thread numbers. (DST files store no colors at all, so this sheet is the only record.)

Also the garment mockup picture, for customer approval and shop listings.
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from . import threads
from .pipeline import export


def plan_of(result: dict) -> dict:
    """Rebuild a drawable plan from a finished job's result."""
    blocks = []
    for b in result["blocks"]:
        h = b["hex"].lstrip("#")
        rgb = tuple(int(h[k:k + 2], 16) for k in (0, 2, 4))
        blocks.append({"rgb": rgb, "segments": [list(zip(s[0::2], s[1::2])) for s in b["segments"]]})
    return {"blocks": blocks, "bounds": tuple(result["bounds"])}


def _rgb(hex_color: str) -> tuple:
    h = (hex_color or "#eeece6").lstrip("#")
    try:
        return tuple(int(h[k:k + 2], 16) for k in (0, 2, 4))
    except ValueError:
        return (238, 236, 230)


def thread_label(color: dict, brand_id: str) -> str:
    t = (color.get("threads") or {}).get(brand_id)
    if not t:
        return ""
    return f"{t['brand']} {t['label']}".strip()


# --------------------------------------------------------------------------------- PDF

def build_pdf(result: dict, preview_png: Path, brand_id: str = threads.DEFAULT_BRAND,
              machine: str = "", file_name: str = "") -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    W, H = letter
    m = 0.6 * inch
    s = result["stats"]

    c.setFont("Helvetica-Bold", 20)
    c.drawString(m, H - m - 6, result["name"][:48])
    c.setFont("Helvetica", 10)
    c.setFillColor(colors.grey)
    c.drawString(m, H - m - 22, "Embroidery worksheet" + (f"  ·  {file_name}" if file_name else ""))
    c.setFillColor(colors.black)

    # picture
    box = 3.6 * inch
    top = H - m - 40
    try:
        img = ImageReader(str(preview_png))
        iw, ih = img.getSize()
        k = min(box / iw, box / ih)
        c.drawImage(img, m + (box - iw * k) / 2, top - box + (box - ih * k) / 2, iw * k, ih * k)
    except Exception:  # noqa: BLE001 - no picture is better than no worksheet
        pass
    c.setStrokeColor(colors.lightgrey)
    c.rect(m, top - box, box, box)

    # facts
    rows = [
        ("Size", f"{s['width_in']} × {s['height_in']} in   ({s['width_mm']} × {s['height_mm']} mm)"),
        ("Stitches", f"{s['stitches']:,}"),
        ("Thread colors", str(s["colors"])),
        ("Thread cuts", str(s["trims"])),
        ("Sewing time", f"about {s['sew_minutes']} minutes"),
        ("Hoop", f"{result['check']['hoop']} or larger" if result["check"]["hoop"] else "does not fit a hoop"),
        ("Going on", result["fabric"]["name"]),
        ("Stabilizer", result["fabric"]["stabilizer"]),
    ]
    if machine:
        rows.append(("Machine", machine))
    x = m + box + 0.35 * inch
    y = top - 14
    for label, value in rows:
        c.setFont("Helvetica", 9)
        c.setFillColor(colors.grey)
        c.drawString(x, y, label.upper())
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 11)
        for n, line in enumerate(_wrap(value, 34)):
            c.drawString(x, y - 13 - 13 * n, line)
            extra = 13 * n
        y -= 32 + extra

    # thread order
    y = top - box - 30
    c.setFont("Helvetica-Bold", 13)
    c.drawString(m, y, "Thread order")
    brand = next((b for b in threads.brands() if b["id"] == brand_id), None)
    if brand:
        c.setFont("Helvetica", 9)
        c.setFillColor(colors.grey)
        c.drawString(m + 1.25 * inch, y, f"closest {brand['line']} colors. Check against a real thread card.")
        c.setFillColor(colors.black)
    y -= 22
    for n, col in enumerate(result["colors"]):
        r, g, b = _rgb(col["hex"])
        c.setFillColorRGB(r / 255, g / 255, b / 255)
        c.setStrokeColor(colors.grey)
        c.rect(m, y - 4, 0.42 * inch, 0.26 * inch, fill=1)
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(m + 0.55 * inch, y + 2, f"{n + 1}.")
        c.setFont("Helvetica", 11)
        label = col.get("label") or thread_label(col, brand_id) or col["hex"]
        c.drawString(m + 0.85 * inch, y + 2, label[:70])
        c.setFillColor(colors.grey)
        c.drawRightString(W - m, y + 2, f"{col['stitches']:,} stitches")
        c.setFillColor(colors.black)
        y -= 0.36 * inch
        if y < m + 80:
            break

    # notes
    notes = list(result["fabric"].get("notes", []))
    notes += [msg["text"] for msg in result["check"]["messages"]]
    if notes and y > m + 40:
        y -= 6
        c.setFont("Helvetica-Bold", 13)
        c.drawString(m, y, "Before you sew")
        y -= 18
        c.setFont("Helvetica", 10)
        for note in notes:
            for line in _wrap("- " + note, 100):
                if y < m:
                    break
                c.drawString(m, y, line)
                y -= 13
    c.showPage()
    c.save()
    return buf.getvalue()


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    return lines + [cur] if cur else lines or [""]


# ------------------------------------------------------------------------------ mockup

SHIRT_MM = (640, 640)        # the whole picture, in millimetres of garment
# A T-shirt laid flat, as (x, y) in mm from the top-left of the picture.
SHIRT = [(228, 48), (262, 56), (320, 60), (378, 56), (412, 48), (522, 88), (606, 218),
         (526, 264), (484, 228), (484, 600), (156, 600), (156, 228), (114, 264), (34, 218),
         (118, 88)]
NECK = ((320, 50), (60, 30))  # centre and half-axes of the neck opening


def build_mockup(result: dict, garment_hex: str) -> bytes:
    """A PNG of the design on a garment-colored T-shirt (or a swatch of fabric for things
    that aren't shirts), to scale."""
    garment = _rgb(garment_hex)
    plan = plan_of(result)
    design = export.render_preview(plan, px_per_mm=8.0, fabric_rgb=garment)
    minx, miny, maxx, maxy = plan["bounds"]
    w_mm, h_mm = maxx - minx, maxy - miny
    ppm = 3.0
    dark = tuple(int(v * 0.78) for v in garment)[::-1]
    page = (240, 244, 246)        # BGR

    shirt = result["fabric"]["id"] in ("tshirt", "hoodie", "polo", "fleece")
    if shirt:
        W, Hh = int(SHIRT_MM[0] * ppm), int(SHIRT_MM[1] * ppm)
        img = np.full((Hh, W, 3), page, np.uint8)
        pts = (np.array(SHIRT) * ppm).astype(np.int32)
        cv2.fillPoly(img, [pts], garment[::-1], lineType=cv2.LINE_AA)
        cv2.polylines(img, [pts], True, dark, 3, cv2.LINE_AA)
        centre = (int(NECK[0][0] * ppm), int(NECK[0][1] * ppm))
        axes = (int(NECK[1][0] * ppm), int(NECK[1][1] * ppm))
        cv2.ellipse(img, centre, axes, 0, 0, 180, dark, -1, cv2.LINE_AA)           # neck opening
        cv2.ellipse(img, centre, (axes[0] + 12, axes[1] + 12), 0, 0, 180, dark, 3, cv2.LINE_AA)
        if max(w_mm, h_mm) <= 115:           # left chest: on the wearer's left, our right
            cx, cy = 400.0, 190.0
        else:                                # full front: centred, a little below the collar
            cx, cy = 320.0, 130.0 + h_mm / 2
    else:
        side = max(w_mm, h_mm) + 120
        W = Hh = int(side * ppm)
        img = np.full((Hh, W, 3), garment[::-1], np.uint8)
        cx = cy = side / 2

    # the design picture has a 4 mm margin of garment color all round; shrink it to scale
    small = cv2.resize(design, (max(1, int(design.shape[1] * ppm / 8.0)),
                                max(1, int(design.shape[0] * ppm / 8.0))),
                       interpolation=cv2.INTER_AREA)
    x0 = int(cx * ppm - small.shape[1] / 2)
    y0 = int(cy * ppm - small.shape[0] / 2)
    x1, y1 = min(W, x0 + small.shape[1]), min(Hh, y0 + small.shape[0])
    x0c, y0c = max(0, x0), max(0, y0)
    img[y0c:y1, x0c:x1] = small[y0c - y0:y1 - y0, x0c - x0:x1 - x0]
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else b""
