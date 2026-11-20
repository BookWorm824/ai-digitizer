"""Printed pages for in-the-hoop projects:

  * the step-by-step instructions, built from the project's own step list, with a picture of
    the hoop after each step and what to place or trim when the machine stops
  * the cutting templates, drawn at 100% so fabric can be cut to shape before sewing
"""

from __future__ import annotations

import io

import cv2
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch, mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from . import worksheet
from .pipeline import export


def _wrap(c, text: str, x: float, y: float, width: float, font="Helvetica", size=10.5,
          leading=13.5) -> float:
    """Draw text wrapped to a width. Returns the y below the last line."""
    c.setFont(font, size)
    line = ""
    for word in text.split():
        trial = (line + " " + word).strip()
        if c.stringWidth(trial, font, size) > width and line:
            c.drawString(x, y, line)
            y -= leading
            line = word
        else:
            line = trial
    if line:
        c.drawString(x, y, line)
        y -= leading
    return y


def instructions_pdf(result: dict) -> bytes:
    project = result["project"]
    plan = worksheet.plan_of(result)
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    W, H = letter
    m = 0.6 * inch

    c.setFont("Helvetica-Bold", 20)
    c.drawString(m, H - m - 6, f"{project['name']}: {result['name']}"[:60])
    c.setFont("Helvetica", 10)
    c.setFillColor(colors.grey)
    w_in, h_in = (v / 25.4 for v in project["body_mm"])
    c.drawString(m, H - m - 22, f"Finished size about {w_in:.1f} × {h_in:.1f} in   ·   "
                                f"hoop: {result['check']['hoop'] or 'none fits'}   ·   "
                                f"{result['stats']['stitches']:,} stitches")
    c.setFillColor(colors.black)

    y = H - m - 48
    c.setFont("Helvetica-Bold", 12)
    c.drawString(m, y, "You will need")
    y -= 16
    for item in project["materials"]:
        y = _wrap(c, "- " + item, m + 6, y, W - 2 * m - 6)
    y -= 4
    y = _wrap(c, "The machine stops after every step. Do what the step says, then start it again.",
              m, y, W - 2 * m, "Helvetica-Oblique")
    y -= 8

    pic = 1.45 * inch
    for n, step in enumerate(project["steps"]):
        if y - pic < m:
            c.showPage()
            y = H - m
        shown = dict(plan, blocks=plan["blocks"][:n + 1])
        ok, png = cv2.imencode(".png", export.render_preview(shown, px_per_mm=5.0))
        if ok:
            img = ImageReader(io.BytesIO(png.tobytes()))
            iw, ih = img.getSize()
            k = min(pic / iw, pic / ih)
            c.drawImage(img, m + (pic - iw * k) / 2, y - pic + (pic - ih * k) / 2, iw * k, ih * k)
        c.setStrokeColor(colors.lightgrey)
        c.rect(m, y - pic, pic, pic)
        x = m + pic + 0.25 * inch
        width = W - m - x
        ty = y - 12
        c.setFont("Helvetica-Bold", 12)
        c.drawString(x, ty, f"Step {step['number']} — {step['title']}")
        c.setFont("Helvetica", 9)
        c.setFillColor(colors.grey)
        c.drawString(x, ty - 12, f"Thread: {step['thread']}")
        c.setFillColor(colors.black)
        ty -= 28
        if step["before"]:
            ty = _wrap(c, "Before it sews: " + step["before"], x, ty, width)
            ty -= 3
        if step["after"]:
            ty = _wrap(c, "When it stops: " + step["after"], x, ty, width)
        y -= pic + 10

    if project.get("notes") and y > m + 40:
        y -= 4
        c.setFont("Helvetica-Bold", 12)
        c.drawString(m, y, "Good to know")
        y -= 16
        for note in project["notes"]:
            y = _wrap(c, "- " + note, m + 6, y, W - 2 * m - 6)
    c.showPage()
    c.save()
    return buf.getvalue()


def templates_pdf(result: dict) -> bytes:
    """One page per fabric piece, at 100% scale, each with a 1-inch square to check the
    printer didn't shrink it."""
    project = result["project"]
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    W, H = letter
    m = 0.5 * inch
    for piece in project["pieces"]:
        ring = piece["ring"]
        xs, ys = ring[0::2], ring[1::2]
        w, h = (max(xs) - min(xs)) * mm, (max(ys) - min(ys)) * mm
        c.setFont("Helvetica-Bold", 15)
        c.drawString(m, H - m - 4, f"Cut 1: {piece['name']}")
        c.setFont("Helvetica", 10)
        c.setFillColor(colors.grey)
        c.drawString(m, H - m - 20, f"{project['name']}: {result['name']}   ·   print at 100% "
                                    "(\"actual size\"), not \"fit to page\"")
        c.setFillColor(colors.black)
        c.rect(W - m - inch, H - m - inch - 4, inch, inch)
        c.setFont("Helvetica", 8)
        c.drawCentredString(W - m - inch / 2, H - m - inch / 2 - 6, "1 inch")
        c.drawCentredString(W - m - inch / 2, H - m - inch / 2 - 16, "check me")

        room_w, room_h = W - 2 * m, H - 2 * m - 1.3 * inch
        if w > room_w or h > room_h:
            c.setFont("Helvetica", 12)
            c.drawString(m, H / 2, f"This piece is bigger than one page. Cut a rectangle "
                                   f"{w / inch + 0.5:.1f} × {h / inch + 0.5:.1f} in instead.")
        else:
            ox = (W - w) / 2 - min(xs) * mm
            oy = m + (room_h - h) / 2 + max(ys) * mm           # PDF y runs upward
            path = c.beginPath()
            path.moveTo(ox + xs[0] * mm, oy - ys[0] * mm)
            for x, y in zip(xs[1:], ys[1:]):
                path.lineTo(ox + x * mm, oy - y * mm)
            path.close()
            c.setLineWidth(1.2)
            c.setDash(6, 3)
            c.drawPath(path)
            c.setDash()
            c.setFont("Helvetica", 9)
            c.drawCentredString(W / 2, m + (room_h - h) / 2 - 14,
                                f"about {w / inch:.1f} × {h / inch:.1f} in. Cut on the dashed line.")
        c.showPage()
    c.save()
    return buf.getvalue()
