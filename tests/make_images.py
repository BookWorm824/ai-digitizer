"""Draws the Phase 1 test pictures into tests/images/ and the sample logo into web/samples/.

Run:  .venv\\Scripts\\python.exe tests\\make_images.py

They are simple, clean logos on purpose: Phase 1 is about the stitch engine. Cartoon, text,
photo and real-world pictures join the set in later phases (BRIEF.md wants 12 in total).
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
IMAGES = HERE / "images"
SAMPLES = HERE.parent / "web" / "samples"

NAVY, RED, GOLD, BLACK, WHITE = (26, 54, 110), (200, 40, 45), (240, 180, 40), (20, 20, 20), (255, 255, 255)
S = 4  # draw big, then shrink, for smooth edges


def _font(size):
    for name in ("arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _star(cx, cy, r_out, r_in, points=5):
    pts = []
    for i in range(points * 2):
        r = r_out if i % 2 == 0 else r_in
        a = -math.pi / 2 + i * math.pi / points
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def badge(size=1000, transparent=False):
    """Sew-out test badge: a solid circle, a star-shaped hole, a second color that touches
    the first, a thin outline ring, and lettering."""
    n = size * S
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0) if transparent else WHITE + (255,))
    d = ImageDraw.Draw(img)
    c = n / 2
    r = n * 0.36
    d.ellipse([c - r, c - r * 1.08, c + r, c + r * 0.92], fill=NAVY + (255,))
    hole = (0, 0, 0, 0) if transparent else WHITE + (255,)
    d.polygon(_star(c, c - r * 0.28, r * 0.42, r * 0.17), fill=hole)
    # thin ring around the circle (should become a running stitch)
    rr = r * 1.13
    d.ellipse([c - rr, c - rr - r * 0.08, c + rr, c + rr - r * 0.08],
              outline=BLACK + (255,), width=max(2, int(n * 0.007)))
    # banner touching the circle
    d.rounded_rectangle([c - r * 1.25, c + r * 0.30, c + r * 1.25, c + r * 0.86],
                        radius=int(r * 0.1), fill=RED + (255,))
    f = _font(int(r * 0.40))
    d.text((c, c + r * 0.575), "TEST", font=f, fill=(WHITE + (255,)), anchor="mm")
    return img.resize((size, size), Image.LANCZOS)


def blocks(size=900):
    """Two colors side by side sharing an edge, plus a ring: checks gaps between colors
    and shapes with holes."""
    n = size * S
    img = Image.new("RGBA", (n, n), WHITE + (255,))
    d = ImageDraw.Draw(img)
    d.rectangle([n * 0.12, n * 0.2, n * 0.5, n * 0.8], fill=NAVY + (255,))
    d.rectangle([n * 0.5, n * 0.2, n * 0.88, n * 0.8], fill=GOLD + (255,))
    d.ellipse([n * 0.3, n * 0.3, n * 0.7, n * 0.7], fill=RED + (255,))
    d.ellipse([n * 0.41, n * 0.41, n * 0.59, n * 0.59], fill=WHITE + (255,))
    return img.resize((size, size), Image.LANCZOS)


def line_art(size=900):
    """Thin black lines only: everything should become running stitches."""
    n = size * S
    img = Image.new("RGBA", (n, n), WHITE + (255,))
    d = ImageDraw.Draw(img)
    w = int(n * 0.012)
    d.ellipse([n * 0.15, n * 0.15, n * 0.85, n * 0.85], outline=BLACK + (255,), width=w)
    d.line([(n * 0.3, n * 0.5), (n * 0.7, n * 0.5)], fill=BLACK + (255,), width=w)
    d.line([(n * 0.5, n * 0.3), (n * 0.5, n * 0.7)], fill=BLACK + (255,), width=w)
    return img.resize((size, size), Image.LANCZOS)


def main():
    IMAGES.mkdir(parents=True, exist_ok=True)
    SAMPLES.mkdir(parents=True, exist_ok=True)
    badge().convert("RGB").save(IMAGES / "logo_badge.png")
    badge(transparent=True).save(IMAGES / "logo_badge_transparent.png")
    blocks().convert("RGB").save(IMAGES / "logo_blocks.png")
    blocks().convert("RGB").save(IMAGES / "logo_blocks.jpg", quality=80)
    line_art().convert("RGB").save(IMAGES / "logo_lines.png")
    badge().convert("RGB").save(SAMPLES / "sample_logo.png")
    print("wrote", len(list(IMAGES.glob("*"))), "test images")


if __name__ == "__main__":
    main()
