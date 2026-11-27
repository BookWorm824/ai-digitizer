"""Draw a varied set of original test designs for the training bench, so every change is
checked on many kinds of art, not one logo:

    .venv\\Scripts\\python.exe tools\\make_bench_designs.py

Each design is made here from simple shapes and fonts already on this PC (no real brand, no
customer art) and saved to the app's data folder, bench\\<name>.png, with bench\\<name>.txt
holding its words for the lettering check. Existing files are left alone unless --force.

  script_bakery     a script name with a small sans tagline and a simple cupcake
  varsity_outline   two-colour outlined block lettering (a team name and number)
  mascot_bear       a cartoon face: outlined shapes, small eyes, a muzzle
  monogram          three interlocking serif letters in a ring
  small_text        a business-card style design: name, tagline, phone, small icons
  gradient_sunset   a sunset scene with smooth shading and a palm silhouette
  line_icon         a thin-line coffee cup icon with steam, single line weight
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import config  # noqa: E402

FONTS = Path("C:/Windows/Fonts")
W, H = 1600, 1200


def font(name: str, size: int):
    try:
        return ImageFont.truetype(str(FONTS / name), size)
    except OSError:
        return ImageFont.load_default()


def centered(d: ImageDraw.ImageDraw, y: int, text: str, f, fill, stroke=0, stroke_fill=None):
    w = d.textlength(text, font=f)
    d.text(((W - w) / 2, y), text, font=f, fill=fill, stroke_width=stroke,
           stroke_fill=stroke_fill)


def script_bakery():
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    pink, brown = (232, 96, 140), (92, 52, 30)
    # a cupcake: wrapper with ridges, frosting swirl, cherry
    d.polygon([(640, 430), (960, 430), (920, 640), (680, 640)], fill=brown)
    for x in range(670, 940, 40):
        d.line([(x, 440), (x + 10, 630)], fill="white", width=8)
    d.ellipse((600, 300, 1000, 470), fill=pink)
    d.ellipse((680, 220, 920, 360), fill=pink)
    d.ellipse((770, 160, 830, 220), fill=(200, 30, 50))
    centered(d, 650, "Bella's Bakery", font("segoescb.ttf", 190), brown)
    centered(d, 950, "FRESH BAKED DAILY", font("bahnschrift.ttf", 70), pink)
    return img, ["Bella's Bakery", "FRESH BAKED DAILY"]


def varsity_outline():
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    navy, gold = (20, 36, 90), (240, 180, 30)
    centered(d, 180, "EAGLES", font("COOPBL.TTF", 300), gold, stroke=18, stroke_fill=navy)
    centered(d, 600, "23", font("impact.ttf", 420), navy, stroke=16, stroke_fill=gold)
    return img, ["EAGLES", "23"]


def mascot_bear():
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    brown, tan, black = (140, 82, 40), (226, 182, 130), (20, 20, 20)
    for cx in (520, 1080):                      # ears
        d.ellipse((cx - 140, 120, cx + 140, 400), fill=brown, outline=black, width=16)
        d.ellipse((cx - 70, 190, cx + 70, 330), fill=tan)
    d.ellipse((380, 200, 1220, 1080), fill=brown, outline=black, width=18)       # head
    d.ellipse((600, 620, 1000, 960), fill=tan, outline=black, width=12)          # muzzle
    d.ellipse((740, 640, 860, 730), fill=black)                                   # nose
    d.arc((690, 700, 910, 900), 20, 160, fill=black, width=12)                    # mouth
    for cx in (620, 980):                                                         # eyes
        d.ellipse((cx - 50, 470, cx + 50, 570), fill=black)
        d.ellipse((cx - 20, 485, cx + 5, 510), fill="white")
    return img, []


def monogram():
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    green = (20, 90, 60)
    d.ellipse((300, 100, 1300, 1100), outline=green, width=26)
    d.ellipse((350, 150, 1250, 1050), outline=green, width=8)
    f_big, f_side = font("georgiab.ttf", 560), font("georgiab.ttf", 360)
    d.text((440, 380), "J", font=f_side, fill=green)
    d.text((630, 240), "M", font=f_big, fill=green)
    d.text((1010, 380), "R", font=f_side, fill=green)
    return img, ["JMR", "J M R"]


def small_text():
    img = Image.new("RGB", (W, 800), "white")
    d = ImageDraw.Draw(img)
    blue, grey = (30, 80, 160), (60, 60, 60)
    d.rounded_rectangle((80, 120, 340, 380), radius=40, fill=blue)               # icon
    d.polygon([(130, 330), (210, 170), (290, 330)], fill="white")
    f1, f2, f3 = font("arialbd.ttf", 130), font("arial.ttf", 64), font("arial.ttf", 52)
    d.text((400, 110), "Summit Plumbing", font=f1, fill=blue)
    d.text((400, 270), "Repairs  -  Installs  -  24/7 Service", font=f2, fill=grey)
    d.text((400, 380), "(555) 214-7788   summitplumbing.example", font=f3, fill=grey)
    return img, ["Summit Plumbing", "Repairs - Installs - 24/7 Service",
                 "(555) 214-7788 summitplumbing.example"]


def gradient_sunset():
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    cx, cy, r = W // 2, 700, 520
    top, bottom = (250, 90, 60), (255, 210, 80)
    for y in range(cy - r, cy + 1):                                               # sky disc
        t = (y - (cy - r)) / r
        col = tuple(int(top[k] + (bottom[k] - top[k]) * t) for k in range(3))
        half = int(math.sqrt(max(r * r - (y - cy) ** 2, 0)))
        d.line([(cx - half, y), (cx + half, y)], fill=col)
    d.ellipse((cx - 160, cy - 260, cx + 160, cy + 60), fill=(255, 240, 170))       # sun
    for k, y in enumerate(range(cy + 20, cy + 200, 45)):                          # sea bands
        d.rectangle((cx - r + 60 + k * 30, y, cx + r - 60 - k * 30, y + 22), fill=(40, 110, 170))
    d.line([(1050, cy), (1000, 380)], fill=(30, 30, 30), width=26)                 # palm
    for a in range(0, 360, 60):
        ex = 1000 + 200 * math.cos(math.radians(a))
        ey = 380 + 90 * math.sin(math.radians(a))
        d.line([(1000, 380), (ex, ey)], fill=(30, 30, 30), width=30)
    return img, []


def line_icon():
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    ink, wgt = (30, 30, 30), 18
    d.rounded_rectangle((520, 480, 1000, 980), radius=60, outline=ink, width=wgt)  # cup
    d.arc((930, 580, 1150, 840), 270, 90, fill=ink, width=wgt)                      # handle
    d.line([(440, 1040), (1080, 1040)], fill=ink, width=wgt)                        # saucer
    for x in (620, 760, 900):                                                       # steam
        pts = [(x + 40 * math.sin(t / 3.0), 420 - t * 12) for t in range(0, 22)]
        d.line(pts, fill=ink, width=wgt, joint="curve")
    return img, []


DESIGNS = {
    "script_bakery": script_bakery,
    "varsity_outline": varsity_outline,
    "mascot_bear": mascot_bear,
    "monogram": monogram,
    "small_text": small_text,
    "gradient_sunset": gradient_sunset,
    "line_icon": line_icon,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="redraw designs that already exist")
    args = ap.parse_args()
    bench = config.data_dir() / "bench"
    bench.mkdir(parents=True, exist_ok=True)
    for name, make in DESIGNS.items():
        png = bench / f"{name}.png"
        if png.exists() and not args.force:
            print("kept", png.name)
            continue
        img, words = make()
        img.save(png)
        txt = bench / f"{name}.txt"
        if words:
            txt.write_text("\n".join(words) + "\n", encoding="utf-8")
        elif txt.exists():
            txt.unlink()
        print("made", png.name)


if __name__ == "__main__":
    main()
