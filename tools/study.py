"""Learn from finished stitch files: measure how designs that were really sewn are built, and
compare our own output with them.

    .venv\\Scripts\\python.exe tools\\study.py "C:\\path\\to\\folder of DST or PES files"
    .venv\\Scripts\\python.exe tools\\study.py folder --ours     # the same numbers for the bench's last run

A stitch file has no objects, only needle points, so every stitch is sorted by what it does:

  satin   zig-zag: each stitch turns back on the one before (more than 140 degrees)
  fill    rows: several stitches in a straight line, then a short step and back
  run     a line of stitches that keeps going (turns of less than 60 degrees)

and the numbers a digitizer sets are read back from the geometry: satin width (the zig-zag's
stitch length), satin spacing (how far it advances per stitch), fill stitch length and row
spacing, run stitch length, trims, jumps, stitches per square inch.

Nothing from the studied files is copied or kept: only the summary numbers are written, to
the app's data folder (bench\\study.json). The files may be customers' designs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pyembroidery

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402

EXTS = {".dst", ".pes", ".jef", ".exp", ".vp3", ".xxx", ".hus", ".pec"}


def read_segments(path: Path) -> tuple[list[np.ndarray], dict]:
    """Thread segments (mm) between trims, jumps and color changes, and the counts of each."""
    p = pyembroidery.read(str(path))
    if p is None:
        return [], {}
    segs, cur = [], []
    trims = jumps = colors = 0
    last_cmd = None
    jump_run = 0
    for x, y, cmd in p.stitches:
        c = cmd & pyembroidery.COMMAND_MASK
        if c == pyembroidery.STITCH:
            if jump_run >= 2 or last_cmd in (pyembroidery.TRIM, pyembroidery.COLOR_CHANGE,
                                             pyembroidery.STOP):
                if len(cur) > 1:
                    segs.append(np.asarray(cur, float) / 10.0)
                cur = []
                if jump_run >= 2 and last_cmd == pyembroidery.JUMP:
                    trims += 1                    # DST writes a trim as a run of jumps
            cur.append((x, y))
            jump_run = 0
        elif c == pyembroidery.JUMP:
            jump_run += 1
            jumps += 1
        elif c == pyembroidery.TRIM:
            trims += 1
            jump_run = 2
        elif c in (pyembroidery.COLOR_CHANGE, pyembroidery.STOP):
            colors += 1
            jump_run = 2
        last_cmd = c
    if len(cur) > 1:
        segs.append(np.asarray(cur, float) / 10.0)
    return segs, {"trims": trims, "jumps": jumps, "color_changes": colors}


def classify(seg: np.ndarray) -> dict:
    """Sort one thread segment's stitches into satin, fill and run, and measure each."""
    out = {"satin_w": [], "satin_sp": [], "fill_len": [], "fill_sp": [], "run_len": [],
           "len": {"satin": 0.0, "fill": 0.0, "run": 0.0, "bean": 0.0}, "all_len": []}
    v = np.diff(seg, axis=0)
    L = np.linalg.norm(v, axis=1)
    keep = L > 0.05
    v, L = v[keep], L[keep]
    pts = np.vstack([seg[:1], seg[1:][keep]])
    n = len(L)
    out["all_len"] = L.tolist()
    if n < 3:
        out["len"]["run"] += float(L.sum())
        out["run_len"] += L.tolist()
        return out
    ang = np.arctan2(v[:, 1], v[:, 0])
    turn = np.abs((np.diff(ang) + math.pi) % (2 * math.pi) - math.pi)      # 0 straight, pi back
    back = turn > math.radians(140)
    kind = np.array(["run"] * n, dtype=object)
    # satin: a stitch with a reversal before and after it (zig-zag), longer than 0.8 mm
    for i in range(n):
        b0 = back[i - 1] if i > 0 else False
        b1 = back[i] if i < n - 1 else False
        if (b0 and b1) or ((b0 or b1) and L[i] >= 1.0 and i in (0, n - 1)):
            kind[i] = "satin"
    # fill: straight runs of 2+ stitches that end in a reversal (rows)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and turn[j] < math.radians(25):
            j += 1
        if j > i:                                   # stitches i..j are in one straight line
            ends_back = (j < n - 1 and turn[j] > math.radians(100)) or \
                        (i > 0 and turn[i - 1] > math.radians(100))
            # a row end: a short step then back, or straight back
            near_back = False
            if j + 2 < n and L[j + 1] < 1.5 and \
                    abs((ang[j + 2] - ang[j] + math.pi) % (2 * math.pi) - math.pi) > math.radians(150):
                near_back = True
            if ends_back or near_back:
                kind[i:j + 1] = "fill"
        i = j + 1
    # bean (triple run): the needle comes back to a hole it has just left (forward, back,
    # forward over each stitch); a satin moves on by its spacing instead. Checked last: a
    # bean along a straight line also looks like short rows.
    for i in range(n):
        back_to_start = i + 2 <= n and i + 2 < len(pts) and np.linalg.norm(pts[i + 2] - pts[i]) < 0.12
        back_to_end = i >= 1 and np.linalg.norm(pts[i + 1] - pts[i - 1]) < 0.12
        if back_to_start or back_to_end:
            kind[i] = "bean"
    for i in range(n):
        out["len"][kind[i]] += float(L[i])
    # measurements
    sat = np.nonzero(kind == "satin")[0]
    for i in sat:
        out["satin_w"].append(float(L[i]))
        if i + 2 < n and kind[i + 1] == "satin":
            adv = np.linalg.norm(pts[i + 2] - pts[i])
            out["satin_sp"].append(float(adv / 2.0))
    out["run_len"] += [float(L[i]) for i in range(n) if kind[i] in ("run", "bean")]
    # fill rows: consecutive fill stitches in a line make a row; row spacing is the distance
    # from one row's line to the next row's start
    rows, cur = [], [0] if kind[0] == "fill" else []
    for i in range(1, n):
        if kind[i] == "fill" and kind[i - 1] == "fill" and turn[i - 1] < math.radians(25):
            cur.append(i)
        else:
            if len(cur) >= 2:
                rows.append(cur)
            cur = [i] if kind[i] == "fill" else []
    if len(cur) >= 2:
        rows.append(cur)
    for r in rows:
        out["fill_len"] += [float(L[i]) for i in r[1:-1]] or [float(np.mean(L[r]))]
    for a, b in zip(rows, rows[1:]):
        if b[0] - a[-1] > 3:
            continue
        p0, p1 = pts[a[0]], pts[a[-1] + 1]
        d = p1 - p0
        nd = np.linalg.norm(d)
        if nd < 1e-6:
            continue
        q = pts[b[0] + 1] if b[0] + 1 < len(pts) else pts[b[0]]
        dist = abs(d[0] * (q[1] - p0[1]) - d[1] * (q[0] - p0[0])) / nd
        if 0.1 < dist < 3.0:
            out["fill_sp"].append(float(dist))
    return out


def pct(a, q):
    return round(float(np.percentile(a, q)), 2) if len(a) else None


def measure(segs: list[np.ndarray], counts: dict) -> dict | None:
    if not segs:
        return None
    allp = np.vstack(segs)
    w, h = np.ptp(allp[:, 0]), np.ptp(allp[:, 1])
    stitches = int(sum(len(s) for s in segs))
    acc = {"satin_w": [], "satin_sp": [], "fill_len": [], "fill_sp": [], "run_len": [], "all_len": []}
    lens = {"satin": 0.0, "fill": 0.0, "run": 0.0, "bean": 0.0}
    for s in segs:
        c = classify(s)
        for k in acc:
            acc[k] += c[k]
        for k in lens:
            lens[k] += c["len"][k]
    total = sum(lens.values()) or 1.0
    al = np.asarray(acc["all_len"])
    return {
        "width_mm": round(float(w), 1), "height_mm": round(float(h), 1),
        "stitches": stitches, "segments": len(segs),
        "trims": counts.get("trims", len(segs) - 1), "colors": counts.get("color_changes", 0) + 1,
        "trims_per_1000": round(1000.0 * counts.get("trims", len(segs) - 1) / max(stitches, 1), 2),
        "stitches_per_in2": round(stitches / max(w * h / 645.16, 0.01)),
        "share": {k: round(v / total, 3) for k, v in lens.items()},
        "satin_width": {"p10": pct(acc["satin_w"], 10), "p50": pct(acc["satin_w"], 50),
                        "p90": pct(acc["satin_w"], 90), "max": pct(acc["satin_w"], 99.5)},
        "satin_spacing": {"p25": pct(acc["satin_sp"], 25), "p50": pct(acc["satin_sp"], 50),
                          "p75": pct(acc["satin_sp"], 75)},
        "fill_stitch": {"p50": pct(acc["fill_len"], 50), "p90": pct(acc["fill_len"], 90)},
        "fill_spacing": {"p25": pct(acc["fill_sp"], 25), "p50": pct(acc["fill_sp"], 50),
                         "p75": pct(acc["fill_sp"], 75)},
        "run_stitch": {"p10": pct(acc["run_len"], 10), "p50": pct(acc["run_len"], 50),
                       "p90": pct(acc["run_len"], 90)},
        "short_under_0.5": round(float((al < 0.5).mean()), 4) if len(al) else None,
        "long_over_7": round(float((al > 7.0).mean()), 4) if len(al) else None,
    }


def summarise(rows: list[dict]) -> dict:
    """Typical values across files (medians, with the middle half)."""
    def col(f):
        v = [f(r) for r in rows]
        v = [x for x in v if x is not None]
        if not v:
            return None
        return {"p25": pct(v, 25), "p50": pct(v, 50), "p75": pct(v, 75), "n": len(v)}
    return {
        "files": len(rows),
        "max_side_mm": col(lambda r: max(r["width_mm"], r["height_mm"])),
        "stitches": col(lambda r: r["stitches"]),
        "trims_per_1000": col(lambda r: r["trims_per_1000"]),
        "trims": col(lambda r: r["trims"]),
        "colors": col(lambda r: r["colors"]),
        "stitches_per_in2": col(lambda r: r["stitches_per_in2"]),
        "share_satin": col(lambda r: r["share"]["satin"]),
        "share_fill": col(lambda r: r["share"]["fill"]),
        "share_run": col(lambda r: r["share"]["run"]),
        "share_bean": col(lambda r: r["share"]["bean"]),
        "satin_width_p10": col(lambda r: r["satin_width"]["p10"]),
        "satin_width_p50": col(lambda r: r["satin_width"]["p50"]),
        "satin_width_p90": col(lambda r: r["satin_width"]["p90"]),
        "satin_width_max": col(lambda r: r["satin_width"]["max"]),
        "satin_spacing_p50": col(lambda r: r["satin_spacing"]["p50"]),
        "fill_stitch_p50": col(lambda r: r["fill_stitch"]["p50"]),
        "fill_spacing_p50": col(lambda r: r["fill_spacing"]["p50"]),
        "run_stitch_p50": col(lambda r: r["run_stitch"]["p50"]),
        "run_stitch_p10": col(lambda r: r["run_stitch"]["p10"]),
        "short_under_0.5": col(lambda r: r["short_under_0.5"]),
        "long_over_7": col(lambda r: r["long_over_7"]),
    }


def show(title: str, s: dict) -> None:
    print(f"\n{title}  ({s['files']} files)")
    for k, v in s.items():
        if isinstance(v, dict):
            print(f"  {k:20s} {v['p50']!s:>8}   (middle half {v['p25']} to {v['p75']})")


def ours_latest() -> list[Path]:
    runs = sorted((config.data_dir() / "bench" / "runs").glob("*/scores.json"))
    if not runs:
        return []
    return sorted(runs[-1].parent.glob("*/*.dst"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--ours", action="store_true", help="also measure the bench's last run")
    ap.add_argument("--min-side", type=float, default=0.0, help="only designs at least this many mm")
    ap.add_argument("--max-side", type=float, default=1e9)
    ap.add_argument("--list", action="store_true", help="print one line per file")
    a = ap.parse_args()
    files = sorted(p for p in Path(a.folder).rglob("*") if p.suffix.lower() in EXTS)
    rows = []
    for p in files:
        try:
            m = measure(*read_segments(p))
        except Exception as e:  # noqa: BLE001 - one bad file must not stop the study
            print(f"  (could not read {p.name}: {e})")
            continue
        if m and a.min_side <= max(m["width_mm"], m["height_mm"]) <= a.max_side:
            rows.append(m)
            if a.list:
                print(f"{max(m['width_mm'], m['height_mm']):6.0f} mm {m['stitches']:6d} st  "
                      f"trims {m['trims']:3d} ({m['trims_per_1000']:5.1f}/1000)  "
                      f"satin {m['share']['satin']:.0%} fill {m['share']['fill']:.0%} run {m['share']['run']:.0%} bean {m['share']['bean']:.0%}  "
                      f"satin w {m['satin_width']['p50']} sp {m['satin_spacing']['p50']}  "
                      f"fill {m['fill_stitch']['p50']}/{m['fill_spacing']['p50']}  run {m['run_stitch']['p50']}")
    if not rows:
        print("No stitch files found.")
        return
    ref = summarise(rows)
    show("Reference files", ref)
    out = {"reference": ref}
    if a.ours:
        mine = [m for m in (measure(*read_segments(p)) for p in ours_latest()) if m]
        if mine:
            out["ours"] = summarise(mine)
            show("Our last bench run", out["ours"])
    dest = config.data_dir() / "bench" / "study.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nSaved the summary numbers to {dest}")


if __name__ == "__main__":
    main()
