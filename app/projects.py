"""The project library: every design is saved automatically so it can be reopened and
changed later, and nothing is lost by closing the app.

A project is a folder in the app's data folder holding the original picture, the cleaned-up
versions, and a small file of the choices made (version, size, colors, changes...).
"""

from __future__ import annotations

import datetime
import json
import re
import shutil
from pathlib import Path

from . import config

ID_RE = re.compile(r"[a-f0-9]{12}")


def _root() -> Path:
    d = config.data_dir() / "projects"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _dir(pid: str) -> Path:
    if not ID_RE.fullmatch(pid or ""):
        raise KeyError(pid)
    return _root() / pid


def files_dir(pid: str) -> Path:
    return _dir(pid) / "files"


def save(pid: str, name: str, cleanup_dir: Path, cleanup_result: dict, fit, fabric: str,
         settings: dict, summary: dict, preview_png: Path | None) -> dict:
    """Create or update a project from a clean-up folder and the current choices."""
    d = _dir(pid)
    files = d / "files"
    files.mkdir(parents=True, exist_ok=True)
    if Path(cleanup_dir).resolve() != files.resolve():
        for p in Path(cleanup_dir).iterdir():
            if p.is_file():
                shutil.copy2(p, files / p.name)
    if preview_png and Path(preview_png).exists():
        shutil.copy2(preview_png, d / "thumb.png")
    meta = {
        "id": pid, "name": name,
        "saved": datetime.datetime.now().isoformat(timespec="seconds"),
        "fit": list(fit), "fabric": fabric, "settings": settings, "summary": summary,
        "cleanup": {k: v for k, v in cleanup_result.items() if k not in ("spend", "flash", "picked")},
    }
    tmp = d / "project.tmp"
    tmp.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    tmp.replace(d / "project.json")
    return _card(meta)


def _card(meta: dict) -> dict:
    return {"id": meta["id"], "name": meta["name"], "saved": meta["saved"],
            "fabric": meta.get("fabric"), "summary": meta.get("summary", {})}


def load(pid: str) -> dict:
    return json.loads((_dir(pid) / "project.json").read_text(encoding="utf-8"))


def list_all() -> list[dict]:
    """Every saved project, newest first."""
    out = []
    for d in _root().iterdir():
        try:
            out.append(_card(json.loads((d / "project.json").read_text(encoding="utf-8"))))
        except (OSError, ValueError, KeyError):
            continue
    return sorted(out, key=lambda c: c["saved"], reverse=True)


def thumb(pid: str) -> Path:
    return _dir(pid) / "thumb.png"


def delete(pid: str) -> None:
    d = _dir(pid)
    if d.is_dir():
        shutil.rmtree(d)
