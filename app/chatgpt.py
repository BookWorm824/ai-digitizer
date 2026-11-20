"""ChatGPT through the user's ChatGPT plan, with no API key and no per-picture charge.

It works by running OpenAI's Codex program (the same way CleanView does): Codex is signed in
with the ChatGPT account and can both look at pictures and generate them. Two calls:

  ask_json()  look at a picture, answer as JSON matching a schema
  redraw()    redraw a picture with the image-generation tool and hand back the PNG

Failures are raised as ai.AIUnavailable by the caller's wrapper; here they are ChatGPTError
with a reason code so this module doesn't depend on app.ai.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from PIL import Image

from . import config

_NOWIN = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
VISION_TIMEOUT = 240
REDRAW_TIMEOUT = 480

_status_cache: tuple[float, str] | None = None


class ChatGPTError(Exception):
    """`.reason` is one of: chatgpt_missing, chatgpt_signed_out, chatgpt_limit, slow, error."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(reason)
        self.reason, self.detail = reason, detail


def find() -> str | None:
    """Where the Codex program is, or None if it isn't installed."""
    exe = shutil.which("codex")
    if exe:
        return exe
    la = Path(os.environ.get("LOCALAPPDATA") or ".")
    ad = Path(os.environ.get("APPDATA") or ".")
    for p in (la / "Programs" / "OpenAI" / "Codex" / "bin" / "codex.exe",
              la / "Microsoft" / "WinGet" / "Links" / "codex.exe",
              ad / "npm" / "codex.cmd",
              Path.home() / ".local" / "bin" / "codex"):
        if p.is_file():
            return str(p)
    return None


def _env() -> dict:
    env = dict(os.environ)
    for k in ("OPENAI_API_KEY", "CODEX_API_KEY"):   # always the ChatGPT plan, never a paid key
        env.pop(k, None)
    return env


def _run(args: list[str], stdin: str, timeout: int, cwd: str) -> tuple[int, str, str]:
    """Run Codex once. (Tests replace this function.)"""
    r = subprocess.run(args, input=stdin, capture_output=True, text=True, encoding="utf8",
                       errors="replace", timeout=timeout, cwd=cwd, env=_env(), **_NOWIN)
    return r.returncode, r.stdout or "", r.stderr or ""


def status(refresh: bool = False) -> str:
    """'ready', 'signed_out' or 'missing'. Checked at most once a minute."""
    global _status_cache
    if not refresh and _status_cache and time.time() - _status_cache[0] < 60:
        return _status_cache[1]
    exe = find()
    if not exe:
        value = "missing"
    else:
        try:
            code, out, err = _run([exe, "login", "status"], "", 30, str(config.data_dir()))
            signed_in = code == 0 and "not logged in" not in (out + err).lower()
        except Exception:  # noqa: BLE001
            signed_in = False
        value = "ready" if signed_in else "signed_out"
    _status_cache = (time.time(), value)
    return value


def forget_status() -> None:
    global _status_cache
    _status_cache = None


def open_sign_in() -> None:
    """Open a small window that walks the user through signing in to ChatGPT."""
    exe = find()
    if not exe:
        raise ChatGPTError("chatgpt_missing")
    forget_status()
    if os.name == "nt":
        subprocess.Popen(
            ["cmd", "/c", "start", "Sign in to ChatGPT", "cmd", "/k",
             "echo Sign in with your ChatGPT account in the browser window that opens. & "
             "echo When it says you are signed in, close this window and go back to the "
             f'digitizer. & echo. & "{exe}" login'], env=_env())
    else:
        subprocess.Popen([exe, "login"], env=_env())


def _problem(code: int, out: str, err: str) -> ChatGPTError:
    msgs = []
    for line in out.splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if isinstance(ev, dict) and ev.get("type") in ("error", "turn.failed"):
            msgs.append(str(ev.get("message") or (ev.get("error") or {}).get("message") or ""))
    detail = (" ".join(msgs) or err or out)[-600:]
    low = detail.lower()
    if any(k in low for k in ("not logged in", "unauthorized", "401", "sign in", "log in")):
        forget_status()
        return ChatGPTError("chatgpt_signed_out", detail)
    if any(k in low for k in ("usage limit", "rate limit", "limit reached", "quota", "429")):
        return ChatGPTError("chatgpt_limit", detail)
    return ChatGPTError("error", detail)


def _base() -> Path:
    """Where Codex gets its scratch folders. This is the Windows temp folder on purpose:
    Codex refuses to write its answer file under some folders (it reports "path contains a
    reparse point" for the app's own data folder), and the temp folder always works."""
    return Path(tempfile.gettempdir()) / "AIDigitizer-chatgpt"


def _workspace() -> tuple[Path, Path]:
    """(a fresh temp folder for this call, an always-empty folder for read-only calls)"""
    base = _base()
    empty = base / "empty"
    empty.mkdir(parents=True, exist_ok=True)
    # A plain folder, not tempfile.mkdtemp(): mkdtemp locks the folder to this user only, and
    # Codex's sandbox (which writes the picture) then can't write into it.
    tmp = base / uuid.uuid4().hex[:12]
    tmp.mkdir()
    return tmp, empty


def _exe() -> str:
    exe = find()
    if not exe:
        raise ChatGPTError("chatgpt_missing")
    return exe


def ask_json(prompt: str, schema: dict, png: bytes) -> dict:
    """Show ChatGPT a picture and get an answer that matches the JSON schema."""
    exe = _exe()
    tmp, empty = _workspace()
    try:
        (tmp / "picture.png").write_bytes(png)
        (tmp / "schema.json").write_text(json.dumps(schema), encoding="utf8")
        answer = tmp / "answer.json"
        args = [exe, "exec", "-s", "read-only", "-C", str(empty), "--color", "never",
                "--skip-git-repo-check", "--ephemeral",
                "--output-schema", str(tmp / "schema.json"), "-o", str(answer), "--json",
                f"--image={tmp / 'picture.png'}", "-"]
        full = (prompt + "\n\nAnswer only from the attached picture. Don't run any commands, "
                         "open files or search the web.")
        try:
            code, out, err = _run(args, full, VISION_TIMEOUT, str(empty))
        except subprocess.TimeoutExpired:
            raise ChatGPTError("slow")
        try:
            data = json.loads(answer.read_text("utf8"))
        except (OSError, ValueError):
            data = None
        if not isinstance(data, dict):
            raise _problem(code, out, err)
        return data
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def redraw(png: bytes, prompt: str) -> bytes:
    """Have ChatGPT redraw a picture with its image-generation tool. Returns PNG bytes."""
    exe = _exe()
    tmp, _empty = _workspace()
    try:
        work = tmp / "work"
        work.mkdir()
        (tmp / "picture.png").write_bytes(png)
        args = [exe, "exec", "-s", "workspace-write", "-C", str(work), "--color", "never",
                "--skip-git-repo-check", "--ephemeral", "-o", str(tmp / "last.txt"), "--json",
                f"--image={tmp / 'picture.png'}", "-"]
        full = ("Use your image generation tool to edit the attached picture.\n" + prompt +
                "\nGenerate exactly one image, then save it as result.png in the current "
                "working directory. Do nothing else and reply with only the word DONE.")
        try:
            code, out, err = _run(args, full, REDRAW_TIMEOUT, str(work))
        except subprocess.TimeoutExpired:
            raise ChatGPTError("slow")
        found = [work / "result.png"] + sorted(
            p for p in work.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
        for path in found:
            try:
                img = Image.open(path)
                img.load()
            except (OSError, ValueError):
                continue
            buf = io.BytesIO()
            img.convert("RGB").save(buf, "PNG")
            return buf.getvalue()
        raise _problem(code, out, err)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
