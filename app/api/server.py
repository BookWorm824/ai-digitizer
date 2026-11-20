"""The local web server: a few JSON endpoints plus the screens in web/.

Flow: a picture is uploaded and cleaned up (locally, and by ChatGPT when a key is set), the
user picks a version, that version is digitized, and the finished files are offered for
download. Slow work runs on background threads so the screen can show progress in plain words.
"""

from __future__ import annotations

import logging
import re
import threading
import traceback
import uuid
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__, ai, cleanup, config, settings
from ..pipeline import PipelineError, run_pipeline

MAX_UPLOAD = 40 * 1024 * 1024
GENERIC_ERROR = ("Something went wrong while making your file. Please try again, "
                 "or try a different picture.")
LOST = "We lost track of that design. Please start again."

app = FastAPI(title="AI Embroidery Digitizer", docs_url=None, redoc_url=None)
CLEANUPS: dict[str, dict] = {}     # the "Clean up" step, one per uploaded picture
JOBS: dict[str, dict] = {}         # the "Stitches" step, one per digitizing run
_lock = threading.Lock()


def _log_error(text: str) -> None:
    """Technical details go to a log file, never to the screen."""
    try:
        with open(config.data_dir() / "log.txt", "a", encoding="utf-8") as f:
            f.write(text + "\n")
    except OSError:
        logging.getLogger(__name__).error(text)


def _safe_name(filename: str | None) -> str:
    stem = Path(filename or "design").stem
    stem = re.sub(r"[^A-Za-z0-9 _-]+", "", stem).strip()[:40]
    return stem or "design"


def _fit_box(size: str, custom_in: float | None) -> tuple[float, float]:
    if size == "custom":
        inches = min(12.0, max(0.5, float(custom_in or 4.0)))
        return inches * config.MM_PER_INCH, inches * config.MM_PER_INCH
    for s in config.sizes():
        if s["id"] == size and "w_in" in s:
            return s["w_in"] * config.MM_PER_INCH, s["h_in"] * config.MM_PER_INCH
    return 3.5 * config.MM_PER_INCH, 3.5 * config.MM_PER_INCH


async def _read_upload(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(400, "That file is empty. Please choose a picture.")
    if len(data) > MAX_UPLOAD:
        raise HTTPException(400, "That picture is too big. Please use one under 40 MB.")
    if (file.filename or "").lower().endswith(".svg"):
        raise HTTPException(400, "SVG pictures aren't supported yet. "
                                 "Please save it as a PNG and try again.")
    return data


def _background(job: dict, work, on_error=None) -> None:
    """Run `work()` on a thread, recording plain-words errors on the job."""
    def runner():
        try:
            work()
            job["status"] = "done"
        except PipelineError as e:
            job["error"] = str(e)
            job["status"] = "error"
        except Exception:  # noqa: BLE001 - anything unexpected: log it, show plain words
            _log_error(traceback.format_exc())
            job["error"] = GENERIC_ERROR
            job["status"] = "error"
    threading.Thread(target=runner, daemon=True).start()


# ------------------------------------------------------------------------------ options

@app.get("/api/options")
def options():
    return {
        "version": __version__,
        "sizes": config.sizes(),
        "fabrics": [{"id": f["id"], "name": f["name"]} for f in config.fabrics().values()],
        "machines": config.machines(),
        "max_colors": config.RULES["max_colors"],
    }


@app.get("/api/settings")
def get_settings():
    return ai.status()


@app.post("/api/settings")
def post_settings(body: dict = Body(...)):
    changes = {}
    if body.get("remove_key"):
        changes["openai_api_key"] = ""
    elif body.get("api_key"):
        key = str(body["api_key"]).strip()
        if not key.startswith("sk-") or len(key) < 20 or " " in key:
            raise HTTPException(400, "That doesn't look like an OpenAI key. "
                                     "It starts with sk- and has no spaces.")
        changes["openai_api_key"] = key
    if body.get("daily_spend_cap_usd") is not None:
        try:
            cap = float(body["daily_spend_cap_usd"])
        except (TypeError, ValueError):
            raise HTTPException(400, "The daily limit must be a number, like 2.")
        if not 0 <= cap <= 100:
            raise HTTPException(400, "The daily limit must be between $0 and $100.")
        changes["daily_spend_cap_usd"] = cap
    if changes:
        settings.save(changes)
    return ai.status()


# ----------------------------------------------------------------------------- clean up

@app.post("/api/cleanup")
async def create_cleanup(file: UploadFile = File(...), size: str = Form("left_chest"),
                         custom_in: float | None = Form(None), fabric: str = Form("tshirt")):
    data = await _read_upload(file)
    cid = uuid.uuid4().hex[:12]
    job = {"status": "working", "label": "Getting started…", "name": _safe_name(file.filename),
           "dir": config.data_dir() / "jobs" / cid, "fit": _fit_box(size, custom_in),
           "fabric": fabric}
    with _lock:
        CLEANUPS[cid] = job

    def work():
        job["result"] = cleanup.run_cleanup(
            data, job["dir"], fit=job["fit"], fabric_id=fabric,
            progress=lambda label: job.__setitem__("label", label))

    _background(job, work)
    return {"id": cid}


def _cleanup(cid: str) -> dict:
    job = CLEANUPS.get(cid)
    if job is None:
        raise HTTPException(404, LOST)
    return job


@app.get("/api/cleanup/{cid}")
def cleanup_status(cid: str):
    job = _cleanup(cid)
    return {k: job.get(k) for k in ("status", "label", "result", "error")}


@app.get("/api/cleanup/{cid}/image/{name}")
def cleanup_image(cid: str, name: str):
    job = _cleanup(cid)
    if not re.fullmatch(r"[a-z0-9_]{1,24}", name):
        raise HTTPException(404, "That picture isn't ready.")
    path = job["dir"] / f"{name}.png"
    if not path.exists():
        raise HTTPException(404, "That picture isn't ready.")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-store"})


def _change_versions(cid: str, label: str, action) -> dict:
    """Shared by "tweak" and "try again": run an AI change in the background, then put the
    updated list of versions (and any plain-words problem) on the clean-up result."""
    job = _cleanup(cid)
    if job.get("status") != "done":
        raise HTTPException(409, "Still working on the last change. Give it a moment.")
    job["status"], job["label"] = "working", label
    result = job["result"]
    result["flash"] = None

    def work():
        try:
            result["picked"] = action(job["dir"])["id"]
        except ai.AIUnavailable as e:
            result["flash"] = e.message
        except PipelineError as e:
            result["flash"] = str(e)
        ctx = cleanup._load_ctx(job["dir"])
        result["versions"] = [cleanup._public(v) for v in ctx["versions"]]
        result["spend"] = ai.status()

    _background(job, work)
    return {"ok": True}


@app.post("/api/cleanup/{cid}/tweak")
def cleanup_tweak(cid: str, version: str = Form(...), change: str = Form(...)):
    if not change.strip():
        raise HTTPException(400, "Type the change you'd like first.")
    return _change_versions(cid, "Making your change…",
                            lambda d: cleanup.run_tweak(d, version, change))


@app.post("/api/cleanup/{cid}/retry")
def cleanup_retry(cid: str, version: str = Form(...)):
    return _change_versions(cid, "Drawing it again…", lambda d: cleanup.run_retry(d, version))


# ------------------------------------------------------------------------------ stitches

@app.post("/api/jobs")
async def create_job(cleanup_id: str | None = Form(None), version: str = Form("local"),
                     colors: int = Form(6), file: UploadFile | None = File(None),
                     size: str = Form("left_chest"), custom_in: float | None = Form(None),
                     fabric: str = Form("tshirt")):
    """Digitize either a version chosen on the Clean up screen (cleanup_id + version), or a
    picture uploaded directly (file + size + fabric)."""
    if cleanup_id:
        src = _cleanup(cleanup_id)
        if "result" not in src:
            raise HTTPException(409, "That picture is still being cleaned up.")
        name, fit, fabric = src["name"], src["fit"], src["fabric"]
        get_data = lambda say: cleanup.picture_to_stitch(src["dir"], version, say)  # noqa: E731
    elif file is not None:
        data = await _read_upload(file)
        name, fit = _safe_name(file.filename), _fit_box(size, custom_in)
        get_data = lambda say: data  # noqa: E731
    else:
        raise HTTPException(400, "Please choose a picture first.")

    job_id = uuid.uuid4().hex[:12]
    job = {"status": "working", "label": "Getting started…"}
    with _lock:
        JOBS[job_id] = job
    colors = int(min(config.RULES["max_colors"], max(1, colors)))

    def work():
        say = lambda label: job.__setitem__("label", label)  # noqa: E731
        result = run_pipeline(get_data(say), config.data_dir() / "jobs" / job_id, name,
                              fit_w_mm=fit[0], fit_h_mm=fit[1], fabric_id=fabric,
                              max_colors=colors, progress=say)
        job["paths"] = {fmt: info.pop("path") for fmt, info in result["files"].items()}
        job["paths"]["preview"] = result.pop("preview")
        job["result"] = result

    _background(job, work)
    return {"id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, LOST)
    return {k: job.get(k) for k in ("status", "label", "result", "error")}


@app.get("/api/jobs/{job_id}/download/{fmt}")
def download(job_id: str, fmt: str):
    job = JOBS.get(job_id)
    if job is None or job.get("status") != "done" or fmt not in job["paths"] or fmt == "preview":
        raise HTTPException(404, "That file isn't ready. Please make the design again.")
    if not job["result"]["files"][fmt]["ok"]:
        raise HTTPException(409, "That file didn't pass its safety check. Please try again.")
    path = Path(job["paths"][fmt])
    return FileResponse(path, filename=path.name, media_type="application/octet-stream")


@app.get("/api/jobs/{job_id}/preview.png")
def preview(job_id: str):
    job = JOBS.get(job_id)
    if job is None or job.get("status") != "done":
        raise HTTPException(404, "That picture isn't ready.")
    return FileResponse(job["paths"]["preview"], media_type="image/png")


@app.exception_handler(HTTPException)
async def http_error(_request, exc: HTTPException):
    return JSONResponse({"error": str(exc.detail)}, status_code=exc.status_code)


@app.exception_handler(Exception)
async def any_error(_request, exc: Exception):
    _log_error("".join(traceback.format_exception(exc)))
    return JSONResponse({"error": GENERIC_ERROR}, status_code=500)


# The screens. Mounted last so /api/... wins.
app.mount("/", StaticFiles(directory=str(config.WEB_DIR), html=True), name="web")
