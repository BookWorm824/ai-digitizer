"""The local web server: a few JSON endpoints plus the screens in web/.

A picture is uploaded, digitized on a background thread (so the screen can show progress in
plain words), and the finished files are offered for download.
"""

from __future__ import annotations

import logging
import re
import threading
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__, config
from ..pipeline import PipelineError, run_pipeline

MAX_UPLOAD = 40 * 1024 * 1024
GENERIC_ERROR = ("Something went wrong while making your file. Please try again, "
                 "or try a different picture.")

app = FastAPI(title="AI Embroidery Digitizer", docs_url=None, redoc_url=None)
JOBS: dict[str, dict] = {}
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


def _work(job_id: str, data: bytes, name: str, fit: tuple, fabric: str, colors: int) -> None:
    job = JOBS[job_id]

    def progress(label: str) -> None:
        job["label"] = label

    try:
        result = run_pipeline(data, config.data_dir() / "jobs" / job_id, name,
                              fit_w_mm=fit[0], fit_h_mm=fit[1], fabric_id=fabric,
                              max_colors=colors, progress=progress)
        job["paths"] = {fmt: info.pop("path") for fmt, info in result["files"].items()}
        job["paths"]["preview"] = result.pop("preview")
        job["result"] = result
        job["status"] = "done"
    except PipelineError as e:
        job["error"] = str(e)
        job["status"] = "error"
    except Exception:  # noqa: BLE001 - anything unexpected: log it, show plain words
        _log_error(f"job {job_id}\n{traceback.format_exc()}")
        job["error"] = GENERIC_ERROR
        job["status"] = "error"


@app.get("/api/options")
def options():
    return {
        "version": __version__,
        "sizes": config.sizes(),
        "fabrics": [{"id": f["id"], "name": f["name"]} for f in config.fabrics().values()],
        "machines": config.machines(),
        "max_colors": config.RULES["max_colors"],
    }


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...), size: str = Form("left_chest"),
                     custom_in: float | None = Form(None), fabric: str = Form("tshirt"),
                     colors: int = Form(6)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "That file is empty. Please choose a picture.")
    if len(data) > MAX_UPLOAD:
        raise HTTPException(400, "That picture is too big. Please use one under 40 MB.")
    if (file.filename or "").lower().endswith(".svg"):
        raise HTTPException(400, "SVG pictures aren't supported yet. "
                                 "Please save it as a PNG and try again.")
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        JOBS[job_id] = {"status": "working", "label": "Getting started…"}
    colors = int(min(config.RULES["max_colors"], max(1, colors)))
    threading.Thread(
        target=_work, daemon=True,
        args=(job_id, data, _safe_name(file.filename), _fit_box(size, custom_in), fabric, colors),
    ).start()
    return {"id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "We lost track of that design. Please start again.")
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
