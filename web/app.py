"""Web prototype: upload a video and run goat counting."""
from __future__ import annotations

import json
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    DEFAULT_CROSSING_DIRECTION,
    DEFAULT_LINE_MODE,
    DEFAULT_LINE_X,
    DEFAULT_LINE_Y,
    DEFAULT_WEIGHTS,
    RESULTS_DIR,
    UPLOAD_DIR,
)
from src.counting_pipeline import CountingPipeline  # noqa: E402

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ASSET_VERSION = "12"

app = FastAPI(title="HerdLine")


class NoCacheHtmlMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type:
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response


app.add_middleware(NoCacheHtmlMiddleware)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")


def _page_context(**extra):
    return {"asset_version": ASSET_VERSION, **extra}


def _weights_path() -> Path | str:
    return DEFAULT_WEIGHTS if DEFAULT_WEIGHTS.is_file() else "yolov8n.pt"


def _parse_direction(line_mode: str, direction: str) -> str:
    if line_mode == "vertical":
        if direction in ("left_to_right", "right_to_left"):
            return direction
        return DEFAULT_CROSSING_DIRECTION
    if direction in ("down", "up"):
        return direction
    return "down"


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    has_model = DEFAULT_WEIGHTS.is_file()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=_page_context(
            active_nav="checkin",
            has_model=has_model,
            default_line_mode=DEFAULT_LINE_MODE,
            default_line_x=DEFAULT_LINE_X,
            default_line_y=DEFAULT_LINE_Y,
            default_direction=DEFAULT_CROSSING_DIRECTION,
        ),
    )


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context=_page_context(active_nav="dashboard"),
    )


@app.get("/checkins", response_class=HTMLResponse)
async def checkins_list(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="checkins.html",
        context=_page_context(active_nav="history"),
    )


@app.get("/checkins/{checkin_id}", response_class=HTMLResponse)
async def checkin_detail(request: Request, checkin_id: str):
    return templates.TemplateResponse(
        request=request,
        name="checkin_detail.html",
        context=_page_context(checkin_id=checkin_id, active_nav="history"),
    )


@app.get("/signup", response_class=HTMLResponse)
async def signup(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="signup.html",
        context=_page_context(active_nav="account"),
    )


@app.get("/login", response_class=HTMLResponse)
async def login(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context=_page_context(active_nav="account"),
    )


@app.post("/count", response_class=HTMLResponse)
async def count_goats(
    request: Request,
    video: UploadFile = File(...),
    herd_size: int = Form(...),
    line_mode: str = Form(DEFAULT_LINE_MODE),
    line_x: float = Form(DEFAULT_LINE_X),
    line_y: float = Form(DEFAULT_LINE_Y),
    direction: str = Form(DEFAULT_CROSSING_DIRECTION),
):
    job_id = uuid.uuid4().hex[:12]
    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    upload_path = UPLOAD_DIR / f"{job_id}{suffix}"
    out_name = f"{job_id}_counted.mp4"
    out_path = RESULTS_DIR / out_name

    with upload_path.open("wb") as f:
        shutil.copyfileobj(video.file, f)

    mode = "vertical" if line_mode == "vertical" else "horizontal"
    crossing = _parse_direction(mode, direction)
    herd = max(0, herd_size)

    pipeline = CountingPipeline(
        weights=_weights_path(),
        line_mode=mode,
        line_y=line_y,
        line_x=line_x,
        direction=crossing,
    )
    result = pipeline.run(
        upload_path,
        output_video=out_path,
        herd_size=herd,
    )

    recorded_at = datetime.now(timezone.utc).isoformat()
    video_url = f"/results/{out_name}"
    status = result.herd_match_status
    accuracy_pct = (
        round(result.counting_accuracy * 100, 1)
        if result.counting_accuracy is not None
        else None
    )
    checkin_payload = {
        "id": job_id,
        "at": recorded_at,
        "herdSize": result.herd_size,
        "counted": result.total_count,
        "missing": result.missing_count,
        "extra": result.extra_count,
        "status": status,
        "videoUrl": video_url,
        "accuracy": accuracy_pct,
        "framesProcessed": result.frames_processed,
    }

    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context=_page_context(
            active_nav="history",
            result=result,
            video_url=video_url,
            checkin_id=job_id,
            recorded_at=recorded_at,
            checkin_json=json.dumps(checkin_payload),
        ),
    )
