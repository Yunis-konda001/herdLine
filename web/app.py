"""HerdLine website: accounts, upload, count, and results.

- This is the main FastAPI backend file.
- It connects the HTML frontend to the Python pipeline (YOLO, tracking, counting).
- It handles routing (sign-up, login, upload, logout) and the file system (saving videos).
"""
from __future__ import annotations

import json
import secrets
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Import the database/auth functions and the counting pipeline.
from src.accounts import authenticate, create_user, get_user, init_db  # noqa: E402
from src.config import (  # noqa: E402
    DEFAULT_CROSSING_DIRECTION,
    DEFAULT_LINE_MODE,
    DEFAULT_LINE_X,
    DEFAULT_LINE_Y,
    DEFAULT_ONNX_WEIGHTS,
    DEFAULT_WEIGHTS,
    OUTPUT_DIR,
    RESULTS_DIR,
    UPLOAD_DIR,
)
from src.counting_pipeline import CountingPipeline 

# Ensure the upload and results folders exist on startup.
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ASSET_VERSION = "18"
SESSION_SECRET_PATH = OUTPUT_DIR / ".session_secret"
_PIPELINE_CACHE: dict[tuple, CountingPipeline] = {}

app = FastAPI(title="HerdLine")
init_db()  # Create the SQLite database table on startup.


class NoCacheHtmlMiddleware(BaseHTTPMiddleware):
    # Prevent the browser from caching HTML pages so users always see fresh data.
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type:
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response


def _session_secret() -> str:
    # Generate or load a secret key used to encrypt session cookies.
    if SESSION_SECRET_PATH.is_file():
        saved = SESSION_SECRET_PATH.read_text(encoding="utf-8").strip()
        if saved:
            return saved
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    secret = secrets.token_hex(32)
    SESSION_SECRET_PATH.write_text(secret, encoding="utf-8")
    return secret


app.add_middleware(NoCacheHtmlMiddleware)
app.add_middleware(
    SessionMiddleware,
    secret_key=_session_secret(),
    session_cookie="herdline_session",
    max_age=60 * 60 * 24 * 14, # Session lasts 14 days.
    same_site="lax",
    https_only=False,
)
# Mount templates and static files (CSS, JS, output videos).
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")


def _page_context(request: Request, **extra):
    # Helper to pass the logged-in user and asset version to every HTML page.
    user_id = request.session.get("user_id")
    user = get_user(int(user_id)) if user_id else None
    return {"asset_version": ASSET_VERSION, "current_user": user, **extra}


def _auth_page(
    # ... (parameters omitted for brevity) ...
) -> HTMLResponse:
    # Renders the sign-up or login page with any error messages.
    return templates.TemplateResponse(
        request=request,
        name="auth.html",
        status_code=status_code,
        context=_page_context(...),
    )


def _weights_path() -> Path | str:
    # Prefer ONNX on deploy hosts (much lower RAM than PyTorch)
    if DEFAULT_ONNX_WEIGHTS.is_file():
        return DEFAULT_ONNX_WEIGHTS
    if DEFAULT_WEIGHTS.is_file():
        return DEFAULT_WEIGHTS
    return "yolov8n.pt"


def _get_pipeline(
    line_mode: str, line_x: float, line_y: float, direction: str
) -> CountingPipeline:
    # Caches the pipeline so we don't reload the AI model on every upload.
    key = (_weights_path(), line_mode, float(line_x), float(line_y), direction)
    pipe = _PIPELINE_CACHE.get(key)
    if pipe is None:
        pipe = CountingPipeline(
            weights=_weights_path(),
            line_mode=line_mode,
            line_y=line_y,
            line_x=line_x,
            direction=direction,
        )
        _PIPELINE_CACHE.clear() # Clear old pipelines to save memory.
        _PIPELINE_CACHE[key] = pipe
    return pipe


def _parse_direction(line_mode: str, direction: str) -> str:
    # Validates the direction input from the form.
    if line_mode == "vertical":
        if direction in ("left_to_right", "right_to_left", "either"):
            return direction
        return DEFAULT_CROSSING_DIRECTION
    if direction in ("down", "up", "either"):
        return direction
    return "either"


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Renders the main check-in page (where the farmer uploads a video).
    has_model = DEFAULT_ONNX_WEIGHTS.is_file() or DEFAULT_WEIGHTS.is_file()
    return templates.TemplateResponse(...)


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    # Renders the dashboard page.
    return templates.TemplateResponse(...)


@app.get("/checkins", response_class=HTMLResponse)
async def checkins_list(request: Request):
    # Renders the history list of past check-ins.
    return templates.TemplateResponse(...)


@app.get("/checkins/{checkin_id}", response_class=HTMLResponse)
async def checkin_detail(request: Request, checkin_id: str):
    # Renders the details of a single past check-in.
    return templates.TemplateResponse(...)


@app.get("/signup", response_class=HTMLResponse)
async def signup(request: Request):
    # Shows the sign-up form.
    return _auth_page(request, mode="signup")


@app.post("/signup", response_class=HTMLResponse)
async def signup_submit(
    # ... (parameters omitted for brevity) ...
):
    # 1. Clean up the inputs.
    email = email.strip()
    farm_name = farm_name.strip()
    farmer_name = farmer_name.strip()
    try:
        herd = int(herd_size)
    except ValueError:
        herd = 0
    
    # 2. Validate the inputs.
    if not email or "@" not in email or not farm_name or not farmer_name or herd < 1:
        return _auth_page(...)
    if len(password) < 8:
        return _auth_page(...)
    
    # 3. Create the user in the SQLite database.
    user = create_user(email, password, farm_name, farmer_name, herd)
    if user is None:
        return _auth_page(...)
    
    # 4. Log the user in by saving their ID in the session.
    request.session["user_id"] = user["id"]
    return RedirectResponse("/dashboard", status_code=303)


@app.get("/login", response_class=HTMLResponse)
async def login(request: Request):
    # Shows the login form.
    return _auth_page(request, mode="signin")


@app.post("/login", response_class=HTMLResponse)
async def login_submit(
):
    # 1. Verify the password against the database.
    user = authenticate(email, password)
    if user is None:
        return _auth_page(...)
    
    # 2. Save the user in the session and redirect.
    request.session["user_id"] = user["id"]
    return RedirectResponse("/dashboard", status_code=303)


@app.post("/logout")
async def logout(request: Request):
    # Clears the session and redirects to login.
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


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
    # 1. Generate a unique job ID for this upload.
    job_id = uuid.uuid4().hex[:12]
    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    
    # 2. Save the uploaded video to the uploads folder.
    upload_path = UPLOAD_DIR / f"{job_id}{suffix}"
    out_name = f"{job_id}_counted.mp4"
    out_path = RESULTS_DIR / out_name

    with upload_path.open("wb") as f:
        shutil.copyfileobj(video.file, f)

    # 3. Prepare the pipeline settings.
    mode = "vertical" if line_mode == "vertical" else "horizontal"
    crossing = _parse_direction(mode, direction)
    herd = max(0, herd_size)

    # 4. Load the pipeline and run the video through it.
    pipeline = _get_pipeline(mode, line_x, line_y, crossing)
    result = pipeline.run(
        upload_path,
        output_video=out_path,
        herd_size=herd,
        session_name=job_id,
    )
    
    # 5. Save a text report to the results folder.
    report_path = RESULTS_DIR / f"{job_id}_report.txt"
    report_path.write_text("\n".join(result.report_lines()) + "\n", encoding="utf-8")

    # 6. Package the result into a JSON payload for the UI.
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
        "rejectionReason": result.rejection_reason,
        "accepted": result.accepted,
    }

    # 7. Render the result page with the count and the video.
    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context=_page_context(
            request,
            active_nav="history",
            result=result,
            video_url=video_url,
            checkin_id=job_id,
            recorded_at=recorded_at,
            checkin_json=json.dumps(checkin_payload),
        ),
    )