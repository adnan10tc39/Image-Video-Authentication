# app.py — Forensics API
# - /predict/image  -> direct image pipeline
# - /predict/video  -> job-based video pipeline (JobManager)
# - /video_jobs/{job_id} -> job status
import os
from pathlib import Path
from uuid import uuid4
from typing import Optional, Any

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    Request,
    HTTPException,
    Depends,
)
from pydantic import BaseModel, Field
from fastapi import UploadFile as FastAPIUploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from backends.image_pipeline import predict_image_pipeline
from backends.video_pipeline import predict_video_pipeline
from backends.job_queue import JobManager, JobRecord


# =========================================================
#   CONFIG & SECURITY
# =========================================================

from dotenv import load_dotenv
load_dotenv()
API_TOKEN = os.getenv("API_TOKEN")

# Security
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the API token from Authorization header."""
    if not API_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="API token not configured on server"
        )
    
    token = credentials.credentials
    if token != API_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API token"
        )
    return token


# =========================================================
#   FASTAPI APP
# =========================================================

app = FastAPI(title="Forensics API", version="6.0")


# =========================================================
#   IMAGE ENDPOINT (DIRECT, NO JOB SYSTEM)
# =========================================================

@app.post("/predict/image")
async def predict_image(
    file: UploadFile = File(...),
    type_hint: str = Form("Background + Splicing + Enhancement"),
    mask_alpha: float = Form(0.5),
    show_boxes: bool = Form(True),
    token: str = Depends(verify_token),  # protect image endpoint as well (optional)
):
    """
    Synchronous image prediction – same behavior as your original app.
    """
    return await predict_image_pipeline(
        file=file,
        type_hint=type_hint,
        mask_alpha=mask_alpha,
        show_boxes=show_boxes,
    )


# =========================================================
#   JOB INFRA FOR VIDEO (USING backends.job_queue)
# =========================================================

# Where to store uploaded video files for async processing
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "video_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Job DB & manager (separate DB just for video jobs)
JOB_DB_PATH = BASE_DIR / "video_jobs.db"
JOB_WORKER_COUNT = int(os.getenv("VIDEO_JOB_WORKERS", "1"))

job_manager = JobManager(db_path=JOB_DB_PATH, worker_count=JOB_WORKER_COUNT)


class VideoJobSubmissionResponse(BaseModel):
    job_id: str
    status_url: Optional[str] = Field(
        default=None,
        description="Endpoint to poll for job status.",
    )


class JobStatusResponse(BaseModel):
    job_id: str
    type: str
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str


# -------- PROCESSOR: runs inside JobManager workers --------
async def video_processor(payload: dict) -> dict:
    """
    This is the processor that JobManager executes in the background.
    It reconstructs an UploadFile-like object from disk and calls
    your existing `predict_video_pipeline`.
    """


    video_path = Path(payload["video_path"])
    type_hint = payload["type_hint"]
    frame_stride = int(payload["frame_stride"])
    max_frames = int(payload["max_frames"])
    batch_size = int(payload["batch_size"])
    mask_alpha = float(payload["mask_alpha"])
    show_boxes = bool(payload["show_boxes"])
    regions_max_per_frame = int(payload["regions_max_per_frame"])
    sample_fps = payload.get("sample_fps", None)
    uniform = bool(payload["uniform"])

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Reopen the file as if it came from a request
    file_obj = video_path.open("rb")
    upload = FastAPIUploadFile(filename=video_path.name, file=file_obj)
    # IMPORTANT:
    #   - Don't pass content_type in __init__
    #   - Don't assign upload.content_type (no setter in your version)

    try:
        result = await predict_video_pipeline(
            file=upload,
            type_hint=type_hint,
            frame_stride=frame_stride,
            max_frames=max_frames,
            batch_size=batch_size,
            mask_alpha=mask_alpha,
            show_boxes=show_boxes,
            regions_max_per_frame=regions_max_per_frame,
            sample_fps=sample_fps,
            uniform=uniform,
        )
    finally:
        file_obj.close()
        # Optional: delete file after processing
        try:
            video_path.unlink(missing_ok=True)
        except Exception:
            pass

    # This dict will be stored in jobs.result (JSON) and returned by /video_jobs/{job_id}
    return result


# Register processor once
job_manager.register_processor("video_inference", video_processor)


# =========================================================
#   LIFECYCLE: START / STOP JOB MANAGER
# =========================================================

@app.on_event("startup")
async def _startup_job_manager() -> None:
    await job_manager.start()


@app.on_event("shutdown")
async def _shutdown_job_manager() -> None:
    await job_manager.stop()


# =========================================================
#   VIDEO ENDPOINT (ENQUEUE JOB LIKE /research)
# =========================================================

@app.post("/predict/video", response_model=VideoJobSubmissionResponse)
async def predict_video(
    request: Request,
    file: UploadFile = File(...),
    type_hint: str = Form("image /action/ background"),
    frame_stride: int = Form(15),
    max_frames: int = Form(60),
    batch_size: int = Form(32),
    mask_alpha: float = Form(0.5),
    show_boxes: bool = Form(True),
    regions_max_per_frame: int = Form(8),
    sample_fps: Optional[float] = Form(None),
    uniform: bool = Form(True),
    token: str = Depends(verify_token),
):
    """
    Job-based video prediction (similar to /research in helping API):
      1. Save uploaded video to disk
      2. Enqueue a background job in JobManager
      3. Return job_id + status_url
    """

    # 1) Save uploaded video to disk
    suffix = Path(file.filename).suffix or ".mp4"
    video_path = UPLOAD_DIR / f"{uuid4().hex}{suffix}"

    with video_path.open("wb") as f:
        f.write(await file.read())

    # 2) Enqueue job with all needed parameters
    payload = {
        "video_path": str(video_path),
        "type_hint": type_hint,
        "frame_stride": frame_stride,
        "max_frames": max_frames,
        "batch_size": batch_size,
        "mask_alpha": mask_alpha,
        "show_boxes": show_boxes,
        "regions_max_per_frame": regions_max_per_frame,
        "sample_fps": sample_fps,
        "uniform": uniform,
    }

    try:
        job_id = await job_manager.enqueue_job("video_inference", payload)
    except Exception as e:
        # Cleanup file if enqueuing fails
        try:
            video_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(e))

    # 3) Return job id + URL to poll
    status_url = str(request.url_for("video_job_status", job_id=job_id))
    return VideoJobSubmissionResponse(
        job_id=job_id,
        status_url=status_url,
    )


# =========================================================
#   JOB STATUS ENDPOINT FOR VIDEO (TOKEN-PROTECTED)
# =========================================================

@app.get(
    "/video_jobs/{job_id}",
    response_model=JobStatusResponse,
    name="video_job_status",
)
async def video_job_status(
    job_id: str,
    token: str = Depends(verify_token),
):
    """
    Poll video job status (similar to /jobs/{job_id} in helping API).
    """
    record: Optional[JobRecord] = await job_manager.fetch_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found.")

    # Only expose result when completed (same pattern as helping API)
    result = record.result if record.status == "completed" else None

    return JobStatusResponse(
        job_id=record.id,
        type=record.type,
        status=record.status,
        result=result,
        error=record.error,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )
