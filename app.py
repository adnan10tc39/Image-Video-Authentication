# app.py — One FastAPI app with two endpoints; logic lives in backends/
from fastapi import FastAPI, UploadFile, File, Form
from backends.image_pipeline import predict_image_pipeline
from backends.video_pipeline import predict_video_pipeline

app = FastAPI(title="Forensics API", version="6.0")

@app.on_event("startup")
def _startup():
    # Models lazy-load on first request via the pipelines.
    pass

@app.post("/predict/image")
async def predict_image(
    file: UploadFile = File(...),
    type_hint: str = Form("Background + Splicing + Enhancement"),
    mask_alpha: float = Form(0.5),
    show_boxes: bool = Form(True),
):
    return await predict_image_pipeline(
        file=file, type_hint=type_hint,
        mask_alpha=mask_alpha, show_boxes=show_boxes
    )

@app.post("/predict/video")
async def predict_video(
    file: UploadFile = File(...),
    type_hint: str = Form("image /action/ background"),
    frame_stride: int = Form(15),
    max_frames: int   = Form(60),
    batch_size: int   = Form(32),
    mask_alpha: float = Form(0.5),
    show_boxes: bool  = Form(True),
    regions_max_per_frame: int = Form(8),
    sample_fps: float = Form(None),           # NEW
    uniform: bool = Form(True),               # NEW
):
    return await predict_video_pipeline(
        file=file, type_hint=type_hint,
        frame_stride=frame_stride, max_frames=max_frames, batch_size=batch_size,
        mask_alpha=mask_alpha, show_boxes=show_boxes,
        regions_max_per_frame=regions_max_per_frame,
        sample_fps=sample_fps, uniform=uniform,      # NEW
    )

