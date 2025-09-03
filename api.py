# api.py — FastAPI for Segmentation + Classification (robust model auto-pick)
import io
import os
import base64
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Optional, Dict, Tuple, Union
from collections import Counter
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from ultralytics import YOLO

# =========================
# Settings (auto-discover weights from ./models)
# =========================
def _models_dir_default() -> Path:
    # Resolves to "<this_file_dir>/models"
    return (Path(__file__).resolve().parent / "models")

class Settings(BaseSettings):
    """
    Auto-resolves model weights from a local 'models/' directory.
    Priority (highest -> lowest):
      1) SEG_WEIGHTS / CLS_WEIGHTS (absolute or relative to MODELS_DIR)
      2) SEG_WEIGHTS_NAME / CLS_WEIGHTS_NAME (relative filenames inside MODELS_DIR)
      3) Auto-pick by inspecting each file's YOLO .task (segment/classify)
      4) Final fallback: newest file in MODELS_DIR
    """
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    MODELS_DIR: Path = Field(default_factory=_models_dir_default)

    SEG_WEIGHTS: Optional[Path] = None
    CLS_WEIGHTS: Optional[Path] = None

    SEG_WEIGHTS_NAME: Optional[str] = None   # e.g., tempering_seg_best.pt
    CLS_WEIGHTS_NAME: Optional[str] = None   # e.g., real_vs_ai_best.pt

    DEVICE: str = "auto"  # "auto", "cpu", or "cuda:0"
    SEG_IMGSZ: int = 512
    CLS_IMGSZ: int = 384
    SEG_CONF: float = 0.25
    SEG_IOU: float = 0.45
    FRAME_STRIDE: int = 15
    MAX_FRAMES: int = 60
    BATCH_SIZE: int = 32
    RETURN_OVERLAY: bool = True

    _allowed_exts: List[str] = [".pt", ".onnx", ".engine", ".tflite", ".safetensors"]

    def model_post_init(self, __context) -> None:
        self.MODELS_DIR = self.MODELS_DIR.resolve()
        # Normalize to absolute if user provided paths
        if self.SEG_WEIGHTS is not None:
            self.SEG_WEIGHTS = self._normalize_to_path(self.SEG_WEIGHTS)
        if self.CLS_WEIGHTS is not None:
            self.CLS_WEIGHTS = self._normalize_to_path(self.CLS_WEIGHTS)
        # If names were given, map to absolute paths
        if self.SEG_WEIGHTS is None and self.SEG_WEIGHTS_NAME:
            self.SEG_WEIGHTS = (self.MODELS_DIR / self.SEG_WEIGHTS_NAME).resolve()
        if self.CLS_WEIGHTS is None and self.CLS_WEIGHTS_NAME:
            self.CLS_WEIGHTS = (self.MODELS_DIR / self.CLS_WEIGHTS_NAME).resolve()

    def _normalize_to_path(self, p: Union[str, Path]) -> Path:
        p = Path(p)
        if not p.is_absolute():
            p = self.MODELS_DIR / p
        return p.resolve()

settings = Settings()

def _resolve_device(choice: str):
    if choice.lower().startswith("auto"):
        return 0 if torch.cuda.is_available() else "cpu"
    return choice

DEVICE = _resolve_device(settings.DEVICE)

# =========================
# Helpers (shared)
# =========================
def _ensure_file(p: Union[str, Path], label: str):
    p = Path(p)
    if not p.is_file():
        raise RuntimeError(f"{label} weights not found: {p}")

def _pil_from_bytes(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")

def _encode_png_rgb(arr_rgb: np.ndarray) -> str:
    """Return base64 PNG data from an RGB numpy image."""
    bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Failed to encode overlay.")
    return base64.b64encode(buf.tobytes()).decode("utf-8")

def _as_numpy_scores(probs_obj) -> Optional[np.ndarray]:
    """
    Ultralytics classification 'probs' can be:
      - Probs object with .data (torch.Tensor)
      - torch.Tensor directly
      - None for non-classifier models
    """
    if probs_obj is None:
        return None
    scores = getattr(probs_obj, "data", probs_obj)
    try:
        if torch.is_tensor(scores):
            return scores.detach().float().cpu().numpy().reshape(-1)
        if isinstance(scores, np.ndarray):
            return scores.reshape(-1)
        if isinstance(scores, (list, tuple)):
            return np.array(scores, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    return None

def _topk_from_cls_result(result, k=2, conf_thresh=0.0):
    scores = _as_numpy_scores(getattr(result, "probs", None))
    names = getattr(result, "names", None)
    if scores is None or names is None:
        # Not a classification result or no probabilities available
        return []
    idxs = scores.argsort()[::-1][:int(k)]
    out = []
    for i in idxs:
        p = float(scores[i])
        if p >= conf_thresh:
            label = names[i] if isinstance(names, dict) else str(i)
            out.append({"label": str(label), "prob": p})
    return out

def _annotate_to_rgb(result, mask_alpha=0.5, boxes=True):
    """Robust overlay across Ultralytics versions (mask_alpha may not exist)."""
    import inspect
    try:
        sig = inspect.signature(result.plot)
        params = sig.parameters.keys()
        kwargs = {"labels": True}
        if "boxes" in params:
            kwargs["boxes"] = boxes
        if "mask_alpha" in params:
            kwargs["mask_alpha"] = mask_alpha
            bgr = result.plot(**kwargs)
        else:
            bgr = result.plot(**kwargs)
    except Exception:
        bgr = result.plot()
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def _seg_counts(result) -> Dict[str, int]:
    if getattr(result, "boxes", None) is None or result.boxes.cls is None:
        return {}
    names = result.names
    cls_ids = result.boxes.cls.int().cpu().tolist()
    return dict(Counter([names[int(c)] for c in cls_ids]))

def _sample_video_frames(file_bytes: bytes, frame_stride: int, max_frames: int):
    with NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(file_bytes); tmp_path = tmp.name
    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        os.remove(tmp_path)
        raise RuntimeError("Could not open video. Check codecs.")
    frames_rgb, idx = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if idx % int(frame_stride) == 0 and fr is not None:
            frames_rgb.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            if len(frames_rgb) >= int(max_frames):
                break
        idx += 1
    cap.release(); os.remove(tmp_path)
    if not frames_rgb:
        raise RuntimeError("No frames sampled. Adjust stride/video.")
    return frames_rgb

# Verdict helpers
TAMPER_KEYS = ("fake","tamper","forg","splice","edit","manip")
AI_KEYS = ("ai","fake","generated","synth")

def _is_tampered(counts: Dict[str,int]) -> bool:
    if not counts: return False
    keys = [k.lower() for k in counts.keys()]
    return any(any(tk in k for tk in TAMPER_KEYS) for k in keys)

def _is_ai_label(lbl: str) -> bool:
    l = lbl.lower()
    return any(k in l for k in AI_KEYS)

def consistency_message(seg_counts: Dict[str,int], cls_items: List[dict]):
    tampered = _is_tampered(seg_counts)
    top1 = cls_items[0] if cls_items else {"label": "", "prob": 0.0}
    is_ai = _is_ai_label(top1["label"])
    if is_ai and not tampered:   return "Likely AI-generated original (no localized edits found)."
    if not is_ai and tampered:   return "Likely real image with localized edits."
    if is_ai and tampered:       return "AI-generated and also edited (both signals present)."
    return "Likely unmodified natural image (no edits detected; classified as Real)."

# =========================
# Model discovery by *actual* task
# =========================
def _yolo_task_of(weights_path: Path) -> Optional[str]:
    try:
        m = YOLO(str(weights_path))
        return getattr(m, "task", None)
    except Exception:
        return None

def _find_weight_by_task(models_dir: Path, allowed_exts: List[str], expected_task: str) -> Optional[Path]:
    if not models_dir.exists():
        return None
    candidates = [p for p in models_dir.glob("*") if p.suffix.lower() in allowed_exts and p.is_file()]
    # Try newest first to keep behavior predictable
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    for p in candidates:
        if _yolo_task_of(p) == expected_task:
            return p.resolve()
    return None

# =========================
# App & Model Load
# =========================
app = FastAPI(title="Forensics API", version="1.1")

SEG_MODEL = None
CLS_MODEL = None
SEG_TASK = None
CLS_TASK = None

@app.on_event("startup")
def load_models():
    global SEG_MODEL, CLS_MODEL, SEG_TASK, CLS_TASK

    # --- Resolve Segmentation ---
    seg_path = settings.SEG_WEIGHTS
    if seg_path is None:
        # Try by actual task
        seg_path = _find_weight_by_task(settings.MODELS_DIR, settings._allowed_exts, expected_task="segment")
        if seg_path is None:
            # Last resort: newest file (will likely fail at runtime if not a segmenter)
            candidates = [p for p in settings.MODELS_DIR.glob("*") if p.suffix.lower() in settings._allowed_exts and p.is_file()]
            if candidates:
                seg_path = max(candidates, key=lambda x: x.stat().st_mtime).resolve()
    _ensure_file(seg_path, "Segmentation")
    seg_model = YOLO(str(seg_path))
    SEG_TASK = getattr(seg_model, "task", None)
    if SEG_TASK != "segment":
        raise RuntimeError(f"Expected a segmentation model, but '{seg_path.name}' is task='{SEG_TASK}'. "
                           f"Pin the correct file via SEG_WEIGHTS or SEG_WEIGHTS_NAME.")

    # --- Resolve Classification ---
    cls_path = settings.CLS_WEIGHTS
    if cls_path is None:
        # Try by actual task
        cls_path = _find_weight_by_task(settings.MODELS_DIR, settings._allowed_exts, expected_task="classify")
        if cls_path is None:
            candidates = [p for p in settings.MODELS_DIR.glob("*") if p.suffix.lower() in settings._allowed_exts and p.is_file()]
            if candidates:
                cls_path = max(candidates, key=lambda x: x.stat().st_mtime).resolve()
    _ensure_file(cls_path, "Classification")
    cls_model = YOLO(str(cls_path))
    CLS_TASK = getattr(cls_model, "task", None)
    if CLS_TASK != "classify":
        # Helpful error so it's obvious why no probs/topk appear
        raise RuntimeError(f"Expected a classification model, but '{cls_path.name}' is task='{CLS_TASK}'. "
                           f"Pin the correct file via CLS_WEIGHTS or CLS_WEIGHTS_NAME.")

    # Assign globals
    SEG_MODEL, CLS_MODEL = seg_model, cls_model

    # --- light warmup (small synthetic input) ---
    dummy = np.zeros((settings.SEG_IMGSZ, settings.SEG_IMGSZ, 3), dtype=np.uint8)
    with torch.no_grad():
        SEG_MODEL.predict(dummy, imgsz=settings.SEG_IMGSZ, device=DEVICE, verbose=False,
                          conf=settings.SEG_CONF, iou=settings.SEG_IOU)
        # for classifier, imgsz can differ
        CLS_MODEL.predict(dummy, imgsz=settings.CLS_IMGSZ, device=DEVICE, verbose=False)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "models_dir": str(settings.MODELS_DIR),
        "seg_weights": str(getattr(SEG_MODEL, "ckpt_path", "n/a")),
        "cls_weights": str(getattr(CLS_MODEL, "ckpt_path", "n/a")),
        "seg_task": SEG_TASK,
        "cls_task": CLS_TASK,
    }

# =========================
# Image endpoints
# =========================
class ClsParams(BaseModel):
    topk: int = 2
    min_prob: float = 0.15

@app.post("/predict/image/seg")
async def predict_image_seg(
    file: UploadFile = File(...),
    conf: float = Form(settings.SEG_CONF),
    iou: float = Form(settings.SEG_IOU),
    imgsz: int = Form(settings.SEG_IMGSZ),
    mask_alpha: float = Form(0.5),
    show_boxes: bool = Form(True),
    return_overlay: bool = Form(True)
):
    try:
        img = _pil_from_bytes(await file.read())
        with torch.no_grad():
            results = SEG_MODEL.predict(img, imgsz=imgsz, device=DEVICE, verbose=False, conf=conf, iou=iou)
        r = results[0]
        counts = _seg_counts(r)
        payload = {"counts": counts}
        if return_overlay or settings.RETURN_OVERLAY:
            rgb = _annotate_to_rgb(r, mask_alpha=mask_alpha, boxes=show_boxes)
            payload["overlay_png_base64"] = _encode_png_rgb(rgb)
        return payload
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/image/cls")
async def predict_image_cls(
    file: UploadFile = File(...),
    topk: int = Form(2),
    min_prob: float = Form(0.15),
    imgsz: int = Form(settings.CLS_IMGSZ)
):
    try:
        img = _pil_from_bytes(await file.read())
        with torch.no_grad():
            results = CLS_MODEL.predict(img, imgsz=imgsz, device=DEVICE, verbose=False)
        r = results[0]
        items = _topk_from_cls_result(r, k=topk, conf_thresh=min_prob)
        # If no probs, surface a helpful hint
        if not items and getattr(r, "probs", None) is None:
            raise RuntimeError("No classification probabilities found. "
                               "Loaded model is not a classifier or is misconfigured.")
        return {"topk": items}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/image/compare")
async def predict_image_compare(
    file: UploadFile = File(...),
    seg_conf: float = Form(settings.SEG_CONF),
    seg_iou: float = Form(settings.SEG_IOU),
    seg_imgsz: int = Form(settings.SEG_IMGSZ),
    cls_imgsz: int = Form(settings.CLS_IMGSZ),
    topk: int = Form(2),
    min_prob: float = Form(0.15),
    return_overlay: bool = Form(True),
    mask_alpha: float = Form(0.5),
    show_boxes: bool = Form(True)
):
    try:
        img_bytes = await file.read()
        img = _pil_from_bytes(img_bytes)

        with torch.no_grad():
            seg_res = SEG_MODEL.predict(img, imgsz=seg_imgsz, device=DEVICE, verbose=False,
                                        conf=seg_conf, iou=seg_iou)[0]
            cls_res = CLS_MODEL.predict(img, imgsz=cls_imgsz, device=DEVICE, verbose=False)[0]

        counts = _seg_counts(seg_res)
        items = _topk_from_cls_result(cls_res, k=topk, conf_thresh=min_prob)
        if not items and getattr(cls_res, "probs", None) is None:
            raise RuntimeError("No classification probabilities found. "
                               "Loaded model is not a classifier or is misconfigured.")
        verdict = consistency_message(counts, items)

        out = {"seg_counts": counts, "cls_topk": items, "verdict": verdict}
        if return_overlay or settings.RETURN_OVERLAY:
            rgb = _annotate_to_rgb(seg_res, mask_alpha=mask_alpha, boxes=show_boxes)
            out["overlay_png_base64"] = _encode_png_rgb(rgb)
        return out
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# =========================
# Video endpoints
# =========================
@app.post("/predict/video/seg")
async def predict_video_seg(
    file: UploadFile = File(...),
    frame_stride: int = Form(settings.FRAME_STRIDE),
    max_frames: int = Form(settings.MAX_FRAMES),
    batch_size: int = Form(settings.BATCH_SIZE),
    conf: float = Form(settings.SEG_CONF),
    iou: float = Form(settings.SEG_IOU),
    imgsz: int = Form(settings.SEG_IMGSZ),
    return_overlays: bool = Form(True),
    mask_alpha: float = Form(0.5),
    show_boxes: bool = Form(True)
):
    try:
        bytes_ = await file.read()
        frames = _sample_video_frames(bytes_, frame_stride, max_frames)

        ann_frames = []
        total = Counter()
        with torch.no_grad():
            for s in range(0, len(frames), int(batch_size)):
                batch = frames[s:s+int(batch_size)]
                results = SEG_MODEL.predict(batch, imgsz=imgsz, device=DEVICE, verbose=False, conf=conf, iou=iou)
                for r in results:
                    total.update(_seg_counts(r))
                    if return_overlays or settings.RETURN_OVERLAY:
                        rgb = _annotate_to_rgb(r, mask_alpha=mask_alpha, boxes=show_boxes)
                        ann_frames.append(_encode_png_rgb(rgb))

        return {"counts": dict(total),
                "overlays_png_base64": ann_frames if (return_overlays or settings.RETURN_OVERLAY) else []}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/video/cls")
async def predict_video_cls(
    file: UploadFile = File(...),
    frame_stride: int = Form(settings.FRAME_STRIDE),
    max_frames: int = Form(settings.MAX_FRAMES),
    batch_size: int = Form(settings.BATCH_SIZE),
    imgsz: int = Form(settings.CLS_IMGSZ),
    topk: int = Form(2),
    min_prob: float = Form(0.15),
    return_overlays: bool = Form(True)
):
    try:
        bytes_ = await file.read()
        frames = _sample_video_frames(bytes_, frame_stride, max_frames)

        overlays = []
        counts = Counter()
        with torch.no_grad():
            for s in range(0, len(frames), int(batch_size)):
                batch = frames[s:s+int(batch_size)]
                results = CLS_MODEL.predict(batch, imgsz=imgsz, device=DEVICE, verbose=False)
                for rgb, r in zip(batch, results):
                    items = _topk_from_cls_result(r, k=topk, conf_thresh=min_prob)
                    if not items and getattr(r, "probs", None) is None:
                        raise RuntimeError("No classification probabilities found. "
                                           "Loaded model is not a classifier or is misconfigured.")
                    if items:
                        counts.update([items[0]["label"]])
                    if return_overlays or settings.RETURN_OVERLAY:
                        # draw top1 on frame
                        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
                        if items:
                            text = f"{items[0]['label']} ({items[0]['prob']*100:.1f}%)"
                            w = max(110, 14 * len(text))
                            cv2.rectangle(bgr, (10 - 6, 30 - 24), (10 + w, 30 + 8), (0, 0, 0), -1)
                            cv2.putText(bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
                        overlays.append(base64.b64encode(cv2.imencode(".png", bgr)[1]).decode("utf-8"))

        return {"top1_counts": dict(counts),
                "overlays_png_base64": overlays if (return_overlays or settings.RETURN_OVERLAY) else []}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/video/compare")
async def predict_video_compare(
    file: UploadFile = File(...),
    frame_stride: int = Form(settings.FRAME_STRIDE),
    max_frames: int = Form(settings.MAX_FRAMES),
    batch_size: int = Form(settings.BATCH_SIZE),
    seg_imgsz: int = Form(settings.SEG_IMGSZ),
    seg_conf: float = Form(settings.SEG_CONF),
    seg_iou: float = Form(settings.SEG_IOU),
    cls_imgsz: int = Form(settings.CLS_IMGSZ),
    topk: int = Form(2),
    min_prob: float = Form(0.15),
    return_overlays: bool = Form(True)
):
    try:
        bytes_ = await file.read()
        frames = _sample_video_frames(bytes_, frame_stride, max_frames)

        seg_counts = Counter()
        cls_counts = Counter()
        seg_overlays, cls_overlays = [], []

        # SEG
        with torch.no_grad():
            for s in range(0, len(frames), int(batch_size)):
                batch = frames[s:s+int(batch_size)]
                seg_results = SEG_MODEL.predict(batch, imgsz=seg_imgsz, device=DEVICE, verbose=False, conf=seg_conf, iou=seg_iou)
                for r in seg_results:
                    seg_counts.update(_seg_counts(r))
                    if return_overlays or settings.RETURN_OVERLAY:
                        rgb = _annotate_to_rgb(r, mask_alpha=0.5, boxes=True)
                        seg_overlays.append(_encode_png_rgb(rgb))

        # CLS
        with torch.no_grad():
            for s in range(0, len(frames), int(batch_size)):
                batch = frames[s:s+int(batch_size)]
                cls_results = CLS_MODEL.predict(batch, imgsz=cls_imgsz, device=DEVICE, verbose=False)
                for rgb, r in zip(batch, cls_results):
                    items = _topk_from_cls_result(r, k=topk, conf_thresh=min_prob)
                    if not items and getattr(r, "probs", None) is None:
                        raise RuntimeError("No classification probabilities found. "
                                           "Loaded model is not a classifier or is misconfigured.")
                    if items:
                        cls_counts.update([items[0]["label"]])
                        if return_overlays or settings.RETURN_OVERLAY:
                            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
                            text = f"{items[0]['label']} ({items[0]['prob']*100:.1f}%)"
                            w = max(110, 14 * len(text))
                            cv2.rectangle(bgr, (10 - 6, 30 - 24), (10 + w, 30 + 8), (0, 0, 0), -1)
                            cv2.putText(bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
                            cls_overlays.append(base64.b64encode(cv2.imencode(".png", bgr)[1]).decode("utf-8"))

        verdict = "Frames appear natural with no detected edits."
        tampered = _is_tampered(dict(seg_counts))
        ai_like = any(_is_ai_label(k) for k in cls_counts.keys())
        if ai_like and not tampered:
            verdict = "Video frames mostly AI-generated; no edits localized."
        elif not ai_like and tampered:
            verdict = "Real video with localized edits in frames."
        elif ai_like and tampered:
            verdict = "AI-generated content with additional edits."

        out = {
            "seg_counts": dict(seg_counts),
            "cls_top1_counts": dict(cls_counts),
            "verdict": verdict
        }
        if return_overlays or settings.RETURN_OVERLAY:
            out["seg_overlays_png_base64"] = seg_overlays
            out["cls_overlays_png_base64"] = cls_overlays
        return out
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
