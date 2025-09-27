# backends/utils.py — shared settings, models, ELA+YOLO-bg helpers, and utilities
import io, os, base64, cv2, torch, numpy as np, datetime, tempfile, logging
from pathlib import Path
from typing import List, Optional, Dict, Union, Tuple
from collections import Counter
from PIL import Image, ImageOps, ImageChops
from tempfile import NamedTemporaryFile

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from ultralytics import YOLO

log = logging.getLogger("forensics-utils")
logging.basicConfig(level=logging.INFO)

# -------- Settings
def _models_dir_default() -> Path:
    return (Path(__file__).resolve().parent.parent / "models")

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    MODELS_DIR: Path = Field(default_factory=_models_dir_default)

    SEG_WEIGHTS: Optional[Path] = None   # YOLO seg/det model for faces (and labels like fake/real if present)
    CLS_WEIGHTS: Optional[Path] = None   # YOLO cls model for ai-vs-real
    FORG_WEIGHTS: Optional[Path] = None  # YOLO seg/det for forgery types (splicing, etc.)
    BG_WEIGHTS: Optional[Path] = None    # 4th model: YOLO classifier (.pt) for background tampering

    SEG_WEIGHTS_NAME: Optional[str] = None
    CLS_WEIGHTS_NAME: Optional[str] = None
    FORG_WEIGHTS_NAME: Optional[str] = None
    BG_WEIGHTS_NAME: Optional[str] = "background_forgery.pt"  # updated to YOLO .pt

    DEVICE: str = "auto"

    SEG_IMGSZ: int = 640
    SEG_CONF: float = 0.25
    SEG_IOU: float = 0.45

    CLS_IMGSZ: int = 384
    CLS_TOPK: int = 2
    CLS_MIN_PROB: float = 0.15

    FORG_IMGSZ: int = 960
    FORG_CONF: float = 0.25
    FORG_IOU: float = 0.45

    BG_IMGSZ: int = 224  # YOLO cls size for background model

    FRAME_STRIDE: int = 15
    MAX_FRAMES: int = 60
    BATCH_SIZE: int = 32

    DEFAULT_MASK_ALPHA: float = 0.5
    DEFAULT_SHOW_BOXES: bool = True
    REGIONS_MAX_PER_FRAME: int = 8

    # Class name mapping for background model (align to your new infer_one_yolo.py)
    BG_NAME_MAP: Dict[str, str] = Field(default_factory=lambda: {
        "Original": "Original Background",
        "Forged":   "Forged Background",
    })

    def model_post_init(self, __context) -> None:
        self.MODELS_DIR = self.MODELS_DIR.resolve()
        def _rel(p): return (self.MODELS_DIR / str(p)).resolve()
        for attr in ("SEG_WEIGHTS","CLS_WEIGHTS","FORG_WEIGHTS","BG_WEIGHTS"):
            p = getattr(self, attr)
            if p is not None and not Path(p).is_absolute():
                setattr(self, attr, _rel(p))
        if self.SEG_WEIGHTS is None and self.SEG_WEIGHTS_NAME:
            self.SEG_WEIGHTS = _rel(self.SEG_WEIGHTS_NAME)
        if self.CLS_WEIGHTS is None and self.CLS_WEIGHTS_NAME:
            self.CLS_WEIGHTS = _rel(self.CLS_WEIGHTS_NAME)
        if self.FORG_WEIGHTS is None and self.FORG_WEIGHTS_NAME:
            self.FORG_WEIGHTS = _rel(self.FORG_WEIGHTS_NAME)
        if self.BG_WEIGHTS is None and self.BG_WEIGHTS_NAME:
            self.BG_WEIGHTS = _rel(self.BG_WEIGHTS_NAME)

settings = Settings()

def resolve_device(choice: str):
    if choice and choice.lower().startswith("auto"):
        return 0 if torch.cuda.is_available() else "cpu"
    return choice or ("cpu")

DEVICE = resolve_device(settings.DEVICE)

# -------- Global model holders (lazy)
SEG_MODEL = None
CLS_MODEL = None
FORG_MODEL = None
BG_MODEL = None  # YOLO classification for background tampering

def _resolve(weights: Optional[Path], name: Optional[str]) -> Path:
    if weights: return Path(weights)
    if name:
        p = settings.MODELS_DIR / name
        if p.is_file(): return p
    hint = " | ".join([p.name for p in settings.MODELS_DIR.glob('*') if p.is_file()])
    raise RuntimeError(f"Model weights not set. MODELS_DIR hint: {hint}")

def get_models():
    global SEG_MODEL, CLS_MODEL, FORG_MODEL, BG_MODEL
    if SEG_MODEL is None:
        SEG_MODEL  = YOLO(str(_resolve(settings.SEG_WEIGHTS,  settings.SEG_WEIGHTS_NAME)))
    if CLS_MODEL is None:
        CLS_MODEL  = YOLO(str(_resolve(settings.CLS_WEIGHTS,  settings.CLS_WEIGHTS_NAME)))
    if FORG_MODEL is None:
        FORG_MODEL = YOLO(str(_resolve(settings.FORG_WEIGHTS, settings.FORG_WEIGHTS_NAME)))
    if BG_MODEL is None:
        BG_MODEL   = YOLO(str(_resolve(settings.BG_WEIGHTS,   settings.BG_WEIGHTS_NAME)))  # YOLO cls
    return SEG_MODEL, CLS_MODEL, FORG_MODEL, BG_MODEL

# -------- ELA helpers used by BG YOLO model
def ela_image(path_or_image, quality=90, scale=10) -> Image.Image:
    """Return PIL 'L' (grayscale) ELA image with boosted differences."""
    if isinstance(path_or_image, (str, bytes, Path)):
        img = Image.open(path_or_image)
        img = ImageOps.exif_transpose(img).convert("RGB")
    else:
        img = ImageOps.exif_transpose(path_or_image)
        if img.mode != "RGB":
            img = img.convert("RGB")
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    jpeg = Image.open(buf).convert("RGB")
    diff = ImageChops.difference(img, jpeg).convert("L")
    _minv, maxv = diff.getextrema() or (0, 0)
    if maxv == 0:
        return diff
    factor = (255.0 / maxv) * scale
    return diff.point(lambda p: min(255, int(p * factor)))

def bg_prepare_pil_rgb_for_yolo(pil_or_path, target_size=(224, 224)) -> Image.Image:
    """Create ELA ('L') → to RGB → resize for YOLO cls."""
    ela = ela_image(pil_or_path)
    ela = ela.resize(target_size, Image.Resampling.LANCZOS)
    rgb = ela.convert("RGB")
    return rgb

def bg_predict_image_yolo(model, pil_or_path) -> Tuple[str, float, Dict[str, float]]:
    """Return (pretty_label, top1_conf, probs_dict) using YOLO cls + ELA preprocessing."""
    img = bg_prepare_pil_rgb_for_yolo(pil_or_path, (settings.BG_IMGSZ, settings.BG_IMGSZ))
    results = model.predict(source=[img], imgsz=settings.BG_IMGSZ, verbose=False)
    r = results[0]
    top1_idx   = int(r.probs.top1)
    top1_conf  = float(r.probs.top1conf)
    names      = r.names  # dict {idx: "ClassName"}
    top1_name  = names[top1_idx]
    pretty_lbl = settings.BG_NAME_MAP.get(top1_name, top1_name)

    # full prob dict
    probs = r.probs.data.detach().cpu().numpy().tolist()
    prob_dict = {}
    for i, p in enumerate(probs):
        raw_name = names[i]
        prob_dict[settings.BG_NAME_MAP.get(raw_name, raw_name)] = float(p)

    return pretty_lbl, top1_conf, prob_dict

def bg_predict_batch_yolo(model, pil_list: List[Image.Image]) -> np.ndarray:
    """Return N-length np.ndarray of 'Forged Background' probabilities."""
    if not pil_list:
        return np.array([], dtype=float)
    imgs = [bg_prepare_pil_rgb_for_yolo(pil, (settings.BG_IMGSZ, settings.BG_IMGSZ)) for pil in pil_list]
    results = model.predict(source=imgs, imgsz=settings.BG_IMGSZ, verbose=False)
    out = []
    # Find the index/name that maps to 'Forged Background'
    forged_aliases = {"Forged Background", "Forged"}
    for r in results:
        names = r.names
        probs = r.probs.data.detach().cpu().numpy().reshape(-1)
        # map class index -> pretty name, pick prob for "Forged Background"
        forged_prob = 0.0
        for i, p in enumerate(probs):
            pretty = settings.BG_NAME_MAP.get(names[i], names[i])
            if pretty in forged_aliases:
                forged_prob = float(p); break
        out.append(forged_prob)
    return np.array(out, dtype=float)

# -------- General helpers
FORGERY_CANON = ("copy-move", "copy_move", "splicing", "enhancement", "removal")

def encode_png_rgb(arr_rgb: np.ndarray) -> str:
    bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Failed to encode overlay.")
    return base64.b64encode(buf.tobytes()).decode("utf-8")

def save_overlay_png(rgb: np.ndarray, prefix: str) -> str:
    os.makedirs("overlays", exist_ok=True)
    fname = f"{prefix}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
    path = os.path.join("overlays", fname)
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return os.path.basename(path)

def as_numpy_scores(probs_obj):
    if probs_obj is None: return None
    scores = getattr(probs_obj, "data", probs_obj)
    if torch.is_tensor(scores): return scores.detach().float().cpu().numpy().reshape(-1)
    if isinstance(scores, np.ndarray): return scores.reshape(-1)
    if isinstance(scores, (list, tuple)): return np.array(scores, dtype=np.float32).reshape(-1)
    return None

def topk_from_cls_result(result, k=2, conf_thresh=0.0):
    scores = as_numpy_scores(getattr(result, "probs", None))
    names = getattr(result, "names", None)
    if scores is None or names is None: return []
    idxs = scores.argsort()[::-1][:int(k)]
    out = []
    for i in idxs:
        p = float(scores[i])
        if p >= conf_thresh:
            label = names[i] if isinstance(names, dict) else str(i)
            out.append({"label": str(label), "prob": p})
    return out

def annotate_to_rgb(result, mask_alpha=0.5, boxes=True):
    import inspect
    try:
        sig = inspect.signature(result.plot)
        params = sig.parameters.keys()
        kwargs = {"labels": True}
        if "boxes" in params: kwargs["boxes"] = boxes
        if "mask_alpha" in params: kwargs["mask_alpha"] = mask_alpha
        bgr = result.plot(**kwargs)
    except Exception:
        bgr = result.plot()
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def seg_counts(result) -> Dict[str, int]:
    if getattr(result, "boxes", None) is None or result.boxes.cls is None:
        return {}
    names = result.names
    cls_ids = result.boxes.cls.int().cpu().tolist()
    return dict(Counter([names[int(c)] for c in cls_ids]))

def regions_from_result(result, limit: Optional[int] = None) -> List[Dict]:
    regions: List[Dict] = []
    names = getattr(result, "names", {})
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    polys = []
    if masks is not None and getattr(masks, "xy", None) is not None:
        for poly in masks.xy:
            try:
                polys.append([[float(x), float(y)] for x, y in poly])
            except Exception:
                polys.append([])
    if boxes is None or boxes.cls is None:
        return regions
    cls_ids = boxes.cls.int().cpu().tolist()
    confs = boxes.conf.detach().cpu().tolist() if getattr(boxes, "conf", None) is not None else [None]*len(cls_ids)
    xyxy = boxes.xyxy.detach().cpu().tolist() if getattr(boxes, "xyxy", None) is not None else []
    for i, cid in enumerate(cls_ids):
        label = names[int(cid)] if isinstance(names, dict) else str(cid)
        conf = float(confs[i]) if i < len(confs) and confs[i] is not None else None
        bb = xyxy[i] if i < len(xyxy) else None
        poly = polys[i] if i < len(polys) else None
        regions.append({"label": label, "conf": conf, "bbox_xyxy": bb, "polygon": poly})
        if limit is not None and len(regions) >= limit:
            break
    return regions

def seg_face_counts_from_regions(regions: List[Dict]) -> Dict[str, int]:
    faces = [r for r in regions if "face" in str(r.get("label","")).lower()]
    fake = [r for r in regions if "fake" in str(r.get("label","")).lower()]
    real = [r for r in regions if "real" in str(r.get("label","")).lower()]
    return {
        "faces_detected": len(faces) if faces else (len(regions) if regions else 0),
        "fake_faces": len(fake),
        "real_faces": len(real)
    }

def forgery_primary_label(regions: List[Dict], counts: Dict[str,int]) -> Dict[str, Union[str,float,int]]:
    cand = None
    for r in regions:
        lbl = str(r.get("label","")).lower().replace("-", "_")
        if any(lbl.startswith(k) or k in lbl for k in ("copy_move","copy-move","splicing","enhancement","removal")):
            if cand is None or (r.get("conf") or 0) > (cand.get("conf") or 0):
                cand = r
    if cand:
        return {"label": cand["label"], "conf": float(cand.get("conf") or 0), "regions": sum(counts.values()) or 1}
    if counts:
        valid = [(k, v) for k, v in counts.items()
                 if any(k.lower().replace("-", "_").startswith(c) or c in k.lower().replace("-", "_")
                        for c in ("copy_move","copy-move","splicing","enhancement","removal"))]
        if valid:
            best = max(valid, key=lambda kv: kv[1])
            return {"label": best[0], "conf": 0.0, "regions": int(best[1])}
    return {"label": "None", "conf": 0.0, "regions": 0}

def op_cls_ai(topk: List[dict]) -> str:
    if not topk: return "No classification signal."
    t = topk[0]; return f"Top-1: {str(t['label']).strip()} ({t['prob']*100:.1f}%)."

def op_face(regions: List[Dict]) -> str:
    cnts = seg_face_counts_from_regions(regions)
    if cnts["faces_detected"] == 0:
        return "No faces detected."
    parts = []
    if cnts["fake_faces"]: parts.append(f"{cnts['fake_faces']} fake")
    if cnts["real_faces"]: parts.append(f"{cnts['real_faces']} real")
    return " | ".join(parts) if parts else f"{cnts['faces_detected']} face(s) detected."

def op_forgery(primary: Dict[str,Union[str,float,int]], counts: Dict[str,int]) -> str:
    if primary["label"] == "None" and not counts:
        return "No forgery artifact detected."
    if primary["label"] != "None" and primary["conf"] > 0:
        return f"{primary['label']} ({primary['conf']:.2f}) detected."
    nonzero = [k for k, v in counts.items() if v > 0]
    return (", ".join(nonzero) + " detected.") if nonzero else "Forgery artifacts detected."

def overall_analysis(is_ai: bool, bg_forged: bool, forg_present: bool) -> str:
    if is_ai and (bg_forged or forg_present):
        return "The uploaded image is AI-generated with background tampering and additional edits."
    if is_ai and not (bg_forged or forg_present):
        return "The uploaded image is AI-generated with no localized edits detected."
    if (bg_forged or forg_present) and not is_ai:
        return "The uploaded image is natural with background tampering / localized edits."
    return "The uploaded image appears natural with no detected edits."

def burn_caption(rgb: np.ndarray, lines: List[str]) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    text = " | ".join(lines)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(bgr, (10-6, 30-24), (10+tw+8, 30+8), (0,0,0), -1)
    cv2.putText(bgr, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def format_date_from_path(path: str) -> str:
    try: ts = os.path.getctime(path)
    except Exception: ts = os.path.getmtime(path)
    return datetime.datetime.fromtimestamp(ts).strftime("%d/%m/%Y")

def sample_video_frames(file_bytes: bytes, frame_stride: int, max_frames: int):
    with NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(file_bytes); tmp_path = tmp.name
    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        os.remove(tmp_path); raise RuntimeError("Could not open video.")
    frames_rgb, idx = [], 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        if idx % int(frame_stride) == 0 and fr is not None:
            frames_rgb.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            if len(frames_rgb) >= int(max_frames): break
        idx += 1
    cap.release(); os.remove(tmp_path)
    if not frames_rgb: raise RuntimeError("No frames sampled.")
    return frames_rgb
