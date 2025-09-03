# Forensics API — Quick Start (Simple README)

A minimal guide to run the FastAPI that does **Segmentation** + **Classification** using Ultralytics YOLO.  
_No extra fluff — just what you need._

---

## 1) Setup

```bash
# create & activate a virtual env (recommended)
conda create -p myenv python==3.11.5 -y
conda activate myenv

# install dependencies
pip install -r requirements.txt
```

---

## 2) Project structure

```
your_project/
├─ api.py
├─ requirements.txt
├─ models/
│  ├─ tempering_seg_best.pt       # segmentation model
│  └─ real_vs_ai_best.pt          # classification model
└─ .env                           # optional (see below)
```

- Put your weights inside **models/**.  
- The API **auto-detects** correct files by actual task:  
  - `segment` → segmentation model  
  - `classify` → classification model

---

## 3) Optional: .env (override defaults)

Create a `.env` file if you want to customize:

```
# Change models folder (optional)
MODELS_DIR=/abs/path/to/models

# Pin filenames (optional, relative to MODELS_DIR)
SEG_WEIGHTS_NAME=tempering_seg_best.pt
CLS_WEIGHTS_NAME=real_vs_ai_best.pt

# Or hard override with absolute/relative paths
# SEG_WEIGHTS=/abs/path/to/seg.pt
# CLS_WEIGHTS=/abs/path/to/cls.pt
```

> If the wrong file is picked, the API will raise a clear error at startup (e.g. “expected classify but got detect/segment”).

---

## 4) Run the server

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
# Expect: seg_task="segment", cls_task="classify"
```

---

## 5) Endpoints (Postman / cURL)

All image/video endpoints expect **form-data** with key `file` (type: File).  
Default parameters are sensible; you can override via form fields.

### A) Image — Classification
```
POST /predict/image/cls
form-data:
  file: <image.jpg/png>
  topk: 2            (optional)
  min_prob: 0.15     (optional)
  imgsz: 384         (optional)
```
**Response:**
```json
{ "topk": [ {"label":"Real","prob":0.93}, {"label":"Fake","prob":0.07} ] }
```

### B) Image — Segmentation
```
POST /predict/image/seg
form-data:
  file: <image.jpg/png>
  conf: 0.25         (optional)
  iou: 0.45          (optional)
  imgsz: 512         (optional)
  mask_alpha: 0.5    (optional)
  show_boxes: true   (optional)
  return_overlay: true (optional)
```
**Response:**
```json
{
  "counts": {"Fake": 3},
  "overlay_png_base64": "<base64-png>"
}
```

### C) Image — Compare (Seg + Cls)
```
POST /predict/image/compare
form-data:
  file: <image.jpg/png>
  seg_conf: 0.25, seg_iou: 0.45, seg_imgsz: 512
  cls_imgsz: 384
  topk: 2, min_prob: 0.15
  return_overlay: true, mask_alpha: 0.5, show_boxes: true
```
**Response:**
```json
{
  "seg_counts": {"Fake": 2},
  "cls_topk": [{"label":"AI","prob":0.81},{"label":"Real","prob":0.19}],
  "verdict": "AI-generated and also edited (both signals present).",
  "overlay_png_base64": "<base64-png>"
}
```

### D) Video — Segmentation
```
POST /predict/video/seg
form-data:
  file: <video.mp4>
  frame_stride: 15, max_frames: 60, batch_size: 32
  conf: 0.25, iou: 0.45, imgsz: 512
  return_overlays: true, mask_alpha: 0.5, show_boxes: true
```

### E) Video — Classification
```
POST /predict/video/cls
form-data:
  file: <video.mp4>
  frame_stride: 15, max_frames: 60, batch_size: 32
  imgsz: 384
  topk: 2, min_prob: 0.15
  return_overlays: true
```

### F) Video — Compare (Seg + Cls)
```
POST /predict/video/compare
form-data:
  file: <video.mp4>
  frame_stride: 15, max_frames: 60, batch_size: 32
  seg_imgsz: 512, seg_conf: 0.25, seg_iou: 0.45
  cls_imgsz: 384
  topk: 2, min_prob: 0.15
  return_overlays: true
```

---

## 6) Quick cURL examples

```bash
# Image classification
curl -X POST http://localhost:8000/predict/image/cls   -F "file=@/path/to/image.jpg"   -F "topk=2" -F "min_prob=0.15"

# Image segmentation
curl -X POST http://localhost:8000/predict/image/seg   -F "file=@/path/to/image.jpg"   -F "conf=0.25" -F "iou=0.45" -F "imgsz=512"
```

---

## 7) Troubleshooting (short)

- **/health shows cls_task != classify**  
  Your classification weights are not a classifier. Pin correct file via `.env` (e.g., `CLS_WEIGHTS_NAME=real_vs_ai_best.pt`).

- **Empty `topk` in /predict/image/cls**  
  Decrease `min_prob`, or confirm the model really is a **classification** model.

- **CUDA not used**  
  Set `DEVICE=auto` (default) or `cuda:0` in `.env`, and ensure CUDA is available.
# Image-Video-Authentication
