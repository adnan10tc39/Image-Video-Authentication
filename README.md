# Forensics API — Quick Start

FastAPI service for **multimodal forensics** with 4 models:  
- **AI vs Real Classification**  
- **Background Tampering Detection** (YOLO cls + ELA)  
- **Face Segmentation & Authentication** (real vs fake faces)  
- **Forgery Types Segmentation** (splicing, copy-move, enhancement, removal)  

Supports both **images** and **videos**.

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
├─ app.py                     # main FastAPI entrypoint (2 endpoints only)
├─ backends/
│  ├─ image_pipeline.py       # image logic
│  ├─ video_pipeline.py       # video logic
│  └─ utils.py                # shared helpers (model loading, ELA, etc)
├─ models/
│  ├─ tempering_seg_best.pt        # segmentation model
│  ├─ real_vs_ai_best.pt           # AI-vs-Real classification
│  ├─ forgery_detector.pt          # forgery detection
│  └─ background_forgery.pt        # background tampering (YOLO cls on ELA)
├─ requirements.txt
└─ .env
```

---

## 3) .env (optional)

You can configure models and parameters here:

```env
# Models directory
MODELS_DIR=/abs/path/to/models

# Pin weights by filename (relative to MODELS_DIR)
SEG_WEIGHTS_NAME=tempering_seg_best.pt
CLS_WEIGHTS_NAME=real_vs_ai_best.pt
FORG_WEIGHTS_NAME=forgery_detector.pt
BG_WEIGHTS_NAME=background_forgery.pt

# Device: auto, cpu, or cuda:0
DEVICE=auto
```

---

## 4) Run the server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

---

## 5) Endpoints

### A) Image

```
POST /predict/image
form-data:
  file: <image.jpg/png>
  type_hint: "image or background"   # optional
```

**Response (example):**

```json
{
  "predictions": {
    "classification_ai_vs_real": {
      "summary": { "frames_total": 1, "frames_ai": 1, "frames_real": 0 },
      "opinion": "Top-1: AI (92.1%)."
    },
    "classification_background_tampering": {
      "summary": { "frames_total": 1, "frames_forged": 1, "frames_original": 0 },
      "opinion": "Background tampering detected (87%)."
    },
    "segmentation_face_authentication": {
      "counts": { "total_faces": 2, "fake_faces": 1, "real_faces": 1 },
      "opinion": "1 fake and 1 real face detected."
    },
    "segmentation_forgery_types": {
      "counts": { "splicing": 2, "enhancement": 1, "copy_move": 0, "removal": 0 },
      "opinion": "Splicing + Enhancement detected."
    }
  },
  "analysis": "AI-generated with background tampering and edits.",
  "card": {
    "Images": {
      "Photo": "92% AI",
      "Creator": "AI",
      "Manipulation": "Yes",
      "Type": "Background + Splicing + Enhancement",
      "Date created": "27/09/2025"
    }
  },
  "overlays": {
    "face_overlay": "face_analysis_overlay.png",
    "forgery_overlay": "forensic_overlay.png"
  }
}
```

---

### B) Video

```
POST /predict/video
form-data:
  file: <video.mp4>
  type_hint: "background / action"   # optional
  frame_stride: 15, max_frames: 60, batch_size: 32  # sampling config
```

**Response (example):**

```json
{
  "predictions": {
    "classification_ai_vs_real": {
      "summary": { "frames_total": 30, "frames_ai": 18, "frames_real": 12 },
      "opinion": "Top-1 across frames: AI (~82%)."
    },
    "classification_background_tampering": {
      "summary": { "frames_total": 30, "frames_forged": 11, "frames_original": 19 },
      "opinion": "Background tampering detected."
    },
    "segmentation_face_authentication": {
      "counts": { "total_faces": 45, "fake_faces": 14, "real_faces": 31 },
      "opinion": "Faces detected in 20/30 frames."
    },
    "segmentation_forgery_types": {
      "counts": { "splicing": 7, "copy_move": 2, "enhancement": 5, "removal": 0 },
      "opinion": "Splicing and Enhancement detected in several frames."
    }
  },
  "analysis": "AI-generated video with background tampering and edits.",
  "card": {
    "Video": {
      "Images": "82% AI",
      "Creator": "AI",
      "Manipulation": "Yes",
      "Type": "Background + Splicing + Enhancement",
      "Date created": "27/09/2025"
    }
  },
  "overlays": {}
}
```

---

## 6) Quick examples

```bash
# Image
curl -X POST http://localhost:8000/predict/image   -F "file=@/path/to/image.jpg"

# Video
curl -X POST http://localhost:8000/predict/video   -F "file=@/path/to/video.mp4"
```

---

## 7) Notes

- Video responses give **frame-level counts** (e.g. 18 AI vs 12 Real).  
- Overlays are only returned for **images** (PNG base64).  
- If wrong model file is provided, API raises a clear error at startup.  
- Device is auto-detected (`cuda` if available, else `cpu`).  

---

## 8) Demo JSON Samples

### Image Dummy Response
```json
{
  "predictions": {
    "classification_ai_vs_real": {
      "summary": { "frames_total": 1, "frames_ai": 1, "frames_real": 0 },
      "opinion": "Top-1: AI (92.1%)."
    },
    "classification_background_tampering": {
      "summary": { "frames_total": 1, "frames_forged": 1, "frames_original": 0 },
      "opinion": "Background tampering detected (87%)."
    },
    "segmentation_face_authentication": {
      "counts": { "total_faces": 2, "fake_faces": 1, "real_faces": 1 },
      "opinion": "1 fake and 1 real face detected."
    },
    "segmentation_forgery_types": {
      "counts": { "splicing": 2, "enhancement": 1, "copy_move": 0, "removal": 0 },
      "opinion": "Splicing + Enhancement detected."
    }
  },
  "analysis": "AI-generated with background tampering and edits.",
  "card": { "Images": { "Photo": "92% AI", "Creator": "AI", "Manipulation": "Yes" } },
  "overlays": { "face_overlay": "face_analysis_overlay.png" }
}
```

### Video Dummy Response
```json
{
  "predictions": {
    "classification_ai_vs_real": {
      "summary": { "frames_total": 30, "frames_ai": 18, "frames_real": 12 },
      "opinion": "Top-1 across frames: AI (~82%)."
    },
    "classification_background_tampering": {
      "summary": { "frames_total": 30, "frames_forged": 11, "frames_original": 19 },
      "opinion": "Background tampering detected."
    },
    "segmentation_face_authentication": {
      "counts": { "total_faces": 45, "fake_faces": 14, "real_faces": 31 },
      "opinion": "Faces detected in 20/30 frames."
    },
    "segmentation_forgery_types": {
      "counts": { "splicing": 7, "copy_move": 2, "enhancement": 5, "removal": 0 },
      "opinion": "Splicing and Enhancement detected in several frames."
    }
  },
  "analysis": "AI-generated video with background tampering and edits.",
  "card": { "Video": { "Images": "82% AI", "Creator": "AI", "Manipulation": "Yes" } },
  "overlays": {}
}
```
