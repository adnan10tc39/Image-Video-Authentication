import os, tempfile
from pathlib import Path
from fastapi import UploadFile
from PIL import Image
import numpy as np
import torch
from collections import Counter

from .utils import (
    settings, DEVICE, get_models,
    sample_video_frames, seg_counts, regions_from_result, forgery_primary_label,
    bg_predict_batch_yolo, cls_predict_simple, to_data_uri_png_from_result,
    burn_caption, encode_png_rgb,
)

FAKE_KEYWORD = "fake"  # <-- two-class model: "real", "fake"

async def predict_video_pipeline(
    file: UploadFile,
    type_hint: str,
    frame_stride: int,
    max_frames: int,
    batch_size: int,
    mask_alpha: float,
    show_boxes: bool,
    regions_max_per_frame: int,
    sample_fps: float = None,   # if None, we’ll default to 1.0 fps (every 1s)
    uniform: bool = True,
):
    # --- Save to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "upload.mp4").suffix) as tmp:
        content = await file.read(); tmp.write(content); tmp_path = tmp.name

    # --- Sampling rate
    # If caller doesn't provide a positive fps, default to 1.0 fps (1 frame / second).
    effective_sample_fps = sample_fps if (sample_fps is not None and sample_fps > 0) else 1.0
    time_step_sec = 1.0 / float(effective_sample_fps)  # used for per-frame timestamps

    # High quality: read full-resolution frames (models resize internally)
    frames = sample_video_frames(
        content,
        frame_stride=frame_stride,          # kept for compatibility; unused when sample_fps is set
        max_frames=max_frames,
        sample_fps=effective_sample_fps,
        uniform=True                        # uniform sampling across duration
    )

    total_frames = len(frames)
    if total_frames == 0:
        try: os.remove(tmp_path)
        except Exception: pass
        return {"error": "No frames sampled from video."}

    SEG_MODEL, CLS_MODEL, FORG_MODEL, BG_MODEL = get_models()

    # --- Aggregate counters
    frames_fake = 0
    frames_real = 0
    frames_bg_forged = 0
    frames_bg_original = 0

    total_faces = 0
    total_fake_faces = 0
    total_real_faces = 0
    frames_with_any_face = 0
    frames_with_fake_faces = 0
    frames_with_real_faces = 0

    canon_map = {"copy-move": "copy_move", "copy_move": "copy_move", "splicing": "splicing",
                 "enhancement": "enhancement", "removal": "removal"}
    frames_with_type = {"splicing": 0, "copy_move": 0, "enhancement": 0, "removal": 0}
    regions_total_by_type = {"splicing": 0, "copy_move": 0, "enhancement": 0, "removal": 0}

    # still useful if you later want distribution insight
    cls_top1_total = Counter()

    # --- overlays per sampled frame (similar to image pipeline)
    per_frame_images = []

    # --- Process in batches
    for s in range(0, total_frames, int(batch_size)):
        batch_np  = frames[s:s+int(batch_size)]
        batch_pil = [Image.fromarray(fr) for fr in batch_np]

        with torch.no_grad():
            seg_results  = SEG_MODEL.predict(
                batch_np, imgsz=settings.SEG_IMGSZ, device=DEVICE, verbose=False,
                conf=settings.SEG_CONF, iou=settings.SEG_IOU
            )
            forg_results = FORG_MODEL.predict(
                batch_np, imgsz=settings.FORG_IMGSZ, device=DEVICE, verbose=False,
                conf=settings.FORG_CONF, iou=settings.FORG_IOU
            )
            # background cls on ELA-prepared RGB via util
            bg_probs = bg_predict_batch_yolo(BG_MODEL, batch_pil)

        # classification PER FRAME via cls_predict_simple
        cls_labels = []
        cls_confs  = []
        for pil in batch_pil:
            lbl, conf = cls_predict_simple(CLS_MODEL, pil)
            cls_labels.append(lbl)
            cls_confs.append(conf)

        # --- Reduce per frame
        for i, (seg_r, forg_r) in enumerate(zip(seg_results, forg_results)):
            global_index = s + i

            # classification counters (strict "fake"/"real")
            raw_lbl = str(cls_labels[i]).strip().lower()
            conf = float(cls_confs[i])
            cls_top1_total.update([raw_lbl])
            if raw_lbl == FAKE_KEYWORD:
                frames_fake += 1
                pretty_lbl = "FAKE"
                label_norm = "fake"
            else:
                frames_real += 1
                pretty_lbl = "REAL"
                label_norm = "real"

            # background per frame
            if i < len(bg_probs) and float(bg_probs[i]) >= 0.5:
                frames_bg_forged += 1
            else:
                frames_bg_original += 1

            # faces
            regions = regions_from_result(seg_r)
            faces_this = [r for r in regions if "face" in str(r.get("label","")).lower()]
            fake_this  = [r for r in regions if "fake" in str(r.get("label","")).lower()]
            real_this  = [r for r in regions if "real" in str(r.get("label","")).lower()]
            if faces_this: frames_with_any_face += 1
            if fake_this:  frames_with_fake_faces += 1
            if real_this:  frames_with_real_faces += 1
            total_faces      += len(faces_this)
            total_fake_faces += len(fake_this)
            total_real_faces += len(real_this)

            # forgery types
            forg_c = seg_counts(forg_r)
            seen = set()
            for k, v in forg_c.items():
                kk = k.lower().replace("-", "_")
                for src, dst in canon_map.items():
                    if src in kk and v > 0:
                        seen.add(dst); regions_total_by_type[dst] += int(v)
                        break
            for t in seen:
                frames_with_type[t] += 1

            # --- overlays for this frame (same generator as image pipeline)
            faces_overlay_png   = to_data_uri_png_from_result(seg_r,  mask_alpha=mask_alpha, boxes=show_boxes)
            forgery_overlay_png = to_data_uri_png_from_result(forg_r, mask_alpha=mask_alpha, boxes=show_boxes)

            # NEW: classification overlay burned onto the raw frame with label+conf
            frame_rgb = np.array(batch_np[i])  # RGB
            caption   = [f"AI vs Real: {pretty_lbl} ({conf*100:.1f}%)"]
            class_rgb = burn_caption(frame_rgb, caption)  # returns RGB
            class_overlay_png = f"data:image/png;base64,{encode_png_rgb(class_rgb)}"

            # timestamp derived from sampling fps
            time_sec = float(global_index) * time_step_sec

            per_frame_images.append({
                "frame_index": int(global_index),
                "time_sec": time_sec,
                "classification_label": label_norm,          # "fake" | "real"
                "classification_conf": conf,                 # 0..1
                "classification_overlay_png": class_overlay_png,
                "faces_overlay_png":   faces_overlay_png,
                "forgery_overlay_png": forgery_overlay_png,
            })

    # --- Video-level primary forgery (counts only)
    total_counts = {k: regions_total_by_type[k] for k in regions_total_by_type}
    forg_primary_total = forgery_primary_label([], total_counts)
    if forg_primary_total["regions"] == 0:
        forg_primary_total["label"] = "None"

    try: os.remove(tmp_path)
    except Exception: pass

    return {
        "predictions": {
            "classification_ai_vs_real": {   # keeping name for API compatibility
                "primary_label": ("fake" if frames_fake >= frames_real else "real"),
                "frames_total": total_frames,
                "frames_fake": frames_fake,
                "frames_real": frames_real
            },
            "classification_background_tampering": {
                "primary_label": ("forged" if frames_bg_forged >= frames_bg_original else "original"),
                "frames_total": total_frames,
                "frames_forged": frames_bg_forged,
                "frames_original": frames_bg_original
            },
            "segmentation_face_authentication": {
                "counts": {
                    "total_faces": total_faces,
                    "fake_faces": total_fake_faces,
                    "real_faces": total_real_faces
                },
                "frames_with_any_face": frames_with_any_face,
                "frames_with_fake_faces": frames_with_fake_faces,
                "frames_with_real_faces": frames_with_real_faces
            },
            "segmentation_forgery_types": {
                "frames_with": {
                    "splicing":   int(frames_with_type["splicing"]),
                    "copy_move":  int(frames_with_type["copy_move"]),
                    "enhancement":int(frames_with_type["enhancement"]),
                    "removal":    int(frames_with_type["removal"]),
                },
                "regions_total": {
                    "splicing":   int(regions_total_by_type["splicing"]),
                    "copy_move":  int(regions_total_by_type["copy_move"]),
                    "enhancement":int(regions_total_by_type["enhancement"]),
                    "removal":    int(regions_total_by_type["removal"]),
                },
                "primary_label": forg_primary_total["label"],
                "primary_conf": float(forg_primary_total["conf"]),
            }
        },
        # --- images similar to image pipeline (but per-frame array)
        "images": {
            "frames": per_frame_images
        }
    }

