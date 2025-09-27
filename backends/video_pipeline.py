# backends/video_pipeline.py — /predict/video logic (4 models, BG via YOLO cls + ELA)
# User-friendly output: counts per frames (no batch/stride exposure, no overlays).

import os, math, tempfile
from pathlib import Path
from typing import Dict, List
from fastapi import UploadFile
from PIL import Image
import numpy as np
import torch
from collections import Counter

from .utils import (
    settings, DEVICE, get_models,
    sample_video_frames, seg_counts, regions_from_result, forgery_primary_label, op_forgery,
    format_date_from_path, bg_predict_batch_yolo
)

AI_KEYWORD = "ai"  # treat top-1 label containing 'ai' (case-insensitive) as AI

async def predict_video_pipeline(
    file: UploadFile,
    type_hint: str,
    frame_stride: int,
    max_frames: int,
    batch_size: int,
    mask_alpha: float,      # kept for signature compatibility (unused in video summary)
    show_boxes: bool,       # kept for signature compatibility (unused in video summary)
    regions_max_per_frame: int,  # kept for signature compatibility (unused in video summary)
):
    # --- save upload temporarily for metadata
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "upload.mp4").suffix) as tmp:
        content = await file.read(); tmp.write(content); tmp_path = tmp.name
    date_created = format_date_from_path(tmp_path)

    # --- sample frames
    frames = sample_video_frames(content, frame_stride, max_frames)
    total_frames = len(frames)
    if total_frames == 0:
        return {"error": "No frames sampled from video."}

    # --- load models
    SEG_MODEL, CLS_MODEL, FORG_MODEL, BG_MODEL = get_models()

    # -----------------------------------------
    # ACCUMULATORS (per-frame user-facing counts)
    # -----------------------------------------
    # AI vs Real (classification)
    frames_ai = 0
    frames_real = 0

    # Background tampering (YOLO cls on ELA)
    frames_bg_forged = 0
    frames_bg_original = 0

    # Face authentication (segmentation result interpreted for faces)
    total_faces = 0
    total_fake_faces = 0
    total_real_faces = 0
    frames_with_any_face = 0
    frames_with_fake_faces = 0
    frames_with_real_faces = 0

    # Forgery types (segmentation of splicing/copy-move/enhancement/removal)
    canon_map = {"copy-move": "copy_move", "copy_move": "copy_move", "splicing": "splicing",
                 "enhancement": "enhancement", "removal": "removal"}
    # count in how many FRAMES each type appears at least once
    frames_with_type = {"splicing": 0, "copy_move": 0, "enhancement": 0, "removal": 0}
    # total regions across all frames (optional, adds depth)
    regions_total_by_type = {"splicing": 0, "copy_move": 0, "enhancement": 0, "removal": 0}

    # also keep some aggregates to build opinions/analysis
    cls_top1_total = Counter()
    avg_cls_prob_accum, avg_cls_prob_n = 0.0, 0
    forg_counts_total = Counter()
    bg_prob_accum, bg_prob_n = 0.0, 0

    # -----------------------------------------
    # PROCESS FRAMES IN BATCHES (internal only)
    # -----------------------------------------
    for s in range(0, total_frames, int(batch_size)):
        batch_np  = frames[s:s+int(batch_size)]
        batch_pil = [Image.fromarray(fr) for fr in batch_np]

        with torch.no_grad():
            seg_results  = SEG_MODEL.predict(batch_np, imgsz=settings.SEG_IMGSZ, device=DEVICE, verbose=False,
                                             conf=settings.SEG_CONF, iou=settings.SEG_IOU)
            cls_results  = CLS_MODEL.predict(batch_np, imgsz=settings.CLS_IMGSZ, device=DEVICE, verbose=False)
            forg_results = FORG_MODEL.predict(batch_np, imgsz=settings.FORG_IMGSZ, device=DEVICE, verbose=False,
                                              conf=settings.FORG_CONF, iou=settings.FORG_IOU)

        # background YOLO cls over batch (prob of 'Forged Background')
        bg_probs = bg_predict_batch_yolo(BG_MODEL, batch_pil)

        for i, (rgb_in, seg_r, cls_r, forg_r) in enumerate(zip(batch_np, seg_results, cls_results, forg_results)):
            # -------- Classification: AI vs Real (top-1)
            scores = getattr(cls_r, "probs", None)
            if scores is not None and scores.data is not None:
                scores_np = scores.data.detach().float().cpu().numpy().reshape(-1)
                names = getattr(cls_r, "names", None)
                if names is not None and scores_np.size:
                    idx = int(np.argmax(scores_np))
                    lbl = str(names[idx])
                    prob = float(scores_np[idx])
                    cls_top1_total.update([lbl])
                    avg_cls_prob_accum += prob; avg_cls_prob_n += 1
                    if AI_KEYWORD in lbl.lower():
                        frames_ai += 1
                    else:
                        frames_real += 1

            # -------- Background tampering: forged vs original (per-frame threshold 0.5)
            if i < len(bg_probs):
                p_forged = float(bg_probs[i])
                bg_prob_accum += p_forged; bg_prob_n += 1
                if p_forged >= 0.5:
                    frames_bg_forged += 1
                else:
                    frames_bg_original += 1

            # -------- Face authentication (roughly from seg_r regions)
            regions = regions_from_result(seg_r)
            faces_this_frame = [r for r in regions if "face" in str(r.get("label","")).lower()]
            fake_this = [r for r in regions if "fake" in str(r.get("label","")).lower()]
            real_this = [r for r in regions if "real" in str(r.get("label","")).lower()]
            if faces_this_frame:
                frames_with_any_face += 1
            if fake_this:
                frames_with_fake_faces += 1
            if real_this:
                frames_with_real_faces += 1
            total_faces      += len(faces_this_frame) if faces_this_frame else 0
            total_fake_faces += len(fake_this)
            total_real_faces += len(real_this)

            # -------- Forgery types (counts + frames with)
            forg_c = seg_counts(forg_r)
            forg_counts_total.update(forg_c)
            # Did this frame contain each canonical type?
            seen_types_this_frame = set()
            for k, v in forg_c.items():
                kk = k.lower().replace("-", "_")
                for src, dst in canon_map.items():
                    if src in kk and v > 0:
                        seen_types_this_frame.add(dst)
                        regions_total_by_type[dst] += int(v)
                        break
            for t in seen_types_this_frame:
                frames_with_type[t] += 1

    # -----------------------------------------
    # Aggregate opinions / analysis
    # -----------------------------------------
    ai_like = any(AI_KEYWORD in k.lower() for k in cls_top1_total.keys())
    avg_ai_prob = (avg_cls_prob_accum / max(avg_cls_prob_n, 1))
    bg_avg_prob = (bg_prob_accum / max(bg_prob_n, 1))
    bg_is_forged_majority = (frames_bg_forged >= frames_bg_original)

    # Build a simple cls opinion (video-level)
    if avg_cls_prob_n:
        cls_opinion = f"Top-1 across frames: {'AI' if ai_like else 'Real'} (~{avg_ai_prob*100:.1f}%)."
    else:
        cls_opinion = "No classification signal."

    # Video-level forgery primary label (counts only)
    forg_primary_total = forgery_primary_label([], forg_counts_total)
    if forg_primary_total["regions"] == 0:
        forg_primary_total["label"] = "None"
    forg_opinion = ("No forgery artifact detected."
                    if forg_primary_total["label"] == "None" else
                    f"{forg_primary_total['label']} ({forg_primary_total['conf']:.2f}) detected.")

    # Background opinion line
    cbt_opinion = ("Background tampering detected."
                   if bg_is_forged_majority else
                   "No background tampering detected.")

    # Overall analysis (concise, user-facing)
    tampered_any_type = any(v > 0 for v in frames_with_type.values())
    if ai_like and not (tampered_any_type or bg_is_forged_majority):
        analysis = "Video frames mostly AI-generated; no localized edits."
    elif not ai_like and (tampered_any_type or bg_is_forged_majority):
        analysis = "Real video with background tampering / localized edits in frames."
    elif ai_like and (tampered_any_type or bg_is_forged_majority):
        analysis = "AI-generated content with background tampering / additional edits."
    else:
        analysis = "Frames appear natural with no detected edits."

    # Card summary line
    images_line = f"{int(round(avg_ai_prob*100) if avg_cls_prob_n>0 else 0)}% {'AI' if ai_like else 'Real'}"
    card = {"Video": {
        "Images": images_line,
        "Creator": "AI" if ai_like else "Human",
        "Manipulation": "Yes" if (tampered_any_type or bg_is_forged_majority) else "No",
        "Type": type_hint,
        "Date created": format_date_from_path(tmp_path)
    }}

    # clean temp
    try: os.remove(tmp_path)
    except Exception: pass

    # -----------------------------------------
    # USER-FRIENDLY PREDICTIONS (by frames)
    # -----------------------------------------
    return {
        "predictions": {
            # 1) AI vs Real — tell the user counts across frames
            "classification_ai_vs_real": {
                "topk": [{"label": ("ai" if ai_like else "real"), "prob": float(avg_ai_prob)}] if avg_cls_prob_n else [],
                "primary_label": ("ai" if ai_like else ("real" if avg_cls_prob_n else "unknown")),
                "primary_conf": float(avg_ai_prob if avg_cls_prob_n else 0.0),
                "opinion": cls_opinion,
                "summary": {
                    "frames_total": total_frames,
                    "frames_ai": frames_ai,
                    "frames_real": frames_real
                }
            },

            # 2) Background tampering — how many frames look forged vs original
            "classification_background_tampering": {
                "topk": [{"label": ("forged" if bg_is_forged_majority else "original"),
                          "prob": float(bg_avg_prob)}] if bg_prob_n else [],
                "primary_label": ("forged" if bg_is_forged_majority else "original"),
                "primary_conf": float(bg_avg_prob if bg_prob_n else 0.0),
                "focus": "background",
                "opinion": cbt_opinion,
                "summary": {
                    "frames_total": total_frames,
                    "frames_forged": frames_bg_forged,
                    "frames_original": frames_bg_original
                }
            },

            # 3) Face authentication — totals across frames + in how many frames faces appeared
            "segmentation_face_authentication": {
                "counts": {
                    "total_faces": total_faces,
                    "fake_faces": total_fake_faces,
                    "real_faces": total_real_faces
                },
                "regions": [],  # omitted for video summary
                "opinion": (
                    "Faces detected in "
                    f"{frames_with_any_face}/{total_frames} frames "
                    f"({frames_with_fake_faces} frames with fake faces, "
                    f"{frames_with_real_faces} frames with real faces)."
                )
            },

            # 4) Forgery types — per-type frame counts + total regions (extra context)
            "segmentation_forgery_types": {
                "counts": {
                    # number of frames where each type appeared at least once
                    "splicing":   int(frames_with_type["splicing"]),
                    "copy_move":  int(frames_with_type["copy_move"]),
                    "enhancement":int(frames_with_type["enhancement"]),
                    "removal":    int(frames_with_type["removal"]),
                    # optional: total regions across the whole video (extra info)
                    "regions_total": {
                        "splicing":   int(regions_total_by_type["splicing"]),
                        "copy_move":  int(regions_total_by_type["copy_move"]),
                        "enhancement":int(regions_total_by_type["enhancement"]),
                        "removal":    int(regions_total_by_type["removal"]),
                    }
                },
                "regions": [],  # per-frame polygons omitted for concise video output
                "primary_label": forg_primary_total["label"],
                "primary_conf": float(forg_primary_total["conf"]),
                "opinion": (
                    f"Detected in frames — "
                    f"Splicing: {frames_with_type['splicing']}, "
                    f"Copy-move: {frames_with_type['copy_move']}, "
                    f"Enhancement: {frames_with_type['enhancement']}, "
                    f"Removal: {frames_with_type['removal']}."
                )
            }
        },

        # One-liner human summary
        "analysis": analysis,

        # Card (high-level)
        "card": card,

        # No overlays for video (as requested)
        "overlays": {}
    }
