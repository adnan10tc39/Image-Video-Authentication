# backends/image_pipeline.py — /predict/image logic (4 models, BG via YOLO cls + ELA)
import os, tempfile
from pathlib import Path
from typing import Dict, List
from PIL import Image
from fastapi import UploadFile
import torch
import numpy as np

from .utils import (
    settings, DEVICE, get_models,
    regions_from_result, seg_counts, seg_face_counts_from_regions,
    topk_from_cls_result, annotate_to_rgb, burn_caption,
    forgery_primary_label, op_cls_ai, op_face, op_forgery,
    overall_analysis, save_overlay_png, format_date_from_path,
    bg_predict_image_yolo
)

async def predict_image_pipeline(
    file: UploadFile,
    type_hint: str,
    mask_alpha: float,
    show_boxes: bool,
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "upload").suffix) as tmp:
        content = await file.read(); tmp.write(content); tmp_path = tmp.name
    date_created = format_date_from_path(tmp_path)

    img = Image.open(tmp_path).convert("RGB")

    SEG_MODEL, CLS_MODEL, FORG_MODEL, BG_MODEL = get_models()
    with torch.no_grad():
        seg_res  = SEG_MODEL.predict(img,  imgsz=settings.SEG_IMGSZ,  device=DEVICE, verbose=False,
                                     conf=settings.SEG_CONF, iou=settings.SEG_IOU)[0]
        cls_res  = CLS_MODEL.predict(img,  imgsz=settings.CLS_IMGSZ,  device=DEVICE, verbose=False)[0]
        forg_res = FORG_MODEL.predict(img, imgsz=settings.FORG_IMGSZ, device=DEVICE, verbose=False,
                                      conf=settings.FORG_CONF, iou=settings.FORG_IOU)[0]

    # background (4th) with YOLO cls + ELA
    bg_label, bg_conf, _probs = bg_predict_image_yolo(BG_MODEL, img)
    bg_is_forged = ("forged" in bg_label.lower())

    # structures
    seg_regions = regions_from_result(seg_res)
    forg_regions = regions_from_result(forg_res)
    forg_counts  = seg_counts(forg_res)

    face_counts = seg_face_counts_from_regions(seg_regions)
    face_regions_compact = []
    for r in seg_regions:
        if "bbox_xyxy" in r and r["bbox_xyxy"]:
            face_regions_compact.append({
                "label": r.get("label"),
                "conf": r.get("conf"),
                "bbox_xyxy": r.get("bbox_xyxy")
            })

    cls_items = topk_from_cls_result(cls_res, k=settings.CLS_TOPK, conf_thresh=settings.CLS_MIN_PROB)
    cls_top1 = cls_items[0] if cls_items else None

    # opinions
    cls_opinion = op_cls_ai(cls_items)
    face_opinion = op_face(seg_regions)
    forg_primary = forgery_primary_label(forg_regions, forg_counts)
    forg_opinion = op_forgery(forg_primary, forg_counts)

    # required JSON fields
    cavr_topk = [{"label": str(cls_top1["label"]).lower(), "prob": float(cls_top1["prob"])}] if cls_top1 else []
    cavr_label = (str(cls_top1["label"]).lower() if cls_top1 else "unknown")
    cavr_conf  = (float(cls_top1["prob"]) if cls_top1 else 0.0)

    cbt_label = "forged" if bg_is_forged else "original"
    cbt_conf  = float(bg_conf)
    cbt_topk  = [{"label": cbt_label, "prob": cbt_conf}]

    # normalize forgery-type counts
    canon_map = {"copy-move": "copy_move", "copy_move": "copy_move", "splicing": "splicing",
                 "enhancement": "enhancement", "removal": "removal"}
    sft_counts = {"splicing": 0, "copy_move": 0, "enhancement": 0, "removal": 0}
    for k, v in forg_counts.items():
        kk = k.lower().replace("-", "_")
        for src, dst in canon_map.items():
            if src in kk:
                sft_counts[dst] += int(v)
                break

    # overlays
    rgb_seg  = burn_caption(annotate_to_rgb(seg_res,  mask_alpha=mask_alpha, boxes=show_boxes), [face_opinion])
    rgb_forg = burn_caption(annotate_to_rgb(forg_res, mask_alpha=mask_alpha, boxes=show_boxes), [forg_opinion])
    face_overlay_name = save_overlay_png(rgb_seg, "face_analysis_overlay")
    forg_overlay_name = save_overlay_png(rgb_forg, "forensic_overlay")

    # analysis + card
    is_ai_like = ("ai" in (cavr_label or ""))
    forg_present = (forg_primary["label"] != "None") or any(v > 0 for v in sft_counts.values())
    analysis = overall_analysis(is_ai_like, bg_is_forged, forg_present)

    ai_pct = int(round(cavr_conf * 100))
    card = {
        "Images": {
            "Photo": f"{ai_pct}% {'AI' if is_ai_like else 'Real'}",
            "Creator": "AI" if is_ai_like else "Human",
            "Manipulation": "Yes" if (bg_is_forged or forg_present or face_counts['fake_faces']>0) else "No",
            "Type": type_hint,
            "Date created": date_created
        }
    }

    try: os.remove(tmp_path)
    except Exception: pass

    return {
        "predictions": {
            "classification_ai_vs_real": {
                "topk": cavr_topk,
                "primary_label": cavr_label,
                "primary_conf": cavr_conf,
                "opinion": cls_opinion
            },
            "classification_background_tampering": {
                "topk": cbt_topk,
                "primary_label": cbt_label,
                "primary_conf": cbt_conf,
                "focus": "background",
                "opinion": ("Background tampering detected with confidence "
                            f"{cbt_conf*100:.1f}%." if bg_is_forged else
                            f"No background tampering detected ({cbt_conf*100:.1f}%).")
            },
            "segmentation_face_authentication": {
                "counts": {
                    "faces_detected": face_counts["faces_detected"],
                    "fake_faces": face_counts["fake_faces"],
                    "real_faces": face_counts["real_faces"]
                },
                "regions": face_regions_compact,
                "opinion": face_opinion
            },
            "segmentation_forgery_types": {
                "counts": sft_counts,
                "regions": [
                    {k:v for k,v in r.items() if k in ("label","conf","bbox_xyxy","polygon")}
                    for r in forg_regions
                ],
                "primary_label": forg_primary["label"],
                "primary_conf": float(forg_primary["conf"]),
                "opinion": forg_opinion
            }
        },
        "analysis": analysis,
        "card": card,
        "overlays": {
            "face_overlay": face_overlay_name,
            "forgery_overlay": forg_overlay_name
        }
    }
