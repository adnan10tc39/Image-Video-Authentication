# backends/image_pipeline.py — minimal, consistent + returns two overlays
import os, tempfile
from pathlib import Path
from PIL import Image
from fastapi import UploadFile
import torch

from .utils import (
    settings, DEVICE, get_models,
    regions_from_result, seg_counts, forgery_primary_label,
    format_date_from_path, bg_predict_image_yolo, cls_predict_simple,
    to_data_uri_png_from_result
)

async def predict_image_pipeline(
    file: UploadFile,
    type_hint: str,
    mask_alpha: float,
    show_boxes: bool,
):
    # save temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "upload").suffix) as tmp:
        content = await file.read(); tmp.write(content); tmp_path = tmp.name
    _date_created = format_date_from_path(tmp_path)
    img = Image.open(tmp_path).convert("RGB")

    SEG_MODEL, CLS_MODEL, FORG_MODEL, BG_MODEL = get_models()

    with torch.no_grad():
        # 1) classification (parity with your separate tester)
        cavr_label, cavr_conf = cls_predict_simple(CLS_MODEL, img)

        # 2) face/seg (produces overlay image)
        seg_res  = SEG_MODEL.predict(img,  imgsz=settings.SEG_IMGSZ,  device=DEVICE, verbose=False,
                                     conf=settings.SEG_CONF, iou=settings.SEG_IOU)[0]
        faces_overlay_png = to_data_uri_png_from_result(seg_res, mask_alpha=mask_alpha, boxes=show_boxes)

        # 3) forgery types (produces overlay image)
        forg_res = FORG_MODEL.predict(img, imgsz=settings.FORG_IMGSZ, device=DEVICE, verbose=False,
                                      conf=settings.FORG_CONF, iou=settings.FORG_IOU)[0]
        forgery_overlay_png = to_data_uri_png_from_result(forg_res, mask_alpha=mask_alpha, boxes=show_boxes)

        # 4) background tampering (YOLO-cls + ELA)
        bg_label, bg_conf, _ = bg_predict_image_yolo(BG_MODEL, img)

    # compact outputs
    seg_regions  = regions_from_result(seg_res)
    forg_regions = regions_from_result(forg_res)
    forg_counts  = seg_counts(forg_res)
    forg_primary = forgery_primary_label(forg_regions, forg_counts)

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

    try: os.remove(tmp_path)
    except Exception: pass

    return {
        "predictions": {
            "classification_ai_vs_real": {
                "primary_label": str(cavr_label),
                "primary_conf": float(cavr_conf)
            },
            "classification_background_tampering": {
                "primary_label": ("forged" if "forged" in bg_label.lower() else "original"),
                "primary_conf": float(bg_conf)
            },
            "segmentation_face_authentication": {
                "counts": {
                    "faces_detected": sum(1 for r in seg_regions if "fake" in str(r.get("label","")).lower()) + sum(1 for r in seg_regions if "real" in str(r.get("label","")).lower()),
                    "fake_faces":     sum(1 for r in seg_regions if "fake" in str(r.get("label","")).lower()),
                    "real_faces":     sum(1 for r in seg_regions if "real" in str(r.get("label","")).lower()),
                },
                "regions": [
                    {k:v for k,v in r.items() if k in ("label","conf","bbox_xyxy","polygon")}
                    for r in seg_regions
                ],
            },
            "segmentation_forgery_types": {
                "counts": sft_counts,
                "regions": [
                    {k:v for k,v in r.items() if k in ("label","conf","bbox_xyxy","polygon")}
                    for r in forg_regions
                ],
                "primary_label": str(forg_primary["label"]),
                "primary_conf": float(forg_primary["conf"]),
            }
        },
        "images": {
            "faces_overlay_png": faces_overlay_png,       # data:image/png;base64,....
            "forgery_overlay_png": forgery_overlay_png    # data:image/png;base64,....
        }
    }
