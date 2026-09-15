"""
Wraps the RF-DETR bale detector from reference_project/edge_deploy/detect.py
as a lazily-loaded singleton usable from FastAPI request handlers.
"""
from __future__ import annotations

import os
import threading

import numpy as np

from .config import CHECKPOINT_PATH, INFERENCE_RESOLUTION
from .schemas import Detection, Point, Rect

_model = None
_model_lock = threading.Lock()
_model_device = "cpu"
# Serializes calls into model.predict(). Loading is already guarded by
# _model_lock; this guards INFERENCE. Occasional polling (the old /detect use)
# tolerated concurrent calls fine in practice, but stream_annotated calls
# predict() continuously per open viewer - with two feeds open at once that's
# sustained concurrency into the same CUDA context, which torch does not
# guarantee is safe. Cheap insurance: one forward pass on the GPU at a time.
_predict_lock = threading.Lock()


def _optimize_cpu() -> None:
    import torch

    if torch.backends.mkldnn.is_available():
        torch.backends.mkldnn.enabled = True
        torch.set_num_threads(os.cpu_count() or 4)


def get_model():
    """Lazily load and cache the RFDETRNano model (thread-safe, loads once).
    Uses the GPU when available (this dev machine has an RTX 5060 Ti — see
    ``optimize_for_inference``'s own warning that CPU inference leaves an
    ~8x FP16-tensor-core speedup on the table). `RFDETRNano(device=...)`
    doesn't actually move params despite accepting the kwarg, so this moves
    the underlying nn.Module explicitly before optimizing. Falls back to
    the CPU-optimized path (matching the real edge-deployment target
    described in reference_project/edge_deploy/detect.py) when no GPU is
    present."""
    global _model, _model_device
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:
            if not os.path.exists(CHECKPOINT_PATH):
                raise FileNotFoundError(
                    f"Checkpoint not found at {CHECKPOINT_PATH}. "
                    "Expected reference_project/edge_deploy/checkpoint_best_ema.pth"
                )
            import torch
            from rfdetr import RFDETRNano

            model = RFDETRNano(resolution=INFERENCE_RESOLUTION, pretrain_weights=CHECKPOINT_PATH)

            if torch.cuda.is_available():
                _model_device = "cuda:0"
                major, minor = torch.cuda.get_device_capability(0)
                print(
                    f"[model_service] using GPU: {torch.cuda.get_device_name(0)} "
                    f"(sm_{major}{minor})"
                )
                model.model.model.to(_model_device)
                # FP16 tensor-core acceleration is only a win on Ampere (sm_80)
                # and newer. On Turing (sm_75, e.g. GTX 1660 SUPER) half-rate
                # FP16 makes optimize_for_inference slow to trace and can OOM a
                # 6 GB card - use fp32 there.
                use_fp16 = major >= 8
                try:
                    if use_fp16:
                        model.optimize_for_inference(dtype=torch.float16)
                    else:
                        print("[model_service] pre-Ampere GPU - optimizing fp32")
                        model.optimize_for_inference()
                except Exception as e:  # fall back to fp32 if fp16 tracing fails
                    print(f"[model_service] fp16 optimize failed ({e}), retrying fp32")
                    model.optimize_for_inference()
            else:
                _model_device = "cpu"
                _optimize_cpu()
                model.optimize_for_inference()

            _model = model
    return _model


# Checkpoint category order (0-indexed, as returned by model.predict()'s
# class_id - the COCO annotations use 1-indexed category_id, one higher).
CLASS_BALE, CLASS_CRANE_HOOK, CLASS_CRANE_SPIKE = 0, 1, 2


def detect_bales(image_bgr: np.ndarray, threshold: float) -> list[Detection]:
    """Run inference on a full BGR image (numpy array), return bale detections
    only (class_id == CLASS_BALE) in original image pixel coordinates, sorted
    by descending confidence. The checkpoint is multi-class (bale/crane_hook/
    crane_spike, see training/README.md) but the planner only reasons about
    bales today, so hook/spike hits are filtered out here rather than passed
    through as bogus high-confidence "bales". Use detect_all() for the raw,
    unfiltered multi-class output (e.g. for crane-pose sensing)."""
    detections = [d for d in detect_all(image_bgr, threshold) if d.class_id == CLASS_BALE]
    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


def detect_all(image_bgr: np.ndarray, threshold: float) -> list[Detection]:
    """Run inference on a full BGR image (numpy array), return every detection
    (bale, crane_hook, crane_spike) in original image pixel coordinates,
    sorted by descending confidence."""
    model = get_model()
    with _predict_lock:
        result = model.predict(image_bgr, threshold=threshold)

    detections: list[Detection] = []
    xyxy = result.xyxy if len(result) > 0 else []
    confidence = result.confidence if len(result) > 0 else []
    class_id = result.class_id if len(result) > 0 else []
    for i, (box, conf, cid) in enumerate(zip(xyxy, confidence, class_id)):
        x1, y1, x2, y2 = [float(v) for v in box]
        bbox = Rect(x1=x1, y1=y1, x2=x2, y2=y2).normalized()
        detections.append(
            Detection(id=i, bbox=bbox, confidence=float(conf), center=bbox.center, class_id=int(cid))
        )
    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


def warmup() -> None:
    """Trigger model load at server startup instead of on the first request."""
    get_model()
