#!/usr/bin/env python3
"""
Export 2 model YOLOv5 (.pt) sang TorchScript (.pt) để dùng trong C++ (LibTorch).

Đầu vào:
  - LP_detector_nano_61.pt  (model detect biển số, kiểu YOLOv5 custom)
  - LP_ocr_nano_62.pt       (model OCR ký tự trên biển số, kiểu YOLOv5 custom)

Đầu ra:
  - lp_detector_ts.pt
  - lp_ocr_ts.pt

Yêu cầu:
  python3 -m pip install --user yolov5 torch torchvision
"""

import sys

import torch


DET_PT = "LP_detector_nano_61.pt"
OCR_PT = "LP_ocr_nano_62.pt"

DET_TS = "lp_detector_ts.pt"
OCR_TS = "lp_ocr_ts.pt"


def export_one(ckpt_path: str, ts_out: str, img_size: int = 640) -> None:
    """
    Load model YOLOv5 qua torch.hub, bỏ AutoShape, trace phần model chính thành TorchScript.
    """
    print(f"[EXPORT] Loading YOLOv5 checkpoint: {ckpt_path}")
    # Load đúng như trong run_models.py
    model = torch.hub.load("ultralytics/yolov5", "custom", path=ckpt_path, source="github")
    model.eval()

    # YOLOv5 hub model có wrapper AutoShape; phần mạng chính nằm trong model.model
    backbone = model.model
    backbone.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone.to(device)

    # Input giả cho trace: 1 ảnh 3ximg_size x img_size
    dummy = torch.zeros(1, 3, img_size, img_size, device=device)

    print(f"[EXPORT] Tracing to TorchScript on device={device}, size={img_size}x{img_size} ...")
    with torch.no_grad():
        ts = torch.jit.trace(backbone, dummy)

    ts.save(ts_out)
    print(f"[EXPORT] Saved TorchScript to: {ts_out}")


def main() -> int:
    try:
        export_one(DET_PT, DET_TS)
        export_one(OCR_PT, OCR_TS)
    except Exception as e:
        print(f"[ERROR] Export thất bại: {e}", file=sys.stderr)
        return 1
    print("[EXPORT] Hoàn thành. Hãy dùng lp_detector_ts.pt và lp_ocr_ts.pt trong C++.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


