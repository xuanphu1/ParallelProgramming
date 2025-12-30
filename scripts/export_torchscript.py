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
import os
from pathlib import Path

import torch

# Lấy đường dẫn thư mục models
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_DIR / "models"

DET_PT = str(MODELS_DIR / "bestDetect.pt")
OCR_PT = str(MODELS_DIR / "bestOCR.pt")

DET_TS = str(MODELS_DIR / "bestDetect_ts.pt")
OCR_TS = str(MODELS_DIR / "bestOCR_ts.pt")


def export_one(ckpt_path: str, ts_out: str, img_size: int = 640) -> None:
    """
    Load model YOLOv5 qua torch.hub, export sang ONNX (YOLOv5 hỗ trợ tốt).
    """
    print(f"[EXPORT] Loading YOLOv5 checkpoint: {ckpt_path}")
    # Load đúng như trong run_models.py, thử force_reload nếu cần
    try:
        model = torch.hub.load("ultralytics/yolov5", "custom", path=ckpt_path, source="github", force_reload=False)
    except:
        # Thử force_reload nếu lỗi
        model = torch.hub.load("ultralytics/yolov5", "custom", path=ckpt_path, source="github", force_reload=True)
    model.eval()

    # Export sang ONNX (YOLOv5 hỗ trợ tốt hơn TorchScript)
    onnx_path = str(MODELS_DIR / (Path(ts_out).stem + ".onnx"))
    print(f"[EXPORT] Exporting to ONNX: {onnx_path} (size={img_size}) ...")
    
    # Chuyển vào thư mục models để export
    original_cwd = os.getcwd()
    try:
        os.chdir(MODELS_DIR)
        
        # YOLOv5 model có AutoShape wrapper; export() nằm trong model.model
        # Hoặc dùng model trực tiếp nếu nó có export()
        if hasattr(model, 'export'):
            # YOLOv5 export() tạo file với tên mặc định, sẽ rename sau
            model.export(format="onnx", imgsz=img_size, simplify=True, opset=12)
        else:
            # Thử export từ model.model (backbone)
            backbone = model.model
            if hasattr(backbone, 'export'):
                backbone.export(format="onnx", imgsz=img_size, simplify=True, opset=12)
            else:
                # Fallback: dùng torch.onnx.export trực tiếp
                class ExportWrapper(torch.nn.Module):
                    def __init__(self, model):
                        super().__init__()
                        self.model = model.model
                    def forward(self, x):
                        return self.model(x)
                
                wrapper = ExportWrapper(model)
                wrapper.eval()
                dummy = torch.zeros(1, 3, img_size, img_size)
                torch.onnx.export(
                    wrapper,
                    dummy,
                    os.path.basename(onnx_path),
                    input_names=['images'],
                    output_names=['output'],
                    opset_version=12,
                    do_constant_folding=True,
                    dynamic_axes={'images': {0: 'batch'}, 'output': {0: 'batch'}}
                )
        
        # YOLOv5 export() thường tạo file với tên mặc định, cần rename
        import glob
        import shutil
        # Tìm file ONNX vừa tạo (thường là tên model + .onnx)
        onnx_files = glob.glob("*.onnx")
        if onnx_files:
            # Tìm file mới nhất
            latest_onnx = max(onnx_files, key=os.path.getmtime)
            target_name = os.path.basename(onnx_path)
            if latest_onnx != target_name:
                if os.path.exists(target_name):
                    os.remove(target_name)
                shutil.move(latest_onnx, target_name)
                print(f"[EXPORT] Renamed {latest_onnx} -> {target_name}")
        
        if os.path.exists(os.path.basename(onnx_path)):
            print(f"[EXPORT] ✓ ONNX saved: {onnx_path}")
        else:
            print(f"[WARNING] Không tìm thấy file ONNX sau khi export.")
    except Exception as e:
        print(f"[ERROR] Export ONNX thất bại: {e}")
        raise
    finally:
        os.chdir(original_cwd)


def main() -> int:
    try:
        export_one(DET_PT, DET_TS)
        export_one(OCR_PT, OCR_TS)
    except Exception as e:
        print(f"[ERROR] Export thất bại: {e}", file=sys.stderr)
        return 1
    print("[EXPORT] Hoàn thành. Hãy dùng bestDetect_ts.onnx và bestOCR_ts.onnx trong C++.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


