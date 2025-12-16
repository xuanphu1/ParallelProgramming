#!/usr/bin/env python3

"""
Webcam/Video/Ảnh license-plate detection + OCR dùng 2 model YOLOv5:
 - LP_detector_nano_61.pt: phát hiện biển số
 - LP_ocr_nano_62.pt: đọc ký tự trên crop biển số

Phụ thuộc:
  python3 -m pip install --user yolov5 opencv-python numpy torch torchvision
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv5 webcam/video/image license plate detect + OCR")
    parser.add_argument(
        "--source",
        default="0",
        help="Đường dẫn ảnh/video hoặc index camera (mặc định: 0 cho /dev/video0)",
    )
    parser.add_argument(
        "--det-model",
        default="LP_detector_nano_61.pt",
        help="YOLOv5 detection model (.pt) cho phát hiện biển số",
    )
    parser.add_argument(
        "--ocr-model",
        default="LP_ocr_nano_62.pt",
        help="YOLOv5 OCR model (.pt) cho đọc biển số",
    )
    parser.add_argument(
        "--det-conf",
        type=float,
        default=0.35,
        help="Ngưỡng confidence cho model detect",
    )
    parser.add_argument(
        "--ocr-conf",
        type=float,
        default=0.25,
        help="Ngưỡng confidence cho model OCR",
    )
    parser.add_argument("--width", type=int, default=1280, help="Chiều rộng capture")
    parser.add_argument("--height", type=int, default=720, help="Chiều cao capture")
    parser.add_argument(
        "--max-dets",
        type=int,
        default=1,
        help="Số lượng biển tối đa hiển thị trên mỗi frame",
    )
    return parser.parse_args()


def load_models(det_path: str, ocr_path: str):
    """Load YOLOv5 models qua torch.hub (đúng chuẩn YOLOv5 .pt)."""
    try:
        import torch
    except ImportError:
        print(
            "Thiếu torch. Cài bằng:\n"
            "  python3 -m pip install --user torch torchvision",
            file=sys.stderr,
        )
        sys.exit(1)

    if not Path(det_path).exists():
        print(f"Model detector không tồn tại: {det_path}", file=sys.stderr)
        sys.exit(1)
    if not Path(ocr_path).exists():
        print(f"Model OCR không tồn tại: {ocr_path}", file=sys.stderr)
        sys.exit(1)

    try:
        det_model = torch.hub.load("ultralytics/yolov5", "custom", path=det_path, source="github")
        ocr_model = torch.hub.load("ultralytics/yolov5", "custom", path=ocr_path, source="github")
    except Exception as e:
        print(
            f"Lỗi load YOLOv5 models qua torch.hub: {e}\n"
            "Kiểm tra kết nối mạng lần đầu load và phiên bản torch.",
            file=sys.stderr,
        )
        sys.exit(1)

    return det_model, ocr_model


def extract_yolov5_boxes(result):
    """
    Chuẩn hóa output YOLOv5 về list (x1, y1, x2, y2, conf, cls_id).
    Hỗ trợ cả API mới (boxes) và cũ (xyxy).
    """
    boxes = []
    if result is None:
        return boxes

    # API mới
    if hasattr(result, "boxes") and result.boxes is not None:
        for b in result.boxes:
            x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
            conf = float(b.conf[0])
            cls_id = int(b.cls[0])
            boxes.append((x1, y1, x2, y2, conf, cls_id))
        return boxes

    # API cũ
    if hasattr(result, "xyxy") and len(result.xyxy) > 0:
        xyxy = result.xyxy[0].cpu().numpy()
        for row in xyxy:
            x1, y1, x2, y2, conf, cls_id = row.tolist()
            boxes.append((float(x1), float(y1), float(x2), float(y2), float(conf), int(cls_id)))
    return boxes


def sort_ocr_boxes(result) -> List[Tuple[str, float, float, float, float]]:
    """Lấy các bbox OCR và sort từ trái sang phải."""
    detections = []
    if result is None:
        return detections

    boxes = extract_yolov5_boxes(result)
    names = getattr(result, "names", {})

    for x1, y1, x2, y2, conf, cls_id in boxes:
        label = names.get(cls_id, str(cls_id))
        cx = (x1 + x2) / 2.0
        detections.append((label, conf, cx, x1, x2))

    detections.sort(key=lambda x: x[2])
    return detections


def ocr_text_from_crop(ocr_model, crop: np.ndarray, conf: float) -> str:
    """Chạy OCR model trên crop biển số và ghép chuỗi ký tự."""
    if crop is None or crop.size == 0:
        return ""

    results = ocr_model(crop, size=max(crop.shape[:2]))
    result = results if hasattr(results, "xyxy") else (results[0] if results else None)

    detections = [d for d in sort_ocr_boxes(result) if d[1] >= conf]
    text = "".join([d[0] for d in detections])
    return text


def draw_plate_box(frame: np.ndarray, box, text: str) -> None:
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = text if text else "plate"
    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def main() -> int:
    args = parse_args()

    det_model, ocr_model = load_models(args.det_model, args.ocr_model)

    # Nếu source là file ảnh → xử lý 1 ảnh rồi thoát
    if Path(args.source).is_file():
        img = cv2.imread(args.source)
        if img is None:
            print(f"Không đọc được ảnh: {args.source}", file=sys.stderr)
            return 1

        det_results = det_model(img, size=max(img.shape[:2]))
        boxes = [
            b for b in extract_yolov5_boxes(det_results)
            if b[4] >= args.det_conf
        ][: args.max_dets]

        for x1f, y1f, x2f, y2f, conf, cls_id in boxes:
            x1, y1, x2, y2 = map(int, [x1f, y1f, x2f, y2f])
            h, w = img.shape[:2]
            pad = 4
            x1p = max(0, x1 - pad)
            y1p = max(0, y1 - pad)
            x2p = min(w, x2 + pad)
            y2p = min(h, y2 + pad)
            plate_crop = img[y1p:y2p, x1p:x2p]
            plate_text = ocr_text_from_crop(ocr_model, plate_crop, args.ocr_conf)
            draw_plate_box(img, (x1, y1, x2, y2), plate_text)
            print(
                f"[DETECT] plate_text='{plate_text}' "
                f"det_conf={conf:.2f} bbox=({x1},{y1},{x2},{y2})",
                flush=True,
            )

        cv2.imshow("LP detect + OCR (image)", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return 0

    # Ngược lại: coi là camera index hoặc video
    device = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f"Cannot open camera/video {args.source}", file=sys.stderr)
        return 1

    window_name = "License Plate Detect + OCR (press q to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    fps_time = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame", file=sys.stderr)
                break

            det_results = det_model(frame, size=max(args.width, args.height))
            det_boxes = [
                b for b in extract_yolov5_boxes(det_results)
                if b[4] >= args.det_conf
            ][: args.max_dets]

            for x1f, y1f, x2f, y2f, conf, cls_id in det_boxes:
                x1, y1, x2, y2 = map(int, [x1f, y1f, x2f, y2f])
                pad = 4
                h, w = frame.shape[:2]
                x1p = max(0, x1 - pad)
                y1p = max(0, y1 - pad)
                x2p = min(w, x2 + pad)
                y2p = min(h, y2 + pad)
                plate_crop = frame[y1p:y2p, x1p:x2p]
                plate_text = ocr_text_from_crop(ocr_model, plate_crop, args.ocr_conf)
                draw_plate_box(frame, (x1, y1, x2, y2), plate_text)
                print(
                    f"[DETECT] plate_text='{plate_text}' "
                    f"det_conf={conf:.2f} bbox=({x1},{y1},{x2},{y2})",
                    flush=True,
                )

            now = time.time()
            fps = 1.0 / (now - fps_time)
            fps_time = now
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


