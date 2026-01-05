# DetectLicense - License Plate Detection với ONNX Models

Dự án phát hiện và đọc biển số xe sử dụng YOLOv9 (detection) và CCT (OCR) với ONNX Runtime.

## Cấu trúc dự án

```
DetectLicense/
├── include/              # Header files (.h)
│   ├── config.h         # Cấu hình global (RTSP, Gamma, Sobel)
│   ├── types.h          # Structs (OCRConfig, Detection, OCRResult)
│   ├── utils.h          # Utility functions (find_models, load_config)
│   ├── image_processing.h # Image processing (gamma, sobel)
│   ├── license_plate_detector.h # Class LicensePlateDetector
│   └── rtsp_client.h    # RTSP connection
│
├── src/                 # Source files (.cpp)
│   ├── main.cpp         # Main function - vòng lặp RTSP và hiển thị
│   ├── config.cpp       # Định nghĩa config variables
│   ├── utils.cpp        # Implementation utility functions
│   ├── image_processing.cpp # Implementation image processing
│   ├── license_plate_detector.cpp # Implementation detector class
│   └── rtsp_client.cpp  # Implementation RTSP connection
│
├── cuda/                # CUDA code
│   ├── sobel_cuda.cu    # CUDA kernel cho Sobel edge counting
│   └── sobel_cuda.h     # Header cho CUDA functions
│
├── model/               # Models và config
│   ├── yolo-v9-t-640-license-plates-end2end.onnx
│   ├── cct_s_v1_global.onnx
│   └── cct_s_v1_global_plate_config.yaml
│
├── scripts/            # Scripts
│   └── run_test_onnx.sh # Script chạy với LD_LIBRARY_PATH đúng
│
├── Makefile            # Build configuration
└── README.md           # File này
```

## Build

```bash
make clean
make
```

## Chạy

```bash
# Sử dụng script (tự động set LD_LIBRARY_PATH)
./scripts/run_test_onnx.sh

# Hoặc chạy trực tiếp
LD_LIBRARY_PATH=onnxruntime-linux-x64-1.16.3/lib:$LD_LIBRARY_PATH ./test_onnx_models
```

## Cấu hình

Các biến cấu hình có thể được override từ environment variables:

- `USE_GAMMA_CORRECTION`: Bật/tắt Gamma Correction
- `GAMMA_VALUE`: Giá trị gamma (mặc định: 1.5)
- `USE_SOBEL_GATING`: Bật/tắt Sobel Frame Gating
- `USE_SOBEL_OCR_ENHANCEMENT`: Bật/tắt Sobel Edge Enhancement cho OCR
- `SOBEL_ENHANCEMENT_STRENGTH`: Độ mạnh của edge enhancement (0.0-1.0)

Ví dụ:
```bash
USE_SOBEL_GATING=1 USE_SOBEL_OCR_ENHANCEMENT=1 ./test_onnx_models
```

## Tính năng

- ✅ Detection biển số với YOLOv9
- ✅ OCR với CCT model
- ✅ RTSP stream support
- ✅ Sobel Frame Gating (bỏ qua frame không cần thiết)
- ✅ Sobel Edge Enhancement cho OCR
- ✅ Gamma Correction cho OCR preprocessing
- ✅ CUDA acceleration cho Sobel edge counting
- ✅ Fake FPS display khi edge_score < 0.050

## Dependencies

- OpenCV 4.x
- ONNX Runtime 1.16.3+
- CUDA Toolkit (cho Sobel acceleration)
- g++ với C++17 support
- nvcc (CUDA compiler)

