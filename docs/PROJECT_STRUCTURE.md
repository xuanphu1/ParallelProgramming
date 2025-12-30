# Cấu Trúc Dự Án

## Tổng Quan

```
.
├── src/              # Source code C++ (.cpp, .cu)
├── include/          # Header files (.h)
├── build/            # Build artifacts (binary, object files)
├── scripts/          # Build scripts và utility scripts
├── docs/             # Documentation
│   ├── Report/       # Báo cáo LaTeX
│   └── *.md          # Các file markdown
├── models/           # Model files (.onnx, .pt)
├── data/             # Test data (ảnh, video)
├── tests/            # Test files và benchmarks
├── README.md         # Hướng dẫn chính
└── .gitignore        # Git ignore rules
```

## Chi Tiết

### `src/` - Source Code
- `main.cpp` - Entry point
- `Pipeline.cpp` - Pipeline đa luồng
- `LPDetector.cpp` - Phát hiện biển số
- `LPOCR.cpp` - OCR ký tự
- `SobelSIMD.cpp` - Sobel với SIMD
- `SobelCuda.cu` - Sobel với CUDA

### `include/` - Headers
- `Pipeline.h`
- `LPDetector.h`
- `LPOCR.h`
- `SobelSIMD.h`
- `SobelCuda.h`

### `scripts/` - Build Scripts
- `build.sh` - Build với CUDA (nếu có)
- `build_no_cuda.sh` - Build không có CUDA
- `export_torchscript.py` - Export YOLOv5 sang ONNX
- `run_models.py` - Test models

### `docs/` - Documentation
- `Report/` - Báo cáo LaTeX
- `README_BUILD.md` - Hướng dẫn build
- `PARALLEL_OPTIMIZATION_IDEAS.md` - Ý tưởng song song hóa
- `IMPLEMENTED_PARALLELISM.md` - Các kỹ thuật đã triển khai
- `benchmark_results.txt` - Kết quả benchmark

### `models/` - Model Files
- `lp_detector_ts.onnx` - Model detection
- `lp_ocr_ts.onnx` - Model OCR
- `LP_detector_nano_61.pt` - YOLOv5 checkpoint (detection)
- `LP_ocr_nano_62.pt` - YOLOv5 checkpoint (OCR)

### `data/` - Test Data
- `Bienso.jpg` - Ảnh test biển số
- Các ảnh/video test khác

### `tests/` - Test Files
- `benchmark_sobel.cpp` - Benchmark Sobel
- `benchmark_sobel_cuda.cpp` - Benchmark Sobel với CUDA

### `build/` - Build Artifacts
- `lp_main` - Binary chính
- `*.o` - Object files
- `benchmark_sobel*` - Benchmark binaries

## Build Output

Sau khi build, binary sẽ ở: `build/lp_main`

Chạy:
```bash
./build/lp_main 0              # Camera
./build/lp_main data/image.jpg # Ảnh
```

