# Tổng quan song song hóa trong dự án

Dự án áp dụng cả **Data Parallelism** và **Task Parallelism** để tăng tốc phát hiện & đọc biển số. Dưới đây là chi tiết từng kỹ thuật, vị trí mã, cách bật/tắt và mục tiêu hiệu năng.

---

## Task Parallelism (Pipeline đa luồng)

### Kiến trúc & luồng dữ liệu
- File: `src/parallel_pipeline.cpp`, `include/parallel_pipeline.h`
- 4 threads chạy đồng thời, nối nhau qua các queue an toàn luồng:
  1) **Capture**: đọc RTSP liên tục, tính `edge_score` (CUDA/CPU), Frame Gating, đẩy frame vào detection queue và display queue.
  2) **Detection**: YOLOv9 trên frames từ detection queue; lưu kết quả vào buffer, đẩy packet có detections sang OCR queue.
  3) **OCR**: crop ROI từ detections, chạy OCR, cập nhật buffer theo `frame_id`.
  4) **Display**: lấy frame từ display queue, ghép kết quả từ buffer (theo `frame_id`), vẽ bbox + text.

### Đồng bộ & chia sẻ
- Mỗi queue có mutex + condition variable riêng.
- `results_buffer_` (map `frame_id` → `FramePacket`) lưu detection/OCR để Display ghép đúng frame.
- Giới hạn queue nhỏ (`MAX_QUEUE_SIZE`) để giảm trễ và tránh backlog.

### Bật / tắt
- Mặc định bật qua script `scripts/run_test_onnx.sh` (đặt `USE_PARALLEL_PIPELINE=1` và tự thêm `--parallel`).
- CLI: `--parallel` / `--sequential` hoặc env `USE_PARALLEL_PIPELINE=1|0`.

### Mục tiêu hiệu năng
- Overlap Capture/Detection/OCR/Display → giảm latency end-to-end.
- Giữ FPS ổn định khi model tốn thời gian inference.

---

## Data Parallelism

### 1) CUDA
- File: `cuda/sobel_cuda.cu` (gọi từ `calculate_edge_score()` trong `src/image_processing.cpp`).
- Kernel `count_edge_pixels`: mỗi thread xử lý 1 pixel, `atomicAdd` đếm edge vượt ngưỡng.
- Dùng cho Frame Gating: bỏ qua frame “ít cạnh” để giảm tải YOLO.

### 2) OpenMP
- Bật qua Makefile: `-fopenmp`; tự kích hoạt nếu compiler hỗ trợ.
- Các vòng lặp song song:
  - **OCR preprocessing** `src/license_plate_detector.cpp`: HWC → tensor (uint8)  
    `#pragma omp parallel for collapse(2)`
  - **Letterbox preprocessing** `src/image_processing.cpp`: HWC → CHW + normalize  
    `#pragma omp parallel for collapse(3)`
  - **CPU edge counting fallback** `src/image_processing.cpp`: khi CUDA không dùng được  
    `#pragma omp parallel for reduction(+:edge_pixels) collapse(2)`

### 3) ONNX Runtime threading
- File: `src/license_plate_detector.cpp`
  - `SetIntraOpNumThreads(8)`: song song hóa bên trong node (matmul/conv).
  - `SetInterOpNumThreads(4)`: chạy nhiều node song song nếu đồ thị cho phép.
  - `GraphOptimizationLevel::ORT_ENABLE_EXTENDED`: bật tối ưu đồ thị.

---

## Fallback & điều khiển
- **Pipeline mode**: `USE_PARALLEL_PIPELINE` hoặc `--parallel/--sequential`.
- **CUDA**: tự động nếu GPU/driver khả dụng; nếu lỗi hoặc `edge=0` → fallback CPU (OpenMP).
- **OpenMP**: tự bật qua Makefile.
- **Gating**: `edge_score` < ngưỡng thấp → bỏ frame; ngưỡng cao → bắt buộc detect; vùng giữa → detect.

---

## Tóm tắt nhanh (theo tầng song song)
- **Task Parallelism**: 4 threads Pipeline (Capture / Detection / OCR / Display) với buffer sync theo `frame_id`.
- **Data Parallelism**:
  - CUDA: Sobel edge counting cho Frame Gating.
  - OpenMP: OCR preprocessing, letterbox preprocessing, CPU edge counting fallback.
  - ONNX Runtime: IntraOp=8, InterOp=4, graph optimizations.

---

## Cách chạy (song song bật sẵn)s
```bash
# Mặc định parallel pipeline
./scripts/run_test_onnx.sh

# Hoặc bật/tắt thủ công
USE_PARALLEL_PIPELINE=1 ./scripts/run_test_onnx.sh --parallel
USE_PARALLEL_PIPELINE=0 ./scripts/run_test_onnx.sh --sequential
```

