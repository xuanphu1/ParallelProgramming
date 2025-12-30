# Hướng Dẫn Benchmark So Sánh Các Phương Pháp Song Song Hóa

## Tổng Quan

Script này cho phép so sánh hiệu năng của 3 phương pháp song song hóa dữ liệu cho Sobel filter:
1. **OpenMP**: Chỉ dùng OpenMP (multi-threading CPU)
2. **SIMD**: SIMD vectorization (AVX-256) + OpenMP
3. **CUDA**: GPU parallelization với CUDA

## Cách Sử Dụng

### 1. Chạy Benchmark Tự Động

```bash
# Chạy benchmark với webcam (mặc định 30 giây mỗi test)
bash scripts/benchmark.sh

# Chạy với video file
bash scripts/benchmark.sh path/to/video.mp4

# Chỉ định thời gian chạy mỗi test (giây)
bash scripts/benchmark.sh 0 60  # Webcam, 60 giây mỗi test
```

### 2. Build Từng Phương Pháp Riêng Lẻ

```bash
# Build OpenMP only
bash scripts/build_openmp.sh

# Build SIMD + OpenMP
bash scripts/build_simd.sh

# Build CUDA
bash scripts/build.sh
```

### 3. Chạy Từng Binary Riêng

```bash
# OpenMP
./lp_main_openmp 0

# SIMD
./lp_main_simd 0

# CUDA
./lp_main_cuda 0
```

## Kết Quả

Sau khi chạy benchmark, kết quả sẽ được lưu tại:
- `benchmark_results/benchmark_YYYYMMDD_HHMMSS.txt`: File kết quả raw
- `benchmark_results/benchmark_YYYYMMDD_HHMMSS_report.txt`: Báo cáo so sánh

## Định Dạng Báo Cáo

Báo cáo bao gồm:
1. **FPS (Frames Per Second)**: Avg, Min, Max
2. **Thời gian Sobel**: Thời gian trung bình xử lý Sobel filter (ms)
3. **Speedup**: So sánh với OpenMP (baseline)
4. **Kết luận**: Phân tích và đánh giá

## Lưu Ý

- Đảm bảo có webcam hoặc video file để test
- CUDA version yêu cầu GPU NVIDIA và CUDA toolkit
- SIMD version yêu cầu CPU hỗ trợ AVX2
- Mỗi test sẽ chạy trong thời gian chỉ định (mặc định 30 giây)
- Nhấn 'q' trong cửa sổ video để dừng sớm

## Ví Dụ Output

```
==========================================
KẾT QUẢ BENCHMARK
==========================================

1. FPS (Frames Per Second)
----------------------------------------
Method     |   Avg FPS |   Min FPS |   Max FPS | Avg Sobel (ms)
----------------------------------------
OpenMP     |      5.23 |      4.10 |      6.50 |          12.45
SIMD       |      8.15 |      6.20 |     10.30 |           7.89
CUDA       |     15.42 |     12.50 |     18.20 |           2.34

2. PHÂN TÍCH HIỆU NĂNG
----------------------------------------
Phương pháp có FPS cao nhất: CUDA (15.42 FPS)
Phương pháp có thời gian Sobel nhanh nhất: CUDA (2.34 ms)

3. SPEEDUP SO VỚI OPENMP (Baseline)
----------------------------------------
SIMD: 1.56x (8.15 FPS / 5.23 FPS)
CUDA: 2.95x (15.42 FPS / 5.23 FPS)
```

