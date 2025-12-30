# Hướng Dẫn Chạy Detection với 3 Phương Pháp Song Song

## Tổng Quan

Dự án hỗ trợ 3 phương pháp song song hóa dữ liệu cho Sobel filter:
1. **OpenMP**: CPU multi-threading (cơ bản)
2. **SIMD**: AVX-256 vectorization + OpenMP (tối ưu CPU)
3. **CUDA**: GPU parallelization (nhanh nhất)

## Cách 1: Chạy Tự Động (Menu)

```bash
bash scripts/run_all_methods.sh [video_source]
```

Ví dụ:
```bash
# Chạy với webcam
bash scripts/run_all_methods.sh 0

# Chạy với video file
bash scripts/run_all_methods.sh data/video.mp4
```

Script sẽ hiển thị menu để bạn chọn:
- Chọn 1: Chạy OpenMP
- Chọn 2: Chạy SIMD
- Chọn 3: Chạy CUDA
- Chọn 4: Chạy tất cả 3 phương pháp lần lượt (mỗi cái 30 giây)

## Cách 2: Chạy Từng Binary Riêng

### Bước 1: Build tất cả versions

```bash
# Build OpenMP
bash scripts/build_openmp.sh

# Build SIMD
bash scripts/build_simd.sh

# Build CUDA
bash scripts/build.sh
```

### Bước 2: Chạy từng binary

```bash
# 1. OpenMP (CPU multi-threading)
./lp_main_openmp 0

# 2. SIMD + OpenMP (AVX-256 vectorization)
./lp_main_simd 0

# 3. CUDA (GPU)
./lp_main_cuda 0
```

**Lưu ý**: Nhấn `q` trong cửa sổ video để thoát.

## Cách 3: Chạy Benchmark Tự Động

Để so sánh hiệu năng tự động và tạo báo cáo:

```bash
# Chạy benchmark (30 giây mỗi test)
bash scripts/benchmark.sh

# Hoặc chỉ định thời gian
bash scripts/benchmark.sh 0 60  # 60 giây mỗi test
```

Kết quả sẽ được lưu tại `benchmark_results/` với báo cáo chi tiết.

## So Sánh Hiệu Năng

Dựa trên kết quả benchmark (ảnh 800x600):

| Phương Pháp | Thời Gian Sobel | FPS | Speedup |
|------------|-----------------|-----|---------|
| OpenMP     | 7.14 ms         | 140 FPS | 1.0x (baseline) |
| SIMD       | 1.89 ms         | 530 FPS | 3.79x |
| CUDA       | 0.49 ms         | 2025 FPS | 14.47x |

## Lưu Ý

1. **OpenMP**: Hoạt động trên mọi CPU, không cần GPU
2. **SIMD**: Yêu cầu CPU hỗ trợ AVX2 (hầu hết CPU hiện đại)
3. **CUDA**: Yêu cầu GPU NVIDIA và CUDA toolkit

## Kiểm Tra Binary Đã Build

```bash
ls -lh lp_main*
```

Kết quả mong đợi:
- `lp_main_openmp`: OpenMP version
- `lp_main_simd`: SIMD version  
- `lp_main_cuda`: CUDA version

## Troubleshooting

### Lỗi: Binary không tìm thấy
```bash
# Build lại binary cần thiết
bash scripts/build_openmp.sh  # hoặc build_simd.sh, build.sh
```

### Lỗi: CUDA không hoạt động
- Kiểm tra GPU: `nvidia-smi`
- Kiểm tra CUDA: `nvcc --version`
- Nếu không có GPU, chỉ dùng OpenMP hoặc SIMD

### Lỗi: SIMD không hoạt động
- Kiểm tra CPU hỗ trợ AVX2: `grep avx2 /proc/cpuinfo`
- Nếu không hỗ trợ, dùng OpenMP

