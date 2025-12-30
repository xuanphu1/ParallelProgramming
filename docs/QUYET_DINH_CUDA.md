# QUYẾT ĐỊNH: SỬ DỤNG CUDA CHO SOBEL FILTER

**Ngày quyết định**: 30/12/2024  
**Trạng thái**: ✅ Đã chốt và triển khai

---

## 1. QUYẾT ĐỊNH

**Phương pháp được chọn**: **CUDA (GPU Parallelization)**

**Lý do**:
- Hiệu năng cao nhất: **14.47x** so với OpenMP baseline
- Phù hợp cho production system với yêu cầu real-time
- Hệ thống đã có GPU NVIDIA (Quadro K2100M) và CUDA toolkit

---

## 2. THAY ĐỔI CODE

### 2.1. Đã Loại Bỏ
- ❌ OpenMP-only implementation
- ❌ SIMD implementation  
- ❌ Fallback mechanisms
- ❌ Enum `SobelMethod` và switch logic

### 2.2. Code Hiện Tại
- ✅ **Chỉ dùng CUDA** cho Sobel filter
- ✅ Build bắt buộc phải có CUDA support
- ✅ Error nếu không có CUDA: `USE_CUDA_SOBEL must be defined`

### 2.3. Files Đã Sửa
- `src/Pipeline.cpp`: Xóa switch logic, chỉ dùng `sobelCuda()`
- `include/Pipeline.h`: Xóa enum và các method không cần thiết
- `scripts/build.sh`: Output về `lp_main` (không phải `lp_main_cuda`)

---

## 3. CÁCH SỬ DỤNG

### Build
```bash
bash scripts/build.sh
```

### Chạy
```bash
./lp_main 0              # Webcam
./lp_main data/image.jpg # Ảnh
```

### Yêu Cầu
- ✅ GPU NVIDIA
- ✅ CUDA toolkit
- ✅ CUDA driver

---

## 4. HIỆU NĂNG

| Metric | Value |
|--------|-------|
| **Thời gian Sobel** | 0.494 ms |
| **FPS Sobel** | 2025.5 FPS |
| **Speedup vs OpenMP** | 14.47x |
| **Speedup vs SIMD** | 3.82x |

---

## 5. LƯU Ý

### Nếu Không Có GPU
- Code sẽ **không build được** (bắt buộc CUDA)
- Cần có GPU NVIDIA để chạy

### Nếu Cần Fallback
- Có thể tạo branch riêng với fallback logic
- Hoặc dùng các build scripts cũ (`build_openmp.sh`, `build_simd.sh`)

---

## 6. TÀI LIỆU LIÊN QUAN

- `docs/BAO_CAO_DANH_GIA.md`: Báo cáo đánh giá chi tiết
- `docs/benchmark_results.txt`: Kết quả benchmark
- `scripts/build.sh`: Build script chính

---

**Tác giả**: AI Assistant  
**Phiên bản**: 1.0  
**Ngày cập nhật**: 30/12/2024

