# Hướng Dẫn Build Dự Án

## Các Script Build

### 1. `build.sh` - Build với CUDA (khuyến nghị)

Script này tự động:
- ✅ Phát hiện CUDA compiler (nvcc)
- ✅ Build với CUDA nếu có
- ✅ Fallback về SIMD + OpenMP nếu không có CUDA
- ✅ Tự động detect GPU architecture

**Cách dùng:**
```bash
./build.sh
```

**Yêu cầu:**
- CUDA toolkit (nếu muốn dùng CUDA)
- ONNX Runtime tại `/tmp/onnxruntime-linux-x64-1.16.3`
- OpenCV development libraries

### 2. `build_no_cuda.sh` - Build không có CUDA

Script này build chỉ với:
- ✅ SIMD AVX-256
- ✅ OpenMP

**Cách dùng:**
```bash
./build_no_cuda.sh
```

**Yêu cầu:**
- ONNX Runtime tại `/tmp/onnxruntime-linux-x64-1.16.3`
- OpenCV development libraries
- CPU hỗ trợ AVX2

## Thứ Tự Ưu Tiên Sobel Filter

Khi chạy, pipeline sẽ thử các phương pháp theo thứ tự:

1. **CUDA** (nếu build với `USE_CUDA_SOBEL`)
   - Nhanh nhất: ~0.5ms/frame
   - Yêu cầu GPU NVIDIA

2. **SIMD AVX-256** (fallback)
   - Nhanh: ~1.9ms/frame
   - Yêu cầu CPU hỗ trợ AVX2

3. **OpenMP Scalar** (fallback cuối)
   - Chậm nhất: ~7ms/frame
   - Hoạt động trên mọi CPU

## Kiểm Tra Build

Sau khi build, kiểm tra:
```bash
./lp_main --help  # Nếu có
# hoặc
./lp_main 0       # Chạy với camera
```

Log sẽ hiển thị:
- `[Pipeline] CUDA support: ENABLED` - Nếu build với CUDA
- `[Pipeline] CUDA support: DISABLED` - Nếu không có CUDA

## Troubleshooting

### Lỗi: CUDA không tìm thấy
```bash
# Kiểm tra CUDA
which nvcc
nvcc --version

# Nếu không có, cài CUDA toolkit hoặc dùng build_no_cuda.sh
```

### Lỗi: ONNX Runtime không tìm thấy
```bash
# Tải ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
tar -xzf onnxruntime-linux-x64-1.16.3.tgz -C /tmp/
```

### Lỗi: OpenCV không tìm thấy
```bash
sudo apt-get install libopencv-dev
# hoặc
pkg-config --modversion opencv4
```

