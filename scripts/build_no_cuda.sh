#!/bin/bash
# Script build dự án KHÔNG có CUDA (chỉ SIMD + OpenMP)

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ONNX_RUNTIME_DIR="/tmp/onnxruntime-linux-x64-1.16.3"

echo "=========================================="
echo "Build Pipeline (NO CUDA) - SIMD + OpenMP"
echo "=========================================="
echo ""

# Kiểm tra ONNX Runtime
if [ ! -d "$ONNX_RUNTIME_DIR" ]; then
    echo "❌ ONNX Runtime không tìm thấy tại: $ONNX_RUNTIME_DIR"
    exit 1
fi

echo "✅ ONNX Runtime: $ONNX_RUNTIME_DIR"
echo ""

# Build
g++ -std=c++17 -fopenmp -mavx2 -O3 \
    -I"$PROJECT_DIR/include" \
    -I"$ONNX_RUNTIME_DIR/include" \
    "$PROJECT_DIR/src/main.cpp" \
    "$PROJECT_DIR/src/Pipeline.cpp" \
    "$PROJECT_DIR/src/LPDetector.cpp" \
    "$PROJECT_DIR/src/LPOCR.cpp" \
    "$PROJECT_DIR/src/SobelSIMD.cpp" \
    -o "$PROJECT_DIR/build/lp_main" \
    $(pkg-config --cflags --libs opencv4) \
    -L"$ONNX_RUNTIME_DIR/lib" -lonnxruntime \
    -Wl,-rpath,"$ONNX_RUNTIME_DIR/lib" \
    -fopenmp

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build thành công (SIMD + OpenMP, không có CUDA)"
    echo "   Binary: $PROJECT_DIR/build/lp_main"
else
    echo "❌ Build thất bại!"
    exit 1
fi

