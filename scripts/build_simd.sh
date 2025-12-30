#!/bin/bash
# Build với SIMD + OpenMP (không CUDA)

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build_simd"
ONNX_RUNTIME_DIR="/tmp/onnxruntime-linux-x64-1.16.3"

echo "=========================================="
echo "Build Pipeline - SIMD + OpenMP"
echo "=========================================="
echo ""

# Kiểm tra ONNX Runtime
if [ ! -d "$ONNX_RUNTIME_DIR" ]; then
    echo "❌ ONNX Runtime không tìm thấy tại: $ONNX_RUNTIME_DIR"
    exit 1
fi

echo "✅ ONNX Runtime: $ONNX_RUNTIME_DIR"
echo ""

# Tạo thư mục build
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Compile C++ sources
echo "=========================================="
echo "Compiling C++ sources (SIMD + OpenMP)..."
echo "=========================================="

CXX_FLAGS="-std=c++17 -O3 -Wall"
CXX_FLAGS="$CXX_FLAGS -fopenmp"  # OpenMP
CXX_FLAGS="$CXX_FLAGS -mavx2"    # AVX-256 for SIMD
CXX_FLAGS="$CXX_FLAGS -DUSE_SIMD_SOBEL"  # Flag để dùng SIMD

INCLUDES="-I$PROJECT_DIR/include"
INCLUDES="$INCLUDES -I$ONNX_RUNTIME_DIR/include"
INCLUDES="$INCLUDES $(pkg-config --cflags opencv4)"

SOURCES=(
    "$PROJECT_DIR/src/main.cpp"
    "$PROJECT_DIR/src/Pipeline.cpp"
    "$PROJECT_DIR/src/LPDetector.cpp"
    "$PROJECT_DIR/src/LPOCR.cpp"
    "$PROJECT_DIR/src/SobelSIMD.cpp"
)

LDFLAGS="$(pkg-config --libs opencv4)"
LDFLAGS="$LDFLAGS -L$ONNX_RUNTIME_DIR/lib -lonnxruntime"
LDFLAGS="$LDFLAGS -Wl,-rpath,$ONNX_RUNTIME_DIR/lib"

# OpenMP linking
if pkg-config --exists openmp; then
    LDFLAGS="$LDFLAGS $(pkg-config --libs openmp)"
else
    LDFLAGS="$LDFLAGS -fopenmp"
fi

# Build command
BUILD_CMD="g++ $CXX_FLAGS $INCLUDES"
BUILD_CMD="$BUILD_CMD ${SOURCES[@]}"
BUILD_CMD="$BUILD_CMD -o $PROJECT_DIR/lp_main_simd $LDFLAGS"

echo "Build command:"
echo "$BUILD_CMD" | sed 's/ -/ \\\n    -/g'
echo ""

eval $BUILD_CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Build thành công (SIMD + OpenMP)!"
    echo "=========================================="
    echo ""
    echo "Binary: $PROJECT_DIR/lp_main_simd"
    echo ""
else
    echo ""
    echo "❌ Build thất bại!"
    exit 1
fi

