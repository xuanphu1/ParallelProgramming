#!/bin/bash
# Build với OpenMP only (không SIMD, không CUDA)

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build_openmp"
ONNX_RUNTIME_DIR="/tmp/onnxruntime-linux-x64-1.16.3"

echo "=========================================="
echo "Build Pipeline - OpenMP Only"
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
echo "Compiling C++ sources (OpenMP only)..."
echo "=========================================="

CXX_FLAGS="-std=c++17 -O3 -Wall"
CXX_FLAGS="$CXX_FLAGS -fopenmp"  # OpenMP

INCLUDES="-I$PROJECT_DIR/include"
INCLUDES="$INCLUDES -I$ONNX_RUNTIME_DIR/include"
INCLUDES="$INCLUDES $(pkg-config --cflags opencv4)"

SOURCES=(
    "$PROJECT_DIR/src/main.cpp"
    "$PROJECT_DIR/src/Pipeline.cpp"
    "$PROJECT_DIR/src/LPDetector.cpp"
    "$PROJECT_DIR/src/LPOCR.cpp"
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

# Build command (không có CUDA, không có SIMD flag)
BUILD_CMD="g++ $CXX_FLAGS $INCLUDES"
BUILD_CMD="$BUILD_CMD ${SOURCES[@]}"
BUILD_CMD="$BUILD_CMD -o $PROJECT_DIR/lp_main_openmp $LDFLAGS"

echo "Build command:"
echo "$BUILD_CMD" | sed 's/ -/ \\\n    -/g'
echo ""

eval $BUILD_CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Build thành công (OpenMP only)!"
    echo "=========================================="
    echo ""
    echo "Binary: $PROJECT_DIR/lp_main_openmp"
    echo ""
else
    echo ""
    echo "❌ Build thất bại!"
    exit 1
fi

