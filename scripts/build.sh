#!/bin/bash
# Script build dự án với hỗ trợ CUDA (nếu có)

set -e  # Dừng nếu có lỗi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build"
ONNX_RUNTIME_DIR="/tmp/onnxruntime-linux-x64-1.16.3"

echo "=========================================="
echo "Build Pipeline - License Plate Detection"
echo "=========================================="
echo ""

# Kiểm tra CUDA
HAS_CUDA=false
CUDA_ARCH="sm_50"  # Default architecture, có thể thay đổi

if command -v nvcc &> /dev/null; then
    HAS_CUDA=true
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    echo "✅ CUDA compiler tìm thấy: version $CUDA_VERSION"
    
    # Xác định CUDA architecture dựa trên GPU (nếu có)
    if command -v nvidia-smi &> /dev/null; then
        GPU_MODEL=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
        if [ ! -z "$GPU_MODEL" ]; then
            echo "   GPU Compute Capability: $GPU_MODEL"
            # Convert compute capability to architecture (ví dụ: 7.5 -> sm_75)
            MAJOR=$(echo $GPU_MODEL | cut -d. -f1)
            MINOR=$(echo $GPU_MODEL | cut -d. -f2)
            CUDA_ARCH="sm_${MAJOR}${MINOR}"
            echo "   Sử dụng CUDA architecture: $CUDA_ARCH"
        fi
    fi
    echo ""
else
    echo "⚠️  CUDA compiler không tìm thấy"
    echo "   Sẽ build không có CUDA support"
    echo ""
fi

# Kiểm tra ONNX Runtime
if [ ! -d "$ONNX_RUNTIME_DIR" ]; then
    echo "❌ ONNX Runtime không tìm thấy tại: $ONNX_RUNTIME_DIR"
    echo "   Tải từ: https://github.com/microsoft/onnxruntime/releases"
    echo "   Giải nén vào: $ONNX_RUNTIME_DIR"
    exit 1
fi

echo "✅ ONNX Runtime: $ONNX_RUNTIME_DIR"
echo ""

# Tạo thư mục build
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Compile CUDA nếu có
CUDA_OBJECT=""
if [ "$HAS_CUDA" = true ]; then
    echo "=========================================="
    echo "Compiling CUDA kernel..."
    echo "=========================================="
    
    nvcc -c "$PROJECT_DIR/src/SobelCuda.cu" \
         -o "$BUILD_DIR/SobelCuda.o" \
         -I/usr/local/cuda/include \
         $(pkg-config --cflags opencv4) \
         -arch=$CUDA_ARCH \
         -O3 \
         2>&1 | tee "$BUILD_DIR/cuda_compile.log"
    
    if [ $? -eq 0 ]; then
        CUDA_OBJECT="$BUILD_DIR/SobelCuda.o"
        CUDA_FLAGS="-DUSE_CUDA_SOBEL"
        CUDA_LIBS="-L/usr/local/cuda/lib64 -lcudart"
        echo "✅ CUDA kernel compiled successfully"
        echo ""
    else
        echo "⚠️  CUDA compilation failed, building without CUDA"
        HAS_CUDA=false
        echo ""
    fi
fi

# Compile C++ sources
echo "=========================================="
echo "Compiling C++ sources..."
echo "=========================================="

CXX_FLAGS="-std=c++17 -O3 -Wall"
CXX_FLAGS="$CXX_FLAGS -fopenmp"  # OpenMP
CXX_FLAGS="$CXX_FLAGS -mavx2"    # AVX-256 for SIMD

INCLUDES="-I$PROJECT_DIR/include"
INCLUDES="$INCLUDES -I$ONNX_RUNTIME_DIR/include"
if [ "$HAS_CUDA" = true ]; then
    INCLUDES="$INCLUDES -I/usr/local/cuda/include"
fi

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

if [ "$HAS_CUDA" = true ] && [ ! -z "$CUDA_OBJECT" ]; then
    LDFLAGS="$LDFLAGS $CUDA_LIBS"
fi

# OpenMP linking
if pkg-config --exists openmp; then
    LDFLAGS="$LDFLAGS $(pkg-config --libs openmp)"
else
    LDFLAGS="$LDFLAGS -fopenmp"
fi

# Build command
BUILD_CMD="g++ $CXX_FLAGS $CUDA_FLAGS $INCLUDES"
BUILD_CMD="$BUILD_CMD ${SOURCES[@]}"
if [ ! -z "$CUDA_OBJECT" ]; then
    BUILD_CMD="$BUILD_CMD $CUDA_OBJECT"
fi
BUILD_CMD="$BUILD_CMD -o $PROJECT_DIR/lp_main $LDFLAGS"

echo "Build command:"
echo "$BUILD_CMD" | sed 's/ -/ \\\n    -/g'
echo ""

eval $BUILD_CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Build thành công!"
    echo "=========================================="
    echo ""
    echo "Binary: $PROJECT_DIR/build/lp_main"
    echo ""
    
    if [ "$HAS_CUDA" = true ] && [ ! -z "$CUDA_OBJECT" ]; then
        echo "✅ CUDA support: ENABLED"
        echo "   CUDA architecture: $CUDA_ARCH"
    else
        echo "⚠️  CUDA support: DISABLED"
        echo "   Sử dụng: SIMD AVX-256 + OpenMP"
    fi
    echo ""
    echo "Chạy:"
    echo "  ./build/lp_main 0              # Camera"
    echo "  ./build/lp_main data/image.jpg # Ảnh"
    echo ""
else
    echo ""
    echo "❌ Build thất bại!"
    exit 1
fi

