#!/bin/bash
# Script để build benchmark với CUDA support

echo "Building benchmark với CUDA support..."

# Kiểm tra CUDA
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc (CUDA compiler) không tìm thấy!"
    echo "Cần cài CUDA toolkit trước."
    exit 1
fi

# Compile CUDA file
echo "Compiling SobelCuda.cu..."
nvcc -c Main/SobelCuda.cu -o SobelCuda.o -I/usr/local/cuda/include \
     `pkg-config --cflags opencv4` -arch=sm_50 2>&1

if [ $? -ne 0 ]; then
    echo "ERROR: Không thể compile CUDA code"
    exit 1
fi

# Compile C++ và link với CUDA
echo "Compiling và linking benchmark..."
g++ -std=c++17 -fopenmp -mavx2 -DUSE_CUDA_SOBEL \
    Main/benchmark_sobel_cuda.cpp Main/SobelSIMD.cpp SobelCuda.o \
    -o benchmark_sobel_cuda \
    `pkg-config --cflags --libs opencv4` \
    -L/usr/local/cuda/lib64 -lcudart 2>&1

if [ $? -ne 0 ]; then
    echo "ERROR: Không thể link CUDA"
    exit 1
fi

echo "✅ Build thành công: benchmark_sobel_cuda"
echo "Chạy: ./benchmark_sobel_cuda Main/Bienso.jpg"

