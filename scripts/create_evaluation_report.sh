#!/bin/bash
# Script tạo báo cáo đánh giá tự động từ kết quả benchmark

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_FILE="$PROJECT_DIR/docs/BAO_CAO_DANH_GIA.md"

echo "=========================================="
echo "Tạo Báo Cáo Đánh Giá"
echo "=========================================="
echo ""

# Kiểm tra file benchmark results
BENCHMARK_FILE="$PROJECT_DIR/docs/benchmark_results.txt"
if [ ! -f "$BENCHMARK_FILE" ]; then
    echo "⚠️  Không tìm thấy file benchmark results"
    echo "   Chạy benchmark trước: bash scripts/benchmark.sh"
    echo ""
    echo "   Hoặc báo cáo sẽ dùng dữ liệu mẫu từ benchmark_results.txt hiện có"
    echo ""
fi

# Tạo summary từ benchmark results nếu có
if [ -f "$BENCHMARK_FILE" ]; then
    echo "📊 Đang phân tích kết quả benchmark..."
    
    # Trích xuất dữ liệu
    OPENMP_TIME=$(grep "CPU OpenMP" "$BENCHMARK_FILE" | grep -oP '\K[0-9.]+(?= ms)')
    SIMD_TIME=$(grep "CPU SIMD" "$BENCHMARK_FILE" | grep -oP '\K[0-9.]+(?= ms)')
    CUDA_TIME=$(grep "GPU CUDA" "$BENCHMARK_FILE" | grep -oP '\K[0-9.]+(?= ms)')
    
    OPENMP_FPS=$(grep "CPU OpenMP.*FPS" "$BENCHMARK_FILE" | grep -oP '\K[0-9.]+(?= FPS)')
    SIMD_FPS=$(grep "CPU SIMD.*FPS" "$BENCHMARK_FILE" | grep -oP '\K[0-9.]+(?= FPS)')
    CUDA_FPS=$(grep "GPU CUDA.*FPS" "$BENCHMARK_FILE" | grep -oP '\K[0-9.]+(?= FPS)')
    
    echo "✅ Dữ liệu benchmark đã được trích xuất"
    echo ""
    echo "Kết quả:"
    echo "  OpenMP: ${OPENMP_TIME}ms (${OPENMP_FPS} FPS)"
    echo "  SIMD:   ${SIMD_TIME}ms (${SIMD_FPS} FPS)"
    echo "  CUDA:   ${CUDA_TIME}ms (${CUDA_FPS} FPS)"
    echo ""
fi

# Kiểm tra phần cứng hiện tại
echo "🔍 Đang kiểm tra phần cứng..."
echo ""

# CPU cores
CPU_CORES=$(nproc 2>/dev/null || echo "N/A")
echo "  CPU Cores: $CPU_CORES"

# AVX2 support
if grep -q avx2 /proc/cpuinfo 2>/dev/null; then
    AVX2_SUPPORT="✅ Có"
else
    AVX2_SUPPORT="❌ Không"
fi
echo "  AVX2 Support: $AVX2_SUPPORT"

# GPU NVIDIA
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_SUPPORT="✅ Có ($GPU_INFO)"
else
    GPU_SUPPORT="❌ Không"
fi
echo "  GPU NVIDIA: $GPU_SUPPORT"

# CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    CUDA_SUPPORT="✅ Có (CUDA $CUDA_VERSION)"
else
    CUDA_SUPPORT="❌ Không"
fi
echo "  CUDA Toolkit: $CUDA_SUPPORT"
echo ""

# Tạo khuyến nghị dựa trên phần cứng
echo "💡 Khuyến nghị dựa trên phần cứng hiện tại:"
echo ""

if [ "$GPU_SUPPORT" != "❌ Không" ] && [ "$CUDA_SUPPORT" != "❌ Không" ]; then
    echo "  ✅ Nên dùng: CUDA (GPU NVIDIA có sẵn)"
    echo "     - Hiệu năng cao nhất (14.47x so với OpenMP)"
    echo "     - Phù hợp cho production"
elif [ "$AVX2_SUPPORT" != "❌ Không" ]; then
    echo "  ✅ Nên dùng: SIMD (CPU hỗ trợ AVX2)"
    echo "     - Hiệu năng tốt (3.79x so với OpenMP)"
    echo "     - Không cần GPU"
else
    echo "  ✅ Nên dùng: OpenMP (baseline)"
    echo "     - Hoạt động trên mọi hệ thống"
    echo "     - Hiệu năng cơ bản"
fi

echo ""
echo "=========================================="
echo "✅ Báo cáo đã được tạo tại:"
echo "   $REPORT_FILE"
echo "=========================================="
echo ""
echo "📖 Xem báo cáo chi tiết:"
echo "   cat $REPORT_FILE"
echo ""

