#!/bin/bash
# Script benchmark để so sánh OpenMP, SIMD và CUDA

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="$PROJECT_DIR/benchmark_results"
VIDEO_SOURCE="${1:-0}"  # Mặc định webcam (0)
DURATION="${2:-30}"     # Thời gian chạy mỗi test (giây)

echo "=========================================="
echo "Benchmark: So sánh OpenMP, SIMD và CUDA"
echo "=========================================="
echo ""
echo "Video source: $VIDEO_SOURCE"
echo "Duration per test: ${DURATION}s"
echo ""

# Tạo thư mục kết quả
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="$RESULTS_DIR/benchmark_${TIMESTAMP}.txt"

echo "Results will be saved to: $RESULT_FILE"
echo ""

# Hàm chạy benchmark cho một phương pháp
run_benchmark() {
    local method=$1
    local binary=$2
    local output_file="$RESULTS_DIR/${method}_${TIMESTAMP}.log"
    
    echo "----------------------------------------"
    echo "Testing: $method"
    echo "----------------------------------------"
    
    if [ ! -f "$binary" ]; then
        echo "❌ Binary not found: $binary"
        echo "   Please build first: bash scripts/build_${method,,}.sh"
        return 1
    fi
    
    echo "Running $binary for ${DURATION}s..."
    echo "Press 'q' to quit early"
    echo ""
    
    # Chạy và ghi log, timeout sau DURATION+5 giây
    timeout $((DURATION + 5)) "$binary" "$VIDEO_SOURCE" 2>&1 | tee "$output_file" &
    local pid=$!
    
    # Đợi DURATION giây
    sleep "$DURATION"
    
    # Gửi tín hiệu để dừng (nếu vẫn chạy)
    kill -INT "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    
    echo ""
    echo "✅ Completed: $method"
    echo ""
    
    # Trích xuất FPS từ log (tìm dòng có "FPS: ")
    local fps_values=$(grep "FPS:" "$output_file" | grep -oP 'FPS: \K[0-9.]+' | head -50)
    if [ -z "$fps_values" ]; then
        # Thử cách khác: tìm trong stdout
        fps_values=$(grep -E "FPS: [0-9.]+" "$output_file" | grep -oE "[0-9]+\.[0-9]+" | head -50)
    fi
    
    local avg_fps="N/A"
    local max_fps="N/A"
    local min_fps="N/A"
    
    if [ -n "$fps_values" ]; then
        avg_fps=$(echo "$fps_values" | awk '{sum+=$1; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}')
        max_fps=$(echo "$fps_values" | awk 'BEGIN{max=0} {if($1>max) max=$1} END {if(max>0) printf "%.2f", max; else print "N/A"}')
        min_fps=$(echo "$fps_values" | awk 'BEGIN{min=999} {if($1<min && $1>0) min=$1} END {if(min<999) printf "%.2f", min; else print "N/A"}')
    fi
    
    # Trích xuất thời gian Sobel
    local sobel_values=$(grep -E "Sobel\([^)]+\):" "$output_file" | grep -oP 'Sobel\([^)]+\): \K[0-9.]+' | head -50)
    local avg_sobel="N/A"
    if [ -n "$sobel_values" ]; then
        avg_sobel=$(echo "$sobel_values" | awk '{sum+=$1; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}')
    fi
    
    echo "$method|$avg_fps|$min_fps|$max_fps|$avg_sobel" >> "$RESULT_FILE"
    
    return 0
}

# Ghi header vào file kết quả
echo "Method|Avg_FPS|Min_FPS|Max_FPS|Avg_Sobel_Time_ms" > "$RESULT_FILE"

# Chạy benchmark cho từng phương pháp
echo "=========================================="
echo "Step 1/3: Building all versions..."
echo "=========================================="
echo ""

# Build OpenMP
echo "Building OpenMP version..."
bash "$PROJECT_DIR/scripts/build_openmp.sh" > /dev/null 2>&1 || {
    echo "⚠️  OpenMP build failed, skipping..."
}

# Build SIMD
echo "Building SIMD version..."
bash "$PROJECT_DIR/scripts/build_simd.sh" > /dev/null 2>&1 || {
    echo "⚠️  SIMD build failed, skipping..."
}

# Build CUDA
echo "Building CUDA version..."
bash "$PROJECT_DIR/scripts/build.sh" > /dev/null 2>&1 || {
    echo "⚠️  CUDA build failed, skipping..."
}

echo ""
echo "=========================================="
echo "Step 2/3: Running benchmarks..."
echo "=========================================="
echo ""

# Chạy benchmark
run_benchmark "OpenMP" "$PROJECT_DIR/lp_main_openmp"
sleep 2  # Nghỉ giữa các test

run_benchmark "SIMD" "$PROJECT_DIR/lp_main_simd"
sleep 2

run_benchmark "CUDA" "$PROJECT_DIR/lp_main_cuda"
sleep 2

echo ""
echo "=========================================="
echo "Step 3/3: Generating report..."
echo "=========================================="
echo ""

# Tạo báo cáo
bash "$PROJECT_DIR/scripts/generate_report.sh" "$RESULT_FILE"

echo ""
echo "✅ Benchmark completed!"
echo "Results: $RESULT_FILE"
echo ""

