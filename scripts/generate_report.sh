#!/bin/bash
# Script tạo báo cáo so sánh từ kết quả benchmark

RESULT_FILE="$1"

if [ ! -f "$RESULT_FILE" ]; then
    echo "❌ Result file not found: $RESULT_FILE"
    exit 1
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_FILE="${RESULT_FILE%.txt}_report.txt"

echo "=========================================="
echo "BÁO CÁO SO SÁNH CÁC PHƯƠNG PHÁP SONG SONG HÓA"
echo "=========================================="
echo ""
echo "File kết quả: $RESULT_FILE"
echo "Ngày tạo: $(date)"
echo ""

# Đọc và parse kết quả
declare -A fps_avg fps_min fps_max sobel_avg

while IFS='|' read -r method avg_fps min_fps max_fps avg_sobel; do
    if [ "$method" != "Method" ]; then
        fps_avg["$method"]="$avg_fps"
        fps_min["$method"]="$min_fps"
        fps_max["$method"]="$max_fps"
        sobel_avg["$method"]="$avg_sobel"
    fi
done < "$RESULT_FILE"

# Tạo báo cáo
{
    echo "=========================================="
    echo "KẾT QUẢ BENCHMARK"
    echo "=========================================="
    echo ""
    echo "1. FPS (Frames Per Second)"
    echo "----------------------------------------"
    printf "%-10s | %10s | %10s | %10s | %15s\n" "Method" "Avg FPS" "Min FPS" "Max FPS" "Avg Sobel (ms)"
    echo "----------------------------------------"
    
    for method in "OpenMP" "SIMD" "CUDA"; do
        if [ -n "${fps_avg[$method]}" ]; then
            printf "%-10s | %10s | %10s | %10s | %15s\n" \
                "$method" \
                "${fps_avg[$method]}" \
                "${fps_min[$method]}" \
                "${fps_max[$method]}" \
                "${sobel_avg[$method]}"
        fi
    done
    echo ""
    
    echo "2. PHÂN TÍCH HIỆU NĂNG"
    echo "----------------------------------------"
    
    # Tìm phương pháp tốt nhất
    best_fps_method=""
    best_fps=0
    best_sobel_method=""
    best_sobel=999999
    
    for method in "OpenMP" "SIMD" "CUDA"; do
        if [ -n "${fps_avg[$method]}" ] && [ "${fps_avg[$method]}" != "N/A" ]; then
            fps_val=$(echo "${fps_avg[$method]}" | awk '{print int($1)}')
            if (( $(echo "$fps_val > $best_fps" | bc -l) )); then
                best_fps=$fps_val
                best_fps_method=$method
            fi
        fi
        
        if [ -n "${sobel_avg[$method]}" ] && [ "${sobel_avg[$method]}" != "N/A" ]; then
            sobel_val=$(echo "${sobel_avg[$method]}" | awk '{print int($1)}')
            if (( $(echo "$sobel_val < $best_sobel" | bc -l) )); then
                best_sobel=$sobel_val
                best_sobel_method=$method
            fi
        fi
    done
    
    echo "Phương pháp có FPS cao nhất: $best_fps_method (${fps_avg[$best_fps_method]} FPS)"
    echo "Phương pháp có thời gian Sobel nhanh nhất: $best_sobel_method (${sobel_avg[$best_sobel_method]} ms)"
    echo ""
    
    # Tính speedup
    if [ -n "${fps_avg[OpenMP]}" ] && [ "${fps_avg[OpenMP]}" != "N/A" ]; then
        baseline_fps=$(echo "${fps_avg[OpenMP]}" | awk '{print $1+0}')
        
        if (( $(echo "$baseline_fps > 0" | bc -l 2>/dev/null || echo "0") )); then
            echo "3. SPEEDUP SO VỚI OPENMP (Baseline)"
            echo "----------------------------------------"
            
            for method in "SIMD" "CUDA"; do
                if [ -n "${fps_avg[$method]}" ] && [ "${fps_avg[$method]}" != "N/A" ]; then
                    method_fps=$(echo "${fps_avg[$method]}" | awk '{print $1+0}')
                    if (( $(echo "$method_fps > 0" | bc -l 2>/dev/null || echo "0") )); then
                        speedup=$(echo "scale=2; $method_fps / $baseline_fps" | bc 2>/dev/null || echo "N/A")
                        echo "$method: ${speedup}x (${fps_avg[$method]} FPS / ${fps_avg[OpenMP]} FPS)"
                    fi
                fi
            done
            echo ""
        fi
    fi
    
    echo "4. KẾT LUẬN"
    echo "----------------------------------------"
    echo "Dựa trên kết quả benchmark:"
    echo "- OpenMP: Phương pháp cơ bản, dễ triển khai"
    echo "- SIMD: Tận dụng vectorization của CPU, cải thiện hiệu năng"
    echo "- CUDA: Tận dụng GPU, hiệu năng cao nhất (nếu có GPU)"
    echo ""
    
} | tee "$REPORT_FILE"

echo ""
echo "✅ Báo cáo đã được tạo: $REPORT_FILE"
echo ""

