#!/bin/bash
# Script chạy detection với 3 phương pháp song song khác nhau

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIDEO_SOURCE="${1:-0}"  # Mặc định webcam (0)

echo "=========================================="
echo "Chạy Detection với 3 Phương Pháp Song Song"
echo "=========================================="
echo ""
echo "Video source: $VIDEO_SOURCE"
echo ""

# Kiểm tra các binary
check_binary() {
    local binary=$1
    local method=$2
    
    if [ ! -f "$binary" ]; then
        echo "❌ Binary không tìm thấy: $binary"
        echo "   Vui lòng build trước: bash scripts/build_${method,,}.sh"
        return 1
    fi
    return 0
}

# Menu chọn phương pháp
echo "Chọn phương pháp để chạy:"
echo "1. OpenMP (CPU multi-threading)"
echo "2. SIMD + OpenMP (AVX-256 vectorization)"
echo "3. CUDA (GPU parallelization)"
echo "4. Chạy tất cả 3 phương pháp lần lượt"
echo ""
read -p "Nhập lựa chọn (1-4): " choice

case $choice in
    1)
        echo ""
        echo "=========================================="
        echo "Chạy với OpenMP..."
        echo "=========================================="
        check_binary "$PROJECT_DIR/lp_main_openmp" "openmp" || exit 1
        echo "Nhấn 'q' để thoát"
        echo ""
        "$PROJECT_DIR/lp_main_openmp" "$VIDEO_SOURCE"
        ;;
    2)
        echo ""
        echo "=========================================="
        echo "Chạy với SIMD + OpenMP..."
        echo "=========================================="
        check_binary "$PROJECT_DIR/lp_main_simd" "simd" || exit 1
        echo "Nhấn 'q' để thoát"
        echo ""
        "$PROJECT_DIR/lp_main_simd" "$VIDEO_SOURCE"
        ;;
    3)
        echo ""
        echo "=========================================="
        echo "Chạy với CUDA..."
        echo "=========================================="
        check_binary "$PROJECT_DIR/lp_main_cuda" "cuda" || exit 1
        echo "Nhấn 'q' để thoát"
        echo ""
        "$PROJECT_DIR/lp_main_cuda" "$VIDEO_SOURCE"
        ;;
    4)
        echo ""
        echo "=========================================="
        echo "Chạy tất cả 3 phương pháp lần lượt..."
        echo "=========================================="
        echo ""
        
        # OpenMP
        if check_binary "$PROJECT_DIR/lp_main_openmp" "openmp"; then
            echo "----------------------------------------"
            echo "1/3: OpenMP"
            echo "----------------------------------------"
            echo "Nhấn 'q' để chuyển sang phương pháp tiếp theo"
            echo ""
            timeout 30 "$PROJECT_DIR/lp_main_openmp" "$VIDEO_SOURCE" 2>/dev/null || true
            sleep 2
        fi
        
        # SIMD
        if check_binary "$PROJECT_DIR/lp_main_simd" "simd"; then
            echo ""
            echo "----------------------------------------"
            echo "2/3: SIMD + OpenMP"
            echo "----------------------------------------"
            echo "Nhấn 'q' để chuyển sang phương pháp tiếp theo"
            echo ""
            timeout 30 "$PROJECT_DIR/lp_main_simd" "$VIDEO_SOURCE" 2>/dev/null || true
            sleep 2
        fi
        
        # CUDA
        if check_binary "$PROJECT_DIR/lp_main_cuda" "cuda"; then
            echo ""
            echo "----------------------------------------"
            echo "3/3: CUDA"
            echo "----------------------------------------"
            echo "Nhấn 'q' để kết thúc"
            echo ""
            timeout 30 "$PROJECT_DIR/lp_main_cuda" "$VIDEO_SOURCE" 2>/dev/null || true
        fi
        
        echo ""
        echo "=========================================="
        echo "✅ Đã chạy xong cả 3 phương pháp!"
        echo "=========================================="
        ;;
    *)
        echo "❌ Lựa chọn không hợp lệ!"
        exit 1
        ;;
esac

