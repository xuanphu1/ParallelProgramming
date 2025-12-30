#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script hiển thị bảng benchmark results đẹp để chụp ảnh vào báo cáo
"""

import sys
import os
from pathlib import Path

# Dữ liệu benchmark từ benchmark_results.txt
BENCHMARK_DATA = {
    "CPU OpenMP (scalar)": {
        "time_ms": 7.144,
        "speedup": 1.0,
        "fps": 140.0,
        "throughput": "~140 MP/s"
    },
    "CPU SIMD (AVX-256)": {
        "time_ms": 1.885,
        "speedup": 3.79,
        "fps": 530.4,
        "throughput": "~530 MP/s"
    },
    "GPU CUDA": {
        "time_ms": 0.494,
        "speedup": 14.47,
        "fps": 2025.5,
        "throughput": "~2025 MP/s"
    }
}

def print_separator(char="=", width=80):
    """In đường phân cách"""
    print(char * width)

def print_header(title, width=80):
    """In tiêu đề"""
    print()
    print_separator("=", width)
    print(title.center(width))
    print_separator("=", width)
    print()

def print_table1():
    """In bảng so sánh hiệu năng chính"""
    print_header("BẢNG 1: SO SÁNH HIỆU NĂNG CÁC PHƯƠNG PHÁP SOBEL")
    print("(100 runs average, ảnh 800x600 pixels, 0.48 MP)")
    print()
    
    # Header
    header = f"{'Phương pháp':<25} {'Thời gian (ms)':<18} {'Speedup':<12} {'FPS':<12} {'Tốc độ xử lý':<15}"
    print(header)
    print("-" * len(header))
    
    # Data rows
    for method, data in BENCHMARK_DATA.items():
        speedup_str = f"{data['speedup']:.2f}x"
        if data['speedup'] == 1.0:
            speedup_str += " (baseline)"
        
        row = f"{method:<25} {data['time_ms']:<18.3f} {speedup_str:<12} {data['fps']:<12.1f} {data['throughput']:<15}"
        print(row)
    
    print()
    print_separator("=", 80)

def print_table2():
    """In bảng so sánh chi tiết giữa các phương pháp"""
    print_header("BẢNG 2: SO SÁNH CHI TIẾT GIỮA CÁC PHƯƠNG PHÁP")
    print()
    
    # Header
    header = f"{'So sánh':<30} {'Tỷ lệ':<20} {'Giải thích':<30}"
    print(header)
    print("-" * len(header))
    
    # Comparisons
    comparisons = [
        ("CUDA vs CPU scalar", "14.47x nhanh hơn", "GPU có hàng nghìn cores vs CPU có vài cores"),
        ("CUDA vs SIMD", "3.82x nhanh hơn", "GPU parallelism vượt trội so với SIMD"),
        ("SIMD vs CPU scalar", "3.79x nhanh hơn", "SIMD xử lý 8 pixels cùng lúc")
    ]
    
    for comp, ratio, explanation in comparisons:
        row = f"{comp:<30} {ratio:<20} {explanation:<30}"
        print(row)
    
    print()
    print_separator("=", 80)

def print_detailed_analysis():
    """In phân tích chi tiết"""
    print_header("PHÂN TÍCH CHI TIẾT KẾT QUẢ")
    print()
    
    methods = [
        ("CPU OpenMP (scalar)", BENCHMARK_DATA["CPU OpenMP (scalar)"]),
        ("CPU SIMD (AVX-256)", BENCHMARK_DATA["CPU SIMD (AVX-256)"]),
        ("GPU CUDA", BENCHMARK_DATA["GPU CUDA"])
    ]
    
    for method_name, data in methods:
        print(f"📊 {method_name}:")
        print(f"   • Thời gian: {data['time_ms']:.3f} ms/frame")
        print(f"   • FPS: {data['fps']:.1f} FPS")
        if data['speedup'] != 1.0:
            print(f"   • Speedup: {data['speedup']:.2f}x so với OpenMP")
        else:
            print(f"   • Speedup: {data['speedup']:.2f}x (baseline)")
        print(f"   • Tốc độ xử lý: {data['throughput']}")
        print()
    
    print_separator("=", 80)

def print_conclusion():
    """In kết luận"""
    print_header("KẾT LUẬN")
    print()
    print("• SIMD vectorization cho thấy 3.79x cải thiện hiệu năng so với scalar code.")
    print("• CUDA GPU cho thấy 14.47x cải thiện so với CPU scalar và 3.82x so với SIMD.")
    print("• GPU CUDA là phương pháp nhanh nhất, phù hợp cho xử lý ảnh real-time")
    print("  với throughput cao.")
    print("• Với ảnh 800x600 (0.48 MP), CUDA có thể xử lý hơn 2000 FPS, đủ cho")
    print("  nhiều ứng dụng real-time.")
    print()
    print_separator("=", 80)

def main():
    """Hàm chính"""
    # Clear screen
    os.system('clear' if os.name != 'nt' else 'cls')
    
    print_separator("=", 80)
    print("BENCHMARK SOBEL FILTER - DATA PARALLELISM".center(80))
    print_separator("=", 80)
    print()
    print("Image: 800x600 pixels (0.48 MP)")
    print("Runs: 100 runs average")
    print()
    
    # Print tables
    print_table1()
    print()
    print_table2()
    print()
    print_detailed_analysis()
    print()
    print_conclusion()
    print()
    print("💡 Tip: Chụp ảnh màn hình để chèn vào báo cáo LaTeX")
    print()

if __name__ == "__main__":
    main()

