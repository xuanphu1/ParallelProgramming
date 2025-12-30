#include <iostream>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "SobelSIMD.h"

// Sobel CPU với OpenMP (phiên bản gốc - scalar)
void sobelCPU_scalar(const cv::Mat& src, cv::Mat& dst) {
    CV_Assert(src.type() == CV_8UC1);
    dst = cv::Mat::zeros(src.size(), CV_8UC1);
    
    int rows = src.rows;
    int cols = src.cols;
    
    #pragma omp parallel for
    for (int y = 1; y < rows - 1; ++y) {
        for (int x = 1; x < cols - 1; ++x) {
            int p00 = src.at<uchar>(y - 1, x - 1);
            int p01 = src.at<uchar>(y - 1, x    );
            int p02 = src.at<uchar>(y - 1, x + 1);
            int p10 = src.at<uchar>(y    , x - 1);
            int p12 = src.at<uchar>(y    , x + 1);
            int p20 = src.at<uchar>(y + 1, x - 1);
            int p21 = src.at<uchar>(y + 1, x    );
            int p22 = src.at<uchar>(y + 1, x + 1);
            
            int gx = -p00 - 2*p10 - p20 + p02 + 2*p12 + p22;
            int gy = -p00 - 2*p01 - p02 + p20 + 2*p21 + p22;
            
            float mag = std::sqrt(static_cast<float>(gx * gx + gy * gy));
            if (mag > 255.0f) mag = 255.0f;
            dst.at<uchar>(y, x) = static_cast<uchar>(mag);
        }
    }
}

// Wrapper cho các hàm Sobel khác nhau
template<typename Func>
double benchmark_sobel_impl(const std::string& name, 
                            Func sobel_func,
                            const cv::Mat& input, 
                            int warmup_runs = 5, 
                            int test_runs = 100) {
    cv::Mat output;
    
    // Warmup
    for (int i = 0; i < warmup_runs; ++i) {
        sobel_func(input, output);
    }
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < test_runs; ++i) {
        sobel_func(input, output);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_time_ms = (duration.count() / 1000.0) / test_runs;
    
    std::cout << "  " << std::left << std::setw(25) << name << ": " 
              << std::fixed << std::setprecision(3) << std::right << std::setw(8) << avg_time_ms << " ms"
              << " (" << std::setw(6) << std::setprecision(1) << (1000.0 / avg_time_ms) << " FPS)" << std::endl;
    
    return avg_time_ms;
}

int main(int argc, char** argv) {
    std::string imagePath = "../bienso.jpg";
    if (argc > 1) {
        imagePath = argv[1];
    }
    
    // Đọc ảnh
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Khong the doc anh: " << imagePath << std::endl;
        std::cerr << "Thu tim trong thu muc hien tai..." << std::endl;
        img = cv::imread("bienso.jpg", cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cerr << "Khong tim thay bienso.jpg" << std::endl;
            return -1;
        }
        imagePath = "bienso.jpg";
    }
    
    std::cout << "========================================" << std::endl;
    std::cout << "BENCHMARK SOBEL FILTER - DATA PARALLELISM" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Anh: " << imagePath << std::endl;
    std::cout << "Kich thuoc: " << img.cols << "x" << img.rows << " pixels" << std::endl;
    std::cout << "So pixel: " << std::fixed << std::setprecision(2) 
              << (img.cols * img.rows) / 1000000.0 << " MP" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Chay benchmark (100 runs, warmup 5 runs)..." << std::endl;
    std::cout << std::endl;
    
    // Benchmark các phiên bản
    double time_cpu = benchmark_sobel_impl("CPU OpenMP (scalar)", sobelCPU_scalar, img);
    double time_simd = benchmark_sobel_impl("CPU SIMD (AVX-256)", sobelSIMD, img);
    
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "KET QUA SO SANH" << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "CPU OpenMP (scalar): " << std::setw(8) << time_cpu << " ms (baseline = 1.0x)" << std::endl;
    std::cout << "CPU SIMD (AVX-256):  " << std::setw(8) << time_simd << " ms (x" 
              << std::setprecision(2) << (time_cpu / time_simd) << " nhanh hon)" << std::endl;
    
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Lưu kết quả vào file
    std::ofstream out("benchmark_results.txt");
    out << "BENCHMARK SOBEL FILTER - DATA PARALLELISM\n";
    out << "==========================================\n";
    out << "Image: " << imagePath << "\n";
    out << "Size: " << img.cols << "x" << img.rows << " pixels\n";
    out << "Megapixels: " << std::fixed << std::setprecision(2) 
        << (img.cols * img.rows) / 1000000.0 << " MP\n";
    out << "\n";
    out << "Results (ms/frame, 100 runs average):\n";
    out << "  CPU OpenMP (scalar): " << std::setprecision(3) << time_cpu << " ms\n";
    out << "  CPU SIMD (AVX-256):  " << time_simd << " ms (speedup: " 
        << std::setprecision(2) << (time_cpu / time_simd) << "x)\n";
    out << "\n";
    out << "Conclusion:\n";
    out << "  SIMD vectorization cho thay " << std::setprecision(2) << (time_cpu / time_simd) 
        << "x cai thien hieu nang so voi scalar code.\n";
    out.close();
    
    std::cout << "Ket qua da luu vao: benchmark_results.txt" << std::endl;
    
    return 0;
}
