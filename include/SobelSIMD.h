#pragma once

#include <opencv2/opencv.hpp>

// Sobel filter với SIMD vectorization (Data Parallelism)
// Sử dụng AVX/SSE để xử lý nhiều pixel cùng lúc
bool sobelSIMD(const cv::Mat& src, cv::Mat& dst);

