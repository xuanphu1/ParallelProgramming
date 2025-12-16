#pragma once

#include <opencv2/opencv.hpp>

// Sobel filter chạy trên GPU (CUDA thuần, không dùng OpenCV CUDA)
// - srcGray: ảnh xám CV_8UC1
// - dstEdge: ảnh biên CV_8UC1
// Trả về true nếu chạy thành công.
bool sobelCuda(const cv::Mat& srcGray, cv::Mat& dstEdge);


