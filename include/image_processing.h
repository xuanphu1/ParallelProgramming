#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <opencv2/opencv.hpp>
#include <string>

// Hàm áp dụng Gamma Correction
cv::Mat apply_gamma_correction_parallel(const cv::Mat& image, double gamma, const std::string& save_path = "");

// Hàm áp dụng Sobel Edge Enhancement cho OCR (làm rõ edges của text)
cv::Mat apply_sobel_edge_enhancement(const cv::Mat& image, double strength = 0.3);

// Hàm tính edge score (edge density) với CUDA
double calculate_edge_score(const cv::Mat& image, double threshold = 50.0);

// Preprocess letterbox giống Python
std::tuple<std::vector<float>, float, std::pair<float, float>> preprocess_letterbox(const cv::Mat& image, int input_size);

#endif // IMAGE_PROCESSING_H

