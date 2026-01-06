#include "../include/image_processing.h"
#include "../include/config.h"
#include "../include/utils.h"
#include "../cuda/sobel_cuda.h"
#include <vector>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace cv;

// Hàm áp dụng Gamma Correction
Mat apply_gamma_correction_parallel(const Mat& image, double gamma, const std::string& save_path) {
    if (image.empty()) {
        return image;
    }
    
    // Tạo lookup table cho gamma correction
    Mat lookup_table(1, 256, CV_8U);
    uchar* p = lookup_table.ptr();
    
    for (int i = 0; i < 256; i++) {
        p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
    }
    
    // Áp dụng lookup table cho từng channel 
    Mat result;
    if (image.channels() == 3) {
        // BGR image
        std::vector<Mat> channels;
        split(image, channels);
        
        for (int c = 0; c < 3; c++) {
            LUT(channels[c], lookup_table, channels[c]);
        }
        
        merge(channels, result);
    } else {
        // Grayscale image
        LUT(image, lookup_table, result);
        // Convert to BGR để giữ format
        if (result.channels() == 1) {
            cvtColor(result, result, COLOR_GRAY2BGR);
        }
    }
    
    // Lưu ảnh nếu được yêu cầu
    if (!save_path.empty() && SAVE_FILTERED_IMAGES) {
        ensure_directory_exists(FILTERED_OUTPUT_DIR);
        imwrite(save_path, result);
    }
    
    return result;
}

// Hàm áp dụng Sobel Edge Enhancement cho OCR (làm rõ edges của text)
Mat apply_sobel_edge_enhancement(const Mat& image, double strength) {
    if (image.empty() || strength <= 0.0) return image;
    
    Mat gray, grad_x, grad_y, magnitude;
    Mat abs_grad_x, abs_grad_y;
    
    // Convert to grayscale
    if (image.channels() == 3) {
        cvtColor(image, gray, COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    // Tính Sobel gradients
    Sobel(gray, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
    Sobel(gray, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);
    
    // Tính magnitude (edge strength)
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, magnitude);
    
    // Normalize magnitude về [0, 255]
    normalize(magnitude, magnitude, 0, 255, NORM_MINMAX, CV_8U);
    
    // Blend với ảnh gốc để tăng cường edges
    Mat enhanced;
    if (image.channels() == 3) {
        // Convert magnitude to BGR
        Mat magnitude_bgr;
        cvtColor(magnitude, magnitude_bgr, COLOR_GRAY2BGR);
        
        // Blend: enhanced = original + strength * magnitude
        addWeighted(image, 1.0, magnitude_bgr, strength, 0, enhanced);
    } else {
        addWeighted(image, 1.0, magnitude, strength, 0, enhanced);
    }
    
    return enhanced;
}

// Hàm tính edge score (edge density) với CUDA
double calculate_edge_score(const Mat& image, double threshold) {
    if (image.empty()) return 0.0;
    
    Mat gray, grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    
    // Convert to grayscale
    if (image.channels() == 3) {
        cvtColor(image, gray, COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    // Apply Sobel filter
    Sobel(gray, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
    Sobel(gray, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);
    
    // Tính edge density với CUDA
    int width = gray.cols;
    int height = gray.rows;
    int total_pixels = width * height;
    int edge_pixels = 0;
    
    // Kiểm tra dữ liệu đầu vào
    if (abs_grad_x.empty() || abs_grad_y.empty() || total_pixels == 0) {
        return 0.0;
    }
    
    // Thử CUDA trước
    cudaError_t cuda_err = cudaGetLastError(); // Clear previous errors
    cuda_count_edges(abs_grad_x.data, abs_grad_y.data, &edge_pixels, width, height, threshold);
    cuda_err = cudaGetLastError();
    
    // Nếu CUDA fail hoặc kết quả = 0 (có thể do CUDA không chạy), fallback về CPU với OpenMP
    if (cuda_err != cudaSuccess || edge_pixels == 0) {
        // Fallback: Tính trên CPU với OpenMP để song song hóa
        edge_pixels = 0;
        #ifdef _OPENMP
        #pragma omp parallel for reduction(+:edge_pixels) collapse(2)
        #endif
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double magnitude = abs_grad_x.at<uchar>(y, x) + abs_grad_y.at<uchar>(y, x);
                if (magnitude > threshold) {
                    edge_pixels++;
                }
            }
        }
    }
    return (double)edge_pixels / total_pixels;
}

// Preprocess letterbox giống Python
std::tuple<std::vector<float>, float, std::pair<float, float>> preprocess_letterbox(const Mat& image, int input_size) {
    int h = image.rows;
    int w = image.cols;
    
    // Tính scale ratio
    float r = std::min((float)input_size / h, (float)input_size / w);
    
    // Tính kích thước mới
    int new_unpad_w = round(w * r);
    int new_unpad_h = round(h * r);
    
    // Tính padding
    float dw = (input_size - new_unpad_w) / 2.0f;
    float dh = (input_size - new_unpad_h) / 2.0f;
    
    // Resize ảnh
    Mat img_resized;
    if (w != new_unpad_w || h != new_unpad_h) {
        resize(image, img_resized, Size(new_unpad_w, new_unpad_h), 0, 0, INTER_LINEAR);
    } else {
        img_resized = image.clone();
    }
    
    // Thêm padding
    int top = round(dh - 0.1);
    int bottom = round(dh + 0.1);
    int left = round(dw - 0.1);
    int right = round(dw + 0.1);
    Mat img_padded;
    copyMakeBorder(img_resized, img_padded, top, bottom, left, right, BORDER_CONSTANT, Scalar(114, 114, 114));
    
    // BGR to RGB
    Mat img_rgb;
    cvtColor(img_padded, img_rgb, COLOR_BGR2RGB);
    
    // HWC to CHW và normalize về [0, 1] - Song song hóa với OpenMP
    std::vector<float> input_tensor(input_size * input_size * 3);
    
    #ifdef _OPENMP
    #pragma omp parallel for collapse(3)
    #endif
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < input_size; h++) {
            for (int w = 0; w < input_size; w++) {
                int idx = c * input_size * input_size + h * input_size + w;
                input_tensor[idx] = img_rgb.at<Vec3b>(h, w)[c] / 255.0f;
            }
        }
    }
    
    return std::make_tuple(input_tensor, r, std::make_pair(dw, dh));
}

