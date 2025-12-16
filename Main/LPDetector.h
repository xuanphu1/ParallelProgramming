#pragma once

#include <opencv2/opencv.hpp>
#include <torch/script.h>  // LibTorch TorchScript

// Lớp LPDetector: chịu trách nhiệm phát hiện vùng biển số trong ảnh
// Dùng model TorchScript export từ LP_detector_nano_61.pt (YOLOv5)
class LPDetector {
public:
    LPDetector() = default;
    virtual ~LPDetector() = default;

    // Khởi tạo / load model, trả về true nếu thành công
    virtual bool initialize(const std::string& modelPath);

    // Phát hiện biển số từ frame đầu vào
    // Trả về danh sách bounding box vùng biển số
    virtual std::vector<cv::Rect> detect(const cv::Mat& frame);

private:
    torch::jit::script::Module module_;
    bool initialized_ = false;
    int inputSize_ = 640;      // phải khớp với kích thước khi export TorchScript
    float confThresh_ = 0.35f; // ngưỡng confidence cho bbox
};


