#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>  // ONNX Runtime C++ API

// Lớp LPDetector: chịu trách nhiệm phát hiện vùng biển số trong ảnh
// Dùng model ONNX export từ LP_detector_nano_61.pt (YOLOv5)
class LPDetector {
public:
    LPDetector();
    virtual ~LPDetector();

    // Khởi tạo / load model, trả về true nếu thành công
    virtual bool initialize(const std::string& modelPath);

    // Phát hiện biển số từ frame đầu vào
    // Trả về danh sách bounding box vùng biển số
    virtual std::vector<cv::Rect> detect(const cv::Mat& frame);

private:
    Ort::Env env_;
    Ort::Session* session_ = nullptr;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::vector<const char*> inputNames_;
    std::vector<const char*> outputNames_;
    std::string inputNameStr_;  // giữ ownership của string
    std::string outputNameStr_;
    std::vector<int64_t> inputShape_;
    bool initialized_ = false;
    int inputSize_ = 640;      // phải khớp với kích thước khi export ONNX
    float confThresh_ = 0.35f; // ngưỡng confidence cho bbox
};


