#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>

// Lớp LPOCR: chịu trách nhiệm đọc ký tự biển số từ vùng ảnh đã cắt
// Dùng model ONNX export từ LP_ocr_nano_62.pt (YOLOv5 OCR)
class LPOCR {
public:
    LPOCR();
    virtual ~LPOCR();

    // Khởi tạo / load model, trả về true nếu thành công
    virtual bool initialize(const std::string& modelPath);

    // Nhận diện text biển số từ ảnh vùng biển số (crop)
    virtual std::string recognize(const cv::Mat& plateRoi);

private:
    Ort::Env env_;
    Ort::Session* session_ = nullptr;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::vector<const char*> inputNames_;
    std::vector<const char*> outputNames_;
    std::string inputNameStr_;  // giữ ownership của string
    std::string outputNameStr_;
    bool initialized_ = false;
    int inputSize_ = 320;        // kích thước input cho OCR (có thể chỉnh nếu cần)
    float confThresh_ = 0.25f;   // ngưỡng confidence ký tự
    // Tên lớp cho OCR: nếu bạn có alphabet cụ thể, hãy điền vào đây.
    // Mặc định: nếu rỗng, sẽ dùng chỉ số lớp (cls_id) dưới dạng chuỗi.
    std::vector<std::string> classNames_;
};


