#ifndef LICENSE_PLATE_DETECTOR_H
#define LICENSE_PLATE_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include "types.h"

class LicensePlateDetector {
private:
    Ort::Env env;
    Ort::Session* detector_session;
    Ort::Session* ocr_session;
    Ort::MemoryInfo memory_info;
    Ort::AllocatorWithDefaultOptions allocator;
    
    std::vector<std::string> detector_input_names;
    std::vector<std::string> detector_output_names;
    std::vector<std::string> ocr_input_names;
    std::vector<std::string> ocr_output_names;
    
    int detector_input_size;
    OCRConfig ocr_config;
    
public:
    LicensePlateDetector(const std::string& detector_path, const std::string& ocr_path, const OCRConfig& config);
    ~LicensePlateDetector();
    
    // Detect license plates - format: [X, x1, y1, x2, y2, class_id, score]
    std::vector<Detection> detect(const cv::Mat& frame, float conf_threshold = 0.4f);
    
    // OCR preprocessing
    std::vector<uint8_t> preprocess_ocr(const cv::Mat& plate_roi, int plate_id = -1);
    
    // OCR inference
    OCRResult ocr(const cv::Mat& plate_roi, int plate_id = -1);
};

#endif // LICENSE_PLATE_DETECTOR_H

