#include "LPOCR.h"

#include <algorithm>
#include <iostream>
#include <memory>

LPOCR::LPOCR() : env_(ORT_LOGGING_LEVEL_WARNING, "LPOCR") {
}

LPOCR::~LPOCR() {
    if (session_) {
        delete session_;
        session_ = nullptr;
    }
}

bool LPOCR::initialize(const std::string& modelPath) {
    try {
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        session_ = new Ort::Session(env_, modelPath.c_str(), sessionOptions);

        // Lấy thông tin input/output
        size_t numInputNodes = session_->GetInputCount();
        size_t numOutputNodes = session_->GetOutputCount();

        if (numInputNodes == 0 || numOutputNodes == 0) {
            std::cerr << "[LPOCR] Model khong co input/output hop le." << std::endl;
            return false;
        }

        // YOLOv5 ONNX export thường dùng tên "images" và "output"
        // Dùng tên cố định thay vì lấy từ model (tránh lỗi API)
        inputNameStr_ = "images";
        outputNameStr_ = "output";
        inputNames_.push_back(inputNameStr_.c_str());
        outputNames_.push_back(outputNameStr_.c_str());

        // Lấy shape input
        Ort::TypeInfo inputTypeInfo = session_->GetInputTypeInfo(0);
        auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        auto inputShape = inputTensorInfo.GetShape();

        if (inputShape.size() == 4 && inputShape[2] > 0 && inputShape[3] > 0) {
            inputSize_ = static_cast<int>(inputShape[2]);
        }

        initialized_ = true;
        std::cout << "[LPOCR] Loaded ONNX model: " << modelPath 
                  << " (input size: " << inputSize_ << "x" << inputSize_ << ")" << std::endl;

        // TODO: nếu bạn có danh sách class cụ thể (alphabet) hãy điền ở đây,
        // ví dụ: classNames_ = {"0","1","2","3","4","5","6","7","8","9","A","B",...};
        classNames_.clear();
    } catch (const Ort::Exception& e) {
        std::cerr << "[LPOCR] Loi load ONNX model: " << e.what() << std::endl;
        initialized_ = false;
        return false;
    }
    return true;
}

std::string LPOCR::recognize(const cv::Mat& plateRoi) {
    if (!initialized_ || plateRoi.empty() || !session_) {
        return {};
    }

    cv::Mat img;
    cv::resize(plateRoi, img, cv::Size(inputSize_, inputSize_));
    if (img.channels() == 3) {
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    }
    img.convertTo(img, CV_32F, 1.0 / 255.0);

    // Tạo tensor input cho ONNX: [1, 3, H, W]
    const size_t inputTensorSize = 1 * 3 * inputSize_ * inputSize_;
    std::vector<float> inputTensorValues(inputTensorSize);
    
    // Chuyển từ HWC sang CHW
    const int channels = img.channels();
    const int h = img.rows;
    const int w = img.cols;
    float* imgData = reinterpret_cast<float*>(img.data);
    
    for (int c = 0; c < channels; ++c) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                inputTensorValues[c * h * w + y * w + x] = imgData[y * w * channels + x * channels + c];
            }
        }
    }

    // Tạo input tensor
    std::vector<int64_t> inputDims = {1, 3, inputSize_, inputSize_};
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorSize,
        inputDims.data(), inputDims.size());

    // Chạy inference
    try {
        auto outputTensors = session_->Run(Ort::RunOptions{nullptr},
                                           inputNames_.data(), &inputTensor, 1,
                                           outputNames_.data(), 1);

        if (outputTensors.empty() || !outputTensors.front().IsTensor()) {
            return {};
        }

        auto& outputTensor = outputTensors.front();
        float* outputData = outputTensor.GetTensorMutableData<float>();
        auto outputShape = outputTensor.GetTensorTypeAndShapeInfo().GetShape();

        // YOLOv5 OCR output thường là [1, N, 6] hoặc [N, 6]
        int numDetections = 1;
        int dims = static_cast<int>(outputShape.size());
        if (dims == 3) {
            numDetections = static_cast<int>(outputShape[1]);
        } else if (dims == 2) {
            numDetections = static_cast<int>(outputShape[0]);
        }

        struct CharDet {
            float cx;
            float conf;
            int cls;
        };
        std::vector<CharDet> chars;

        for (int i = 0; i < numDetections; ++i) {
            int offset = i * 6;
            float x1 = outputData[offset + 0];
            float x2 = outputData[offset + 2];
            float conf = outputData[offset + 4];
            int clsId = static_cast<int>(outputData[offset + 5]);

            if (conf < confThresh_) continue;

            float cx = (x1 + x2) * 0.5f;
            chars.push_back({cx, conf, clsId});
        }

        if (chars.empty()) {
            return {};
        }

        std::sort(chars.begin(), chars.end(),
                  [](const CharDet& a, const CharDet& b) { return a.cx < b.cx; });

        std::string result;
        for (const auto& ch : chars) {
            std::string label;
            if (!classNames_.empty() && ch.cls >= 0 &&
                static_cast<size_t>(ch.cls) < classNames_.size()) {
                label = classNames_[ch.cls];
            } else {
                // fallback: dùng mã lớp dạng số
                label = std::to_string(ch.cls);
            }
            result += label;
        }

        return result;
    } catch (const Ort::Exception& e) {
        std::cerr << "[LPOCR] Loi forward ONNX: " << e.what() << std::endl;
        return {};
    }
}

