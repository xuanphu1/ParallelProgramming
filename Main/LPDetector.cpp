#include "LPDetector.h"

#include <iostream>
#include <memory>

LPDetector::LPDetector() : env_(ORT_LOGGING_LEVEL_WARNING, "LPDetector") {
}

LPDetector::~LPDetector() {
    if (session_) {
        delete session_;
        session_ = nullptr;
    }
}

bool LPDetector::initialize(const std::string& modelPath) {
    try {
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        session_ = new Ort::Session(env_, modelPath.c_str(), sessionOptions);

        // Lấy thông tin input/output
        size_t numInputNodes = session_->GetInputCount();
        size_t numOutputNodes = session_->GetOutputCount();

        if (numInputNodes == 0 || numOutputNodes == 0) {
            std::cerr << "[LPDetector] Model khong co input/output hop le." << std::endl;
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
        inputShape_ = inputTensorInfo.GetShape();

        if (inputShape_.size() == 4 && inputShape_[2] > 0 && inputShape_[3] > 0) {
            inputSize_ = static_cast<int>(inputShape_[2]);
        }

        initialized_ = true;
        std::cout << "[LPDetector] Loaded ONNX model: " << modelPath 
                  << " (input size: " << inputSize_ << "x" << inputSize_ << ")" << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "[LPDetector] Loi load ONNX model: " << e.what() << std::endl;
        initialized_ = false;
    }
    return initialized_;
}

std::vector<cv::Rect> LPDetector::detect(const cv::Mat& frame) {
    std::vector<cv::Rect> boxes;
    if (!initialized_ || frame.empty() || !session_) {
        return boxes;
    }

    // Tiền xử lý: resize về inputSize_ x inputSize_, BGR->RGB, [0,1]
    cv::Mat img;
    cv::resize(frame, img, cv::Size(inputSize_, inputSize_));
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
            return boxes;
        }

        auto& outputTensor = outputTensors.front();
        float* outputData = outputTensor.GetTensorMutableData<float>();
        auto outputShape = outputTensor.GetTensorTypeAndShapeInfo().GetShape();

        // YOLOv5 output thường là [1, N, 6] hoặc [N, 6] với format: x1, y1, x2, y2, conf, cls
        int numDetections = 1;
        int dims = static_cast<int>(outputShape.size());
        if (dims == 3) {
            numDetections = static_cast<int>(outputShape[1]);
        } else if (dims == 2) {
            numDetections = static_cast<int>(outputShape[0]);
        }

        const int origW = frame.cols;
        const int origH = frame.rows;
        const float scaleX = static_cast<float>(origW) / static_cast<float>(inputSize_);
        const float scaleY = static_cast<float>(origH) / static_cast<float>(inputSize_);

        for (int i = 0; i < numDetections; ++i) {
            int offset = i * 6;
            float x1 = outputData[offset + 0];
            float y1 = outputData[offset + 1];
            float x2 = outputData[offset + 2];
            float y2 = outputData[offset + 3];
            float conf = outputData[offset + 4];
            // float cls = outputData[offset + 5];

            if (conf < confThresh_) {
                continue;
            }

            int ix1 = static_cast<int>(x1 * scaleX);
            int iy1 = static_cast<int>(y1 * scaleY);
            int ix2 = static_cast<int>(x2 * scaleX);
            int iy2 = static_cast<int>(y2 * scaleY);

            ix1 = std::max(0, std::min(ix1, origW - 1));
            iy1 = std::max(0, std::min(iy1, origH - 1));
            ix2 = std::max(0, std::min(ix2, origW - 1));
            iy2 = std::max(0, std::min(iy2, origH - 1));

            int w = std::max(0, ix2 - ix1);
            int h = std::max(0, iy2 - iy1);
            if (w > 0 && h > 0) {
                boxes.emplace_back(ix1, iy1, w, h);
            }
        }
    } catch (const Ort::Exception& e) {
        std::cerr << "[LPDetector] Loi forward ONNX: " << e.what() << std::endl;
    }

    return boxes;
}

