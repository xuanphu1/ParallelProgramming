#include "LPDetector.h"

#include <iostream>

bool LPDetector::initialize(const std::string& modelPath) {
    try {
        module_ = torch::jit::load(modelPath);
        module_.eval();
        initialized_ = true;
    } catch (const c10::Error& e) {
        std::cerr << "[LPDetector] Loi load TorchScript model: " << e.what() << std::endl;
        initialized_ = false;
    }
    return initialized_;
}

std::vector<cv::Rect> LPDetector::detect(const cv::Mat& frame) {
    std::vector<cv::Rect> boxes;
    if (!initialized_ || frame.empty()) {
        return boxes;
    }

    // Tiền xử lý: resize về inputSize_ x inputSize_, BGR->RGB, [0,1]
    cv::Mat img;
    cv::resize(frame, img, cv::Size(inputSize_, inputSize_));
    if (img.channels() == 3) {
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    }
    img.convertTo(img, CV_32F, 1.0 / 255.0);

    auto tensor = torch::from_blob(
        img.data,
        {1, inputSize_, inputSize_, img.channels()},
        torch::kFloat32);
    tensor = tensor.permute({0, 3, 1, 2}); // [1,3,H,W]

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor);

    torch::NoGradGuard no_grad;
    torch::Tensor out;
    try {
        auto iv = module_.forward(inputs);
        if (iv.isTensor()) {
            out = iv.toTensor();
        } else if (iv.isTuple()) {
            auto tup = iv.toTuple();
            if (!tup->elements().empty() && tup->elements()[0].isTensor()) {
                out = tup->elements()[0].toTensor();
            } else {
                return boxes;
            }
        } else {
            return boxes;
        }
    } catch (const c10::Error& e) {
        std::cerr << "[LPDetector] Loi forward TorchScript: " << e.what() << std::endl;
        return boxes;
    }

    if (out.dim() == 3) {
        // [1, N, 6] -> [N, 6]
        out = out.squeeze(0);
    }
    if (out.dim() != 2 || out.size(1) < 6) {
        return boxes;
    }

    auto cpu = out.to(torch::kCPU);
    const int num = static_cast<int>(cpu.size(0));
    auto acc = cpu.accessor<float, 2>();

    const int origW = frame.cols;
    const int origH = frame.rows;
    const float scaleX = static_cast<float>(origW) / static_cast<float>(inputSize_);
    const float scaleY = static_cast<float>(origH) / static_cast<float>(inputSize_);

    for (int i = 0; i < num; ++i) {
        float x1 = acc[i][0];
        float y1 = acc[i][1];
        float x2 = acc[i][2];
        float y2 = acc[i][3];
        float conf = acc[i][4];
        // float cls  = acc[i][5]; // nếu cần phân loại

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

    return boxes;
}

