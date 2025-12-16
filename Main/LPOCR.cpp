#include "LPOCR.h"

#include <algorithm>
#include <iostream>

bool LPOCR::initialize(const std::string& modelPath) {
    try {
        module_ = torch::jit::load(modelPath);
        module_.eval();
        initialized_ = true;
    } catch (const c10::Error& e) {
        std::cerr << "[LPOCR] Loi load TorchScript model: " << e.what() << std::endl;
        initialized_ = false;
        return false;
    }

    // TODO: nếu bạn có danh sách class cụ thể (alphabet) hãy điền ở đây,
    // ví dụ: classNames_ = {"0","1","2","3","4","5","6","7","8","9","A","B",...};
    classNames_.clear();
    return true;
}

std::string LPOCR::recognize(const cv::Mat& plateRoi) {
    if (!initialized_ || plateRoi.empty()) {
        return {};
    }

    cv::Mat img;
    cv::resize(plateRoi, img, cv::Size(inputSize_, inputSize_));
    if (img.channels() == 3) {
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    }
    img.convertTo(img, CV_32F, 1.0 / 255.0);

    auto tensor = torch::from_blob(
        img.data,
        {1, inputSize_, inputSize_, img.channels()},
        torch::kFloat32);
    tensor = tensor.permute({0, 3, 1, 2}); // [1,C,H,W]

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
                return {};
            }
        } else {
            return {};
        }
    } catch (const c10::Error& e) {
        std::cerr << "[LPOCR] Loi forward TorchScript: " << e.what() << std::endl;
        return {};
    }

    if (out.dim() == 3) {
        out = out.squeeze(0);
    }
    if (out.dim() != 2 || out.size(1) < 6) {
        return {};
    }

    auto cpu = out.to(torch::kCPU);
    const int num = static_cast<int>(cpu.size(0));
    auto acc = cpu.accessor<float, 2>();

    struct CharDet {
        float cx;
        float conf;
        int cls;
    };
    std::vector<CharDet> chars;

    for (int i = 0; i < num; ++i) {
        float x1 = acc[i][0];
        float x2 = acc[i][2];
        float conf = acc[i][4];
        int clsId = static_cast<int>(acc[i][5]);

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
}

