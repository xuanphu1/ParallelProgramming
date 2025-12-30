#include <iostream>
#include "Pipeline.h"

int main(int argc, char** argv) {
    // Mặc định: Webcam local (video0)
    std::string source = "0";
    if (argc > 1) {
        source = argv[1]; // có thể override bằng RTSP URL hoặc đường dẫn video/ảnh
    }

    // Đường dẫn model ONNX mới (đã convert từ bestDetect.pt và bestOCR.pt)
    std::string detectorModel = "models/bestDetect.onnx";
    std::string ocrModel      = "models/bestOCR.onnx";

    Pipeline pipeline;
    if (!pipeline.initialize(source, detectorModel, ocrModel)) {
        std::cerr << "Khoi tao pipeline that bai." << std::endl;
        return -1;
    }

    pipeline.run();
    return 0;
}



