#include <iostream>
#include "Pipeline.h"

int main(int argc, char** argv) {
    std::string source = "0"; // mặc định: camera
    if (argc > 1) {
        source = argv[1]; // có thể là "0" hoặc đường dẫn video
    }

    // Đường dẫn model ONNX (export từ export_torchscript.py)
    // Lưu ý: lp_detector_ts.onnx & lp_ocr_ts.onnx nằm trong thư mục Main/
    std::string detectorModel = "Main/lp_detector_ts.onnx";
    std::string ocrModel      = "Main/lp_ocr_ts.onnx";

    Pipeline pipeline;
    if (!pipeline.initialize(source, detectorModel, ocrModel)) {
        std::cerr << "Khoi tao pipeline that bai." << std::endl;
        return -1;
    }

    pipeline.run();
    return 0;
}



