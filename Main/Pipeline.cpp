#include "Pipeline.h"

#include <iostream>

Pipeline::Pipeline() = default;

Pipeline::~Pipeline() {
    stop();
}

bool Pipeline::initialize(const std::string& source,
                          const std::string& detectorModelPath,
                          const std::string& ocrModelPath) {
    source_ = source;

    // Mở nguồn video/camera
    if (source == "0" || source.empty()) {
        if (!cap_.open(0)) {
            std::cerr << "Khong the mo camera." << std::endl;
            return false;
        }
    } else {
        if (!cap_.open(source)) {
            std::cerr << "Khong the mo video: " << source << std::endl;
            return false;
        }
    }

    // GPU: dùng CUDA Sobel nếu được build với hỗ trợ CUDA (macro USE_CUDA_SOBEL)
#ifdef USE_CUDA_SOBEL
    useGPU_ = true;
    std::cout << "[Pipeline] GPU Sobel: ON (CUDA)" << std::endl;
#else
    useGPU_ = false;
    std::cout << "[Pipeline] GPU Sobel: OFF (dùng CPU OpenMP)" << std::endl;
#endif

    // Khởi tạo models
    if (!detector_.initialize(detectorModelPath)) {
        std::cerr << "Khong the khoi tao LPDetector tu: " << detectorModelPath << std::endl;
        return false;
    }
    if (!ocr_.initialize(ocrModelPath)) {
        std::cerr << "Khong the khoi tao LPOCR tu: " << ocrModelPath << std::endl;
        return false;
    }

    running_.store(true);
    return true;
}

void Pipeline::run() {
    if (!running_.load()) {
        std::cerr << "Pipeline chua duoc initialize()." << std::endl;
        return;
    }

    cv::namedWindow(windowName_, cv::WINDOW_NORMAL);

    // Khởi động các luồng
    tCapture_ = std::thread(&Pipeline::captureLoop, this);
    tSobel_   = std::thread(&Pipeline::sobelLoop, this);
    tDetOcr_  = std::thread(&Pipeline::detectOcrLoop, this);
    tRender_  = std::thread(&Pipeline::renderLoop, this);

    // Chờ các luồng kết thúc
    tCapture_.join();
    tSobel_.join();
    tDetOcr_.join();
    tRender_.join();

    cv::destroyWindow(windowName_);
}

void Pipeline::stop() {
    if (running_.load()) {
        running_.store(false);
    }
}

// ==================== Các loop ====================

void Pipeline::captureLoop() {
    while (running_.load()) {
        FramePacket pkt;
        if (!cap_.read(pkt.frame) || pkt.frame.empty()) {
            std::cerr << "[Capture] Het frame hoac loi doc." << std::endl;
            running_.store(false);
            break;
        }
        pkt.frameId = frameCounter_++;
        qCapture_.push(pkt);
    }
}

void Pipeline::sobelLoop() {
    while (running_.load() || !qCapture_.empty()) {
        FramePacket pkt;
        if (!qCapture_.pop(pkt)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // Chuyển thành ảnh xám
        cv::Mat gray;
        if (pkt.frame.channels() == 3) {
            cv::cvtColor(pkt.frame, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = pkt.frame;
        }

        if (useGPU_) {
            if (!sobelCuda(gray, pkt.sobel)) {
                // fallback CPU nếu CUDA lỗi
                sobelCPU(gray, pkt.sobel);
            }
        } else {
            sobelCPU(gray, pkt.sobel);
        }

        qSobel_.push(pkt);
    }
}

void Pipeline::detectOcrLoop() {
    while (running_.load() || !qSobel_.empty()) {
        FramePacket pkt;
        if (!qSobel_.pop(pkt)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // Dùng frame gốc để detect (hoặc có thể dùng sobel như thêm kênh thông tin)
        pkt.plates = detector_.detect(pkt.frame);

        // Lấy vùng biển số đầu tiên để OCR (có thể mở rộng nhiều vùng)
        if (!pkt.plates.empty()) {
            const cv::Rect& r = pkt.plates[0];
            cv::Rect roiRect = r & cv::Rect(0, 0, pkt.frame.cols, pkt.frame.rows);
            cv::Mat plateRoi = pkt.frame(roiRect).clone();
            pkt.plateText = ocr_.recognize(plateRoi);
        }

        qDetOcr_.push(pkt);
    }
}

void Pipeline::renderLoop() {
    while (running_.load() || !qDetOcr_.empty()) {
        FramePacket pkt;
        if (!qDetOcr_.pop(pkt)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        cv::Mat display = pkt.frame.clone();

        // Vẽ bounding box biển số
        for (const auto& r : pkt.plates) {
            cv::rectangle(display, r, cv::Scalar(0, 255, 0), 2);
        }

        // Vẽ text biển số
        if (!pkt.plateText.empty()) {
            cv::putText(display, pkt.plateText, cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

            // Ở đây: gửi MQTT một lần khi có biển số (bạn tự tích hợp)
            std::cout << "[Frame " << pkt.frameId << "] Bien so: " << pkt.plateText << std::endl;
        }

        cv::imshow(windowName_, display);
        if (cv::waitKey(1) == 'q') {
            running_.store(false);
            break;
        }
    }
}

// ==================== Sobel GPU/CPU ====================

void Pipeline::sobelGPU(const cv::Mat& src, cv::Mat& dst) {
    // Hàm wrapper nếu sau này muốn gọi trực tiếp trong code khác
    if (!sobelCuda(src, dst)) {
        sobelCPU(src, dst);
    }
}

void Pipeline::sobelCPU(const cv::Mat& src, cv::Mat& dst) {
    CV_Assert(src.type() == CV_8UC1);
    dst = cv::Mat::zeros(src.size(), CV_8UC1);

    int rows = src.rows;
    int cols = src.cols;

    #pragma omp parallel for
    for (int y = 1; y < rows - 1; ++y) {
        for (int x = 1; x < cols - 1; ++x) {
            int p00 = src.at<uchar>(y - 1, x - 1);
            int p01 = src.at<uchar>(y - 1, x    );
            int p02 = src.at<uchar>(y - 1, x + 1);
            int p10 = src.at<uchar>(y    , x - 1);
            int p11 = src.at<uchar>(y    , x    );
            int p12 = src.at<uchar>(y    , x + 1);
            int p20 = src.at<uchar>(y + 1, x - 1);
            int p21 = src.at<uchar>(y + 1, x    );
            int p22 = src.at<uchar>(y + 1, x + 1);

            int gx = -p00 - 2*p10 - p20 + p02 + 2*p12 + p22;
            int gy = -p00 - 2*p01 - p02 + p20 + 2*p21 + p22;

            float mag = std::sqrt(static_cast<float>(gx * gx + gy * gy));
            if (mag > 255.0f) mag = 255.0f;

            dst.at<uchar>(y, x) = static_cast<uchar>(mag);
        }
    }
}


