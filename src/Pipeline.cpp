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

    // Kiểm tra xem source có phải là file ảnh không
    bool isImageFile = false;
    if (source != "0" && !source.empty()) {
        std::string ext = source.substr(source.find_last_of(".") + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        isImageFile = (ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp");
    }

    // Mở nguồn video/camera (bỏ qua nếu là ảnh)
    if (!isImageFile && (source == "0" || source.empty())) {
        // Thử mở camera với V4L2 backend
        if (!cap_.open(0, cv::CAP_V4L2)) {
            // Fallback: thử mở không chỉ định backend
            if (!cap_.open(0)) {
                std::cerr << "Khong the mo camera. Kiem tra:" << std::endl;
                std::cerr << "  - Camera co ket noi khong?" << std::endl;
                std::cerr << "  - Quyen truy cap /dev/video0?" << std::endl;
                std::cerr << "  - Camera co dang duoc dung boi ung dung khac?" << std::endl;
                return false;
            }
        }
        // Đặt resolution và FPS
        cap_.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        cap_.set(cv::CAP_PROP_FPS, 30);
        std::cout << "[Pipeline] Camera da mo thanh cong (V4L2)." << std::endl;
    } else if (!isImageFile) {
        // Mở file video (không phải ảnh)
        if (!cap_.open(source)) {
            std::cerr << "Khong the mo video: " << source << std::endl;
            return false;
        }
    } else {
        // Là file ảnh, không cần mở VideoCapture
        // Sẽ đọc trong captureLoop()
        std::cout << "[Pipeline] Source la file anh: " << source << std::endl;
    }

    // GPU: dùng CUDA Sobel nếu được build với hỗ trợ CUDA (macro USE_CUDA_SOBEL)
    #ifdef USE_CUDA_SOBEL
    useGPU_ = true;
    std::cout << "[Pipeline] CUDA support: ENABLED" << std::endl;
    #else
    useGPU_ = false;
    std::cout << "[Pipeline] CUDA support: DISABLED (using SIMD + OpenMP)" << std::endl;
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

    // Khởi động các luồng (Task Parallelism: tách Detection và OCR)
    tCapture_ = std::thread(&Pipeline::captureLoop, this);
    tSobel_   = std::thread(&Pipeline::sobelLoop, this);
    tDetect_  = std::thread(&Pipeline::detectLoop, this);   // Thread riêng cho detection
    tOCR_     = std::thread(&Pipeline::ocrLoop, this);      // Thread riêng cho OCR
    tRender_  = std::thread(&Pipeline::renderLoop, this);

    // Chờ các luồng kết thúc
    tCapture_.join();
    tSobel_.join();
    tDetect_.join();
    tOCR_.join();
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
    // Nếu source là file ảnh (không phải camera/video)
    if (source_ != "0" && !source_.empty() && !cap_.isOpened()) {
        // Kiểm tra xem có phải là file ảnh không
        std::string ext = source_.substr(source_.find_last_of(".") + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        
        if (ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp") {
            // Đọc ảnh tĩnh
            cv::Mat frame = cv::imread(source_);
            if (frame.empty()) {
                std::cerr << "[Capture] Khong the doc anh: " << source_ << std::endl;
                running_.store(false);
                return;
            }
            
            std::cout << "[Capture] Doc anh thanh cong: " << source_ 
                      << " (" << frame.cols << "x" << frame.rows << ")" << std::endl;
            
            FramePacket pkt;
            pkt.frame = frame;
            pkt.frameId = frameCounter_++;
            qCapture_.push(pkt);
            
            // Đối với ảnh tĩnh, chỉ xử lý 1 lần rồi dừng
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            running_.store(false);
            return;
        }
    }
    
    // Xử lý video/camera stream
    int consecutiveFailures = 0;
    const int maxFailures = 10;
    
    while (running_.load()) {
        FramePacket pkt;
        if (!cap_.read(pkt.frame) || pkt.frame.empty()) {
            consecutiveFailures++;
            if (consecutiveFailures >= maxFailures) {
                std::cerr << "[Capture] Lien tuc khong doc duoc frame. Dung capture loop." << std::endl;
                running_.store(false);
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        
        consecutiveFailures = 0; // Reset counter khi đọc được frame
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

        // Data Parallelism: Thử CUDA -> SIMD -> CPU OpenMP
        bool processed = false;
        
        #ifdef USE_CUDA_SOBEL
        if (useGPU_ && sobelCuda(gray, pkt.sobel)) {
            processed = true;
        }
        #endif
        
        if (!processed && !sobelSIMD(gray, pkt.sobel)) {
            sobelCPU(gray, pkt.sobel);
        }

        qSobel_.push(pkt);
    }
}

// Task Parallelism: Tách Detection và OCR thành 2 threads riêng
void Pipeline::detectLoop() {
    while (running_.load() || !qSobel_.empty()) {
        FramePacket pkt;
        if (!qSobel_.pop(pkt)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // Chỉ làm detection ở đây
        pkt.plates = detector_.detect(pkt.frame);

        // Đẩy vào queue OCR (có thể có nhiều biển số)
        qDetect_.push(pkt);
    }
}

void Pipeline::ocrLoop() {
    while (running_.load() || !qDetect_.empty()) {
        FramePacket pkt;
        if (!qDetect_.pop(pkt)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // Data Parallelism: Xử lý nhiều ROI song song (nếu có nhiều biển số)
        if (!pkt.plates.empty()) {
            std::vector<std::string> ocrResults(pkt.plates.size());
            
            // Task Parallelism: Xử lý tất cả ROI song song
            #pragma omp parallel for
            for (size_t i = 0; i < pkt.plates.size(); ++i) {
                const cv::Rect& r = pkt.plates[i];
                cv::Rect roiRect = r & cv::Rect(0, 0, pkt.frame.cols, pkt.frame.rows);
                cv::Mat plateRoi = pkt.frame(roiRect).clone();
                ocrResults[i] = ocr_.recognize(plateRoi);
            }
            
            // Ghép kết quả (lấy biển số đầu tiên có text)
            for (const auto& text : ocrResults) {
                if (!text.empty()) {
                    pkt.plateText = text;
                    break;
                }
            }
        }

        qRender_.push(pkt);
    }
}

void Pipeline::renderLoop() {
    while (running_.load() || !qRender_.empty()) {
        FramePacket pkt;
        if (!qRender_.pop(pkt)) {
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
    // Tạm thời fallback về CPU (CUDA cần build riêng)
    sobelCPU(src, dst);
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


