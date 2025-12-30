#include "Pipeline.h"
#ifdef USE_CUDA_SOBEL
#include "SobelCuda.h"
#endif

#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>

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
        // Mở file video hoặc RTSP stream
        // Kiểm tra xem có phải RTSP stream không
        bool isRTSP = (source.find("rtsp://") == 0);
        
        if (isRTSP) {
            // RTSP stream: dùng GStreamer pipeline để xử lý authentication tốt hơn
            std::cout << "[Pipeline] Dang ket noi RTSP stream: " << source << std::endl;
            
            bool opened = false;
            
            // Thử với GStreamer pipeline trước (xử lý authentication tốt hơn)
            // Parse username và password từ URL
            std::string username, password, rtspUrl;
            size_t userPos = source.find("://");
            size_t atPos = source.find('@');
            if (userPos != std::string::npos && atPos != std::string::npos) {
                std::string authPart = source.substr(userPos + 3, atPos - userPos - 3);
                size_t colonPos = authPart.find(':');
                if (colonPos != std::string::npos) {
                    username = authPart.substr(0, colonPos);
                    password = authPart.substr(colonPos + 1);
                    rtspUrl = "rtsp://" + source.substr(atPos + 1);
                }
            }
            
            // GStreamer pipeline với user-id và user-pw
            std::string gstPipeline;
            if (!username.empty() && !password.empty()) {
                gstPipeline = "rtspsrc location=" + rtspUrl + 
                    " user-id=" + username + " user-pw=" + password +
                    " latency=0 ! decodebin ! videoconvert ! appsink";
            } else {
                gstPipeline = "rtspsrc location=" + source + 
                    " latency=0 ! decodebin ! videoconvert ! appsink";
            }
            
            std::cout << "[Pipeline] Thu GStreamer pipeline..." << std::endl;
            if (cap_.open(gstPipeline, cv::CAP_GSTREAMER)) {
                opened = true;
                std::cout << "[Pipeline] Ket noi RTSP thanh cong voi GStreamer pipeline." << std::endl;
            } else {
                // Thử với FFmpeg backend
                std::cout << "[Pipeline] GStreamer that bai, thu FFmpeg backend..." << std::endl;
                if (cap_.open(source, cv::CAP_FFMPEG)) {
                    opened = true;
                    std::cout << "[Pipeline] Ket noi RTSP thanh cong voi FFmpeg backend." << std::endl;
                } else {
                    // Thử các format URL RTSP phổ biến khác
                    std::vector<std::string> rtspVariants;
                    
                    // Tách base URL
                    size_t atPos = source.find('@');
                    if (atPos != std::string::npos) {
                        std::string base = source.substr(atPos + 1);  // ip:port
                        std::string auth = source.substr(0, atPos + 1);  // rtsp://user:pass@
                        
                        // Các format phổ biến
                        rtspVariants.push_back(auth + base + "/stream1");
                        rtspVariants.push_back(auth + base + "/cam/realmonitor");
                        rtspVariants.push_back(auth + base + "/h264");
                        rtspVariants.push_back(auth + base + "/live");
                    }
                    
                    // Thử từng variant với GStreamer
                    for (const auto& variant : rtspVariants) {
                        std::cout << "[Pipeline] Thu format khac voi GStreamer: " << variant << std::endl;
                        // Parse variant URL
                        std::string varUsername, varPassword, varRtspUrl;
                        size_t varUserPos = variant.find("://");
                        size_t varAtPos = variant.find('@');
                        if (varUserPos != std::string::npos && varAtPos != std::string::npos) {
                            std::string varAuthPart = variant.substr(varUserPos + 3, varAtPos - varUserPos - 3);
                            size_t varColonPos = varAuthPart.find(':');
                            if (varColonPos != std::string::npos) {
                                varUsername = varAuthPart.substr(0, varColonPos);
                                varPassword = varAuthPart.substr(varColonPos + 1);
                                varRtspUrl = "rtsp://" + variant.substr(varAtPos + 1);
                            }
                        }
                        
                        std::string gstVariant;
                        if (!varUsername.empty() && !varPassword.empty()) {
                            gstVariant = "rtspsrc location=" + varRtspUrl + 
                                " user-id=" + varUsername + " user-pw=" + varPassword +
                                " latency=0 ! decodebin ! videoconvert ! appsink";
                        } else {
                            gstVariant = "rtspsrc location=" + variant + 
                                " latency=0 ! decodebin ! videoconvert ! appsink";
                        }
                        
                        if (cap_.open(gstVariant, cv::CAP_GSTREAMER)) {
                            opened = true;
                            std::cout << "[Pipeline] Ket noi RTSP thanh cong voi format: " << variant << std::endl;
                            break;
                        }
                    }
                    
                    // Fallback: thử với backend mặc định
                    if (!opened) {
                        std::cout << "[Pipeline] Thu backend mac dinh..." << std::endl;
                        if (cap_.open(source)) {
                            opened = true;
                        }
                    }
                }
            }
            
            if (!opened) {
                std::cerr << "Khong the mo RTSP stream: " << source << std::endl;
                std::cerr << "Kiem tra:" << std::endl;
                std::cerr << "  - Username/password co dung khong?" << std::endl;
                std::cerr << "  - Camera co dang chay khong? (test voi: ffplay " << source << ")" << std::endl;
                std::cerr << "  - URL RTSP co dung format khong?" << std::endl;
                std::cerr << "  - Co the thu cac format:" << std::endl;
                std::cerr << "    * " << source << "/stream1" << std::endl;
                std::cerr << "    * " << source << "/cam/realmonitor" << std::endl;
                std::cerr << "    * " << source << "/h264" << std::endl;
                return false;
            }
            
            // Đặt các thuộc tính cho RTSP stream
            cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);  // Giảm buffer để giảm latency
            std::cout << "[Pipeline] RTSP stream da ket noi thanh cong." << std::endl;
        } else {
            // File video thông thường, HTTP stream, UDP stream, hoặc named pipe
            bool isHTTP = (source.find("http://") == 0 || source.find("https://") == 0);
            bool isUDP = (source.find("udp://") == 0);
            bool isPipe = (source.find("/tmp/rtsp_fifo") == 0 || source.find("/tmp/") == 0);
            
            if (isUDP) {
                // UDP stream từ VLC bridge: thử nhiều format
                std::cout << "[Pipeline] Doc UDP stream: " << source << std::endl;
                std::cout << "[Pipeline] Dang cho UDP stream san sang (co the mat 5-10 giay)..." << std::endl;
                
                // Đợi một chút để VLC bắt đầu stream
                std::this_thread::sleep_for(std::chrono::seconds(3));
                
                // Thử 1: FFmpeg backend với format mặc định
                if (cap_.open(source, cv::CAP_FFMPEG)) {
                    std::cout << "[Pipeline] UDP stream da ket noi thanh cong voi FFmpeg." << std::endl;
                } else {
                    // Thử 2: Format với buffer size
                    std::string udpWithBuffer = source + "?buffer_size=65536";
                    if (cap_.open(udpWithBuffer, cv::CAP_FFMPEG)) {
                        std::cout << "[Pipeline] UDP stream da ket noi thanh cong voi buffer." << std::endl;
                    } else {
                        // Thử 3: GStreamer pipeline
                        std::string gstPipeline = "udpsrc port=" + source.substr(source.find_last_of(':') + 1) + 
                            " ! application/x-rtp,encoding-name=H264,payload=96 ! rtph264depay ! decodebin ! videoconvert ! appsink";
                        std::cout << "[Pipeline] Thu GStreamer pipeline cho UDP stream..." << std::endl;
                        if (cap_.open(gstPipeline, cv::CAP_GSTREAMER)) {
                            std::cout << "[Pipeline] UDP stream da ket noi thanh cong voi GStreamer." << std::endl;
                        } else {
                            std::cerr << "Khong the mo UDP stream: " << source << std::endl;
                            std::cerr << "Kiem tra VLC bridge co dang chay khong?" << std::endl;
                            return false;
                        }
                    }
                }
            } else if (isHTTP) {
                // HTTP stream từ VLC bridge: thử nhiều cách
                std::cout << "[Pipeline] Doc HTTP stream: " << source << std::endl;
                
                // Thử 1: FFmpeg backend
                if (cap_.open(source, cv::CAP_FFMPEG)) {
                    std::cout << "[Pipeline] HTTP stream da ket noi thanh cong voi FFmpeg." << std::endl;
                } else {
                    // Thử 2: GStreamer pipeline với decodebin (tự động detect codec)
                    std::string gstPipeline = "souphttpsrc location=" + source + 
                        " ! decodebin ! videoconvert ! appsink";
                    std::cout << "[Pipeline] Thu GStreamer pipeline cho HTTP stream..." << std::endl;
                    if (cap_.open(gstPipeline, cv::CAP_GSTREAMER)) {
                        std::cout << "[Pipeline] HTTP stream da ket noi thanh cong voi GStreamer." << std::endl;
                    } else {
                        std::cerr << "Khong the mo HTTP stream: " << source << std::endl;
                        std::cerr << "Kiem tra VLC bridge co dang chay khong?" << std::endl;
                        std::cerr << "  ps aux | grep vlc" << std::endl;
                        std::cerr << "  netstat -tlnp | grep 8080" << std::endl;
                        return false;
                    }
                }
            } else if (isPipe) {
                std::cout << "[Pipeline] Doc tu named pipe (ffmpeg bridge): " << source << std::endl;
                // Đọc raw video từ pipe (format: rawvideo, bgr24)
                // Cần biết resolution trước, tạm thời dùng 1920x1080
                // Hoặc có thể đọc header từ pipe
                if (!cap_.open(source)) {
                    std::cerr << "Khong the mo named pipe: " << source << std::endl;
                    return false;
                }
            } else {
                // File video thông thường
                if (!cap_.open(source)) {
                    std::cerr << "Khong the mo video: " << source << std::endl;
                    return false;
                }
            }
        }
    } else {
        // Là file ảnh, không cần mở VideoCapture
        // Sẽ đọc trong captureLoop()
        std::cout << "[Pipeline] Source la file anh: " << source << std::endl;
    }

    // Data Parallelism: Chỉ dùng CUDA (bắt buộc)
    #ifdef USE_CUDA_SOBEL
    useGPU_ = true;
    std::cout << "[Pipeline] Data Parallelism: CUDA (GPU)" << std::endl;
    #else
    std::cerr << "[Pipeline] ERROR: USE_CUDA_SOBEL must be defined. Build with CUDA support." << std::endl;
    return false;
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

        // Data Parallelism: Chỉ dùng CUDA (bắt buộc)
        auto sobelStart = std::chrono::high_resolution_clock::now();
        
        #ifdef USE_CUDA_SOBEL
        if (!sobelCuda(gray, pkt.sobel)) {
            std::cerr << "[Sobel] CUDA Sobel failed for frame " << pkt.frameId << std::endl;
            continue;  // Bỏ qua frame nếu CUDA lỗi
        }
        #else
        #error "USE_CUDA_SOBEL must be defined. Build with CUDA support."
        #endif
        
        auto sobelEnd = std::chrono::high_resolution_clock::now();
        auto sobelTime = std::chrono::duration_cast<std::chrono::microseconds>(sobelEnd - sobelStart).count();
        pkt.sobelTimeMs = sobelTime / 1000.0;

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
        auto detectStart = std::chrono::high_resolution_clock::now();
        pkt.plates = detector_.detect(pkt.frame);
        auto detectEnd = std::chrono::high_resolution_clock::now();
        auto detectTime = std::chrono::duration_cast<std::chrono::microseconds>(detectEnd - detectStart).count();
        pkt.detectTimeMs = detectTime / 1000.0;

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
        auto ocrStart = std::chrono::high_resolution_clock::now();
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
        auto ocrEnd = std::chrono::high_resolution_clock::now();
        auto ocrTime = std::chrono::duration_cast<std::chrono::microseconds>(ocrEnd - ocrStart).count();
        pkt.ocrTimeMs = ocrTime / 1000.0;

        qRender_.push(pkt);
    }
}

void Pipeline::renderLoop() {
    // Khởi tạo FPS tracking
    auto lastFpsTime = std::chrono::high_resolution_clock::now();
    int fpsFrameCount = 0;
    double currentFps = 0.0;
    const double fpsUpdateInterval = 1.0; // Cập nhật FPS mỗi 1 giây
    
    while (running_.load() || !qRender_.empty()) {
        FramePacket pkt;
        if (!qRender_.pop(pkt)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // Tính FPS
        auto currentTime = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastFpsTime).count() / 1000.0;
        fpsFrameCount++;
        
        if (elapsed >= fpsUpdateInterval) {
            currentFps = fpsFrameCount / elapsed;
            fpsFrameCount = 0;
            lastFpsTime = currentTime;
        }

        cv::Mat display = pkt.frame.clone();

        // Vẽ bounding box biển số
        for (const auto& r : pkt.plates) {
            cv::rectangle(display, r, cv::Scalar(0, 255, 0), 2);
        }

        // Hiển thị FPS và profiling info (góc trên bên phải)
        std::stringstream fpsText;
        fpsText << "FPS: " << std::fixed << std::setprecision(1) << currentFps;
        cv::putText(display, fpsText.str(), cv::Point(display.cols - 200, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        
        // Hiển thị thời gian từng stage (nếu có)
        int yOffset = 60;
        if (pkt.sobelTimeMs > 0) {
            std::stringstream sobelText;
            sobelText << "Sobel(CUDA): " << std::fixed << std::setprecision(2) << pkt.sobelTimeMs << "ms";
            cv::putText(display, sobelText.str(), cv::Point(display.cols - 250, yOffset),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
            yOffset += 25;
        }
        if (pkt.detectTimeMs > 0) {
            std::stringstream detectText;
            detectText << "Detect(CPU): " << std::fixed << std::setprecision(2) << pkt.detectTimeMs << "ms";
            cv::putText(display, detectText.str(), cv::Point(display.cols - 250, yOffset),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
            yOffset += 25;
        }
        if (pkt.ocrTimeMs > 0) {
            std::stringstream ocrText;
            ocrText << "OCR(CPU): " << std::fixed << std::setprecision(2) << pkt.ocrTimeMs << "ms";
            cv::putText(display, ocrText.str(), cv::Point(display.cols - 250, yOffset),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
        }

        // Vẽ text biển số (góc trên bên trái)
        if (!pkt.plateText.empty()) {
            cv::putText(display, pkt.plateText, cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

            // Ở đây: gửi MQTT một lần khi có biển số (bạn tự tích hợp)
            std::cout << "[Frame " << pkt.frameId << "] Bien so: " << pkt.plateText 
                      << " | FPS: " << std::fixed << std::setprecision(1) << currentFps << std::endl;
        }

        cv::imshow(windowName_, display);
        if (cv::waitKey(1) == 'q') {
            running_.store(false);
            break;
        }
    }
}

// Sobel filter chỉ dùng CUDA (xem SobelCuda.cu)


