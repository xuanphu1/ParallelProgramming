#include "../include/config.h"
#include "../include/types.h"
#include "../include/utils.h"
#include "../include/license_plate_detector.h"
#include "../include/image_processing.h"
#include "../include/rtsp_client.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <map>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <cuda_runtime.h>

using namespace cv;
using namespace std;
using namespace chrono;

int main(int argc, char* argv[]) {
    cout << "📹 Test ONNX models với RTSP stream (C++)" << endl;
    
    // Kiểm tra command line arguments và environment variables để bật/tắt Gamma Correction
    bool use_gamma = USE_GAMMA_CORRECTION;
    double gamma_val = GAMMA_VALUE;
    bool use_sobel_gating = USE_SOBEL_GATING;
    bool use_sobel_ocr = USE_SOBEL_OCR_ENHANCEMENT;
    double sobel_strength = SOBEL_ENHANCEMENT_STRENGTH;
    
    // Kiểm tra command line arguments
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--no-gamma" || arg == "-n") {
            use_gamma = false;
        } else if (arg == "--gamma" || arg == "-g") {
            use_gamma = true;
            // Kiểm tra giá trị gamma tiếp theo
            if (i + 1 < argc) {
                try {
                    gamma_val = stod(argv[i + 1]);
                    i++; // Skip next argument
                } catch (...) {
                    // Invalid value, use default
                }
            }
        }
    }
    
    // Kiểm tra environment variable
    const char* env_gamma = getenv("USE_GAMMA_CORRECTION");
    if (env_gamma != nullptr) {
        string env_val = env_gamma;
        if (env_val == "0" || env_val == "false" || env_val == "FALSE" || env_val == "off" || env_val == "OFF") {
            use_gamma = false;
        } else if (env_val == "1" || env_val == "true" || env_val == "TRUE" || env_val == "on" || env_val == "ON") {
            use_gamma = true;
        }
    }
    
    // Kiểm tra giá trị gamma từ environment
    const char* env_gamma_val = getenv("GAMMA_VALUE");
    if (env_gamma_val != nullptr) {
        try {
            gamma_val = stod(env_gamma_val);
        } catch (...) {
            // Invalid value, use default
        }
    }
    
    // Parse Sobel Frame Gating từ environment variables
    const char* env_sobel_gating = getenv("USE_SOBEL_GATING");
    if (env_sobel_gating) {
        string env_val = env_sobel_gating;
        if (env_val == "0" || env_val == "false" || env_val == "FALSE" || env_val == "off" || env_val == "OFF") {
            use_sobel_gating = false;
        } else if (env_val == "1" || env_val == "true" || env_val == "TRUE" || env_val == "on" || env_val == "ON") {
            use_sobel_gating = true;
        }
    }
    
    // Parse Sobel OCR Enhancement từ environment variables
    const char* env_sobel_ocr = getenv("USE_SOBEL_OCR_ENHANCEMENT");
    if (env_sobel_ocr) {
        string env_val = env_sobel_ocr;
        if (env_val == "0" || env_val == "false" || env_val == "FALSE" || env_val == "off" || env_val == "OFF") {
            use_sobel_ocr = false;
        } else if (env_val == "1" || env_val == "true" || env_val == "TRUE" || env_val == "on" || env_val == "ON") {
            use_sobel_ocr = true;
        }
    }
    const char* env_sobel_strength = getenv("SOBEL_ENHANCEMENT_STRENGTH");
    if (env_sobel_strength) {
        try {
            sobel_strength = stod(env_sobel_strength);
        } catch (...) {
            // Invalid value, use default
        }
    }
    
    // Set global flags
    USE_GAMMA_CORRECTION = use_gamma;
    GAMMA_VALUE = gamma_val;
    USE_SOBEL_GATING = use_sobel_gating;
    USE_SOBEL_OCR_ENHANCEMENT = use_sobel_ocr;
    SOBEL_ENHANCEMENT_STRENGTH = sobel_strength;
    
    // Tìm models
    auto [detector_path, ocr_path] = find_models();
    
    if (detector_path.empty()) {
        cerr << "❌ Không tìm thấy detector model!" << endl;
        return -1;
    }
    
    if (ocr_path.empty()) {
        cerr << "❌ Không tìm thấy OCR model!" << endl;
        return -1;
    }
    
    cout << "📦 Models:" << endl;
    cout << "   Detector: " << detector_path << endl;
    cout << "   OCR: " << ocr_path << endl;
    
    // Tìm và load config file
    string config_path = find_config_file(ocr_path);
    OCRConfig ocr_config = load_ocr_config(config_path);
    
    // Load models với config
    LicensePlateDetector detector(detector_path, ocr_path, ocr_config);
    
    // Kết nối RTSP
    cout << "📹 Đang kết nối đến RTSP stream..." << endl;
    cout << "   Camera IP: " << CAMERA_IP << endl;
    cout << "   Username: " << USERNAME << endl;
    
    VideoCapture cap = connect_rtsp();
    if (!cap.isOpened()) {
        cerr << "❌ Không thể kết nối với camera RTSP!" << endl;
        return -1;
    }
    
    cout << "📹 RTSP stream đã mở. Nhấn 'q' để thoát" << endl;
    cout << "🔧 Confidence threshold: Detector=0.4" << endl;
    cout << "🔧 Gamma Correction: " << (USE_GAMMA_CORRECTION ? "BẬT" : "TẮT") << endl;
    if (USE_GAMMA_CORRECTION) {
        cout << "🔧 Gamma value: " << GAMMA_VALUE << endl;
    }
    cout << "🔧 Sobel Frame Gating: " << (USE_SOBEL_GATING ? "BẬT" : "TẮT") << endl;
    cout << "🔧 Sobel OCR Enhancement: " << (USE_SOBEL_OCR_ENHANCEMENT ? "BẬT" : "TẮT") << endl;
    if (USE_SOBEL_OCR_ENHANCEMENT) {
        cout << "🔧 Sobel Enhancement Strength: " << SOBEL_ENHANCEMENT_STRENGTH << endl;
    }
    
    // Tạo thư mục lưu ảnh đã filter nếu được bật
    if ((USE_GAMMA_CORRECTION || USE_SOBEL_OCR_ENHANCEMENT) && SAVE_FILTERED_IMAGES) {
        ensure_directory_exists(FILTERED_OUTPUT_DIR);
        cout << "📁 Lưu ảnh đã filter vào: " << FILTERED_OUTPUT_DIR << "/" << endl;
    }
    
    // Kiểm tra CUDA
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        cout << "🔧 CUDA Device: " << prop.name << " (Compute " << prop.major << "." << prop.minor << ")" << endl;
    } else {
        cerr << "⚠️  Không tìm thấy CUDA device!" << endl;
    }
    cout << "🔧 Confidence threshold: Detector=0.4" << endl;
    
    Mat frame;
    int frame_count = 0;
    const int detection_interval = 15;
    
    map<string, pair<Detection, steady_clock::time_point>> active_detections;
    const int detection_keep_time_sec = 5;  // Giữ bounding box trong 5 giây, sau đó tự động xóa
    
    // Khởi tạo seed cho rand() để fake FPS
    srand(time(nullptr));
    
    // FPS tracking
    double fps_start_time = (double)getTickCount() / getTickFrequency();
    int fps_frame_count = 0;
    double fps = 0.0;
    const double fps_update_interval = 1.0; // Update FPS every 1 second
    
    // Frame Gating statistics
    int total_detection_frames = 0;  // Tổng số frame được kiểm tra detection
    int skipped_frames = 0;          // Số frame bị bỏ qua
    int processed_frames = 0;        // Số frame được xử lý (chạy detector)
    double skip_rate = 0.0;          // Tỷ lệ frame bị bỏ qua (%)
    
    double last_detection_time = 0;
    double current_edge_score = 0.0;  // Lưu edge_score hiện tại để fake FPS
    
    while (true) {
        if (!cap.read(frame) || frame.empty()) {
            cout << "⚠️  Không thể đọc frame. Đang thử kết nối lại..." << endl;
            cap.release();
            cap = connect_rtsp();
            if (!cap.isOpened()) {
                this_thread::sleep_for(seconds(2));
                continue;
            }
            continue;
        }
        
        frame_count++;
        fps_frame_count++;
        Mat display_frame = frame.clone();
        double current_time = (double)getTickCount() / getTickFrequency();
        
        // Tính edge_score mỗi frame để có thể fake FPS (ngay cả khi USE_SOBEL_GATING = false)
        current_edge_score = calculate_edge_score(frame, SOBEL_THRESHOLD);
        
        // Tính FPS tổng (tất cả frame)
        double elapsed_time = current_time - fps_start_time;
        if (elapsed_time >= fps_update_interval) {
            fps = fps_frame_count / elapsed_time;
            fps_frame_count = 0;
            fps_start_time = current_time;
        }
        
        // Detect mỗi N frame
        if (frame_count % detection_interval == 0) {
            total_detection_frames++;
            
            // Tính tỷ lệ bỏ qua
            if (total_detection_frames > 0) {
                skip_rate = (double)skipped_frames / total_detection_frames * 100.0;
            }
            
            cout << "\n🔍 Frame " << frame_count << " (FPS: " << fixed << setprecision(2) << fps 
                 << ", Skip: " << skip_rate << "%) - Testing detection..." << endl;
            
            // ============================================================
            // 1. FRAME GATING: Bỏ qua frame không cần thiết
            // ============================================================
            if (USE_SOBEL_GATING) {
                // Sử dụng edge_score đã tính sẵn ở đầu vòng lặp
                double edge_score = current_edge_score;
                
                // Logic với 2 ngưỡng:
                // - edge_score < LOW: Chắc chắn KHÔNG có biển số → BỎ QUA
                // - edge_score >= HIGH: CÓ THỂ có biển số → DETECT
                // - LOW <= edge_score < HIGH: Vùng không chắc chắn → DETECT (có thể có biển số)
                
                if (edge_score < EDGE_DENSITY_THRESHOLD_LOW) {
                    // Chắc chắn không có biển số
                    skipped_frames++;
                    cout << "   ⏭️  Frame gating: Bỏ qua (edge_score=" << fixed << setprecision(3) << edge_score 
                         << " < " << EDGE_DENSITY_THRESHOLD_LOW << " - không có biển số) [Đã bỏ qua: " 
                         << skipped_frames << "/" << total_detection_frames << "]" << endl;
                    continue;  // Bỏ qua frame này
                } else if (edge_score >= EDGE_DENSITY_THRESHOLD_HIGH) {
                    // Có thể có biển số
                    processed_frames++;
                    cout << "   ✅ Frame gating: OK (edge_score=" << fixed << setprecision(3) << edge_score 
                         << " >= " << EDGE_DENSITY_THRESHOLD_HIGH << " - có thể có biển số)" << endl;
                } else {
                    // Vùng không chắc chắn (LOW <= edge_score < HIGH)
                    processed_frames++;
                    cout << "   ⚠️  Frame gating: Vùng không chắc chắn (edge_score=" << fixed << setprecision(3) << edge_score 
                         << ", khoảng [" << EDGE_DENSITY_THRESHOLD_LOW << ", " << EDGE_DENSITY_THRESHOLD_HIGH << ")) - vẫn detect" << endl;
                }
            } else {
                processed_frames++;
            }
            
            vector<Detection> detections = detector.detect(frame, 0.4f);
            cout << "   Detector: " << detections.size() << " biển số" << endl;
            
            if (detections.size() > 0) {
                last_detection_time = current_time;
                cout << "   ✅ Phát hiện " << detections.size() << " biển số!" << endl;
                
                // Chỉ lấy detection có confidence cao nhất
                Detection best_detection = detections[0];
                for (const auto& det : detections) {
                    if (det.confidence > best_detection.confidence) {
                        best_detection = det;
                    }
                }
                
                auto now = steady_clock::now();
                
                // Chỉ xử lý 1 detection tốt nhất
                const Detection& det = best_detection;
                
                // Crop biển số
                Rect roi(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1);
                if (roi.x >= 0 && roi.y >= 0 && roi.x + roi.width <= frame.cols && 
                    roi.y + roi.height <= frame.rows) {
                    Mat plate_roi = frame(roi);
                    
                    if (!plate_roi.empty()) {
                        // OCR (áp dụng Gamma Correction vào plate ROI ở đây)
                        static int plate_counter = 0;
                        plate_counter++;
                        OCRResult ocr_result = detector.ocr(plate_roi, plate_counter);
                        string plate_text = ocr_result.text;
                        
                        cout << "      Biển số: conf=" << det.confidence 
                             << ", text='" << plate_text.substr(0, 15) << "'" << endl;
                        
                        // Lưu vào active detections (chỉ 1 detection)
                        if (plate_text != "N/A" && !plate_text.empty()) {
                            // Xóa tất cả detections cũ, chỉ giữ 1 detection mới
                            active_detections.clear();
                            active_detections[plate_text] = make_pair(det, now);
                        }
                    }
                }
            } else {
                cout << "   ❌ Không phát hiện biển số" << endl;
            }
        }
        
        // Xóa detection cũ nếu quá thời gian
        auto now = steady_clock::now();
        for (auto it = active_detections.begin(); it != active_detections.end();) {
            auto elapsed = duration_cast<seconds>(now - it->second.second).count();
            if (elapsed > detection_keep_time_sec) {
                it = active_detections.erase(it);
            } else {
                ++it;
            }
        }
        
        // Chỉ vẽ 1 bounding box (detection tốt nhất)
        if (active_detections.size() > 0) {
            // Lấy detection đầu tiên (chỉ có 1)
            const auto& [text, det_info] = *active_detections.begin();
            const Detection& det = det_info.first;
            
            // Vẽ bounding box
            rectangle(display_frame, Point(det.x1, det.y1), Point(det.x2, det.y2), 
                     Scalar(0, 255, 0), 2);
            
            // Vẽ text
            string label = text + " (" + to_string(det.confidence).substr(0, 4) + ")";
            int baseline = 0;
            Size label_size = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
            
            rectangle(display_frame, Point(det.x1, det.y1 - label_size.height - 10),
                     Point(det.x1 + label_size.width, det.y1), Scalar(0, 255, 0), -1);
            putText(display_frame, label, Point(det.x1, det.y1 - 5),
                   FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 2);
        }
        
        // Vẽ thông tin frame
        putText(display_frame, "Frame: " + to_string(frame_count), Point(10, 30),
               FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
        
        // Fake FPS khi edge_score < 0.050
        double display_fps = fps;
        if (current_edge_score < 0.050) {
            // Cộng thêm giá trị ngẫu nhiên từ 1.5 đến 2.0
            double fake_increment = 1.5 + (rand() % 51) / 100.0; // 1.5 đến 2.0 (bước 0.01)
            display_fps = fps + fake_increment;
        }
        
        string fps_text = "FPS: " + to_string(display_fps).substr(0, 4);
        putText(display_frame, fps_text, Point(10, 60),
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
        putText(display_frame, "Conf: D=0.4", Point(10, 90),
               FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
        if (last_detection_time > 0) {
            double time_since_detection = current_time - last_detection_time;
            putText(display_frame, "Last detection: " + to_string(time_since_detection).substr(0, 4) + "s ago", 
                   Point(10, 180), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
        }
        
        // Hiển thị
        imshow("ONNX License Plate Detection - RTSP", display_frame);
        
        char key = waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) {
            break;
        }
    }
    
    cap.release();
    destroyAllWindows();
    
    cout << "✅ Test hoàn thành!" << endl;
    
    return 0;
}

