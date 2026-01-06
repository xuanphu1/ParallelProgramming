#include "../include/parallel_pipeline.h"
#include "../include/image_processing.h"
#include "../include/config.h"
#include "../include/utils.h"
#include <iostream>
#include <chrono>
#include <map>

using namespace cv;
using namespace std;
using namespace chrono;

ParallelPipeline::ParallelPipeline(LicensePlateDetector* detector, cv::VideoCapture* cap)
    : detector_(detector), cap_(cap) {
}

ParallelPipeline::~ParallelPipeline() {
    stop();
    wait();
}

void ParallelPipeline::start() {
    running_.store(true);
    stop_requested_.store(false);
    
    // Khởi động các threads
    capture_thread_ = thread(&ParallelPipeline::captureLoop, this);
    detection_thread_ = thread(&ParallelPipeline::detectionLoop, this);
    ocr_thread_ = thread(&ParallelPipeline::ocrLoop, this);
    display_thread_ = thread(&ParallelPipeline::displayLoop, this);
    
    cout << "🚀 Parallel Pipeline đã khởi động với 4 threads:" << endl;
    cout << "   - Capture Thread: Đọc frames từ RTSP" << endl;
    cout << "   - Detection Thread: Chạy YOLOv9 detection" << endl;
    cout << "   - OCR Thread: Xử lý OCR cho các ROI" << endl;
    cout << "   - Display Thread: Hiển thị kết quả" << endl;
}

void ParallelPipeline::stop() {
    stop_requested_.store(true);
    running_.store(false);
    
    // Notify all threads
    detection_cv_.notify_all();
    ocr_cv_.notify_all();
    display_cv_.notify_all();
}

void ParallelPipeline::wait() {
    if (capture_thread_.joinable()) capture_thread_.join();
    if (detection_thread_.joinable()) detection_thread_.join();
    if (ocr_thread_.joinable()) ocr_thread_.join();
    if (display_thread_.joinable()) display_thread_.join();
}

void ParallelPipeline::captureLoop() {
    int frame_id = 0;
    
    while (running_.load() && !stop_requested_.load()) {
        Mat frame;
        if (!cap_->read(frame) || frame.empty()) {
            this_thread::sleep_for(milliseconds(10));
            continue;
        }
        
        frame_id++;
        
        // Tính edge score cho mỗi frame (có thể dùng CUDA)
        double edge_score = calculate_edge_score(frame, SOBEL_THRESHOLD);
        
        // Chỉ gửi frame để detection mỗi N frame
        if (frame_id % DETECTION_INTERVAL == 0) {
            // Frame gating: bỏ qua frame không cần thiết
            if (USE_SOBEL_GATING && edge_score < EDGE_DENSITY_THRESHOLD_LOW) {
                continue;  // Bỏ qua frame này
            }
            
            FramePacket packet;
            packet.frame = frame.clone();
            packet.frame_id = frame_id;
            packet.edge_score = edge_score;
            
            // Push vào detection queue
            {
                unique_lock<mutex> lock(detection_mutex_);
                if (detection_queue_.size() < MAX_QUEUE_SIZE) {
                    detection_queue_.push(packet);
                    detection_cv_.notify_one();
                }
            }
        }
        
        // Luôn gửi frame để display (với frame_id để sync)
        FramePacket display_packet;
        display_packet.frame = frame.clone();
        display_packet.frame_id = frame_id;
        display_packet.edge_score = edge_score;
        
        {
            unique_lock<mutex> lock(display_mutex_);
            // Giữ queue nhỏ để hiển thị real-time
            while (display_queue_.size() >= MAX_QUEUE_SIZE) {
                display_queue_.pop();  // Bỏ frame cũ
            }
            display_queue_.push(display_packet);
            display_cv_.notify_one();
        }
    }
}

void ParallelPipeline::detectionLoop() {
    while (running_.load() || !detection_queue_.empty()) {
        FramePacket packet;
        
        {
            unique_lock<mutex> lock(detection_mutex_);
            detection_cv_.wait(lock, [this] {
                return !detection_queue_.empty() || !running_.load();
            });
            
            if (detection_queue_.empty() && !running_.load()) {
                break;
            }
            
            if (!detection_queue_.empty()) {
                packet = detection_queue_.front();  // Copy
                detection_queue_.pop();
            } else {
                continue;
            }
        }
        
        // Kiểm tra frame có hợp lệ không
        if (packet.frame.empty()) {
            cerr << "[Detection Thread] Error: Frame " << packet.frame_id << " is empty!" << endl;
            continue;
        }
        
        // Chạy detection
        packet.detections = detector_->detect(packet.frame, 0.4f);
        
        cout << "[Detection Thread] Frame " << packet.frame_id 
             << ": " << packet.detections.size() << " detections" << endl;
        
        // Lưu kết quả detection vào buffer
        {
            unique_lock<mutex> lock(results_buffer_mutex_);
            results_buffer_[packet.frame_id] = packet;
            
            // Cleanup buffer cũ (giữ lại tối đa 20 frames)
            if (results_buffer_.size() > 20) {
                auto oldest = results_buffer_.begin();
                results_buffer_.erase(oldest);
            }
        }
        
        // Nếu có detections, gửi sang OCR thread
        if (!packet.detections.empty()) {
            {
                unique_lock<mutex> lock(ocr_mutex_);
                if (ocr_queue_.size() < MAX_QUEUE_SIZE) {
                    ocr_queue_.push(packet);  // Copy để giữ frame cho OCR
                    ocr_cv_.notify_one();
                }
            }
        }
    }
}

void ParallelPipeline::ocrLoop() {
    while (running_.load() || !ocr_queue_.empty()) {
        FramePacket packet;
        
        {
            unique_lock<mutex> lock(ocr_mutex_);
            ocr_cv_.wait(lock, [this] {
                return !ocr_queue_.empty() || !running_.load();
            });
            
            if (ocr_queue_.empty() && !running_.load()) {
                break;
            }
            
            if (!ocr_queue_.empty()) {
                packet = ocr_queue_.front();  // Copy
                ocr_queue_.pop();
            } else {
                continue;
            }
        }
        
        // Kiểm tra frame có hợp lệ không
        if (packet.frame.empty()) {
            cerr << "[OCR Thread] Error: Frame " << packet.frame_id << " is empty!" << endl;
            continue;
        }
        
        // Crop các ROI từ detections
        vector<Mat> plate_rois;
        for (const auto& det : packet.detections) {
            Rect roi(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1);
            if (roi.x >= 0 && roi.y >= 0 && 
                roi.x + roi.width <= packet.frame.cols && 
                roi.y + roi.height <= packet.frame.rows) {
                Mat plate_roi = packet.frame(roi).clone();  // Clone để tránh reference issues
                if (!plate_roi.empty()) {
                    plate_rois.push_back(plate_roi);
                }
            }
        }
        
        // Batch OCR - xử lý tất cả ROI song song (nếu có method ocr_batch)
        // Tạm thời xử lý tuần tự từng ROI
        if (!plate_rois.empty()) {
            packet.ocr_results.resize(plate_rois.size());
            for (size_t i = 0; i < plate_rois.size(); i++) {
                static int plate_counter = 0;
                plate_counter++;
                packet.ocr_results[i] = detector_->ocr(plate_rois[i], plate_counter);
            }
            
            cout << "[OCR Thread] Frame " << packet.frame_id 
                 << ": " << packet.ocr_results.size() << " OCR results" << endl;
            for (size_t i = 0; i < packet.ocr_results.size(); i++) {
                cout << "   Plate " << i << ": " << packet.ocr_results[i].text 
                     << " (conf: " << packet.ocr_results[i].confidence << ")" << endl;
            }
        }
        
        packet.processed = true;
        
        // Cập nhật kết quả OCR vào buffer
        {
            unique_lock<mutex> lock(results_buffer_mutex_);
            auto it = results_buffer_.find(packet.frame_id);
            if (it != results_buffer_.end()) {
                // Cập nhật OCR results và detections
                it->second.ocr_results = packet.ocr_results;
                it->second.detections = packet.detections;
                it->second.processed = true;
            } else {
                // Nếu không tìm thấy, thêm mới
                results_buffer_[packet.frame_id] = packet;
            }
        }
    }
}

void ParallelPipeline::displayLoop() {
    while (running_.load() || !display_queue_.empty()) {
        FramePacket packet;
        
        {
            unique_lock<mutex> lock(display_mutex_);
            display_cv_.wait(lock, [this] {
                return !display_queue_.empty() || !running_.load();
            });
            
            if (display_queue_.empty() && !running_.load()) {
                break;
            }
            
            if (!display_queue_.empty()) {
                packet = display_queue_.front();  // Copy
                display_queue_.pop();
            } else {
                continue;
            }
        }
        
        // Lấy kết quả detection/OCR từ buffer (nếu có)
        FramePacket* result_packet = nullptr;
        {
            unique_lock<mutex> lock(results_buffer_mutex_);
            auto it = results_buffer_.find(packet.frame_id);
            if (it != results_buffer_.end()) {
                result_packet = &it->second;
            }
        }
        
        // Hiển thị frame
        Mat display_frame = packet.frame.clone();
        
        // Vẽ bounding boxes và text nếu có kết quả
        if (result_packet && !result_packet->detections.empty()) {
            for (size_t i = 0; i < result_packet->detections.size() && 
                 i < result_packet->ocr_results.size(); i++) {
                const Detection& det = result_packet->detections[i];
                const OCRResult& ocr = result_packet->ocr_results[i];
                
                // Vẽ bounding box
                rectangle(display_frame, Point(det.x1, det.y1), Point(det.x2, det.y2), 
                        Scalar(0, 255, 0), 2);
                
                // Vẽ text biển số
                string label = ocr.text + " (" + to_string(det.confidence).substr(0, 4) + ")";
                int baseline = 0;
                Size label_size = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
                
                rectangle(display_frame, Point(det.x1, det.y1 - label_size.height - 10),
                         Point(det.x1 + label_size.width, det.y1), Scalar(0, 255, 0), -1);
                putText(display_frame, label, Point(det.x1, det.y1 - 5),
                       FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 2);
            }
        }
        
        // Vẽ thông tin frame
        putText(display_frame, "Frame: " + to_string(packet.frame_id), Point(10, 30),
               FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
        
        if (result_packet && result_packet->processed) {
            putText(display_frame, "Detections: " + to_string(result_packet->detections.size()), 
                   Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
        }
        
        putText(display_frame, "Edge Score: " + to_string(packet.edge_score).substr(0, 4), 
               Point(10, 90), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
        
        // Hiển thị
        imshow("ONNX License Plate Detection - RTSP (Parallel)", display_frame);
        
        char key = waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) {
            stop_requested_.store(true);
            break;
        }
    }
}

