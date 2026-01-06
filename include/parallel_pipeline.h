#ifndef PARALLEL_PIPELINE_H
#define PARALLEL_PIPELINE_H

#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <condition_variable>
#include "license_plate_detector.h"
#include "types.h"

// Frame packet để truyền giữa các threads
struct FramePacket {
    cv::Mat frame;
    int frame_id;
    double edge_score;
    std::vector<Detection> detections;
    std::vector<OCRResult> ocr_results;
    bool processed = false;
    
    FramePacket() : frame_id(0), edge_score(0.0), processed(false) {}
    
    // Copy constructor để tránh mất frame khi move
    FramePacket(const FramePacket& other) 
        : frame(other.frame.clone()), frame_id(other.frame_id), 
          edge_score(other.edge_score), detections(other.detections),
          ocr_results(other.ocr_results), processed(other.processed) {}
    
    FramePacket& operator=(const FramePacket& other) {
        if (this != &other) {
            frame = other.frame.clone();
            frame_id = other.frame_id;
            edge_score = other.edge_score;
            detections = other.detections;
            ocr_results = other.ocr_results;
            processed = other.processed;
        }
        return *this;
    }
};

// Pipeline song song với threads riêng cho từng stage
class ParallelPipeline {
private:
    // Threads
    std::thread capture_thread_;
    std::thread detection_thread_;
    std::thread ocr_thread_;
    std::thread display_thread_;
    
    // Queues thread-safe
    std::queue<FramePacket> detection_queue_;
    std::queue<FramePacket> ocr_queue_;
    std::queue<FramePacket> display_queue_;
    
    // Shared buffer để lưu kết quả detection/OCR theo frame_id
    std::map<int, FramePacket> results_buffer_;
    
    // Mutexes và condition variables
    std::mutex detection_mutex_;
    std::mutex ocr_mutex_;
    std::mutex display_mutex_;
    std::mutex results_buffer_mutex_;
    
    std::condition_variable detection_cv_;
    std::condition_variable ocr_cv_;
    std::condition_variable display_cv_;
    
    // Control flags
    std::atomic<bool> running_{false};
    std::atomic<bool> stop_requested_{false};
    
    // Detector và VideoCapture
    LicensePlateDetector* detector_;
    cv::VideoCapture* cap_;
    
    // Queue sizes
    static constexpr size_t MAX_QUEUE_SIZE = 5;
    static constexpr int DETECTION_INTERVAL = 15;
    
    // Thread functions
    void captureLoop();
    void detectionLoop();
    void ocrLoop();
    void displayLoop();
    
public:
    ParallelPipeline(LicensePlateDetector* detector, cv::VideoCapture* cap);
    ~ParallelPipeline();
    
    void start();
    void stop();
    void wait();
};

#endif // PARALLEL_PIPELINE_H

