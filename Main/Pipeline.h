#pragma once

#include <atomic>
#include <string>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

#include <opencv2/opencv.hpp>

#include "LPDetector.h"
#include "LPOCR.h"
#include "SobelCuda.h"
#include "SobelSIMD.h"

// Cấu trúc dữ liệu frame trong pipeline
struct FramePacket {
    cv::Mat frame;
    cv::Mat sobel;
    std::vector<cv::Rect> plates;
    std::string plateText;
    int frameId = 0;
};

// Queue an toàn luồng (template đơn giản)
template<typename T>
class TSQueue {
public:
    explicit TSQueue(size_t maxSize = 10) : maxSize_(maxSize) {}

    void push(const T& value) {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_.wait(lock, [&] { return queue_.size() < maxSize_; });
        queue_.push(value);
        cond_.notify_all();
    }

    bool pop(T& out) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return false;
        }
        out = queue_.front();
        queue_.pop();
        cond_.notify_all();
        return true;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cond_;
    std::queue<T> queue_;
    size_t maxSize_;
};

// Lớp Pipeline chính: quản lý toàn bộ flow CPU/GPU + models
class Pipeline {
public:
    Pipeline();
    ~Pipeline();

    bool initialize(const std::string& source,
                    const std::string& detectorModelPath,
                    const std::string& ocrModelPath);

    void run();   // chạy pipeline (blocking)
    void stop();  // dừng pipeline

private:
    // Các stage
    void captureLoop();
    void sobelLoop();
    void detectLoop();      // Tách riêng detection (Task Parallelism)
    void ocrLoop();         // Tách riêng OCR (Task Parallelism)
    void renderLoop();

    // Sobel GPU/CPU
    // Hiện tại triển khai CPU-only; Sobel GPU có thể thêm lại khi OpenCV CUDA sẵn sàng.
    void sobelGPU(const cv::Mat& src, cv::Mat& dst);
    void sobelCPU(const cv::Mat& src, cv::Mat& dst);

private:
    std::string source_;
    std::string windowName_ = "LP Pipeline";

    cv::VideoCapture cap_;
    bool useGPU_ = false;

    LPDetector detector_;
    LPOCR ocr_;

    std::atomic<bool> running_{false};
    int frameCounter_ = 0;

    TSQueue<FramePacket> qCapture_{5};
    TSQueue<FramePacket> qSobel_{5};
    TSQueue<FramePacket> qDetect_{5};   // Queue riêng cho detection
    TSQueue<FramePacket> qOCR_{5};      // Queue riêng cho OCR
    TSQueue<FramePacket> qRender_{5};   // Queue riêng cho render

    std::thread tCapture_;
    std::thread tSobel_;
    std::thread tDetect_;   // Thread riêng cho detection (Task Parallelism)
    std::thread tOCR_;      // Thread riêng cho OCR (Task Parallelism)
    std::thread tRender_;
};


