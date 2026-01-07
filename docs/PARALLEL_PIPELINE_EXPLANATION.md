# Giải Thích Chi Tiết: parallel_pipeline.cpp

## 📋 Mục Lục
1. [Tổng Quan](#tổng-quan)
2. [Kiến Trúc Pipeline](#kiến-trúc-pipeline)
3. [Cấu Trúc Dữ Liệu](#cấu-trúc-dữ-liệu)
4. [Chi Tiết Từng Hàm](#chi-tiết-từng-hàm)
5. [Luồng Xử Lý](#luồng-xử-lý)
6. [Đồng Bộ Hóa Threads](#đồng-bộ-hóa-threads)
7. [Tối Ưu Hóa](#tối-ưu-hóa)

---

## Tổng Quan

File `parallel_pipeline.cpp` triển khai **Task Parallelism** (Song song hóa tác vụ) cho hệ thống nhận diện biển số xe. Thay vì xử lý tuần tự (Capture → Detection → OCR → Display), pipeline chia thành **4 threads độc lập** chạy song song:

- **Capture Thread**: Đọc frames từ RTSP camera
- **Detection Thread**: Chạy YOLOv9 để phát hiện biển số
- **OCR Thread**: Nhận diện ký tự trên các biển số đã phát hiện
- **Display Thread**: Hiển thị kết quả lên màn hình

### Lợi Ích
- **Giảm Latency**: Các stage chạy song song, không phải đợi nhau
- **Tăng Throughput**: Xử lý nhiều frames đồng thời
- **Real-time Display**: Hiển thị liên tục không bị block bởi detection/OCR

---

## Kiến Trúc Pipeline

```
┌─────────────────┐
│  RTSP Camera    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│     CAPTURE THREAD                  │
│  - Đọc frame từ RTSP                │
│  - Tính edge_score (Sobel)          │
│  - Frame gating (bỏ qua frame xấu)  │
└────────┬────────────────────────────┘
         │
         ├─────────────────┬──────────────────┐
         ▼                 ▼                  ▼
    ┌─────────┐      ┌──────────┐      ┌──────────┐
    │Detection│      │  Display │      │  Buffer  │
    │ Queue   │      │  Queue   │      │ (Map)    │
    └────┬────┘      └────┬─────┘      └────┬─────┘
         │                 │                  │
         ▼                 │                  │
┌─────────────────┐        │                  │
│ DETECTION THREAD │        │                  │
│ - YOLOv9 detect  │        │                  │
│ - Lưu vào buffer │        │                  │
└────────┬────────┘        │                  │
         │                 │                  │
         ▼                 │                  │
    ┌─────────┐            │                  │
    │ OCR     │            │                  │
    │ Queue   │            │                  │
    └────┬────┘            │                  │
         │                 │                  │
         ▼                 │                  │
┌─────────────────┐        │                  │
│   OCR THREAD     │        │                  │
│ - Crop ROI       │        │                  │
│ - OCR từng ROI   │        │                  │
│ - Update buffer  │        │                  │
└────────┬────────┘        │                  │
         │                 │                  │
         └─────────────────┴──────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ DISPLAY THREAD │
                    │ - Lấy kết quả  │
                    │ - Vẽ bbox/text │
                    │ - Hiển thị     │
                    └───────────────┘
```

---

## Cấu Trúc Dữ Liệu

### FramePacket
Struct để truyền dữ liệu giữa các threads:

```cpp
struct FramePacket {
    cv::Mat frame;                      // Frame ảnh (clone để tránh mất dữ liệu)
    int frame_id;                       // ID duy nhất của frame
    double edge_score;                  // Điểm edge từ Sobel filter
    std::vector<Detection> detections;  // Kết quả detection (bounding boxes)
    std::vector<OCRResult> ocr_results; // Kết quả OCR (text biển số)
    bool processed;                     // Đã xử lý xong chưa
};
```

**Lưu ý quan trọng**: Copy constructor và assignment operator đều **clone** frame để tránh mất dữ liệu khi move giữa các threads.

### Queues
- **detection_queue_**: Chứa frames cần detection (mỗi `DETECTION_INTERVAL` frame)
- **ocr_queue_**: Chứa frames đã có detections, cần OCR
- **display_queue_**: Chứa tất cả frames để hiển thị liên tục

### results_buffer_
Map `frame_id → FramePacket` để đồng bộ kết quả:
- Detection thread lưu detections vào buffer
- OCR thread cập nhật OCR results vào buffer
- Display thread lấy kết quả từ buffer theo `frame_id`

---

## Chi Tiết Từng Hàm

### 1. Constructor & Destructor

```cpp
ParallelPipeline::ParallelPipeline(LicensePlateDetector* detector, cv::VideoCapture* cap)
    : detector_(detector), cap_(cap) {
}
```

- Nhận con trỏ đến `LicensePlateDetector` và `VideoCapture`
- Khởi tạo các queues, mutexes, condition variables

```cpp
ParallelPipeline::~ParallelPipeline() {
    stop();
    wait();
}
```

- Tự động dừng và đợi tất cả threads khi object bị hủy

---

### 2. start() - Khởi Động Pipeline

```cpp
void ParallelPipeline::start() {
    running_.store(true);
    stop_requested_.store(false);
    
    // Khởi động các threads
    capture_thread_ = thread(&ParallelPipeline::captureLoop, this);
    detection_thread_ = thread(&ParallelPipeline::detectionLoop, this);
    ocr_thread_ = thread(&ParallelPipeline::ocrLoop, this);
    display_thread_ = thread(&ParallelPipeline::displayLoop, this);
    
    cout << "🚀 Parallel Pipeline đã khởi động với 4 threads:" << endl;
    // ...
}
```

**Chức năng**:
- Set flags `running_` và `stop_requested_`
- Tạo 4 threads, mỗi thread chạy một loop function
- In thông báo khởi động

---

### 3. stop() & wait() - Dừng Pipeline

```cpp
void ParallelPipeline::stop() {
    stop_requested_.store(true);
    running_.store(false);
    
    // Notify all threads
    detection_cv_.notify_all();
    ocr_cv_.notify_all();
    display_cv_.notify_all();
}
```

- Set flags để các threads biết cần dừng
- Notify tất cả condition variables để đánh thức threads đang chờ

```cpp
void ParallelPipeline::wait() {
    if (capture_thread_.joinable()) capture_thread_.join();
    if (detection_thread_.joinable()) detection_thread_.join();
    if (ocr_thread_.joinable()) ocr_thread_.join();
    if (display_thread_.joinable()) display_thread_.join();
}
```

- Đợi tất cả threads kết thúc (join)

---

### 4. captureLoop() - Thread Đọc Frame

**Mục đích**: Đọc frames từ RTSP, tính edge score, áp dụng frame gating, và phân phối frames đến các queues.

```cpp
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
```

**Bước 1**: Đọc frame từ RTSP
- Tăng `frame_id` mỗi frame
- Nếu không đọc được, sleep 10ms và thử lại

**Bước 2**: Tính edge score
- Gọi `calculate_edge_score()` (có thể dùng CUDA hoặc CPU OpenMP)
- Edge score dùng để frame gating (bỏ qua frame không có biển số)

```cpp
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
```

**Bước 3**: Gửi frame đến Detection Queue (mỗi `DETECTION_INTERVAL` frame)
- **Frame Gating**: Nếu `USE_SOBEL_GATING` và `edge_score < EDGE_DENSITY_THRESHOLD_LOW` → bỏ qua frame (tiết kiệm tài nguyên)
- Tạo `FramePacket`, **clone frame** để tránh mất dữ liệu
- Push vào `detection_queue_` (thread-safe với mutex)
- Notify detection thread bằng condition variable

**Tại sao mỗi N frame?**
- Detection (YOLOv9) tốn nhiều tài nguyên
- Không cần detect mọi frame (biển số không thay đổi nhanh)
- `DETECTION_INTERVAL = 15` → detect mỗi 15 frames

```cpp
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
```

**Bước 4**: Gửi frame đến Display Queue (mọi frame)
- **Luôn gửi** để hiển thị liên tục (real-time)
- Nếu queue đầy, **bỏ frame cũ** để giữ queue nhỏ (tránh lag)
- Notify display thread

---

### 5. detectionLoop() - Thread Detection

**Mục đích**: Nhận frames từ detection queue, chạy YOLOv9, lưu kết quả vào buffer, và gửi sang OCR queue nếu có detections.

```cpp
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
```

**Bước 1**: Chờ và lấy frame từ queue
- **Condition Variable**: Chờ đến khi queue không rỗng hoặc `running_ = false`
- **Copy** packet (không move) để giữ frame cho OCR thread sau này
- Nếu queue rỗng và đã dừng → break

```cpp
        // Kiểm tra frame có hợp lệ không
        if (packet.frame.empty()) {
            cerr << "[Detection Thread] Error: Frame " << packet.frame_id << " is empty!" << endl;
            continue;
        }
        
        // Chạy detection
        packet.detections = detector_->detect(packet.frame, 0.4f);
        
        cout << "[Detection Thread] Frame " << packet.frame_id 
             << ": " << packet.detections.size() << " detections" << endl;
```

**Bước 2**: Chạy YOLOv9 Detection
- Kiểm tra frame hợp lệ
- Gọi `detector_->detect()` với confidence threshold 0.4
- Lưu kết quả vào `packet.detections` (vector các bounding boxes)

```cpp
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
```

**Bước 3**: Lưu vào results_buffer_
- Lưu packet (có detections) vào buffer theo `frame_id`
- **Cleanup**: Giữ tối đa 20 frames để tránh memory leak

```cpp
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
```

**Bước 4**: Gửi sang OCR Queue
- Chỉ gửi nếu có detections
- **Copy** packet (không move) để giữ frame cho OCR crop ROI
- Notify OCR thread

---

### 6. ocrLoop() - Thread OCR

**Mục đích**: Nhận frames có detections, crop các ROI (Region of Interest), chạy OCR, và cập nhật buffer.

```cpp
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
```

**Bước 1**: Chờ và lấy frame từ OCR queue (tương tự detection loop)

```cpp
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
```

**Bước 2**: Crop ROI từ detections
- Duyệt qua tất cả detections
- Tạo `Rect` từ bounding box (x1, y1, x2, y2)
- **Validate**: Kiểm tra ROI nằm trong frame
- **Clone** ROI để tránh reference issues (frame có thể bị giải phóng)

```cpp
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
```

**Bước 3**: Chạy OCR cho từng ROI
- **Hiện tại**: Xử lý tuần tự từng ROI (có thể tối ưu bằng `ocr_batch()`)
- Gọi `detector_->ocr()` cho mỗi ROI
- Lưu kết quả vào `packet.ocr_results`

**Tối ưu tương lai**: Có thể dùng `detector_->ocr_batch(plate_rois)` để xử lý song song nhiều ROI.

```cpp
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
```

**Bước 4**: Cập nhật buffer
- Đánh dấu `processed = true`
- Tìm packet trong buffer theo `frame_id`
- **Cập nhật** OCR results và detections (không ghi đè frame)
- Nếu không tìm thấy → thêm mới

---

### 7. displayLoop() - Thread Hiển Thị

**Mục đích**: Nhận frames từ display queue, lấy kết quả detection/OCR từ buffer, vẽ bounding boxes và text, hiển thị lên màn hình.

```cpp
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
```

**Bước 1**: Chờ và lấy frame từ display queue

```cpp
        // Lấy kết quả detection/OCR từ buffer (nếu có)
        FramePacket* result_packet = nullptr;
        {
            unique_lock<mutex> lock(results_buffer_mutex_);
            auto it = results_buffer_.find(packet.frame_id);
            if (it != results_buffer_.end()) {
                result_packet = &it->second;
            }
        }
```

**Bước 2**: Lấy kết quả từ buffer
- Tìm packet trong buffer theo `frame_id`
- Lưu con trỏ để truy cập detections và OCR results

**Tại sao dùng frame_id?**
- Display thread nhận frame **ngay lập tức** (real-time)
- Detection/OCR có thể **chậm hơn** (mất vài frames)
- Dùng `frame_id` để **đồng bộ** kết quả với frame hiển thị

```cpp
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
```

**Bước 3**: Vẽ bounding boxes và text
- Clone frame để vẽ
- Nếu có kết quả detection/OCR:
  - Vẽ **bounding box** màu xanh lá
  - Vẽ **text biển số** + confidence trên bounding box
  - Background màu xanh lá, text màu đen

```cpp
        // Vẽ thông tin frame
        putText(display_frame, "Frame: " + to_string(packet.frame_id), Point(10, 30),
               FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
        
        if (result_packet && result_packet->processed) {
            putText(display_frame, "Detections: " + to_string(result_packet->detections.size()), 
                   Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
        }
        
        putText(display_frame, "Edge Score: " + to_string(packet.edge_score).substr(0, 4), 
               Point(10, 90), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
```

**Bước 4**: Vẽ thông tin debug
- Frame ID
- Số detections (nếu đã processed)
- Edge score

```cpp
        // Hiển thị
        imshow("ONNX License Plate Detection - RTSP (Parallel)", display_frame);
        
        char key = waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) {
            stop_requested_.store(true);
            break;
        }
```

**Bước 5**: Hiển thị và xử lý input
- Hiển thị cửa sổ OpenCV
- Nhấn `q` hoặc `Esc` để dừng pipeline

---

## Luồng Xử Lý

### Ví Dụ Timeline

```
Time    Capture    Detection    OCR         Display
─────────────────────────────────────────────────────
T0      Frame 0    -            -           Frame 0 (no results)
T1      Frame 1    -            -           Frame 1 (no results)
T2      Frame 2    -            -           Frame 2 (no results)
...
T15     Frame 15   Frame 15     -           Frame 15 (no results)
T16     Frame 16   [detecting]  -           Frame 16 (no results)
T17     Frame 17   -            -           Frame 17 (no results)
T18     Frame 18   -            Frame 15    Frame 18 (no results)
T19     Frame 19   -            [OCR...]    Frame 19 (no results)
T20     Frame 20   -            -           Frame 20 (no results)
T21     Frame 21   -            -           Frame 15 (with bbox + text!)
T22     Frame 22   Frame 30     -           Frame 22 (no results)
...
```

**Quan sát**:
- Display luôn hiển thị frames mới nhất (real-time)
- Detection/OCR chạy chậm hơn, kết quả hiển thị sau vài frames
- Kết quả được đồng bộ bằng `frame_id`

---

## Đồng Bộ Hóa Threads

### Mutexes
- **detection_mutex_**: Bảo vệ `detection_queue_`
- **ocr_mutex_**: Bảo vệ `ocr_queue_`
- **display_mutex_**: Bảo vệ `display_queue_`
- **results_buffer_mutex_**: Bảo vệ `results_buffer_`

### Condition Variables
- **detection_cv_**: Đánh thức detection thread khi có frame mới
- **ocr_cv_**: Đánh thức OCR thread khi có detections
- **display_cv_**: Đánh thức display thread khi có frame mới

### Atomic Flags
- **running_**: Pipeline đang chạy
- **stop_requested_**: Yêu cầu dừng

### Pattern: Producer-Consumer
- **Capture** → Producer cho Detection và Display
- **Detection** → Consumer của Capture, Producer cho OCR
- **OCR** → Consumer của Detection
- **Display** → Consumer của Capture và Buffer

---

## Tối Ưu Hóa

### Đã Áp Dụng
1. **Frame Gating**: Bỏ qua frames không có biển số (dựa trên edge score)
2. **Detection Interval**: Chỉ detect mỗi N frames (tiết kiệm tài nguyên)
3. **Queue Size Limit**: Giới hạn queue để tránh memory overflow
4. **Buffer Cleanup**: Xóa frames cũ trong buffer (giữ tối đa 20)
5. **Clone Frames**: Tránh mất dữ liệu khi truyền giữa threads

### Có Thể Cải Thiện
1. **Batch OCR**: Dùng `ocr_batch()` thay vì loop tuần tự
2. **Dynamic Detection Interval**: Tự động điều chỉnh dựa trên FPS
3. **Priority Queue**: Ưu tiên frames có edge score cao
4. **GPU Memory Pool**: Tái sử dụng GPU memory cho CUDA operations

---

## Kết Luận

File `parallel_pipeline.cpp` triển khai **Task Parallelism** hiệu quả, cho phép hệ thống xử lý real-time với độ trễ thấp. Kiến trúc 4 threads độc lập với queues và buffer đồng bộ đảm bảo:
- **Real-time display** không bị block
- **Detection/OCR** chạy song song, không chặn capture
- **Đồng bộ kết quả** chính xác bằng `frame_id`

Đây là một ví dụ điển hình của **Pipeline Parallelism** trong xử lý video real-time.

