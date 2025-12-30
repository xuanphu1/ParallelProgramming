# Các Ý Tưởng Song Song Hóa cho Dự Án Nhận Diện Biển Số

## Tổng Quan Pipeline Hiện Tại

```
[Capture] → [Sobel Filter] → [Detection] → [OCR] → [Render]
```

---

## 1. DATA PARALLELISM (Song song hóa theo dữ liệu)

### 1.1. Sobel Filter - Pixel-level Parallelism

**Hiện tại**: OpenMP `#pragma omp parallel for` trên vòng lặp `y`

**Cải tiến có thể:**

#### A. **2D Block Decomposition**
```cpp
// Chia ảnh thành các block 2D (ví dụ 64x64 pixels)
// Mỗi thread xử lý 1 block độc lập
#pragma omp parallel for collapse(2)
for (int by = 0; by < numBlocksY; ++by) {
    for (int bx = 0; bx < numBlocksX; ++bx) {
        // Xử lý block [bx, by]
    }
}
```

#### B. **SIMD Vectorization (AVX/SSE)**
```cpp
// Xử lý 8 pixels cùng lúc với AVX-256
#include <immintrin.h>
__m256i gx_vec = _mm256_loadu_si256(...);
__m256i gy_vec = _mm256_loadu_si256(...);
__m256 mag = _mm256_sqrt_ps(_mm256_add_ps(gx2, gy2));
```

#### C. **GPU CUDA Kernel - Fine-grained Parallelism**
```cuda
// Mỗi thread xử lý 1 pixel
__global__ void sobel_kernel(uchar* src, uchar* dst, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > 0 && x < w-1 && y > 0 && y < h-1) {
        // Tính Sobel cho pixel (x,y)
    }
}
```

### 1.2. Model Inference - Batch Processing

**Hiện tại**: Xử lý 1 frame tại một thời điểm

**Cải tiến:**

#### A. **Batch Inference trên GPU**
```cpp
// Gom nhiều frame lại thành batch [N, 3, H, W]
// Inference 1 lần cho cả batch → tăng throughput
std::vector<cv::Mat> frames; // N frames
// Chuyển thành tensor [N, 3, 640, 640]
// ONNX Runtime: session.Run(..., batch_tensor)
```

#### B. **Multiple ROI Processing**
```cpp
// Nếu detect được nhiều biển số trong 1 frame
// Xử lý OCR cho tất cả ROI song song
#pragma omp parallel for
for (size_t i = 0; i < plateROIs.size(); ++i) {
    ocrResults[i] = ocr_.recognize(plateROIs[i]);
}
```

---

## 2. TASK PARALLELISM (Song song hóa theo tác vụ)

### 2.1. Pipeline Stages - Multi-threading

**Hiện tại**: 4 threads (capture, sobel, detect+ocr, render)

**Cải tiến:**

#### A. **Tách Detection và OCR thành 2 threads riêng**
```
Thread 1: Capture
Thread 2: Sobel
Thread 3: Detection (chỉ detect, không OCR)
Thread 4: OCR (xử lý nhiều ROI song song)
Thread 5: Render
```

#### B. **Pre-processing Pipeline Song Song**
```cpp
// Nhiều pre-processing chạy song song:
// - Resize
// - Normalize
// - Color space conversion
// - Noise reduction
std::vector<std::future<cv::Mat>> futures;
futures.push_back(std::async(std::launch::async, resize, frame));
futures.push_back(std::async(std::launch::async, normalize, frame));
// ...
```

### 2.2. Model Inference - Async Execution

```cpp
// Detection và OCR chạy song song (nếu có 2 GPU hoặc CPU đủ mạnh)
auto detFuture = std::async(std::launch::async, 
    [&]() { return detector_.detect(frame); });
auto ocrFuture = std::async(std::launch::async,
    [&]() { return ocr_.recognize(roi); });
```

---

## 3. PIPELINE PARALLELISM (Song song hóa theo pipeline)

### 3.1. Multi-Frame Pipeline

**Ý tưởng**: Xử lý nhiều frame cùng lúc ở các stage khác nhau

```
Frame N:   [Capture] → [Sobel] → [Detect] → [OCR] → [Render]
Frame N+1: [Capture] → [Sobel] → [Detect] → [OCR] → [Render]
Frame N+2: [Capture] → [Sobel] → [Detect] → [OCR] → [Render]
```

**Triển khai:**
```cpp
// Mỗi stage có buffer riêng
TSQueue<FramePacket> qCapture{10};  // Buffer 10 frames
TSQueue<FramePacket> qSobel{10};
TSQueue<FramePacket> qDetect{10};
TSQueue<FramePacket> qOCR{10};
// → Pipeline có thể xử lý 10 frames "trong không khí" cùng lúc
```

### 3.2. Double Buffering / Triple Buffering

```cpp
// Capture vào buffer A, xử lý buffer B
cv::Mat bufferA, bufferB;
std::thread t1([&]() {
    while (running) {
        cap.read(bufferA);  // Capture vào A
        std::swap(bufferA, bufferB);  // Swap
    }
});
std::thread t2([&]() {
    while (running) {
        process(bufferB);  // Xử lý B
    }
});
```

---

## 4. GPU PARALLELISM (Song song hóa trên GPU)

### 4.1. CUDA Streams - Overlap Computation và Transfer

```cuda
cudaStream_t stream1, stream2, stream3;

// Stream 1: Upload frame N+1 lên GPU
cudaMemcpyAsync(..., stream1);

// Stream 2: Chạy Sobel cho frame N
sobel_kernel<<<..., stream2>>>(...);

// Stream 3: Download kết quả frame N-1 về CPU
cudaMemcpyAsync(..., stream3);

// → 3 operations chạy song song trên GPU
```

### 4.2. Multi-GPU Processing

```cpp
// Nếu có 2 GPU:
// GPU 0: Detection model
// GPU 1: OCR model
// → Chạy song song hoàn toàn
```

### 4.3. Unified Memory (CUDA)

```cuda
// Dùng cudaMallocManaged() để GPU và CPU share memory
// → Không cần copy, GPU truy cập trực tiếp
```

---

## 5. HYBRID PARALLELISM (Kết hợp nhiều kỹ thuật)

### 5.1. CPU + GPU Hybrid

```
CPU Thread Pool:
  - Thread 1-4: Sobel filter (OpenMP, SIMD)
  - Thread 5: Pre-processing (resize, normalize)
  - Thread 6: Post-processing (NMS, parse results)

GPU:
  - CUDA Stream 1: Detection model
  - CUDA Stream 2: OCR model
  - CUDA Stream 3: Sobel (nếu muốn)
```

### 5.2. SIMD + OpenMP + CUDA

```cpp
// Level 1: SIMD trong mỗi thread (vectorization)
// Level 2: OpenMP chia work cho nhiều threads
// Level 3: CUDA cho heavy computation
#pragma omp parallel for simd
for (int i = 0; i < N; ++i) {
    // SIMD instructions tự động
}
```

---

## 6. ADVANCED PARALLELISM TECHNIQUES

### 6.1. Work Stealing

```cpp
// Thread pool với work stealing
// Thread nào xong việc sẽ "ăn cắp" work từ thread khác
class WorkStealingQueue {
    // Lock-free queue
    // Thread local + global queue
};
```

### 6.2. Lock-free Data Structures

```cpp
// Thay vì mutex, dùng atomic operations
std::atomic<FramePacket*> head_;
std::atomic<FramePacket*> tail_;

// Lock-free queue → giảm contention
```

### 6.3. NUMA-aware Parallelism

```cpp
// Nếu có nhiều CPU sockets (NUMA)
// Bind threads vào CPU cores cụ thể
cpu_set_t cpuset;
CPU_ZERO(&cpuset);
CPU_SET(core_id, &cpuset);
pthread_setaffinity_np(thread, sizeof(cpuset), &cpuset);
```

### 6.4. OpenMP Tasks

```cpp
#pragma omp parallel
{
    #pragma omp single
    {
        for (auto& roi : plateROIs) {
            #pragma omp task
            {
                ocrResults.push_back(ocr_.recognize(roi));
            }
        }
    }
    #pragma omp taskwait
}
```

---

## 7. SPECIFIC OPTIMIZATIONS CHO DỰ ÁN

### 7.1. Sobel Filter - Tiled Processing

```cpp
// Chia ảnh thành tiles, xử lý song song
// Mỗi tile có border overlap để tính đúng
const int TILE_SIZE = 64;
const int OVERLAP = 1;
#pragma omp parallel for collapse(2)
for (int ty = 0; ty < numTilesY; ++ty) {
    for (int tx = 0; tx < numTilesX; ++tx) {
        processTile(tx, ty, TILE_SIZE, OVERLAP);
    }
}
```

### 7.2. Detection - Multi-scale Processing

```cpp
// Chạy detection ở nhiều scale song song
std::vector<float> scales = {0.5, 1.0, 1.5, 2.0};
#pragma omp parallel for
for (size_t i = 0; i < scales.size(); ++i) {
    cv::Mat scaled;
    cv::resize(frame, scaled, cv::Size(), scales[i], scales[i]);
    auto boxes = detector_.detect(scaled);
    // Merge results
}
```

### 7.3. OCR - Character-level Parallelism

```cpp
// Nếu OCR model trả về từng ký tự riêng
// Xử lý mỗi ký tự song song
#pragma omp parallel for
for (int i = 0; i < numChars; ++i) {
    charResults[i] = recognizeChar(charRois[i]);
}
```

### 7.4. Frame Queue với Priority

```cpp
// Ưu tiên xử lý frame có biển số (detect được)
class PriorityQueue {
    // Frame có detection → priority cao
    // Frame không có → priority thấp
};
```

---

## 8. MEASUREMENT & PROFILING

### 8.1. Performance Metrics

```cpp
// Đo thời gian từng stage
auto t1 = std::chrono::high_resolution_clock::now();
sobelCPU(frame, sobel);
auto t2 = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
std::cout << "Sobel: " << duration.count() << " us" << std::endl;
```

### 8.2. Throughput Measurement

```cpp
// Đếm số frame/giây (FPS)
int frameCount = 0;
auto start = std::chrono::steady_clock::now();
// ... process frames ...
auto end = std::chrono::steady_clock::now();
double fps = frameCount / std::chrono::duration<double>(end - start).count();
```

---

## 9. IMPLEMENTATION PRIORITY

### Ưu tiên cao (dễ triển khai, hiệu quả):
1. ✅ **OpenMP cho Sobel** (đã có)
2. **SIMD vectorization cho Sobel**
3. **Batch inference cho Detection/OCR**
4. **Tách Detection và OCR thành 2 threads**

### Ưu tiên trung bình:
5. **CUDA streams cho GPU**
6. **Multi-frame pipeline (tăng buffer size)**
7. **OpenMP tasks cho OCR nhiều ROI**

### Ưu tiên thấp (phức tạp, cần nghiên cứu):
8. **Work stealing thread pool**
9. **Lock-free data structures**
10. **NUMA-aware binding**

---

## 10. CODE EXAMPLES SẴN SÀNG TRIỂN KHAI

Xem các file:
- `Main/SobelCuda.cu` - CUDA Sobel kernel
- `Main/Pipeline.cpp` - Multi-threading pipeline
- Có thể thêm: `Main/SobelSIMD.cpp` - SIMD version
- Có thể thêm: `Main/BatchInference.cpp` - Batch processing

---

## Kết Luận

Dự án này có **rất nhiều cơ hội song song hóa**:
- **Data parallelism**: Sobel (pixel-level), Model inference (batch)
- **Task parallelism**: Tách các stage, async execution
- **Pipeline parallelism**: Multi-frame processing
- **GPU parallelism**: CUDA streams, multi-GPU
- **Hybrid**: Kết hợp tất cả các kỹ thuật trên

**Mục tiêu**: Tăng throughput từ ~10 FPS lên **50-100+ FPS** với đầy đủ song song hóa.

