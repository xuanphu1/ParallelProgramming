# So Sánh Hiệu Năng - DetectVehicleLicensePlateSobelFilter

## 📋 Thông Tin Hệ Thống

**Máy tính**: Dell Precision M4800
- **CPU**: Intel Core i7-4910MQ @ 2.90GHz (4 cores, 8 threads, max 3.90GHz)
- **GPU**: NVIDIA Quadro K2100M (576 CUDA cores, 2GB GDDR5 VRAM, CUDA 11.4)
- **RAM**: 16GB DDR3L 1600MHz

## 📋 Mục Lục
1. [So Sánh Sobel Edge Detection](#1-so-sánh-sobel-edge-detection)
2. [So Sánh Task Parallelism](#2-so-sánh-task-parallelism)
3. [So Sánh FPS Tổng Hợp](#3-so-sánh-fps-tổng-hợp)
4. [Kết Luận và Khuyến Nghị](#kết-luận-và-khuyến-nghị)

---

## 1. So Sánh Sobel Edge Detection

### 1.1. Bảng So Sánh Chi Tiết (Dell Precision M4800)

| Phương Pháp | Thời Gian (ms) | Tốc Độ | Số Cores | Memory Bandwidth | Phụ Thuộc |
|-------------|----------------|--------|----------|------------------|-----------|
| **Tuần Tự (Sequential)** | 150-200 | 1x (baseline) | 1 | ~20 GB/s | Không |
| **OpenMP (8 threads)** | 18-25 | **8-11x** | 8 | ~50 GB/s | Compiler support |
| **CUDA (K2100M)** | 5-12 | **12-40x** | 576 | ~80 GB/s | GPU + CUDA |

### 1.2. So Sánh Theo Kích Thước Ảnh

#### Ảnh Nhỏ (640×480 = 307,200 pixels)

| Phương Pháp | Thời Gian (ms) | Tốc Độ | Ghi Chú |
|-------------|----------------|--------|---------|
| Tuần Tự | 25-30 | 1x | Baseline |
| OpenMP (8 threads) | 4-5 | **6-7x** | Hiệu quả tốt với 8 threads |
| CUDA (K2100M) | 3-5 | **5-10x** | GPU cũ, overhead copy memory lớn |

#### Ảnh Trung Bình (1280×720 = 921,600 pixels)

| Phương Pháp | Thời Gian (ms) | Tốc Độ | Ghi Chú |
|-------------|----------------|--------|---------|
| Tuần Tự | 75-100 | 1x | Baseline |
| OpenMP (8 threads) | 10-12 | **7-10x** | Hiệu quả tốt với 8 threads |
| CUDA (K2100M) | 4-8 | **9-25x** | GPU cũ, tốc độ vừa phải |

#### Ảnh Lớn (1920×1080 = 2,073,600 pixels)

| Phương Pháp | Thời Gian (ms) | Tốc Độ | Ghi Chú |
|-------------|----------------|--------|---------|
| Tuần Tự | 150-200 | 1x | Baseline |
| OpenMP (8 threads) | 18-25 | **8-11x** | Hiệu quả tốt với 8 threads |
| CUDA (K2100M) | 5-12 | **12-40x** | GPU cũ (2013), tốc độ chấp nhận được |

### 1.3. Phân Tích Chi Tiết

#### Tuần Tự (Sequential)
```
Ưu điểm:
- Không cần GPU hoặc OpenMP
- Code đơn giản, dễ debug
- Memory footprint nhỏ

Nhược điểm:
- Chậm nhất (150-200ms cho Full HD)
- Chỉ dùng 1 CPU core
- Không tận dụng được đa lõi

Sử dụng khi:
- Hệ thống không có GPU
- Ảnh rất nhỏ (< 320×240)
- Debugging hoặc development
```

#### OpenMP (CPU Parallel)
```
Ưu điểm:
- Nhanh hơn tuần tự 8-11 lần
- Không cần GPU
- Tận dụng đa lõi CPU
- Code đơn giản (chỉ cần #pragma)

Nhược điểm:
- Phụ thuộc vào số CPU cores
- Chậm hơn CUDA 5-10 lần
- Memory bandwidth giới hạn

Sử dụng khi:
- Không có GPU hoặc CUDA không khả dụng
- Ảnh nhỏ-trung bình
- Cần balance giữa tốc độ và độ phức tạp
```

#### CUDA (GPU Parallel - Quadro K2100M)
```
Ưu điểm:
- Nhanh hơn tuần tự 12-40x (với GPU K2100M)
- Xử lý song song 576 CUDA cores
- Giảm tải cho CPU
- Tốc độ không phụ thuộc vào CPU cores

Nhược điểm:
- GPU cũ (2013) → tốc độ không bằng GPU hiện đại
- Overhead copy memory lớn (PCIe 2.0, DDR3)
- Chỉ nhanh hơn OpenMP 2-3x (không phải 50-200x như GPU mới)
- Cần GPU và CUDA driver
- Tiêu thụ điện năng cao hơn

Sử dụng khi:
- Có GPU khả dụng
- Ảnh lớn (> 1MP)
- Cần tốc độ tốt hơn OpenMP một chút
- Note: Với K2100M, OpenMP có thể đủ tốt cho nhiều trường hợp
```

### 1.4. Biểu Đồ So Sánh Thời Gian (Dell Precision M4800)

```
Thời Gian (ms)
│
200│ ████████████████████████████████████ Tuần Tự
   │
150│ ████████████████████████████████████
   │
100│ ████████████████████████████████████
   │
 50│ ████████████████████████████████████
   │
 25│ ████████████████████████████████████ OpenMP (8 threads)
   │
 12│ ████████████████████████████████████ CUDA (K2100M)
   │
  5│ ████████████████████████████████████
   │
   └─────────────────────────────────────
     640×480  1280×720  1920×1080
```

**Lưu ý**: Với GPU Quadro K2100M (2013), CUDA chỉ nhanh hơn OpenMP khoảng **2-3x**, không phải 50-200x như GPU hiện đại. OpenMP với 8 threads trên CPU i7-4910MQ hoạt động rất tốt!

---

## 2. So Sánh Task Parallelism

### 2.1. Bảng So Sánh Tổng Quan

| Phương Pháp | Latency (ms) | Throughput (FPS) | CPU Usage | Memory Usage | Độ Phức Tạp |
|-------------|--------------|------------------|-----------|--------------|-------------|
| **Tuần Tự (Sequential)** | 200-300 | 3-5 | 25-40% | Thấp | Đơn giản |
| **Song Song (Parallel Pipeline)** | 50-100 | 10-20 | 60-80% | Trung bình | Phức tạp |

### 2.2. So Sánh Chi Tiết Từng Stage

#### Sequential Pipeline (Tuần Tự)

```
Timeline:
┌─────────────────────────────────────────────────────────┐
│ Frame 1: Capture → Detection → OCR → Display           │
│          [50ms]  [100ms]    [50ms]  [10ms] = 210ms    │
└─────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│ Frame 2: Capture → Detection → OCR → Display           │
│          [50ms]  [100ms]    [50ms]  [10ms] = 210ms    │
└─────────────────────────────────────────────────────────┘
Total: 420ms cho 2 frames → ~4.76 FPS
```

**Thời gian từng stage (ước tính)**:
- Capture: 50ms (đọc từ RTSP)
- Detection (YOLOv9): 100ms (ONNX inference)
- OCR (CCT): 50ms (ONNX inference)
- Display: 10ms (vẽ bbox + hiển thị)

**Tổng**: ~210ms/frame → **~4.76 FPS**

#### Parallel Pipeline (Song Song)

```
Timeline:
┌─────────────────────────────────────────────────────────┐
│ Capture Thread:  Frame 1  Frame 2  Frame 3  Frame 4  │
│                   [50ms]   [50ms]   [50ms]   [50ms]   │
└─────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│ Detection Thread:        Frame 1    Frame 2            │
│                          [100ms]    [100ms]            │
└─────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│ OCR Thread:                    Frame 1                 │
│                                [50ms]                  │
└─────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│ Display Thread: Frame 1  Frame 2  Frame 3  Frame 4     │
│                 [10ms]   [10ms]   [10ms]   [10ms]     │
└─────────────────────────────────────────────────────────┘
Total: ~200ms cho 4 frames → ~20 FPS
```

**Thời gian từng stage (song song)**:
- Capture: 50ms/frame (chạy liên tục, không bị block)
- Detection: 100ms/frame (chạy song song với Capture)
- OCR: 50ms/frame (chạy song song với Detection)
- Display: 10ms/frame (chạy song song với tất cả)

**Tổng**: ~200ms cho 4 frames → **~20 FPS** (nhanh hơn 4-5 lần!)

### 2.3. Bảng So Sánh Chi Tiết

| Metric | Sequential | Parallel Pipeline | Cải Thiện |
|--------|-----------|-------------------|-----------|
| **Latency (độ trễ)** | 200-300ms | 50-100ms | **2-3x nhanh hơn** |
| **Throughput (FPS)** | 3-5 FPS | 10-20 FPS | **4-5x cao hơn** |
| **CPU Usage** | 25-40% (1 core) | 60-80% (4-8 cores) | Tận dụng đa lõi |
| **Memory Usage** | Thấp (~500MB) | Trung bình (~1GB) | Queue buffers |
| **Frame Drop** | Không | Có thể (nếu queue đầy) | Trade-off |
| **Real-time Display** | Bị block | Liên tục | ✅ Tốt hơn |
| **Code Complexity** | Đơn giản | Phức tạp (threads, sync) | Trade-off |

### 2.4. Phân Tích Chi Tiết

#### Sequential Pipeline

**Ưu điểm**:
- Code đơn giản, dễ hiểu
- Không có race conditions
- Memory footprint nhỏ
- Dễ debug

**Nhược điểm**:
- **Chậm**: Mỗi frame phải đợi tất cả stages hoàn thành
- **Low FPS**: 3-5 FPS (không đủ cho real-time)
- **Display bị block**: Không hiển thị được khi đang detect/OCR
- **Không tận dụng đa lõi**: Chỉ dùng 1 CPU core

**Sử dụng khi**:
- Development/Debugging
- Không cần real-time
- Hệ thống đơn lõi

#### Parallel Pipeline

**Ưu điểm**:
- **Nhanh**: 10-20 FPS (đủ cho real-time)
- **Low latency**: Display không bị block
- **Tận dụng đa lõi**: 4 threads chạy song song
- **Scalable**: Có thể tăng số threads nếu cần

**Nhược điểm**:
- Code phức tạp (threads, mutexes, condition variables)
- Memory overhead (queues, buffers)
- Có thể có frame drops nếu queue đầy
- Khó debug hơn (race conditions, deadlocks)

**Sử dụng khi**:
- Cần real-time performance
- Hệ thống đa lõi
- Production environment

### 2.5. Biểu Đồ So Sánh FPS

```
FPS
│
20│                                    ████ Parallel Pipeline
   │
15│                                    ████
   │
10│                                    ████
   │
 5│ ████ Sequential
   │
 0│ ████
   └─────────────────────────────────────
     Sequential    Parallel
```

---

## 3. So Sánh FPS Tổng Hợp

### 3.1. Bảng So Sánh Tất Cả Trường Hợp

| Sobel Method | Task Method | FPS | Latency (ms) | CPU Usage | GPU Usage | Ghi Chú |
|--------------|-------------|-----|--------------|-----------|-----------|---------|
| **Tuần Tự** | **Sequential** | 3-4 | 250-300 | 25-30% | 0% | Baseline |
| **OpenMP** | **Sequential** | 4-5 | 200-250 | 40-50% | 0% | Sobel nhanh hơn |
| **CUDA (K2100M)** | **Sequential** | 4-5 | 200-250 | 25-30% | 15-25% | Sobel nhanh hơn một chút |
| **Tuần Tự** | **Parallel** | 8-12 | 80-120 | 50-60% | 0% | Pipeline tốt |
| **OpenMP** | **Parallel** | 12-18 | 55-80 | 70-80% | 0% | ⭐⭐ Tốt nhất (CPU only) |
| **CUDA (K2100M)** | **Parallel** | 13-20 | 50-75 | 60-70% | 20-35% | ⭐ Tốt (nhưng không tốt hơn OpenMP nhiều) |

### 3.2. Phân Tích Chi Tiết Từng Trường Hợp

#### 1. Tuần Tự Sobel + Sequential Pipeline
```
Sobel: 150ms (tuần tự)
Pipeline: Sequential
├─ Capture: 50ms
├─ Detection: 100ms
├─ OCR: 50ms
└─ Display: 10ms
Total: ~310ms/frame → ~3.2 FPS

Ưu điểm: Đơn giản nhất
Nhược điểm: Chậm nhất
```

#### 2. OpenMP Sobel + Sequential Pipeline
```
Sobel: 20ms (OpenMP, 8 cores)
Pipeline: Sequential
├─ Capture: 50ms
├─ Detection: 100ms
├─ OCR: 50ms
└─ Display: 10ms
Total: ~230ms/frame → ~4.3 FPS

Ưu điểm: Sobel nhanh hơn, không cần GPU
Nhược điểm: Pipeline vẫn tuần tự
```

#### 3. CUDA Sobel (K2100M) + Sequential Pipeline
```
Sobel: 8ms (CUDA với K2100M - GPU cũ)
Pipeline: Sequential
├─ Capture: 50ms
├─ Detection: 100ms
├─ OCR: 50ms
└─ Display: 10ms
Total: ~218ms/frame → ~4.6 FPS

Ưu điểm: Sobel nhanh hơn tuần tự
Nhược điểm: Pipeline vẫn tuần tự, GPU cũ không nhanh hơn OpenMP nhiều
```

#### 4. Tuần Tự Sobel + Parallel Pipeline
```
Sobel: 150ms (tuần tự, nhưng chạy song song với Detection)
Pipeline: Parallel
├─ Capture: 50ms (song song)
├─ Detection: 100ms (song song với Capture)
├─ OCR: 50ms (song song với Detection)
└─ Display: 10ms (song song với tất cả)
Effective: ~150ms/frame → ~6.7 FPS

Ưu điểm: Pipeline song song tốt
Nhược điểm: Sobel vẫn chậm
```

#### 5. OpenMP Sobel + Parallel Pipeline ⭐⭐
```
Sobel: 20ms (OpenMP với 8 threads, chạy song song với Detection)
Pipeline: Parallel
├─ Capture: 50ms (song song)
├─ Detection: 100ms (song song với Capture)
├─ OCR: 50ms (song song với Detection)
└─ Display: 10ms (song song với tất cả)
Effective: ~100ms/frame → ~10 FPS

Ưu điểm: ⭐⭐ TỐT NHẤT cho Dell Precision M4800
- Sobel nhanh với 8 threads (i7-4910MQ)
- Pipeline song song
- Không cần GPU
- FPS đủ cho real-time (12-18 FPS)
- CPU i7-4910MQ có 8 threads → OpenMP hoạt động rất tốt

Nhược điểm: Cần CPU đa lõi (đã có: 4 cores, 8 threads)
```

#### 6. CUDA Sobel (K2100M) + Parallel Pipeline ⭐
```
Sobel: 8ms (CUDA với K2100M, chạy song song với Detection)
Pipeline: Parallel
├─ Capture: 50ms (song song)
├─ Detection: 100ms (song song với Capture)
├─ OCR: 50ms (song song với Detection)
└─ Display: 10ms (song song với tất cả)
Effective: ~100ms/frame → ~10 FPS

Thực tế: ~13-20 FPS (Sobel nhanh hơn OpenMP một chút, nhưng không nhiều)

Ưu điểm: ⭐ Tốt cho systems có GPU
- Sobel nhanh hơn OpenMP một chút (8ms vs 20ms)
- Pipeline song song
- FPS tốt (13-20 FPS)
- Real-time tốt

Nhược điểm: 
- GPU K2100M cũ (2013) → không nhanh hơn OpenMP nhiều
- Chỉ nhanh hơn OpenMP khoảng 2-3x (không phải 50-200x)
- Với CPU i7-4910MQ có 8 threads, OpenMP đã rất tốt
- Khuyến nghị: Có thể không cần CUDA, OpenMP đủ tốt!
```

### 3.3. Biểu Đồ So Sánh FPS Tổng Hợp

```
FPS (Dell Precision M4800)
│
20│                                                    ████ CUDA + Parallel
   │
15│                                    ████ CUDA + Parallel
   │                                    ████ OpenMP + Parallel
   │
10│            ████ OpenMP + Parallel
   │
 5│ ████ Sequential (baseline)
   │
 0│ ████
   └─────────────────────────────────────────────────────
     Seq+Seq  OMP+Seq  CUDA+Seq  Seq+Par  OMP+Par  CUDA+Par
     
Lưu ý: Với GPU K2100M cũ, CUDA chỉ nhanh hơn OpenMP một chút.
OpenMP với 8 threads trên i7-4910MQ hoạt động rất tốt!
```

### 3.4. Bảng So Sánh Resource Usage

| Configuration | CPU Usage | GPU Usage | Memory | Power |
|---------------|-----------|-----------|--------|-------|
| Sequential + Sequential | 25-30% | 0% | ~500MB | Thấp |
| OpenMP + Sequential | 40-50% | 0% | ~500MB | Trung bình |
| CUDA (K2100M) + Sequential | 25-30% | 15-25% | ~600MB | Trung bình |
| Sequential + Parallel | 50-60% | 0% | ~1GB | Trung bình |
| **OpenMP + Parallel** | **70-80%** | **0%** | **~1GB** | **Cao** |
| **CUDA (K2100M) + Parallel** | **60-70%** | **20-35%** | **~1.2GB** | **Rất cao** |

### 3.5. Khuyến Nghị Sử Dụng

#### Hệ Thống Không Có GPU
```
✅ OpenMP Sobel + Parallel Pipeline
- FPS: 12-18 FPS
- CPU: 70-80%
- Đủ cho real-time
- Không cần GPU
```

#### Hệ Thống Có GPU
```
✅ CUDA Sobel + Parallel Pipeline
- FPS: 15-25 FPS
- CPU: 60-70%
- GPU: 30-50%
- Tốt nhất cho real-time
```

#### Development/Debugging
```
✅ Sequential Sobel + Sequential Pipeline
- FPS: 3-4 FPS
- Đơn giản, dễ debug
- Không cần GPU
```

---

## Kết Luận và Khuyến Nghị

### Tổng Kết Cho Dell Precision M4800

1. **Sobel Edge Detection**:
   - **OpenMP (8 threads)**: 18-25ms → **8-11x** so với tuần tự ⭐⭐ TỐT NHẤT
   - **CUDA (K2100M)**: 5-12ms → **12-40x** so với tuần tự, nhưng chỉ nhanh hơn OpenMP **2-3x**
   - Tuần tự chỉ dùng khi debugging
   - **Khuyến nghị**: Dùng OpenMP! CPU i7-4910MQ có 8 threads → OpenMP hoạt động rất tốt

2. **Task Parallelism**:
   - Parallel Pipeline nhanh hơn **4-5x** so với Sequential
   - FPS tăng từ 3-4 → 12-18 FPS
   - Real-time display không bị block

3. **Kết Hợp Tối Ưu Cho Máy Này**:
   - ⭐⭐ **TỐT NHẤT**: **OpenMP Sobel + Parallel Pipeline** → **12-18 FPS**
     - CPU i7-4910MQ có 8 threads → OpenMP rất hiệu quả
     - Không cần GPU
     - Đơn giản, ổn định
   - ⭐ **TỐT**: **CUDA Sobel + Parallel Pipeline** → **13-20 FPS**
     - Nhanh hơn OpenMP một chút (không nhiều)
     - Cần GPU K2100M
     - Có thể không đáng để phức tạp hóa code

### Khuyến Nghị Cho Dell Precision M4800

1. **Production Environment**:
   - ✅ **Luôn dùng Parallel Pipeline** (tăng FPS từ 3-4 → 12-18)
   - ✅ **Dùng OpenMP Sobel** (tốt nhất cho máy này!)
     - CPU i7-4910MQ có 8 threads → OpenMP hoạt động rất tốt
     - GPU K2100M cũ → CUDA không nhanh hơn OpenMP nhiều
     - OpenMP đơn giản hơn, không cần GPU
   - ⚠️ **CUDA Sobel**: Chỉ dùng nếu muốn thử, nhưng không cần thiết
     - GPU K2100M chỉ nhanh hơn OpenMP khoảng 2-3x
     - Với CPU 8 threads, OpenMP đã đủ tốt

2. **Development Environment**:
   - Có thể dùng Sequential để debug
   - Test với Parallel trước khi deploy

3. **Tối Ưu Thêm**:
   - Tăng `DETECTION_INTERVAL` nếu FPS vẫn thấp (hiện tại: 15)
   - Giảm `MAX_QUEUE_SIZE` nếu memory hạn chế (hiện tại: 5)
   - Tune ONNX Runtime threads:
     - IntraOp: 8 threads (phù hợp với 8 threads CPU)
     - InterOp: 4 threads (phù hợp với 4 cores)

### Benchmark Thực Tế

Để có số liệu chính xác, chạy benchmark:

```bash
# Sequential Sobel + Sequential Pipeline
./test_onnx_models --sequential

# OpenMP Sobel + Sequential Pipeline
./test_onnx_models --sequential  # (OpenMP tự động)

# CUDA Sobel + Sequential Pipeline
./test_onnx_models --sequential  # (CUDA tự động fallback)

# Sequential Sobel + Parallel Pipeline
./test_onnx_models --parallel

# OpenMP Sobel + Parallel Pipeline
./test_onnx_models --parallel  # (OpenMP tự động)

# CUDA Sobel + Parallel Pipeline
./test_onnx_models --parallel  # (CUDA tự động fallback)
```

---

**Lưu ý**: Các số liệu trong tài liệu này là ước tính dựa trên lý thuyết và kinh nghiệm. Để có số liệu chính xác, cần benchmark trên hệ thống thực tế với các cấu hình khác nhau.

