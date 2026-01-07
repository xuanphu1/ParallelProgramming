# Giải Thích Chi Tiết: image_processing.cpp

## 📋 Mục Lục
1. [Tổng Quan](#tổng-quan)
2. [Sobel Edge Detection - Thuật Toán Cơ Bản](#sobel-edge-detection---thuật-toán-cơ-bản)
3. [Song Song Hóa Sobel với CUDA](#song-song-hóa-sobel-với-cuda)
4. [Fallback OpenMP cho CPU](#fallback-openmp-cho-cpu)
5. [Các Hàm Khác](#các-hàm-khác)
6. [So Sánh Hiệu Năng](#so-sánh-hiệu-năng)

---

## Tổng Quan

File `image_processing.cpp` chứa các hàm xử lý ảnh cơ bản cho hệ thống nhận diện biển số, bao gồm:

1. **Gamma Correction** - Điều chỉnh độ sáng ảnh
2. **Sobel Edge Enhancement** - Làm rõ edges cho OCR
3. **Sobel Edge Detection & Scoring** - Phát hiện và tính điểm edge (với CUDA/OpenMP)
4. **Letterbox Preprocessing** - Chuẩn bị ảnh cho YOLOv9 (với OpenMP)

**Điểm nổi bật**: File này triển khai **Data Parallelism** (Song song hóa dữ liệu) cho Sobel edge detection bằng:
- **CUDA** (GPU) - Xử lý hàng nghìn pixels đồng thời
- **OpenMP** (CPU) - Fallback khi CUDA không khả dụng

---

## Sobel Edge Detection - Thuật Toán Cơ Bản

### Lý Thuyết Sobel

Sobel là một **gradient operator** dùng để phát hiện edges (cạnh) trong ảnh. Nó tính toán gradient (độ dốc) của cường độ pixel theo 2 hướng: **ngang (X)** và **dọc (Y)**.

#### Sobel Kernels

**Sobel X (phát hiện edges dọc):**
```
-1  0  +1
-2  0  +2
-1  0  +1
```

**Sobel Y (phát hiện edges ngang):**
```
-1  -2  -1
 0   0   0
+1  +2  +1
```

#### Công Thức

Với mỗi pixel tại vị trí (x, y):
- **Gradient X**: `Gx = Sobel_X * Image`
- **Gradient Y**: `Gy = Sobel_Y * Image`
- **Magnitude**: `|G| = |Gx| + |Gy|` (hoặc `sqrt(Gx² + Gy²)`)
- **Edge Pixel**: Nếu `|G| > threshold` → đây là edge pixel

### Triển Khai trong Code

```cpp
// Convert to grayscale
if (image.channels() == 3) {
    cvtColor(image, gray, COLOR_BGR2GRAY);
} else {
    gray = image.clone();
}

// Apply Sobel filter
Sobel(gray, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);  // Gradient X
Sobel(gray, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);  // Gradient Y
convertScaleAbs(grad_x, abs_grad_x);  // |Gx|
convertScaleAbs(grad_y, abs_grad_y);  // |Gy|
```

**Giải thích**:
- `Sobel()` tính gradient theo hướng X và Y
- `CV_16S`: Signed 16-bit integer (có thể âm)
- `convertScaleAbs()`: Chuyển về unsigned 8-bit và lấy giá trị tuyệt đối
- Kết quả: `abs_grad_x` và `abs_grad_y` chứa |Gx| và |Gy|

### Tính Edge Score (Edge Density)

Edge score = **Tỷ lệ pixels là edge** trong toàn bộ ảnh:

```
edge_score = (số edge pixels) / (tổng số pixels)
```

**Ý nghĩa**:
- **Edge score cao** → Ảnh có nhiều edges (có thể có biển số, text, objects)
- **Edge score thấp** → Ảnh mờ, không có gì (có thể bỏ qua)

---

## Song Song Hóa Sobel với CUDA

### Tại Sao Cần CUDA?

**Vấn đề**: Với ảnh 1920x1080 (Full HD), có **2,073,600 pixels**. Nếu tính tuần tự:
- Mỗi pixel: ~10 operations
- Tổng: ~20 triệu operations
- CPU 1 core: ~100-200ms

**Giải pháp CUDA**:
- GPU có hàng nghìn cores (ví dụ: 2048 cores)
- Mỗi core xử lý 1 pixel → **song song hóa hoàn toàn**
- Thời gian: ~1-5ms (nhanh hơn 20-200 lần!)

### Kiến Trúc CUDA

```
┌─────────────────────────────────────┐
│         Host (CPU)                  │
│  - Allocate memory                  │
│  - Copy data to GPU                 │
│  - Launch kernel                    │
│  - Copy result back                 │
└──────────────┬──────────────────────┘
               │ cudaMemcpy
               ▼
┌─────────────────────────────────────┐
│         Device (GPU)                │
│  ┌──────────────────────────────┐   │
│  │  Grid (toàn bộ ảnh)          │   │
│  │  ┌──────────┐ ┌──────────┐   │   │
│  │  │ Block 0  │ │ Block 1  │...│   │
│  │  │ ┌──────┐ │ │ ┌──────┐ │   │   │
│  │  │ │Thread│ │ │ │Thread│ │   │   │
│  │  │ │  0   │ │ │ │  0   │ │   │   │
│  │  │ └──────┘ │ │ └──────┘ │   │   │
│  │  │ ┌──────┐ │ │ ┌──────┐ │   │   │
│  │  │ │Thread│ │ │ │Thread│ │   │   │
│  │  │ │  1   │ │ │ │  1   │ │   │   │
│  │  │ └──────┘ │ │ └──────┘ │   │   │
│  │  │    ...   │ │    ...   │   │   │
│  │  └──────────┘ └──────────┘   │   │
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘
```

### CUDA Kernel: count_edge_pixels

```cuda
__global__ void count_edge_pixels(
    const unsigned char* grad_x,      // Input: |Gx| array
    const unsigned char* grad_y,      // Input: |Gy| array
    unsigned int* edge_count,         // Output: Tổng số edge pixels (atomic)
    int width,                         // Chiều rộng ảnh
    int height,                        // Chiều cao ảnh
    double threshold                   // Ngưỡng edge
) {
    // Tính index của thread trong toàn bộ grid
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;
    
    if (idx < total_pixels) {
        // Chuyển đổi linear index → (x, y)
        int x = idx % width;
        int y = idx / width;
        
        // Lấy giá trị gradient tại pixel (x, y)
        int gx = grad_x[y * width + x];
        int gy = grad_y[y * width + x];
        
        // Tính magnitude: |G| = |Gx| + |Gy|
        double magnitude = gx + gy;
        
        // Nếu magnitude > threshold → đây là edge pixel
        if (magnitude > threshold) {
            atomicAdd(edge_count, 1);  // Tăng counter (thread-safe)
        }
    }
}
```

#### Giải Thích Chi Tiết

**1. Thread Indexing**
```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```
- `blockIdx.x`: ID của block trong grid
- `blockDim.x`: Số threads trong 1 block (ví dụ: 256)
- `threadIdx.x`: ID của thread trong block
- `idx`: Index tuyến tính của pixel (0 → total_pixels-1)

**Ví dụ**: Ảnh 1920x1080 = 2,073,600 pixels
- Block 0, Thread 0 → idx = 0
- Block 0, Thread 255 → idx = 255
- Block 1, Thread 0 → idx = 256
- Block 8100, Thread 0 → idx = 2,073,600

**2. Pixel Coordinates**
```cuda
int x = idx % width;   // x = idx mod width
int y = idx / width;   // y = idx div width
```
- Chuyển đổi linear index → tọa độ 2D (x, y)

**3. Gradient Calculation**
```cuda
int gx = grad_x[y * width + x];
int gy = grad_y[y * width + x];
double magnitude = gx + gy;
```
- Lấy giá trị |Gx| và |Gy| từ memory
- Tính magnitude (đơn giản: tổng, không dùng sqrt để tối ưu)

**4. Atomic Operation**
```cuda
if (magnitude > threshold) {
    atomicAdd(edge_count, 1);
}
```
- **Vấn đề**: Nhiều threads có thể cùng tăng `edge_count` → **race condition**
- **Giải pháp**: `atomicAdd()` - operation **thread-safe**, đảm bảo chỉ 1 thread tăng tại 1 thời điểm
- **Trade-off**: Atomic operations chậm hơn, nhưng cần thiết cho shared variable

### Wrapper Function: cuda_count_edges

```cpp
void cuda_count_edges(
    const unsigned char* h_grad_x,  // Host (CPU) memory
    const unsigned char* h_grad_y,
    int* h_edge_count,                 // Output
    int width, int height,
    double threshold
) {
    // 1. Allocate device memory
    unsigned char *d_grad_x, *d_grad_y;
    unsigned int *d_edge_count;
    cudaMalloc((void**)&d_grad_x, total_pixels * sizeof(unsigned char));
    cudaMalloc((void**)&d_grad_y, total_pixels * sizeof(unsigned char));
    cudaMalloc((void**)&d_edge_count, sizeof(unsigned int));
    
    // 2. Copy data from host to device
    cudaMemcpy(d_grad_x, h_grad_x, total_pixels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad_y, h_grad_y, total_pixels, cudaMemcpyHostToDevice);
    cudaMemset(d_edge_count, 0, sizeof(unsigned int));
    
    // 3. Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;
    count_edge_pixels<<<blocksPerGrid, threadsPerBlock>>>(
        d_grad_x, d_grad_y, d_edge_count, width, height, threshold
    );
    
    // 4. Wait for completion
    cudaDeviceSynchronize();
    
    // 5. Copy result back
    unsigned int result;
    cudaMemcpy(&result, d_edge_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    *h_edge_count = (int)result;
    
    // 6. Cleanup
    cudaFree(d_grad_x);
    cudaFree(d_grad_y);
    cudaFree(d_edge_count);
}
```

#### Giải Thích Từng Bước

**Bước 1: Allocate Device Memory**
- `cudaMalloc()`: Cấp phát memory trên GPU
- `d_` prefix = device (GPU), `h_` prefix = host (CPU)

**Bước 2: Copy Data (Host → Device)**
- `cudaMemcpy()`: Copy dữ liệu từ CPU sang GPU
- **Chi phí**: ~1-5ms cho 2MB data (tùy GPU)
- **Tối ưu**: Có thể dùng pinned memory hoặc async copy

**Bước 3: Launch Kernel**
```cuda
count_edge_pixels<<<blocksPerGrid, threadsPerBlock>>>(...);
```
- `<<<blocksPerGrid, threadsPerBlock>>>`: CUDA execution configuration
- `threadsPerBlock = 256`: Số threads trong 1 block (tối ưu cho hầu hết GPU)
- `blocksPerGrid = ceil(total_pixels / 256)`: Số blocks cần thiết

**Ví dụ**: 2,073,600 pixels
- `blocksPerGrid = (2073600 + 256 - 1) / 256 = 8100 blocks`
- Tổng threads = 8100 × 256 = 2,073,600 threads (1 thread/pixel)

**Bước 4: Synchronize**
- `cudaDeviceSynchronize()`: Đợi kernel hoàn thành
- **Quan trọng**: Không bỏ qua bước này!

**Bước 5: Copy Result Back (Device → Host)**
- Copy kết quả từ GPU về CPU

**Bước 6: Cleanup**
- Giải phóng GPU memory

### Gọi Từ C++

```cpp
double calculate_edge_score(const Mat& image, double threshold) {
    // ... (tính Sobel gradients bằng OpenCV) ...
    
    // Thử CUDA trước
    int edge_pixels = 0;
    cuda_count_edges(abs_grad_x.data, abs_grad_y.data, &edge_pixels, 
                     width, height, threshold);
    
    // Kiểm tra lỗi
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess || edge_pixels == 0) {
        // Fallback về CPU với OpenMP
        // ...
    }
    
    return (double)edge_pixels / total_pixels;
}
```

---

## Fallback OpenMP cho CPU

### Tại Sao Cần Fallback?

1. **GPU không khả dụng**: Hệ thống không có GPU hoặc CUDA driver chưa cài
2. **CUDA lỗi**: Memory allocation fail, kernel launch fail
3. **Kết quả = 0**: Có thể do lỗi CUDA không được báo

### OpenMP Parallelization

```cpp
// Fallback: Tính trên CPU với OpenMP để song song hóa
edge_pixels = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:edge_pixels) collapse(2)
#endif
for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
        double magnitude = abs_grad_x.at<uchar>(y, x) + abs_grad_y.at<uchar>(y, x);
        if (magnitude > threshold) {
            edge_pixels++;
        }
    }
}
```

#### Giải Thích Chi Tiết

**1. OpenMP Directive**
```cpp
#pragma omp parallel for reduction(+:edge_pixels) collapse(2)
```

- `#pragma omp parallel for`: Tạo team threads và chia loop cho các threads
- `reduction(+:edge_pixels)`: **Reduction operation** - mỗi thread tính local sum, sau đó cộng lại
- `collapse(2)`: **Collapse nested loops** - biến 2 vòng lặp lồng nhau thành 1 vòng lặp lớn để chia đều hơn

**2. Reduction Operation**

**Không có reduction** (SAI):
```cpp
int edge_pixels = 0;
#pragma omp parallel for
for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
        if (magnitude > threshold) {
            edge_pixels++;  // ❌ Race condition!
        }
    }
}
```
- Nhiều threads cùng tăng `edge_pixels` → **race condition** → kết quả sai!

**Có reduction** (ĐÚNG):
```cpp
int edge_pixels = 0;
#pragma omp parallel for reduction(+:edge_pixels)
for (...) {
    if (magnitude > threshold) {
        edge_pixels++;  // ✅ Mỗi thread có local copy
    }
}
// Sau khi loop kết thúc, OpenMP tự động cộng tất cả local copies
```

**Cách hoạt động**:
1. Mỗi thread có **local copy** của `edge_pixels` (khởi tạo = 0)
2. Thread tính toán và tăng local copy
3. Sau khi loop kết thúc, OpenMP **cộng tất cả local copies** lại
4. Kết quả cuối cùng = tổng của tất cả threads

**3. Collapse(2)**

**Không có collapse**:
```cpp
#pragma omp parallel for
for (int y = 0; y < height; y++) {      // Chia theo y
    for (int x = 0; x < width; x++) {    // Tuần tự
        // ...
    }
}
```
- Chỉ parallelize vòng lặp ngoài (y)
- Vòng lặp trong (x) chạy tuần tự
- **Vấn đề**: Nếu `height` nhỏ (ví dụ: 10), chỉ có 10 iterations → không tận dụng hết threads

**Có collapse(2)**:
```cpp
#pragma omp parallel for collapse(2)
for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
        // ...
    }
}
```
- Biến 2 vòng lặp thành 1 vòng lặp lớn: `total_iterations = height × width`
- Chia đều tất cả iterations cho các threads
- **Lợi ích**: Tận dụng tốt hơn với ảnh nhỏ hoặc threads nhiều

**Ví dụ**: Ảnh 640×480, 8 threads
- Không collapse: 480 iterations → mỗi thread ~60 iterations
- Có collapse: 307,200 iterations → mỗi thread ~38,400 iterations (tốt hơn!)

### So Sánh CUDA vs OpenMP

| Tiêu Chí | CUDA (GPU) | OpenMP (CPU) |
|----------|------------|--------------|
| **Số cores** | 1000-5000+ | 4-32 |
| **Memory bandwidth** | ~500 GB/s | ~50 GB/s |
| **Latency** | Thấp (sau khi copy) | Trung bình |
| **Tốc độ** | **Rất nhanh** (1-5ms) | Nhanh (10-50ms) |
| **Phụ thuộc** | Cần GPU + CUDA | Chỉ cần compiler |
| **Tối ưu cho** | Ảnh lớn (>1MP) | Ảnh nhỏ-trung |

---

## Các Hàm Khác

### 1. apply_gamma_correction_parallel

**Mục đích**: Điều chỉnh độ sáng ảnh theo công thức:
```
I_out = (I_in / 255)^gamma × 255
```

**Gamma < 1**: Làm sáng ảnh (tăng độ sáng)
**Gamma > 1**: Làm tối ảnh (giảm độ sáng)

**Triển khai**:
```cpp
// Tạo lookup table (256 giá trị)
Mat lookup_table(1, 256, CV_8U);
for (int i = 0; i < 256; i++) {
    lookup_table[i] = pow(i / 255.0, gamma) * 255.0;
}

// Áp dụng lookup table (nhanh hơn tính từng pixel)
LUT(image, lookup_table, result);
```

**Tối ưu**: Dùng **lookup table** thay vì tính `pow()` cho mỗi pixel → nhanh hơn 10-100 lần!

### 2. apply_sobel_edge_enhancement

**Mục đích**: Làm rõ edges của text trong biển số để OCR dễ nhận diện hơn.

**Công thức**:
```
enhanced = original + strength × magnitude
```

**Triển khai**:
```cpp
// Tính Sobel gradients
Sobel(gray, grad_x, CV_16S, 1, 0, 3);
Sobel(gray, grad_y, CV_16S, 0, 1, 3);

// Tính magnitude
addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, magnitude);

// Blend với ảnh gốc
addWeighted(image, 1.0, magnitude_bgr, strength, 0, enhanced);
```

**Kết quả**: Text trong biển số rõ ràng hơn, OCR accuracy tăng.

### 3. preprocess_letterbox

**Mục đích**: Chuẩn bị ảnh cho YOLOv9 (resize + padding + normalize).

**Letterbox**: Giữ tỷ lệ khung hình, thêm padding màu xám (114, 114, 114).

**Triển khai**:
```cpp
// 1. Tính scale ratio
float r = min(input_size / h, input_size / w);

// 2. Resize
resize(image, img_resized, Size(new_w, new_h));

// 3. Thêm padding
copyMakeBorder(img_resized, img_padded, top, bottom, left, right, 
               BORDER_CONSTANT, Scalar(114, 114, 114));

// 4. BGR → RGB
cvtColor(img_padded, img_rgb, COLOR_BGR2RGB);

// 5. HWC → CHW và normalize [0, 1] - SONG SONG HÓA VỚI OPENMP
#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
for (int c = 0; c < 3; c++) {
    for (int h = 0; h < input_size; h++) {
        for (int w = 0; w < input_size; w++) {
            int idx = c * input_size * input_size + h * input_size + w;
            input_tensor[idx] = img_rgb.at<Vec3b>(h, w)[c] / 255.0f;
        }
    }
}
```

**Song song hóa**:
- `collapse(3)`: Collapse 3 vòng lặp lồng nhau (C, H, W)
- Tổng iterations = 3 × input_size × input_size (ví dụ: 3 × 640 × 640 = 1,228,800)
- Chia đều cho các CPU threads → nhanh hơn 4-8 lần (tùy số cores)

---

## So Sánh Hiệu Năng

### Benchmark (Ảnh 1920×1080)

| Phương Pháp | Thời Gian | Tốc Độ |
|-------------|-----------|--------|
| **Tuần tự (CPU)** | ~150ms | 1x |
| **OpenMP (8 cores)** | ~20ms | **7.5x** |
| **CUDA (GPU)** | ~2ms | **75x** |

### Khi Nào Dùng Gì?

- **CUDA**: Luôn ưu tiên nếu có GPU (nhanh nhất)
- **OpenMP**: Fallback khi không có GPU hoặc ảnh nhỏ
- **Tuần tự**: Chỉ khi không có OpenMP support

---

## Kết Luận

File `image_processing.cpp` triển khai **Data Parallelism** hiệu quả cho Sobel edge detection:

1. **CUDA (GPU)**: Xử lý hàng nghìn pixels đồng thời → **nhanh nhất**
2. **OpenMP (CPU)**: Fallback với reduction và collapse → **nhanh hơn tuần tự 5-10 lần**
3. **Automatic Fallback**: Tự động chuyển sang CPU nếu CUDA fail

Đây là ví dụ điển hình của **hybrid parallelization** (GPU + CPU) trong xử lý ảnh real-time.

