## Dự án: Phát hiện & Đọc Biển Số Xe với Sobel + YOLOv5 (Pipeline C++/CUDA + Python)

Dự án này triển khai một hệ thống **nhận diện biển số xe** với các thành phần:

- **Pipeline C++ (folder `Main/`)**:  
  - Capture frame từ camera/video.  
  - Tiền xử lý biên cạnh bằng **Sobel** (CPU song song bằng OpenMP, có thể bật CUDA nếu môi trường hỗ trợ).  
  - Gọi **model phát hiện biển số** (`LP_detector_nano_61.pt` → TorchScript) để lấy bounding box.  
  - Gọi **model OCR biển số** (`LP_ocr_nano_62.pt` → TorchScript) để đọc chuỗi ký tự.  
  - Vẽ khung + text lên ảnh, in kết quả biển số (và là chỗ bạn có thể gắn MQTT).  

- **Script Python YOLOv5 (file `Main/run_models.py`)**:  
  - Chạy trực tiếp 2 model YOLOv5 `.pt` gốc (det + ocr) bằng `torch.hub` (YOLOv5).  
  - Dùng để kiểm thử nhanh model, demo trên ảnh hoặc webcam.

- **Script export TorchScript (file `Main/export_torchscript.py`)**:  
  - Chuyển 2 file YOLOv5 `.pt` sang **TorchScript** (`lp_detector_ts.pt`, `lp_ocr_ts.pt`) để C++ load được bằng LibTorch.

---

## Cấu trúc thư mục chính

```text
DetectVehicleLicensePlateSobelFIlter/
├── CMakeLists.txt                 # build pipeline C++ với OpenCV + LibTorch (+ CUDA nếu có)
├── Main/
│   ├── main.cpp                   # điểm vào C++, chạy Pipeline
│   ├── Pipeline.h / Pipeline.cpp  # pipeline đa luồng: capture → sobel → detect+OCR → render
│   ├── LPDetector.h / .cpp        # lớp C++ gọi model TorchScript detect biển số
│   ├── LPOCR.h / .cpp             # lớp C++ gọi model TorchScript OCR biển số
│   ├── SobelCuda.h / SobelCuda.cu # Sobel filter chạy trên GPU (CUDA) – tuỳ chọn
│   ├── LP_detector_nano_61.pt     # model YOLOv5 gốc (checkpoint) detect biển số
│   ├── LP_ocr_nano_62.pt          # model YOLOv5 gốc (checkpoint) OCR biển số
│   ├── export_torchscript.py      # script export YOLOv5 .pt → TorchScript .pt
│   └── run_models.py              # demo YOLOv5 thuần Python (không C++)
├── bienso.jpg                     # ảnh mẫu có biển số (dùng để test nhanh)
└── ...
```

---

## Yêu cầu môi trường

### Bắt buộc

- **C++ toolchain**: `g++` (hỗ trợ C++17).  
- **CMake** (>= 3.16):

```bash
sudo apt-get install cmake
```

- **OpenCV (>= 4.x)**:

```bash
sudo apt-get install libopencv-dev
```

- **LibTorch (PyTorch C++ API)**:  
  - Tải từ trang chính thức: `https://pytorch.org/`  
  - Chọn bản: Linux, C++17, CPU only (hoặc CUDA nếu driver cho phép).  
  - Giải nén, ví dụ: `/home/phu/libtorch`

### Tuỳ chọn

- **CUDA Toolkit** (nếu muốn chạy Sobel trên GPU qua `SobelCuda.cu`).  
- **OpenMP** (thường có sẵn với `g++`) để song song hóa Sobel trên CPU.

### Python (chỉ để test model / export)

```bash
python3 -m pip install --user yolov5 torch torchvision opencv-python
```

---

## Bước 1: Export YOLOv5 `.pt` → TorchScript `.pt`

Trong `Main/` đã có sẵn script:

```bash
cd /home/phu/Documents/ParallelProgramming/DetectVehicleLicensePlateSobelFIlter/Main

python3 export_torchscript.py
```

Sau khi chạy thành công sẽ tạo:

- `Main/lp_detector_ts.pt`  – dùng cho `LPDetector` (C++)  
- `Main/lp_ocr_ts.pt`       – dùng cho `LPOCR` (C++)  

Nếu có lỗi, kiểm tra lại:

- Đã cài `torch`, `yolov5` chưa.  
- Có kết nối mạng lần đầu để `torch.hub` tải repo YOLOv5.

---

## Bước 2: Build pipeline C++ với CMake

Từ thư mục gốc project:

```bash
cd /home/phu/Documents/ParallelProgramming/DetectVehicleLicensePlateSobelFIlter

mkdir -p build
cd build

# Giả sử LibTorch nằm ở /home/phu/libtorch
cmake -DCMAKE_PREFIX_PATH=/home/phu/libtorch ..

make -j$(nproc)
```

Nếu mọi thứ OK, CMake sẽ tìm được:

- OpenCV (`OpenCV_VERSION`)  
- LibTorch (`TORCH_INSTALL_PREFIX`)  
- (tuỳ chọn) CUDA, OpenMP

Sau khi `make` xong, sẽ có binary:

- `build/lp_main`

---

## Bước 3: Chạy pipeline C++ (dùng model thật)

Từ thư mục `build/`:

```bash
cd /home/phu/Documents/ParallelProgramming/DetectVehicleLicensePlateSobelFIlter/build

# 1) Dùng camera mặc định (index 0)
./lp_main 0

# 2) Hoặc dùng video file
./lp_main ../video.mp4
```

Pipeline sẽ:

1. **Capture**: đọc frame từ camera/video (`cv::VideoCapture`).  
2. **Sobel**:  
   - Nếu có CUDA & bật `USE_CUDA_SOBEL` → chạy `SobelCuda.cu` trên GPU.  
   - Ngược lại → dùng Sobel CPU với OpenMP (`sobelCPU`).  
3. **Detect + OCR** (C++):  
   - `LPDetector` load `Main/lp_detector_ts.pt`, chạy `forward()` để lấy bbox biển số (`std::vector<cv::Rect>`).  
   - `LPOCR` load `Main/lp_ocr_ts.pt`, chạy `forward()` trên ROI biển số để lấy chuỗi ký tự.  
4. **Render + log**:  
   - Vẽ khung biển số + text lên frame.  
   - In ra console:  
     `"[Frame X] Bien so: 5535Z5A55"`  
   - Đây là chỗ có thể gắn thêm **MQTT** để gửi log ra ngoài (theo yêu cầu của dự án gốc).

Nhấn **`q`** để thoát.

---

## Demo nhanh model YOLOv5 bằng Python (không C++)

File `Main/run_models.py` cho phép test nhanh 2 model `.pt` gốc:

```bash
cd /home/phu/Documents/ParallelProgramming/DetectVehicleLicensePlateSobelFIlter/Main

# Test trên ảnh
python3 run_models.py --source ../bienso.jpg

# Test trên webcam (nếu có, ví dụ /dev/video0)
python3 run_models.py --source 0
```

Script sẽ:

- Dùng `torch.hub.load("ultralytics/yolov5", "custom", path=...)` để load 2 model YOLOv5.  
- Chạy detect + OCR, vẽ lên ảnh, in log dạng:

```text
[DETECT] plate_text='5535Z5A55' det_conf=0.90 bbox=(x1,y1,x2,y2)
```

Phần này hữu ích để:

- Đảm bảo model `.pt` hoạt động đúng trước khi export sang TorchScript.  
- So sánh kết quả giữa Python và C++ (TorchScript) nếu cần tune thêm parse output.

---

## Ghi chú & Hướng phát triển tiếp

- **Lớp `LPDetector` / `LPOCR`** hiện parse output TorchScript với giả định format gần giống YOLOv5 sau NMS (`[N, 6]` với `x1,y1,x2,y2,conf,cls`).  
  - Nếu format thực tế khác, chỉ cần chỉnh lại phần parse trong `LPDetector.cpp` và `LPOCR.cpp`.  
- **`classNames_` trong `LPOCR`**:  
  - Nếu bạn có danh sách alphabet dùng khi train (0–9, A–Z, v.v.), có thể điền vào để tránh phải dùng `cls_id` số.  
- Có thể bổ sung:  
  - **Voting nhiều frame** để ổn định biển số (giảm nhiễu OCR).  
  - Gửi **MQTT** khi phát hiện biển số mới (log một lần cho mỗi biển số).  
  - Benchmark CPU vs GPU Sobel và tối ưu thêm.


