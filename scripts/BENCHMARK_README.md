# Benchmark Scripts - Hướng Dẫn Sử Dụng

## Tổng Quan

Các script này cho phép bạn benchmark hiệu năng của các phương pháp song song hóa khác nhau và tạo báo cáo tự động.

## Files

- `benchmark.sh`: Script chính để chạy benchmark
- `collect_metrics.sh`: Script helper để thu thập metrics (CPU, GPU, Memory)
- `generate_report.sh`: Script để tạo báo cáo markdown từ kết quả JSON

## Yêu Cầu

- `jq`: Để xử lý JSON (cài đặt: `sudo apt-get install jq`)
- `bc`: Để tính toán (thường đã có sẵn)
- `timeout`: Để giới hạn thời gian chạy (thường đã có sẵn)

## Cách Sử Dụng

### 1. Chạy Benchmark

```bash
cd /path/to/project
./scripts/benchmark.sh
```

**Tùy chọn**:
- `BENCHMARK_DURATION=60`: Thời gian chạy mỗi test (mặc định: 30s)
- `NUM_RUNS=5`: Số lần chạy mỗi test để lấy trung bình (mặc định: 3)

**Ví dụ**:
```bash
BENCHMARK_DURATION=60 NUM_RUNS=5 ./scripts/benchmark.sh
```

### 2. Tạo Báo Cáo

Sau khi benchmark hoàn thành, tạo báo cáo:

```bash
./scripts/generate_report.sh benchmark_results/results_YYYYMMDD_HHMMSS.json
```

Báo cáo sẽ được lưu tại: `benchmark_results/report_YYYYMMDD_HHMMSS.md`

## Các Test Cases

Script sẽ chạy 6 test cases:

1. **Sequential Sobel + Sequential Pipeline**
2. **OpenMP Sobel + Sequential Pipeline**
3. **CUDA Sobel + Sequential Pipeline**
4. **Sequential Sobel + Parallel Pipeline**
5. **OpenMP Sobel + Parallel Pipeline**
6. **CUDA Sobel + Parallel Pipeline**

## Kết Quả

- **JSON file**: `benchmark_results/results_YYYYMMDD_HHMMSS.json`
  - Chứa tất cả metrics chi tiết từng run
- **Markdown report**: `benchmark_results/report_YYYYMMDD_HHMMSS.md`
  - Báo cáo dễ đọc với bảng so sánh và khuyến nghị

## Metrics Thu Thập

- **FPS**: Frames per second (ước tính)
- **CPU Usage**: % CPU sử dụng (trung bình)
- **GPU Usage**: % GPU sử dụng (trung bình, nếu có GPU)
- **Memory**: Memory sử dụng (MB, trung bình)

## Lưu Ý

- Benchmark cần RTSP stream đang hoạt động
- Kết quả có thể khác nhau tùy vào tải hệ thống
- Nên chạy khi hệ thống ít tải để có kết quả chính xác nhất
- FPS là ước tính dựa trên thời gian xử lý, không phải số frames thực tế

## So Sánh Kết Quả Giữa Các Máy

Để so sánh kết quả giữa các máy khác nhau:

1. Chạy benchmark trên mỗi máy
2. Copy các file JSON và markdown report
3. So sánh các metrics trong báo cáo

**Ví dụ**:
```bash
# Máy 1
./scripts/benchmark.sh
# Copy: benchmark_results/results_*.json và report_*.md

# Máy 2
./scripts/benchmark.sh
# Copy: benchmark_results/results_*.json và report_*.md

# So sánh các file markdown report
```

