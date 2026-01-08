# Báo Cáo LaTeX - So Sánh Hiệu Năng Song Song Hóa

## Mô Tả

Báo cáo LaTeX này trình bày chi tiết việc so sánh hiệu năng của các kỹ thuật song song hóa trong hệ thống phát hiện và đọc biển số xe.

## Yêu Cầu

Để compile báo cáo, cần cài đặt:

1. **LaTeX Distribution**:
   - TeX Live (Linux): `sudo apt-get install texlive-full`
   - MiKTeX (Windows)
   - MacTeX (macOS)

2. **Packages cần thiết**:
   - `babel` với `vietnamese` option
   - `amsmath`, `amsfonts`, `amssymb`
   - `graphicx`
   - `hyperref`
   - `listings`
   - `xcolor`
   - `geometry`
   - `float`
   - `booktabs`
   - `array`
   - `longtable`
   - `multirow`

## Cách Compile

### Cách 1: Sử dụng pdflatex (Khuyến nghị)

```bash
cd report/
pdflatex main.tex
pdflatex main.tex  # Chạy lại lần 2 để cập nhật references
```

### Cách 2: Sử dụng Makefile

```bash
cd report/
make
```

### Cách 3: Sử dụng latexmk (Tự động)

```bash
cd report/
latexmk -pdf main.tex
```

## Cấu Trúc File

```
report/
├── main.tex          # File LaTeX chính
├── README.md         # File này
└── main.pdf          # Output PDF (sau khi compile)
```

## Nội Dung Báo Cáo

Báo cáo bao gồm các phần chính:

1. **Giới Thiệu**: Mục tiêu, thông tin hệ thống, công nghệ sử dụng
2. **Kiến Trúc Hệ Thống**: 
   - Data Parallelism (CUDA, OpenMP, ONNX Runtime)
   - Task Parallelism (Parallel Pipeline)
3. **So Sánh Hiệu Năng**:
   - So sánh Sobel Edge Detection
   - So sánh Task Parallelism
   - So sánh FPS tổng hợp
4. **Kết Luận và Khuyến Nghị**
5. **Phụ Lục**: Cấu trúc dự án, code examples

## Lưu Ý

- Báo cáo sử dụng tiếng Việt với package `babel`
- Các bảng sử dụng `longtable` để tự động phân trang
- Code listings sử dụng syntax highlighting cho C++
- Hyperlinks được tự động tạo bởi `hyperref`

## Troubleshooting

### Lỗi: Package 'vietnamese' not found

Cài đặt package:
```bash
sudo apt-get install texlive-lang-other
```

### Lỗi: Missing packages

Cài đặt tất cả packages:
```bash
sudo apt-get install texlive-full
```

### Lỗi: Font không hiển thị đúng tiếng Việt

Đảm bảo đã cài đặt:
```bash
sudo apt-get install texlive-lang-other texlive-fonts-extra
```

## Tùy Chỉnh

Để chỉnh sửa báo cáo:

1. Mở file `main.tex`
2. Chỉnh sửa nội dung trong các section tương ứng
3. Compile lại để xem kết quả

## Liên Hệ

Nếu có vấn đề, vui lòng kiểm tra:
- Log file `.log` sau khi compile
- Đảm bảo tất cả packages đã được cài đặt
- Kiểm tra encoding của file (phải là UTF-8)

