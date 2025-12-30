# Báo Cáo LaTeX - Dự Án Nhận Diện Biển Số

## Cấu trúc

- `main.tex`: File LaTeX chính chứa toàn bộ nội dung báo cáo
- `Makefile`: Script để build PDF
- `README.md`: Hướng dẫn này

## Cách Build PDF

### Yêu cầu:
- `pdflatex` (thường có trong gói `texlive-latex-base`)
- `texlive-lang-vietnamese` (cho tiếng Việt)

### Cài đặt (Ubuntu/Debian):
```bash
sudo apt-get install texlive-latex-base texlive-lang-vietnamese texlive-latex-extra
```

### Build:
```bash
cd Report
make
```

File PDF sẽ được tạo: `main.pdf`

### Xem PDF:
```bash
make view
```

### Xóa file build:
```bash
make clean
```

## Nội dung báo cáo

Báo cáo bao gồm:
1. Giới thiệu dự án
2. Kiến trúc hệ thống
3. Các kỹ thuật song song hóa (Data + Task Parallelism)
4. Chi tiết triển khai
5. Kết quả và đánh giá
6. Kết luận

## Tùy chỉnh

Bạn có thể chỉnh sửa `main.tex` để:
- Thêm tên của bạn vào phần author
- Thêm các hình ảnh/sơ đồ
- Thêm kết quả đo lường cụ thể
- Thêm các phần khác theo yêu cầu

