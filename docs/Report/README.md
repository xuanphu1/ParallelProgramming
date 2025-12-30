# Hướng Dẫn Biên Dịch Báo Cáo LaTeX

## Yêu Cầu

Để biên dịch báo cáo LaTeX, bạn cần cài đặt:

1. **TeX Distribution:**
   - Windows: MiKTeX hoặc TeX Live
   - Linux: `sudo apt-get install texlive-full` hoặc `texlive-latex-extra texlive-lang-vietnamese`
   - macOS: MacTeX

2. **Các package cần thiết:**
   - `babel` với option `vietnamese`
   - `amsmath`
   - `graphicx`
   - `float`
   - `listings`
   - `xcolor`
   - `hyperref`
   - `geometry`

## Cách Biên Dịch

### Windows (MiKTeX/TeX Live)

```bash
pdflatex main.tex
pdflatex main.tex  # Chạy lại lần 2 để cập nhật references
```

### Linux

```bash
pdflatex main.tex
pdflatex main.tex  # Chạy lại lần 2
```

### macOS

```bash
pdflatex main.tex
pdflatex main.tex
```

## Cấu Trúc File

- `main.tex`: File LaTeX chính chứa toàn bộ nội dung báo cáo

## Lưu Ý

- Nếu thiếu package, hệ thống sẽ tự động cài đặt (với MiKTeX) hoặc bạn cần cài thủ công
- Chạy `pdflatex` 2 lần để đảm bảo tất cả cross-references và table of contents được cập nhật đúng
- File PDF output sẽ là `main.pdf`

## Troubleshooting

### Lỗi: Package không tìm thấy

**Linux:**
```bash
sudo apt-get install texlive-lang-vietnamese
```

**Windows (MiKTeX):**
- Mở MiKTeX Console và cài package thiếu

### Lỗi: Font không hỗ trợ tiếng Việt

Đảm bảo đã cài `texlive-lang-vietnamese` (Linux) hoặc package `vietnam` (MiKTeX).

