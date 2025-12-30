#!/bin/bash
# Script cài đặt LaTeX để build PDF

echo "=========================================="
echo "Cài đặt LaTeX để build báo cáo PDF"
echo "=========================================="
echo ""

# Kiểm tra quyền sudo
if [ "$EUID" -ne 0 ]; then 
    echo "Cần quyền sudo. Chạy: sudo ./install_latex.sh"
    exit 1
fi

echo "Đang cập nhật package list..."
apt-get update -qq

echo ""
echo "Đang cài đặt LaTeX (có thể mất 5-10 phút)..."
# Cài đặt các package cơ bản
apt-get install -y texlive-latex-base texlive-latex-extra texlive-latex-recommended

# Thử cài package tiếng Việt nếu có
if apt-cache show texlive-lang-vietnamese &> /dev/null; then
    echo "Cài đặt hỗ trợ tiếng Việt..."
    apt-get install -y texlive-lang-vietnamese
elif apt-cache show texlive-lang-other &> /dev/null; then
    echo "Cài đặt hỗ trợ ngôn ngữ khác (có thể bao gồm tiếng Việt)..."
    apt-get install -y texlive-lang-other
else
    echo "⚠️  Không tìm thấy package tiếng Việt, nhưng vẫn có thể build PDF"
    echo "   (babel với vietnamese sẽ dùng fallback)"
fi

echo ""
if command -v pdflatex &> /dev/null; then
    echo "✅ Cài đặt thành công!"
    echo ""
    pdflatex --version | head -1
    echo ""
    echo "Bây giờ bạn có thể chạy: make"
else
    echo "❌ Cài đặt thất bại. Vui lòng kiểm tra lại."
    exit 1
fi

