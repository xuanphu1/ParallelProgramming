# Makefile để build báo cáo LaTeX

LATEX = pdflatex
BIBTEX = bibtex
MAIN = main
BUILD_DIR = build

.PHONY: all clean view

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex
	@mkdir -p $(BUILD_DIR)
	@echo "Đang build PDF..."
	$(LATEX) -output-directory=$(BUILD_DIR) $(MAIN).tex
	$(LATEX) -output-directory=$(BUILD_DIR) $(MAIN).tex
	@if [ -f $(BUILD_DIR)/$(MAIN).pdf ]; then \
		cp $(BUILD_DIR)/$(MAIN).pdf .; \
		echo "✅ PDF đã được tạo: $(MAIN).pdf"; \
	else \
		echo "❌ Lỗi: Không thể tạo PDF"; \
		exit 1; \
	fi

clean:
	rm -rf $(BUILD_DIR)
	rm -f $(MAIN).pdf $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).toc

view: $(MAIN).pdf
	@if command -v xdg-open > /dev/null; then \
		xdg-open $(MAIN).pdf; \
	elif command -v evince > /dev/null; then \
		evince $(MAIN).pdf & \
	else \
		echo "Mở file $(MAIN).pdf bằng PDF viewer của bạn"; \
	fi

help:
	@echo "Các lệnh có sẵn:"
	@echo "  make       - Build PDF"
	@echo "  make clean - Xóa các file build"
	@echo "  make view  - Mở PDF (nếu có X11)"

