# Makefile để compile báo cáo LaTeX

TEXFILE = main.tex
PDFFILE = $(TEXFILE:.tex=.pdf)
AUXFILES = *.aux *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz

.PHONY: all clean pdf view

all: pdf

pdf: $(PDFFILE)

$(PDFFILE): $(TEXFILE)
	pdflatex $(TEXFILE)
	pdflatex $(TEXFILE)  # Chạy lại lần 2 để cập nhật references

clean:
	rm -f $(AUXFILES)

cleanall: clean
	rm -f $(PDFFILE)

view: $(PDFFILE)
	@if command -v xdg-open > /dev/null; then \
		xdg-open $(PDFFILE); \
	elif command -v open > /dev/null; then \
		open $(PDFFILE); \
	else \
		echo "Vui lòng mở $(PDFFILE) thủ công"; \
	fi

help:
	@echo "Các lệnh có sẵn:"
	@echo "  make        - Compile báo cáo PDF"
	@echo "  make clean  - Xóa các file tạm"
	@echo "  make cleanall - Xóa tất cả file (bao gồm PDF)"
	@echo "  make view   - Mở PDF sau khi compile"
	@echo "  make help   - Hiển thị thông tin này"

