#!/bin/bash

# Script để chạy test_onnx_models với LD_LIBRARY_PATH đúng (từ thư mục scripts/)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ONNXRUNTIME_LIB="$PROJECT_DIR/onnxruntime-linux-x64-1.16.3/lib"

# Set LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$ONNXRUNTIME_LIB:$LD_LIBRARY_PATH"

# Chạy chương trình (binary nằm ở thư mục project root)
exec "$PROJECT_DIR/test_onnx_models" "$@"

