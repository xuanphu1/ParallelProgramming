#!/bin/bash

# Script để chạy test_onnx_models với LD_LIBRARY_PATH đúng (từ thư mục scripts/)
# Mặc định sử dụng Parallel Pipeline (Task Parallelism)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ONNXRUNTIME_LIB="$PROJECT_DIR/onnxruntime-linux-x64-1.16.3/lib"

# Set LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$ONNXRUNTIME_LIB:$LD_LIBRARY_PATH"

# Mặc định sử dụng Parallel Pipeline
# Có thể override bằng: USE_PARALLEL_PIPELINE=0 ./run_test_onnx.sh
# Hoặc: ./run_test_onnx.sh --sequential
if [ -z "$USE_PARALLEL_PIPELINE" ]; then
    export USE_PARALLEL_PIPELINE=1
fi

# Kiểm tra arguments - nếu có --sequential thì override
ARGS=()
USE_PARALLEL=true
for arg in "$@"; do
    if [ "$arg" == "--sequential" ] || [ "$arg" == "-s" ]; then
        USE_PARALLEL=false
        export USE_PARALLEL_PIPELINE=0
    elif [ "$arg" == "--parallel" ] || [ "$arg" == "-p" ]; then
        USE_PARALLEL=true
        export USE_PARALLEL_PIPELINE=1
    else
        ARGS+=("$arg")
    fi
done

# Nếu không có --parallel hoặc --sequential trong args, thêm --parallel mặc định
if [ "$USE_PARALLEL" = true ] && [[ ! " $@ " =~ " --parallel " ]] && [[ ! " $@ " =~ " --sequential " ]]; then
    ARGS+=("--parallel")
fi

# Chạy chương trình (binary nằm ở thư mục project root)
exec "$PROJECT_DIR/test_onnx_models" "${ARGS[@]}"

