#!/bin/bash

# Script helper để thu thập metrics trong khi chạy benchmark
# Usage: collect_metrics.sh <pid> <duration_seconds> <output_file>

PID=$1
DURATION=$2
OUTPUT_FILE=$3

if [ -z "$PID" ] || [ -z "$DURATION" ] || [ -z "$OUTPUT_FILE" ]; then
    echo "Usage: $0 <pid> <duration_seconds> <output_file>"
    exit 1
fi

# Tạo file output với header
echo "timestamp,cpu_percent,memory_mb,gpu_percent" > "$OUTPUT_FILE"

# Thu thập metrics mỗi giây
for i in $(seq 1 $DURATION); do
    # CPU và Memory từ /proc
    if [ -d "/proc/$PID" ]; then
        CPU=$(ps -p $PID -o %cpu --no-headers | tr -d ' ')
        MEM=$(ps -p $PID -o rss --no-headers | awk '{print $1/1024}')  # MB
        
        # GPU usage (nếu có)
        if command -v nvidia-smi &> /dev/null; then
            GPU=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
        else
            GPU="N/A"
        fi
        
        echo "$i,$CPU,$MEM,$GPU" >> "$OUTPUT_FILE"
    fi
    sleep 1
done

