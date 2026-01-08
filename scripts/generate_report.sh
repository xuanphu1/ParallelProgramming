#!/bin/bash

# Script để tạo báo cáo markdown từ kết quả benchmark

if [ $# -lt 1 ]; then
    echo "Usage: $0 <results_json_file>"
    exit 1
fi

RESULTS_FILE="$1"
REPORT_FILE="${RESULTS_FILE%.json}.md"

if [ ! -f "$RESULTS_FILE" ]; then
    echo "Error: File not found: $RESULTS_FILE"
    exit 1
fi

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "Error: jq is required but not installed. Install with: sudo apt-get install jq"
    exit 1
fi

echo "Generating report from $RESULTS_FILE..."

cat > "$REPORT_FILE" << EOF
# Benchmark Report - License Plate Detection

**Generated**: $(date)

## System Information

- **CPU**: $(jq -r '.system_info.cpu_model' "$RESULTS_FILE")
- **CPU Cores**: $(jq -r '.system_info.cpu_cores' "$RESULTS_FILE")
- **RAM**: $(jq -r '.system_info.ram' "$RESULTS_FILE")
- **GPU**: $(jq -r '.system_info.gpu_name' "$RESULTS_FILE")
- **CUDA**: $(jq -r '.system_info.cuda_version' "$RESULTS_FILE")

## Benchmark Configuration

- **Duration per test**: $(jq -r '.config.duration' "$RESULTS_FILE")s
- **Number of runs**: $(jq -r '.config.num_runs' "$RESULTS_FILE")
- **Timestamp**: $(jq -r '.timestamp' "$RESULTS_FILE")

## Results Summary

| Test Case | Avg FPS | Avg CPU % | Avg GPU % | Avg Memory MB |
|-----------|---------|-----------|-----------|---------------|
EOF

# Process results and calculate averages
jq -r '.results | group_by(.test_name) | .[] | 
  {
    name: .[0].test_name,
    avg_fps: ([.[].fps] | add / length),
    avg_cpu: ([.[].cpu_usage] | add / length),
    avg_gpu: (if ([.[].gpu_usage] | map(select(. != "N/A")) | length > 0) then ([.[].gpu_usage] | map(select(. != "N/A")) | add / length) else "N/A" end),
    avg_mem: ([.[].memory_mb] | add / length)
  } | 
  "| \(.name) | \(.avg_fps | floor) | \(.avg_cpu | floor) | \(if .avg_gpu == "N/A" then "N/A" else (.avg_gpu | floor) end) | \(.avg_mem | floor) |"' \
  "$RESULTS_FILE" >> "$REPORT_FILE"

cat >> "$REPORT_FILE" << 'EOF'

## Detailed Results

EOF

# Add detailed results
jq -r '.results | group_by(.test_name) | .[] | 
  "### \(.[0].test_name)\n\n" +
  "- **Pipeline**: \(.[0].pipeline_mode)\n" +
  "- **Sobel**: \(.[0].sobel_mode)\n" +
  "- **Average FPS**: \(([.[].fps] | add / length) | floor)\n" +
  "- **Average CPU Usage**: \(([.[].cpu_usage] | add / length) | floor)%\n" +
  "- **Average GPU Usage**: \(if ([.[].gpu_usage] | map(select(. != "N/A")) | length > 0) then ([.[].gpu_usage] | map(select(. != "N/A")) | add / length | floor) else "N/A" end)%\n" +
  "- **Average Memory**: \(([.[].memory_mb] | add / length) | floor) MB\n" +
  "- **Runs**:\n" +
  (.[] | "  - Run \(.run_num): FPS=\(.fps | floor), CPU=\(.cpu_usage | floor)%, GPU=\(.gpu_usage), Memory=\(.memory_mb | floor)MB\n") +
  "\n"' \
  "$RESULTS_FILE" >> "$REPORT_FILE"

cat >> "$REPORT_FILE" << 'EOF'

## Recommendations

Based on the benchmark results:

EOF

# Find best performance
BEST=$(jq -r '.results | group_by(.test_name) | .[] | 
  {
    name: .[0].test_name,
    avg_fps: ([.[].fps] | add / length)
  } | 
  "\(.avg_fps) \(.name)"' "$RESULTS_FILE" | sort -rn | head -1)

BEST_FPS=$(echo "$BEST" | cut -d' ' -f1)
BEST_TEST=$(echo "$BEST" | cut -d' ' -f2-)

cat >> "$REPORT_FILE" << EOF
- **Best Performance**: $BEST_TEST ($BEST_FPS FPS)

## Notes

- FPS values are estimates based on processing time
- Actual FPS may vary depending on RTSP stream quality and network conditions
- GPU usage is only available if NVIDIA GPU is present
- Results may vary between runs due to system load and other factors

EOF

echo "✅ Report generated: $REPORT_FILE"

