#!/bin/bash
# Script dùng ffmpeg làm bridge để đọc RTSP và pipe vào OpenCV
# Vì VLC chạy được nhưng OpenCV không được, dùng ffmpeg làm trung gian

RTSP_URL="${1:-rtsp://admin:SWPLBG@192.168.1.5:554}"
FIFO="/tmp/rtsp_fifo_$$"

# Tạo named pipe
mkfifo "$FIFO"

# Chạy ffmpeg đọc RTSP và ghi vào pipe
# Dùng format rawvideo để OpenCV đọc được
ffmpeg -rtsp_transport tcp -i "$RTSP_URL" \
    -f rawvideo -pix_fmt bgr24 - \
    > "$FIFO" 2>/dev/null &

FFMPEG_PID=$!

# Đợi một chút để ffmpeg khởi động
sleep 2

# Kiểm tra xem ffmpeg còn chạy không
if ! kill -0 $FFMPEG_PID 2>/dev/null; then
    echo "FFmpeg không thể kết nối RTSP stream"
    rm -f "$FIFO"
    exit 1
fi

echo "FFmpeg bridge đang chạy, pipe: $FIFO"
echo "PID: $FFMPEG_PID"
echo ""
echo "Để dùng trong code, đọc từ: $FIFO"
echo "Hoặc chạy: ./lp_main $FIFO"

# Cleanup khi exit
trap "kill $FFMPEG_PID 2>/dev/null; rm -f $FIFO; exit" INT TERM

# Giữ script chạy
wait $FFMPEG_PID

