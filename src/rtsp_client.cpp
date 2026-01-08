#include "../include/rtsp_client.h"
#include "../include/config.h"
#include <iostream>
#include <vector>
#include <cstdlib>

using namespace cv;
using namespace std;

// Kết nối RTSP stream hoặc đọc từ video file
VideoCapture connect_rtsp() {
    // Kiểm tra nếu có VIDEO_FILE environment variable
    const char* video_file = getenv("VIDEO_FILE");
    if (video_file != nullptr && string(video_file) != "") {
        cout << "📹 Đang đọc từ video file: " << video_file << endl;
        VideoCapture cap(video_file);
        if (cap.isOpened()) {
            Mat test_frame;
            if (cap.read(test_frame) && !test_frame.empty()) {
                cout << "   ✅ Video file đã mở thành công!" << endl;
                // Reset về frame đầu để đảm bảo đọc từ đầu
                cap.set(CAP_PROP_POS_FRAMES, 0);
                return cap;
            }
            cap.release();
        } else {
            cerr << "   ❌ Không thể mở video file: " << video_file << endl;
        }
    }
    
    // Nếu không có video file, thử kết nối RTSP
    vector<string> rtsp_urls = {
        "rtsp://" + USERNAME + ":" + PASSWORD + "@" + CAMERA_IP + ":554/cam/realmonitor?channel=1&subtype=0",
        "rtsp://" + USERNAME + ":" + PASSWORD + "@" + CAMERA_IP + ":554/Streaming/Channels/101",
        "rtsp://" + USERNAME + ":" + PASSWORD + "@" + CAMERA_IP + ":554/h264/ch1/main/av_stream",
    };
    
    VideoCapture cap;
    for (const auto& url : rtsp_urls) {
        cout << "   Đang thử: " << url << endl;
        cap.open(url);
        if (cap.isOpened()) {
            Mat test_frame;
            if (cap.read(test_frame) && !test_frame.empty()) {
                cout << "   ✅ Kết nối thành công!" << endl;
                return cap;
            }
            cap.release();
        }
    }
    
    return cap;
}

