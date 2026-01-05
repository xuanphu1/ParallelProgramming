#include "../include/rtsp_client.h"
#include "../include/config.h"
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// Kết nối RTSP stream
VideoCapture connect_rtsp() {
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

