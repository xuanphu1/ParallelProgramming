#include "../include/config.h"

// Cấu hình RTSP
const std::string CAMERA_IP = "192.168.1.5";
const std::string USERNAME = "admin";
const std::string PASSWORD = "SWPLBG";

// Cấu hình Gamma Correction (có thể override từ command line hoặc environment variable)
bool USE_GAMMA_CORRECTION = false;
double GAMMA_VALUE = 1.5;
bool SAVE_FILTERED_IMAGES = false;
std::string FILTERED_OUTPUT_DIR = "filtered_images";

// Cấu hình Sobel Frame Gating
bool USE_SOBEL_GATING = false;
double SOBEL_THRESHOLD = 30.0;
double EDGE_DENSITY_THRESHOLD_LOW = 0.05;
double EDGE_DENSITY_THRESHOLD_HIGH = 0.15;

// Cấu hình Sobel Edge Enhancement cho OCR
bool USE_SOBEL_OCR_ENHANCEMENT = false;
double SOBEL_ENHANCEMENT_STRENGTH = 0.3;

