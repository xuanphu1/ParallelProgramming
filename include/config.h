#ifndef CONFIG_H
#define CONFIG_H

#include <string>

// Cấu hình RTSP
extern const std::string CAMERA_IP;
extern const std::string USERNAME;
extern const std::string PASSWORD;

// Cấu hình Gamma Correction (có thể override từ command line hoặc environment variable)
extern bool USE_GAMMA_CORRECTION;
extern double GAMMA_VALUE;
extern bool SAVE_FILTERED_IMAGES;
extern std::string FILTERED_OUTPUT_DIR;

// Cấu hình Sobel Frame Gating
extern bool USE_SOBEL_GATING;
extern double SOBEL_THRESHOLD;
extern double EDGE_DENSITY_THRESHOLD_LOW;
extern double EDGE_DENSITY_THRESHOLD_HIGH;

// Cấu hình Sobel Edge Enhancement cho OCR
extern bool USE_SOBEL_OCR_ENHANCEMENT;
extern double SOBEL_ENHANCEMENT_STRENGTH;

#endif // CONFIG_H

