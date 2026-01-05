#ifndef TYPES_H
#define TYPES_H

#include <string>

// Struct để lưu OCR config
struct OCRConfig {
    int img_width = 128;
    int img_height = 64;
    std::string alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_";
    int max_plate_slots = 9;
    char pad_char = '_';
    std::string image_color_mode = "rgb";
};

// Struct để lưu detection
struct Detection {
    int x1, y1, x2, y2;
    float confidence;
    int class_id;
};

// Struct để lưu OCR result
struct OCRResult {
    std::string text;
    float confidence;
};

#endif // TYPES_H

