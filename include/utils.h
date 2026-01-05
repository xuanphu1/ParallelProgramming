#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <utility>
#include "types.h"

// Hàm parse YAML đơn giản
OCRConfig load_ocr_config(const std::string& config_path);

// Hàm tìm config file
std::string find_config_file(const std::string& ocr_path);

// Hàm tìm models
std::pair<std::string, std::string> find_models();

// Hàm tạo thư mục nếu chưa tồn tại
void ensure_directory_exists(const std::string& dir_path);

#endif // UTILS_H

