#include "../include/utils.h"
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <iostream>
#include <vector>

// Hàm parse YAML đơn giản
OCRConfig load_ocr_config(const std::string& config_path) {
    OCRConfig config;
    
    std::ifstream file(config_path);
    if (!file.is_open()) {
        std::cout << "⚠️  Không tìm thấy config file: " << config_path << std::endl;
        std::cout << "   Sử dụng config mặc định" << std::endl;
        return config;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // Bỏ qua comment và dòng trống
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // Tìm dấu ':'
        size_t colon_pos = line.find(':');
        if (colon_pos == std::string::npos) {
            continue;
        }
        
        std::string key = line.substr(0, colon_pos);
        std::string value = line.substr(colon_pos + 1);
        
        // Trim whitespace
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);
        
        // Remove quotes nếu có
        if (value.size() >= 2 && value[0] == '\'' && value.back() == '\'') {
            value = value.substr(1, value.size() - 2);
        }
        
        // Parse các giá trị
        if (key == "max_plate_slots") {
            config.max_plate_slots = std::stoi(value);
        } else if (key == "alphabet") {
            config.alphabet = value;
        } else if (key == "pad_char") {
            if (!value.empty()) {
                config.pad_char = value[0];
            }
        } else if (key == "img_height") {
            config.img_height = std::stoi(value);
        } else if (key == "img_width") {
            config.img_width = std::stoi(value);
        } else if (key == "image_color_mode") {
            config.image_color_mode = value;
        }
    }
    
    file.close();
    std::cout << "✅ Loaded config từ " << config_path << std::endl;
    std::cout << "   " << config.img_width << "x" << config.img_height 
         << ", alphabet=" << config.alphabet.size() << " chars" << std::endl;
    
    return config;
}

// Hàm tìm config file
std::string find_config_file(const std::string& ocr_path) {
    // Tìm config file tương ứng với OCR model
    std::vector<std::string> candidates;
    
    if (ocr_path.find("cct_s") != std::string::npos) {
        candidates.push_back("cct_s_v1_global_plate_config.yaml");
        candidates.push_back("model/cct_s_v1_global_plate_config.yaml");
    } else if (ocr_path.find("cct_xs") != std::string::npos) {
        candidates.push_back("cct_xs_v1_global_plate_config.yaml");
        candidates.push_back("model/cct_xs_v1_global_plate_config.yaml");
    }
    
    for (const auto& candidate : candidates) {
        std::ifstream file(candidate);
        if (file.good()) {
            file.close();
            return candidate;
        }
    }
    
    return "";
}

// Hàm tìm models
std::pair<std::string, std::string> find_models() {
    std::vector<std::string> detector_candidates = {
        "yolo-v9-t-640-license-plates-end2end.onnx",
        "yolo-v9-t-384-license-plates-end2end.onnx",
        "model/yolo-v9-t-640-license-plates-end2end.onnx",
        "model/yolo-v9-t-384-license-plates-end2end.onnx"
    };
    
    std::vector<std::string> ocr_candidates = {
        "cct_s_v1_global.onnx",
        "cct_xs_v1_global.onnx",
        "model/cct_s_v1_global.onnx",
        "model/cct_xs_v1_global.onnx"
    };
    
    std::string detector_path, ocr_path;
    
    for (const auto& path : detector_candidates) {
        std::ifstream file(path);
        if (file.good()) {
            file.close();
            detector_path = path;
            break;
        }
    }
    
    for (const auto& path : ocr_candidates) {
        std::ifstream file(path);
        if (file.good()) {
            file.close();
            ocr_path = path;
            break;
        }
    }
    
    return std::make_pair(detector_path, ocr_path);
}

// Hàm tạo thư mục nếu chưa tồn tại
void ensure_directory_exists(const std::string& dir_path) {
    // Kiểm tra và tạo thư mục nếu chưa có
    struct stat info;
    if (stat(dir_path.c_str(), &info) != 0) {
        // Thư mục không tồn tại, tạo mới
        mkdir(dir_path.c_str(), 0755);
    } else if (!(info.st_mode & S_IFDIR)) {
        // Tồn tại nhưng không phải thư mục
        mkdir(dir_path.c_str(), 0755);
    }
}

