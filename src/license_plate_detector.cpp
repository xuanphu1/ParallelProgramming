#include "../include/license_plate_detector.h"
#include "../include/image_processing.h"
#include "../include/config.h"
#include "../include/utils.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace cv;
using namespace std;
using namespace Ort;

LicensePlateDetector::LicensePlateDetector(const string& detector_path, const string& ocr_path, const OCRConfig& config) 
    : env(ORT_LOGGING_LEVEL_WARNING, "LicensePlateDetector"),
      memory_info(MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      ocr_config(config) {
    
    // Session options - Tối đa hóa threading cho ONNX Runtime
    SessionOptions session_options;
    // IntraOp: số threads cho operations trong một node (matrix multiplication, convolution, etc.)
    // Đã set 8 threads để song song hóa các operations trong model
    session_options.SetIntraOpNumThreads(8);
    // InterOp: số threads để chạy các nodes song song (nếu model cho phép)
    // Cho phép chạy nhiều nodes song song khi có thể
    session_options.SetInterOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    // Enable CPU memory arena để tái sử dụng memory hiệu quả hơn
    session_options.EnableCpuMemArena();
    
    // Load detector model
    cout << "🔄 Loading detector model: " << detector_path << endl;
    detector_session = new Session(env, detector_path.c_str(), session_options);
    
    // Get detector input/output names
    size_t num_input_nodes = detector_session->GetInputCount();
    size_t num_output_nodes = detector_session->GetOutputCount();
    
    for (size_t i = 0; i < num_input_nodes; i++) {
        auto input_name = detector_session->GetInputNameAllocated(i, allocator);
        detector_input_names.push_back(input_name.get());
        auto input_shape = detector_session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        if (input_shape.size() == 4 && input_shape[1] == 3) {
            detector_input_size = input_shape[2]; // 640 or 384
        }
    }
    
    for (size_t i = 0; i < num_output_nodes; i++) {
        auto output_name = detector_session->GetOutputNameAllocated(i, allocator);
        detector_output_names.push_back(output_name.get());
    }
    
    cout << "   Detector input size: " << detector_input_size << "x" << detector_input_size << endl;
    
    // Load OCR model
    cout << "🔄 Loading OCR model: " << ocr_path << endl;
    ocr_session = new Session(env, ocr_path.c_str(), session_options);
    
    // Get OCR input/output names
    num_input_nodes = ocr_session->GetInputCount();
    num_output_nodes = ocr_session->GetOutputCount();
    
    for (size_t i = 0; i < num_input_nodes; i++) {
        auto input_name = ocr_session->GetInputNameAllocated(i, allocator);
        ocr_input_names.push_back(input_name.get());
    }
    
    for (size_t i = 0; i < num_output_nodes; i++) {
        auto output_name = ocr_session->GetOutputNameAllocated(i, allocator);
        ocr_output_names.push_back(output_name.get());
    }
    
    cout << "✅ Models loaded successfully!" << endl;
}

LicensePlateDetector::~LicensePlateDetector() {
    delete detector_session;
    delete ocr_session;
}

// Detect license plates - format: [X, x1, y1, x2, y2, class_id, score]
vector<Detection> LicensePlateDetector::detect(const Mat& frame, float conf_threshold) {
    // Detect trên frame gốc (không áp dụng Sobel filter)
    // Preprocess với letterbox
    auto [input_tensor, ratio, padding] = preprocess_letterbox(frame, detector_input_size);
    float ratio_x = ratio, ratio_y = ratio;
    float dw = padding.first, dh = padding.second;
    
    // Tạo input tensor
    vector<int64_t> input_shape = {1, 3, detector_input_size, detector_input_size};
    Value input_tensor_value = Value::CreateTensor<float>(
        memory_info, input_tensor.data(), input_tensor.size(),
        input_shape.data(), input_shape.size());
    
    // Convert string names to const char*
    vector<const char*> detector_input_names_cstr;
    vector<const char*> detector_output_names_cstr;
    for (const auto& name : detector_input_names) {
        detector_input_names_cstr.push_back(name.c_str());
    }
    for (const auto& name : detector_output_names) {
        detector_output_names_cstr.push_back(name.c_str());
    }
    
    // Run inference
    auto output_tensors = detector_session->Run(RunOptions{nullptr},
        detector_input_names_cstr.data(), &input_tensor_value, 1,
        detector_output_names_cstr.data(), 1);
    
    // Get output
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    
    int num_detections = output_shape[0];
    
    vector<Detection> detections;
    
    if (num_detections == 0) {
        return detections;
    }
    
    // Parse output: format [X, x1, y1, x2, y2, class_id, score]
    // Bbox ở index 1:5 (x1, y1, x2, y2)
    // Class ID ở index 5
    // Score ở index 6
    int orig_h = frame.rows;
    int orig_w = frame.cols;
    
    for (int i = 0; i < num_detections; i++) {
        float* det = output_data + i * 7;
        
        // Bỏ qua index 0 (X), lấy bbox từ index 1:5
        float x1 = det[1];
        float y1 = det[2];
        float x2 = det[3];
        float y2 = det[4];
        int class_id = (int)det[5];
        float conf = det[6];
        
        if (conf >= conf_threshold) {
            // Scale về ảnh gốc: (bbox - padding) / ratio
            int x1_scaled = (int)((x1 - dw) / ratio_x);
            int y1_scaled = (int)((y1 - dh) / ratio_y);
            int x2_scaled = (int)((x2 - dw) / ratio_x);
            int y2_scaled = (int)((y2 - dh) / ratio_y);
            
            // Clip về bounds
            x1_scaled = max(0, min(x1_scaled, orig_w));
            y1_scaled = max(0, min(y1_scaled, orig_h));
            x2_scaled = max(0, min(x2_scaled, orig_w));
            y2_scaled = max(0, min(y2_scaled, orig_h));
            
            // Kiểm tra bbox hợp lệ
            if (x2_scaled > x1_scaled && y2_scaled > y1_scaled) {
                int width = x2_scaled - x1_scaled;
                int height = y2_scaled - y1_scaled;
                
                // Kích thước tối thiểu
                if (width >= 10 && height >= 10) {
                    Detection det_obj;
                    det_obj.x1 = x1_scaled;
                    det_obj.y1 = y1_scaled;
                    det_obj.x2 = x2_scaled;
                    det_obj.y2 = y2_scaled;
                    det_obj.confidence = conf;
                    det_obj.class_id = class_id;
                    detections.push_back(det_obj);
                }
            }
        }
    }
    
    return detections;
}

// OCR preprocessing
vector<uint8_t> LicensePlateDetector::preprocess_ocr(const Mat& plate_roi, int plate_id) {
    // Lưu ảnh gốc để so sánh
    if (SAVE_FILTERED_IMAGES && plate_id >= 0) {
        ensure_directory_exists(FILTERED_OUTPUT_DIR);
        string original_path = FILTERED_OUTPUT_DIR + "/ocr_plate_" + to_string(plate_id) + "_original.jpg";
        imwrite(original_path, plate_roi);
    }
    
    Mat processed_roi = plate_roi;
    
    // Áp dụng Gamma Correction SAU KHI ĐÃ CROP (chỉ cho OCR)
    if (USE_GAMMA_CORRECTION) {
        string gamma_path = "";
        if (SAVE_FILTERED_IMAGES && plate_id >= 0) {
            gamma_path = FILTERED_OUTPUT_DIR + "/ocr_plate_" + to_string(plate_id) + "_gamma_" + to_string(GAMMA_VALUE).substr(0, 3) + ".jpg";
        }
        processed_roi = apply_gamma_correction_parallel(processed_roi, GAMMA_VALUE, gamma_path);
    }
    
    // Áp dụng Sobel Edge Enhancement để làm rõ edges của text (SAU Gamma Correction)
    if (USE_SOBEL_OCR_ENHANCEMENT) {
        string sobel_path = "";
        if (SAVE_FILTERED_IMAGES && plate_id >= 0) {
            sobel_path = FILTERED_OUTPUT_DIR + "/ocr_plate_" + to_string(plate_id) + "_sobel_enhanced.jpg";
        }
        processed_roi = apply_sobel_edge_enhancement(processed_roi, SOBEL_ENHANCEMENT_STRENGTH);
        if (!sobel_path.empty()) {
            imwrite(sobel_path, processed_roi);
        }
    }
    
    // Resize về kích thước từ config
    Mat img_resized;
    resize(processed_roi, img_resized, Size(ocr_config.img_width, ocr_config.img_height), 0, 0, INTER_LINEAR);
    
    // Convert BGR -> RGB
    Mat img_rgb;
    if (ocr_config.image_color_mode == "rgb") {
        cvtColor(img_resized, img_rgb, COLOR_BGR2RGB);
    } else {
        Mat gray;
        cvtColor(img_resized, gray, COLOR_BGR2GRAY);
        cvtColor(gray, img_rgb, COLOR_GRAY2RGB);
    }
    
    // Convert to uint8 vector (HWC format) - Song song hóa với OpenMP
    vector<uint8_t> input_tensor(ocr_config.img_height * ocr_config.img_width * 3);
    
    #ifdef _OPENMP
    #pragma omp parallel for collapse(2)
    #endif
    for (int h = 0; h < ocr_config.img_height; h++) {
        for (int w = 0; w < ocr_config.img_width; w++) {
            Vec3b pixel = img_rgb.at<Vec3b>(h, w);
            int idx = h * ocr_config.img_width * 3 + w * 3;
            input_tensor[idx] = pixel[0];     // R
            input_tensor[idx + 1] = pixel[1]; // G
            input_tensor[idx + 2] = pixel[2]; // B
        }
    }
    
    return input_tensor;
}

// OCR inference
OCRResult LicensePlateDetector::ocr(const Mat& plate_roi, int plate_id) {
    // Preprocess (áp dụng Sobel filter ở đây nếu được bật)
    vector<uint8_t> input_tensor = preprocess_ocr(plate_roi, plate_id);
    
    // Tạo input tensor (HWC format: batch, height, width, channels)
    vector<int64_t> input_shape = {1, ocr_config.img_height, ocr_config.img_width, 3};
    Value input_tensor_value = Value::CreateTensor<uint8_t>(
        memory_info, input_tensor.data(), input_tensor.size(),
        input_shape.data(), input_shape.size());
    
    // Convert string names to const char*
    vector<const char*> ocr_input_names_cstr;
    vector<const char*> ocr_output_names_cstr;
    for (const auto& name : ocr_input_names) {
        ocr_input_names_cstr.push_back(name.c_str());
    }
    for (const auto& name : ocr_output_names) {
        ocr_output_names_cstr.push_back(name.c_str());
    }
    
    // Run inference
    auto output_tensors = ocr_session->Run(RunOptions{nullptr},
        ocr_input_names_cstr.data(), &input_tensor_value, 1,
        ocr_output_names_cstr.data(), 1);
    
    // Get output
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    
    // Output shape: [batch, 9, 37]
    int num_slots = output_shape[1];
    int num_classes = output_shape[2];
    
    // Parse kết quả
    string text = "";
    float total_conf = 0.0f;
    int char_count = 0;
    
    for (int slot = 0; slot < num_slots && slot < ocr_config.max_plate_slots; slot++) {
        float* slot_probs = output_data + slot * num_classes;
        
        // Softmax
        vector<float> probs(num_classes);
        float max_prob = *max_element(slot_probs, slot_probs + num_classes);
        float sum = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            probs[i] = exp(slot_probs[i] - max_prob);
            sum += probs[i];
        }
        for (int i = 0; i < num_classes; i++) {
            probs[i] /= sum;
        }
        
        // Lấy class có prob cao nhất
        int best_idx = max_element(probs.begin(), probs.end()) - probs.begin();
        float best_prob = probs[best_idx];
        
        // Bỏ qua padding character
        if (best_idx < (int)ocr_config.alphabet.size() && ocr_config.alphabet[best_idx] != ocr_config.pad_char) {
            text += ocr_config.alphabet[best_idx];
            total_conf += best_prob;
            char_count++;
        } else if (best_idx < (int)ocr_config.alphabet.size() && ocr_config.alphabet[best_idx] == ocr_config.pad_char) {
            // Gặp padding, có thể đã hết ký tự
            break;
        }
    }
    
    OCRResult result;
    result.text = text.empty() ? "N/A" : text;
    result.confidence = char_count > 0 ? total_conf / char_count : 0.0f;
    
    return result;
}

