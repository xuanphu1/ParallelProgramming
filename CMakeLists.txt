cmake_minimum_required(VERSION 3.16)
project(LP_Pipeline_Cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -------- OpenCV --------
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# -------- ONNX Runtime C++ API --------
# Bạn cần cài ONNX Runtime C++ hoặc đặt CMAKE_PREFIX_PATH tới thư mục onnxruntime, ví dụ:
#   cmake -DCMAKE_PREFIX_PATH=/path/to/onnxruntime ..
# Hoặc cài qua package manager: sudo apt-get install libonnxruntime-dev (nếu có)
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    PATHS
        /usr/include
        /usr/local/include
        ${CMAKE_PREFIX_PATH}/include
)

find_library(ONNXRUNTIME_LIB onnxruntime
    PATHS
        /usr/lib
        /usr/local/lib
        ${CMAKE_PREFIX_PATH}/lib
)

if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIB)
    message(STATUS "Found ONNX Runtime: ${ONNXRUNTIME_INCLUDE_DIR}")
    set(HAS_ONNXRUNTIME TRUE)
else()
    message(WARNING "ONNX Runtime not found. Bạn cần cài ONNX Runtime C++ hoặc set CMAKE_PREFIX_PATH.")
    set(HAS_ONNXRUNTIME FALSE)
endif()

# -------- OpenMP (cho Sobel CPU song song) --------
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found")
    set(HAS_OPENMP TRUE)
else()
    message(STATUS "OpenMP not found, build without -fopenmp")
    set(HAS_OPENMP FALSE)
endif()

# -------- CUDA cho SobelCuda.cu (tùy chọn) --------
find_package(CUDA QUIET)
if(CUDA_FOUND)
    message(STATUS "CUDA found: ${CUDA_VERSION}")
    set(HAS_CUDA TRUE)
else()
    message(STATUS "CUDA not found, Sobel sẽ chạy trên CPU")
    set(HAS_CUDA FALSE)
endif()

include_directories(${OpenCV_INCLUDE_DIRS})
if(HAS_ONNXRUNTIME)
    include_directories(${ONNXRUNTIME_INCLUDE_DIR})
endif()

add_executable(lp_main
    Main/main.cpp
    Main/Pipeline.cpp
    Main/LPDetector.cpp
    Main/LPOCR.cpp
)

target_include_directories(lp_main PRIVATE Main)

target_link_libraries(lp_main
    ${OpenCV_LIBS}
)

if(HAS_ONNXRUNTIME)
    target_link_libraries(lp_main ${ONNXRUNTIME_LIB})
else()
    message(FATAL_ERROR "ONNX Runtime is required. Please install it or set CMAKE_PREFIX_PATH.")
endif()

if(HAS_OPENMP)
    target_link_libraries(lp_main OpenMP::OpenMP_CXX)
endif()

if(HAS_CUDA)
    enable_language(CUDA)
    target_sources(lp_main PRIVATE Main/SobelCuda.cu)
    target_compile_definitions(lp_main PRIVATE USE_CUDA_SOBEL)
    target_include_directories(lp_main PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(lp_main ${CUDA_LIBRARIES})
endif()

set_property(TARGET lp_main PROPERTY CXX_STANDARD 17)


