CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -fopenmp -march=native
NVCCFLAGS = -O2 -arch=sm_75 --compiler-options -fPIC -std=c++14
CUDA_LDFLAGS = -lcudart
OPENMP_LDFLAGS = -fopenmp

# Tìm OpenCV
OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv)
OPENCV_LIBS = $(shell pkg-config --libs opencv4 2>/dev/null || pkg-config --libs opencv)

# ONNX Runtime paths - sử dụng thư mục onnxruntime-linux-x64-1.16.3 trong project
ONNXRUNTIME_ROOT = $(shell pwd)/onnxruntime-linux-x64-1.16.3
ONNXRUNTIME_INCLUDE = $(ONNXRUNTIME_ROOT)/include
ONNXRUNTIME_LIB = $(ONNXRUNTIME_ROOT)/lib

# Kiểm tra xem ONNX Runtime có tồn tại không
ifeq ($(wildcard $(ONNXRUNTIME_INCLUDE)/onnxruntime_cxx_api.h),)
    # Thử các đường dẫn khác
    ifeq ($(wildcard /usr/include/onnxruntime_cxx_api.h),)
        ifeq ($(wildcard /usr/local/include/onnxruntime_cxx_api.h),)
            $(error ONNX Runtime not found at $(ONNXRUNTIME_INCLUDE). Please check path or set ONNXRUNTIME_ROOT)
        else
            ONNXRUNTIME_INCLUDE = /usr/local/include
            ONNXRUNTIME_LIB = /usr/local/lib
        endif
    else
        ONNXRUNTIME_INCLUDE = /usr/include
        ONNXRUNTIME_LIB = /usr/lib
    endif
endif

# Include directories
INCLUDES = -I$(ONNXRUNTIME_INCLUDE) -Iinclude -Icuda $(OPENCV_CFLAGS)
LIBS = -L$(ONNXRUNTIME_LIB) -lonnxruntime $(OPENCV_LIBS) -lpthread

# Target executable
TARGET = test_onnx_models

# Source files
SRC_DIR = src
CUDA_DIR = cuda
SRC_FILES = $(SRC_DIR)/main.cpp \
            $(SRC_DIR)/config.cpp \
            $(SRC_DIR)/utils.cpp \
            $(SRC_DIR)/image_processing.cpp \
            $(SRC_DIR)/license_plate_detector.cpp \
            $(SRC_DIR)/rtsp_client.cpp \
            $(SRC_DIR)/parallel_pipeline.cpp

CUDA_SOURCE = $(CUDA_DIR)/sobel_cuda.cu
CUDA_OBJ = $(CUDA_DIR)/sobel_cuda.o

# Object files
OBJ_FILES = $(SRC_FILES:.cpp=.o)

all: $(TARGET)

# Compile CUDA file
$(CUDA_OBJ): $(CUDA_SOURCE)
	$(NVCC) $(NVCCFLAGS) -Iinclude -c $(CUDA_SOURCE) -o $(CUDA_OBJ)

# Compile C++ source files
$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Link everything together
$(TARGET): $(OBJ_FILES) $(CUDA_OBJ)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJ_FILES) $(CUDA_OBJ) $(LIBS) $(CUDA_LDFLAGS) $(OPENMP_LDFLAGS)

run: $(TARGET)
	LD_LIBRARY_PATH=$(ONNXRUNTIME_LIB):$$LD_LIBRARY_PATH ./$(TARGET)

clean:
	rm -f $(TARGET) $(OBJ_FILES) $(CUDA_OBJ)

.PHONY: all clean run
