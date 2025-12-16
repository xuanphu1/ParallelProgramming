#include <cuda_runtime.h>
#include "SobelCuda.h"

// Kernel Sobel đơn giản 3x3 trên ảnh xám 8-bit
__global__ void sobelKernel(const unsigned char* src, unsigned char* dst,
                            int width, int height, int srcStep, int dstStep) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1) {
        return;
    }

    const unsigned char* row0 = src + (y - 1) * srcStep;
    const unsigned char* row1 = src + y * srcStep;
    const unsigned char* row2 = src + (y + 1) * srcStep;

    int p00 = row0[x - 1];
    int p01 = row0[x    ];
    int p02 = row0[x + 1];
    int p10 = row1[x - 1];
    int p11 = row1[x    ];
    int p12 = row1[x + 1];
    int p20 = row2[x - 1];
    int p21 = row2[x    ];
    int p22 = row2[x + 1];

    int gx = -p00 - 2 * p10 - p20 + p02 + 2 * p12 + p22;
    int gy = -p00 - 2 * p01 - p02 + p20 + 2 * p21 + p22;

    float mag = sqrtf((float)(gx * gx + gy * gy));
    if (mag > 255.0f) mag = 255.0f;

    unsigned char* drow = dst + y * dstStep;
    drow[x] = (unsigned char)mag;
}

bool sobelCuda(const cv::Mat& srcGray, cv::Mat& dstEdge) {
    if (srcGray.empty() || srcGray.type() != CV_8UC1) {
        return false;
    }

    int width = srcGray.cols;
    int height = srcGray.rows;

    dstEdge.create(srcGray.size(), CV_8UC1);

    unsigned char *d_src = nullptr, *d_dst = nullptr;
    size_t srcBytes = width * height * sizeof(unsigned char);
    size_t dstBytes = srcBytes;

    cudaError_t err;
    err = cudaMalloc(&d_src, srcBytes);
    if (err != cudaSuccess) return false;
    err = cudaMalloc(&d_dst, dstBytes);
    if (err != cudaSuccess) {
        cudaFree(d_src);
        return false;
    }

    err = cudaMemcpy(d_src, srcGray.data, srcBytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_src);
        cudaFree(d_dst);
        return false;
    }

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    sobelKernel<<<grid, block>>>(d_src, d_dst, width, height, width, width);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_src);
        cudaFree(d_dst);
        return false;
    }

    err = cudaMemcpy(dstEdge.data, d_dst, dstBytes, cudaMemcpyDeviceToHost);
    cudaFree(d_src);
    cudaFree(d_dst);

    return err == cudaSuccess;
}


