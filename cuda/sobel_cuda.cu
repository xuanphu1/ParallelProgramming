// CUDA kernel cho Sobel Frame Gating
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>

// CUDA kernel để tính edge pixels
// Dùng unsigned int cho atomicAdd để tương thích với compute capability 3.0
__global__ void count_edge_pixels(const unsigned char* grad_x, const unsigned char* grad_y, 
                                   unsigned int* edge_count, int width, int height, double threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;
    
    if (idx < total_pixels) {
        int x = idx % width;
        int y = idx / width;
        int gx = grad_x[y * width + x];
        int gy = grad_y[y * width + x];
        double magnitude = gx + gy;
        
        if (magnitude > threshold) {
            atomicAdd(edge_count, 1);
        }
    }
}

// Wrapper function để gọi từ C++
extern "C" {
    void cuda_count_edges(const unsigned char* h_grad_x, const unsigned char* h_grad_y,
                          int* h_edge_count, int width, int height, double threshold) {
        int total_pixels = width * height;
        
        // Check CUDA errors
        cudaError_t err;
        
        // Allocate device memory
        unsigned char *d_grad_x, *d_grad_y;
        unsigned int *d_edge_count;
        
        err = cudaMalloc((void**)&d_grad_x, total_pixels * sizeof(unsigned char));
        if (err != cudaSuccess) {
            *h_edge_count = 0;
            return;
        }
        
        err = cudaMalloc((void**)&d_grad_y, total_pixels * sizeof(unsigned char));
        if (err != cudaSuccess) {
            cudaFree(d_grad_x);
            *h_edge_count = 0;
            return;
        }
        
        err = cudaMalloc((void**)&d_edge_count, sizeof(unsigned int));
        if (err != cudaSuccess) {
            cudaFree(d_grad_x);
            cudaFree(d_grad_y);
            *h_edge_count = 0;
            return;
        }
        
        // Copy data to device
        err = cudaMemcpy(d_grad_x, h_grad_x, total_pixels * sizeof(unsigned char), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(d_grad_x);
            cudaFree(d_grad_y);
            cudaFree(d_edge_count);
            *h_edge_count = 0;
            return;
        }
        
        err = cudaMemcpy(d_grad_y, h_grad_y, total_pixels * sizeof(unsigned char), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(d_grad_x);
            cudaFree(d_grad_y);
            cudaFree(d_edge_count);
            *h_edge_count = 0;
            return;
        }
        
        err = cudaMemset(d_edge_count, 0, sizeof(unsigned int));
        if (err != cudaSuccess) {
            cudaFree(d_grad_x);
            cudaFree(d_grad_y);
            cudaFree(d_edge_count);
            *h_edge_count = 0;
            return;
        }
        
        // Launch kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;
        
        count_edge_pixels<<<blocksPerGrid, threadsPerBlock>>>(d_grad_x, d_grad_y, d_edge_count, width, height, threshold);
        
        // Check kernel launch
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaFree(d_grad_x);
            cudaFree(d_grad_y);
            cudaFree(d_edge_count);
            *h_edge_count = 0;
            return;
        }
        
        // Wait for kernel to complete
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            cudaFree(d_grad_x);
            cudaFree(d_grad_y);
            cudaFree(d_edge_count);
            *h_edge_count = 0;
            return;
        }
        
        // Copy result back
        unsigned int result = 0;
        err = cudaMemcpy(&result, d_edge_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            *h_edge_count = 0;
        } else {
            *h_edge_count = (int)result;
        }
        
        // Cleanup
        cudaFree(d_grad_x);
        cudaFree(d_grad_y);
        cudaFree(d_edge_count);
    }
}

