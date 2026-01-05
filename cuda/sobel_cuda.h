// Header cho CUDA Sobel functions
#ifndef SOBEL_CUDA_H
#define SOBEL_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

void cuda_count_edges(const unsigned char* h_grad_x, const unsigned char* h_grad_y,
                      int* h_edge_count, int width, int height, double threshold);

#ifdef __cplusplus
}
#endif

#endif // SOBEL_CUDA_H

