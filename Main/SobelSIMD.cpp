#include "SobelSIMD.h"
#include <immintrin.h>  // AVX intrinsics
#include <algorithm>

// Sobel filter với SIMD vectorization (Data Parallelism)
// Xử lý 8 pixels cùng lúc với AVX-256
bool sobelSIMD(const cv::Mat& src, cv::Mat& dst) {
    CV_Assert(src.type() == CV_8UC1);
    
    const int rows = src.rows;
    const int cols = src.cols;
    
    if (rows < 3 || cols < 3) {
        return false;
    }
    
    dst = cv::Mat::zeros(src.size(), CV_8UC1);
    
    // Process inner pixels with SIMD (8 pixels at a time)
    #pragma omp parallel for
    for (int y = 1; y < rows - 1; ++y) {
        const uchar* prevRow = src.ptr<uchar>(y - 1);
        const uchar* currRow = src.ptr<uchar>(y);
        const uchar* nextRow = src.ptr<uchar>(y + 1);
        uchar* dstRow = dst.ptr<uchar>(y);
        
        int x = 1;
        
        // SIMD processing: 8 pixels at a time
        for (; x <= cols - 9; x += 8) {
            // Load 8 pixels from each row (need 10 pixels for 8 outputs due to kernel)
            __m256i p00 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(prevRow + x - 1)));
            __m256i p01 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(prevRow + x)));
            __m256i p02 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(prevRow + x + 1)));
            
            __m256i p10 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(currRow + x - 1)));
            __m256i p12 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(currRow + x + 1)));
            
            __m256i p20 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(nextRow + x - 1)));
            __m256i p21 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(nextRow + x)));
            __m256i p22 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(nextRow + x + 1)));
            
            // Convert to float for calculations
            __m256 p00f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(p00, 0)));
            __m256 p02f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(p02, 0)));
            __m256 p10f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(p10, 0)));
            __m256 p12f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(p12, 0)));
            __m256 p20f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(p20, 0)));
            __m256 p22f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(p22, 0)));
            
            // Sobel X: -p00 - 2*p10 - p20 + p02 + 2*p12 + p22
            __m256 gx = _mm256_add_ps(
                _mm256_sub_ps(
                    _mm256_add_ps(p02f, _mm256_mul_ps(p12f, _mm256_set1_ps(2.0f))),
                    _mm256_add_ps(p00f, _mm256_mul_ps(p10f, _mm256_set1_ps(2.0f)))
                ),
                _mm256_sub_ps(p22f, p20f)
            );
            
            // Sobel Y: -p00 - 2*p01 - p02 + p20 + 2*p21 + p22
            __m256 p01f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(p01, 0)));
            __m256 p21f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(p21, 0)));
            
            __m256 gy = _mm256_add_ps(
                _mm256_sub_ps(
                    _mm256_add_ps(p20f, _mm256_mul_ps(p21f, _mm256_set1_ps(2.0f))),
                    _mm256_add_ps(p00f, _mm256_mul_ps(p01f, _mm256_set1_ps(2.0f)))
                ),
                _mm256_sub_ps(p22f, p02f)
            );
            
            // Magnitude: sqrt(gx^2 + gy^2)
            __m256 mag = _mm256_sqrt_ps(
                _mm256_add_ps(
                    _mm256_mul_ps(gx, gx),
                    _mm256_mul_ps(gy, gy)
                )
            );
            
            // Clamp to 255
            mag = _mm256_min_ps(mag, _mm256_set1_ps(255.0f));
            
            // Convert back to uchar and store
            __m256i magi = _mm256_cvtps_epi32(mag);
            __m128i magi16 = _mm_packus_epi32(
                _mm256_extracti128_si256(magi, 0),
                _mm256_extracti128_si256(magi, 1)
            );
            __m128i magi8 = _mm_packus_epi16(magi16, _mm_setzero_si128());
            _mm_storel_epi64((__m128i*)(dstRow + x), magi8);
        }
        
        // Process remaining pixels with scalar code
        for (; x < cols - 1; ++x) {
            int p00 = prevRow[x - 1];
            int p01 = prevRow[x];
            int p02 = prevRow[x + 1];
            int p10 = currRow[x - 1];
            int p12 = currRow[x + 1];
            int p20 = nextRow[x - 1];
            int p21 = nextRow[x];
            int p22 = nextRow[x + 1];
            
            int gx = -p00 - 2*p10 - p20 + p02 + 2*p12 + p22;
            int gy = -p00 - 2*p01 - p02 + p20 + 2*p21 + p22;
            
            float mag = std::sqrt(static_cast<float>(gx * gx + gy * gy));
            dstRow[x] = static_cast<uchar>(std::min(255.0f, mag));
        }
    }
    
    return true;
}

