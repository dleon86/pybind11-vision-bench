// Sobel.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <immintrin.h>

namespace py = pybind11;

int gx[3][3] = {
    {1, 0, -1},
    {2, 0, -2},
    {1, 0, -1}
};

int gy[3][3] = {
    {1, 2, 1},
    {0, 0, 0},
    {-1, -2, -1}
};

// Original Sobel operator (baseline)
py::array_t<uint8_t> sobel_basic(py::array_t<int> img) {
    auto img_mat = img.unchecked<2>();
    auto rows = img_mat.shape(0);
    auto cols = img_mat.shape(1);

    std::vector<py::ssize_t> shape = {rows, cols};
    auto gradient_magnitude = py::array_t<double>(shape);
    auto result = py::array_t<uint8_t>(shape);
    auto gradient_mat = gradient_magnitude.mutable_unchecked<2>();
    auto result_mat = result.mutable_unchecked<2>();

    // First pass: calculate gradient magnitudes
    for (py::ssize_t i = 0; i < rows; ++i) {
        for (py::ssize_t j = 0; j < cols; ++j) {
            gradient_mat(i, j) = 0.0;
            
            if (i == 0 || i == rows - 1 || j == 0 || j == cols - 1) {
                continue;
            }
            
            double gx_sum = 0.0;
            double gy_sum = 0.0;
            
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    double pixel_value = img_mat(i + dx, j + dy);
                    gx_sum += pixel_value * gx[dx + 1][dy + 1];
                    gy_sum += pixel_value * gy[dx + 1][dy + 1];
                }
            }
            gradient_mat(i, j) = std::sqrt(gx_sum * gx_sum + gy_sum * gy_sum);
        }
    }
    
    // Find maximum value for normalization
    double max_magnitude = 0.0;
    for (py::ssize_t i = 0; i < rows; ++i) {
        for (py::ssize_t j = 0; j < cols; ++j) {
            if (gradient_mat(i, j) > max_magnitude) {
                max_magnitude = gradient_mat(i, j);
            }
        }
    }
    
    // Normalize to 0-255 range
    for (py::ssize_t i = 0; i < rows; ++i) {
        for (py::ssize_t j = 0; j < cols; ++j) {
            if (max_magnitude > 0.0) {
                result_mat(i, j) = static_cast<uint8_t>((gradient_mat(i, j) / max_magnitude) * 255.0);
            } else {
                result_mat(i, j) = 0;
            }
        }
    }
    
    return result;
}

// Loop unrolled version
py::array_t<uint8_t> sobel_unrolled(py::array_t<int> img) {
    auto img_mat = img.unchecked<2>();
    auto rows = img_mat.shape(0);
    auto cols = img_mat.shape(1);

    std::vector<py::ssize_t> shape = {rows, cols};
    auto gradient_magnitude = py::array_t<double>(shape);
    auto result = py::array_t<uint8_t>(shape);
    auto gradient_mat = gradient_magnitude.mutable_unchecked<2>();
    auto result_mat = result.mutable_unchecked<2>();

    // Calculate gradient magnitudes with unrolled loops
    for (py::ssize_t i = 1; i < rows - 1; ++i) {
        for (py::ssize_t j = 1; j < cols - 1; ++j) {
            // Manually unrolled Sobel kernel computation
            double p00 = img_mat(i-1, j-1), p01 = img_mat(i-1, j), p02 = img_mat(i-1, j+1);
            double p10 = img_mat(i, j-1),   p11 = img_mat(i, j),   p12 = img_mat(i, j+1);
            double p20 = img_mat(i+1, j-1), p21 = img_mat(i+1, j), p22 = img_mat(i+1, j+1);
            
            // Gx calculation (unrolled)
            double gx_sum = p00 * 1 + p01 * 0 + p02 * (-1) +
                           p10 * 2 + p11 * 0 + p12 * (-2) +
                           p20 * 1 + p21 * 0 + p22 * (-1);
            
            // Gy calculation (unrolled)
            double gy_sum = p00 * 1 + p01 * 2 + p02 * 1 +
                           p10 * 0 + p11 * 0 + p12 * 0 +
                           p20 * (-1) + p21 * (-2) + p22 * (-1);
            
            gradient_mat(i, j) = std::sqrt(gx_sum * gx_sum + gy_sum * gy_sum);
        }
    }
    
    // Set border pixels to zero
    for (py::ssize_t i = 0; i < rows; ++i) {
        gradient_mat(i, 0) = 0.0;
        gradient_mat(i, cols-1) = 0.0;
    }
    for (py::ssize_t j = 0; j < cols; ++j) {
        gradient_mat(0, j) = 0.0;
        gradient_mat(rows-1, j) = 0.0;
    }
    
    // Find maximum value
    double max_magnitude = 0.0;
    for (py::ssize_t i = 0; i < rows; ++i) {
        for (py::ssize_t j = 0; j < cols; ++j) {
            max_magnitude = std::max(max_magnitude, gradient_mat(i, j));
        }
    }
    
    // Normalize
    for (py::ssize_t i = 0; i < rows; ++i) {
        for (py::ssize_t j = 0; j < cols; ++j) {
            if (max_magnitude > 0.0) {
                result_mat(i, j) = static_cast<uint8_t>((gradient_mat(i, j) / max_magnitude) * 255.0);
            } else {
                result_mat(i, j) = 0;
            }
        }
    }
    
    return result;
}

// SIMD optimized version
py::array_t<uint8_t> sobel_simd(py::array_t<int> img) {
    auto img_mat = img.unchecked<2>();
    auto rows = img_mat.shape(0);
    auto cols = img_mat.shape(1);

    std::vector<py::ssize_t> shape = {rows, cols};
    auto gradient_magnitude = py::array_t<double>(shape);
    auto result = py::array_t<uint8_t>(shape);
    auto gradient_mat = gradient_magnitude.mutable_unchecked<2>();
    auto result_mat = result.mutable_unchecked<2>();

    // Process 4 pixels at a time using SIMD where possible
    for (py::ssize_t i = 1; i < rows - 1; ++i) {
        py::ssize_t j = 1;
        
        // Process 4 pixels at a time with SIMD
        for (; j <= cols - 5; j += 4) {
            __m128 gx_sum = _mm_setzero_ps();
            __m128 gy_sum = _mm_setzero_ps();
            
            // Load 4 consecutive pixels from each row
            for (int dy = -1; dy <= 1; ++dy) {
                __m128 p0 = _mm_set_ps(img_mat(i+dy, j+3), img_mat(i+dy, j+2), img_mat(i+dy, j+1), img_mat(i+dy, j));
                __m128 p1 = _mm_set_ps(img_mat(i+dy, j+4), img_mat(i+dy, j+3), img_mat(i+dy, j+2), img_mat(i+dy, j+1));
                __m128 p2 = _mm_set_ps(img_mat(i+dy, j+5), img_mat(i+dy, j+4), img_mat(i+dy, j+3), img_mat(i+dy, j+2));
                
                // Apply Gx kernel weights
                __m128 gx_weight0 = _mm_set1_ps(gx[dy+1][0]);
                __m128 gx_weight1 = _mm_set1_ps(gx[dy+1][1]);
                __m128 gx_weight2 = _mm_set1_ps(gx[dy+1][2]);
                
                gx_sum = _mm_add_ps(gx_sum, _mm_mul_ps(p0, gx_weight0));
                gx_sum = _mm_add_ps(gx_sum, _mm_mul_ps(p1, gx_weight1));
                gx_sum = _mm_add_ps(gx_sum, _mm_mul_ps(p2, gx_weight2));
                
                // Apply Gy kernel weights
                __m128 gy_weight0 = _mm_set1_ps(gy[dy+1][0]);
                __m128 gy_weight1 = _mm_set1_ps(gy[dy+1][1]);
                __m128 gy_weight2 = _mm_set1_ps(gy[dy+1][2]);
                
                gy_sum = _mm_add_ps(gy_sum, _mm_mul_ps(p0, gy_weight0));
                gy_sum = _mm_add_ps(gy_sum, _mm_mul_ps(p1, gy_weight1));
                gy_sum = _mm_add_ps(gy_sum, _mm_mul_ps(p2, gy_weight2));
            }
            
            // Calculate magnitudes
            __m128 gx_sq = _mm_mul_ps(gx_sum, gx_sum);
            __m128 gy_sq = _mm_mul_ps(gy_sum, gy_sum);
            __m128 magnitude = _mm_sqrt_ps(_mm_add_ps(gx_sq, gy_sq));
            
            // Store results
            float mag_array[4];
            _mm_storeu_ps(mag_array, magnitude);
            
            for (int k = 0; k < 4; ++k) {
                gradient_mat(i, j + k) = mag_array[k];
            }
        }
        
        // Handle remaining pixels
        for (; j < cols - 1; ++j) {
            double gx_sum = 0.0, gy_sum = 0.0;
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    double pixel_value = img_mat(i + dx, j + dy);
                    gx_sum += pixel_value * gx[dx + 1][dy + 1];
                    gy_sum += pixel_value * gy[dx + 1][dy + 1];
                }
            }
            gradient_mat(i, j) = std::sqrt(gx_sum * gx_sum + gy_sum * gy_sum);
        }
    }
    
    // Set border pixels to zero
    for (py::ssize_t i = 0; i < rows; ++i) {
        gradient_mat(i, 0) = 0.0;
        gradient_mat(i, cols-1) = 0.0;
    }
    for (py::ssize_t j = 0; j < cols; ++j) {
        gradient_mat(0, j) = 0.0;
        gradient_mat(rows-1, j) = 0.0;
    }
    
    // Find maximum and normalize
    double max_magnitude = 0.0;
    for (py::ssize_t i = 0; i < rows; ++i) {
        for (py::ssize_t j = 0; j < cols; ++j) {
            max_magnitude = std::max(max_magnitude, gradient_mat(i, j));
        }
    }
    
    for (py::ssize_t i = 0; i < rows; ++i) {
        for (py::ssize_t j = 0; j < cols; ++j) {
            if (max_magnitude > 0.0) {
                result_mat(i, j) = static_cast<uint8_t>((gradient_mat(i, j) / max_magnitude) * 255.0);
            } else {
                result_mat(i, j) = 0;
            }
        }
    }
    
    return result;
}

// Blocked/tiled version for better cache performance
py::array_t<uint8_t> sobel_blocked(py::array_t<int> img) {
    auto img_mat = img.unchecked<2>();
    auto rows = img_mat.shape(0);
    auto cols = img_mat.shape(1);

    std::vector<py::ssize_t> shape = {rows, cols};
    auto gradient_magnitude = py::array_t<double>(shape);
    auto result = py::array_t<uint8_t>(shape);
    auto gradient_mat = gradient_magnitude.mutable_unchecked<2>();
    auto result_mat = result.mutable_unchecked<2>();

    const int BLOCK_SIZE = 64; // Cache-friendly block size
    
    // Process image in blocks
    for (py::ssize_t bi = 1; bi < rows - 1; bi += BLOCK_SIZE) {
        for (py::ssize_t bj = 1; bj < cols - 1; bj += BLOCK_SIZE) {
            py::ssize_t i_end = std::min(bi + BLOCK_SIZE, rows - 1);
            py::ssize_t j_end = std::min(bj + BLOCK_SIZE, cols - 1);
            
            // Process block
            for (py::ssize_t i = bi; i < i_end; ++i) {
                for (py::ssize_t j = bj; j < j_end; ++j) {
                    double gx_sum = 0.0, gy_sum = 0.0;
                    
                    // Unrolled kernel application
                    double p00 = img_mat(i-1, j-1), p01 = img_mat(i-1, j), p02 = img_mat(i-1, j+1);
                    double p10 = img_mat(i, j-1),   p11 = img_mat(i, j),   p12 = img_mat(i, j+1);
                    double p20 = img_mat(i+1, j-1), p21 = img_mat(i+1, j), p22 = img_mat(i+1, j+1);
                    
                    gx_sum = p00 + 2*p10 + p20 - p02 - 2*p12 - p22;
                    gy_sum = p00 + 2*p01 + p02 - p20 - 2*p21 - p22;
                    
                    gradient_mat(i, j) = std::sqrt(gx_sum * gx_sum + gy_sum * gy_sum);
                }
            }
        }
    }
    
    // Set border pixels to zero
    for (py::ssize_t i = 0; i < rows; ++i) {
        gradient_mat(i, 0) = 0.0;
        gradient_mat(i, cols-1) = 0.0;
    }
    for (py::ssize_t j = 0; j < cols; ++j) {
        gradient_mat(0, j) = 0.0;
        gradient_mat(rows-1, j) = 0.0;
    }
    
    // Find maximum and normalize
    double max_magnitude = 0.0;
    for (py::ssize_t i = 0; i < rows; ++i) {
        for (py::ssize_t j = 0; j < cols; ++j) {
            max_magnitude = std::max(max_magnitude, gradient_mat(i, j));
        }
    }
    
    for (py::ssize_t i = 0; i < rows; ++i) {
        for (py::ssize_t j = 0; j < cols; ++j) {
            if (max_magnitude > 0.0) {
                result_mat(i, j) = static_cast<uint8_t>((gradient_mat(i, j) / max_magnitude) * 255.0);
            } else {
                result_mat(i, j) = 0;
            }
        }
    }
    
    return result;
}

// Combined optimizations version
py::array_t<uint8_t> sobel_optimized(py::array_t<int> img) {
    auto img_mat = img.unchecked<2>();
    auto rows = img_mat.shape(0);
    auto cols = img_mat.shape(1);

    std::vector<py::ssize_t> shape = {rows, cols};
    auto gradient_magnitude = py::array_t<double>(shape);
    auto result = py::array_t<uint8_t>(shape);
    auto gradient_mat = gradient_magnitude.mutable_unchecked<2>();
    auto result_mat = result.mutable_unchecked<2>();

    const int BLOCK_SIZE = 32;
    
    // Blocked processing with unrolled loops
    for (py::ssize_t bi = 1; bi < rows - 1; bi += BLOCK_SIZE) {
        for (py::ssize_t bj = 1; bj < cols - 1; bj += BLOCK_SIZE) {
            py::ssize_t i_end = std::min(bi + BLOCK_SIZE, rows - 1);
            py::ssize_t j_end = std::min(bj + BLOCK_SIZE, cols - 1);
            
            for (py::ssize_t i = bi; i < i_end; ++i) {
                for (py::ssize_t j = bj; j < j_end; ++j) {
                    // Prefetch next cache line
                    __builtin_prefetch(&img_mat(i, j+8), 0, 3);
                    
                    // Unrolled and optimized kernel
                    double p00 = img_mat(i-1, j-1), p01 = img_mat(i-1, j), p02 = img_mat(i-1, j+1);
                    double p10 = img_mat(i, j-1),                         p12 = img_mat(i, j+1);
                    double p20 = img_mat(i+1, j-1), p21 = img_mat(i+1, j), p22 = img_mat(i+1, j+1);
                    
                    double gx_sum = p00 + 2*p10 + p20 - p02 - 2*p12 - p22;
                    double gy_sum = p00 + 2*p01 + p02 - p20 - 2*p21 - p22;
                    
                    gradient_mat(i, j) = std::sqrt(gx_sum * gx_sum + gy_sum * gy_sum);
                }
            }
        }
    }
    
    // Set borders to zero
    for (py::ssize_t i = 0; i < rows; ++i) {
        gradient_mat(i, 0) = 0.0;
        gradient_mat(i, cols-1) = 0.0;
    }
    for (py::ssize_t j = 0; j < cols; ++j) {
        gradient_mat(0, j) = 0.0;
        gradient_mat(rows-1, j) = 0.0;
    }
    
    // Vectorized max finding and normalization
    double max_magnitude = 0.0;
    for (py::ssize_t i = 0; i < rows; ++i) {
        for (py::ssize_t j = 0; j < cols; ++j) {
            max_magnitude = std::max(max_magnitude, gradient_mat(i, j));
        }
    }
    
    const double inv_max = (max_magnitude > 0.0) ? 255.0 / max_magnitude : 0.0;
    for (py::ssize_t i = 0; i < rows; ++i) {
        for (py::ssize_t j = 0; j < cols; ++j) {
            result_mat(i, j) = static_cast<uint8_t>(gradient_mat(i, j) * inv_max);
        }
    }
    
    return result;
}

// Backward compatibility
py::array_t<uint8_t> sobel(py::array_t<int> img) {
    return sobel_basic(img);
}

PYBIND11_MODULE(sobel, m) {
    m.doc() = "Sobel edge detection with multiple optimization levels";
    m.def("sobel", &sobel, "Basic Sobel edge detection (backward compatibility)");
    m.def("sobel_basic", &sobel_basic, "Basic Sobel edge detection");
    m.def("sobel_unrolled", &sobel_unrolled, "Sobel with loop unrolling");
    m.def("sobel_simd", &sobel_simd, "Sobel with SIMD optimizations");
    m.def("sobel_blocked", &sobel_blocked, "Sobel with cache blocking");
    m.def("sobel_optimized", &sobel_optimized, "Sobel with combined optimizations");
}
