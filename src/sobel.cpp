// Sobel.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <iostream>

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

// Sobel operator
py::array_t<uint8_t> sobel(py::array_t<int> img) {
    auto img_mat = img.unchecked<2>();
    auto rows = img_mat.shape(0);
    auto cols = img_mat.shape(1);

    // Create an output array with the same shape as the input
    std::vector<py::ssize_t> shape = {rows, cols};
    auto gradient_magnitude = py::array_t<double>(shape);
    auto result = py::array_t<uint8_t>(shape);
    auto gradient_mat = gradient_magnitude.mutable_unchecked<2>();
    auto result_mat = result.mutable_unchecked<2>();

    // First pass: calculate gradient magnitudes
    for (py::ssize_t i = 0; i < rows; ++i) {
        for (py::ssize_t j = 0; j < cols; ++j) {
            // Initialize borders to zero
            gradient_mat(i, j) = 0.0;
            
            // Skip border pixels
            if (i == 0 || i == rows - 1 || j == 0 || j == cols - 1) {
                continue;
            }
            
            double gx_sum = 0.0;
            double gy_sum = 0.0;
            
            // Apply Sobel operator
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    double pixel_value = img_mat(i + dx, j + dy);
                    gx_sum += pixel_value * gx[dx + 1][dy + 1];
                    gy_sum += pixel_value * gy[dx + 1][dy + 1];
                }
            }
            // Calculate the gradient magnitude
            gradient_mat(i, j) = std::sqrt(gx_sum * gx_sum + gy_sum * gy_sum);
        }
    }
    
    // Second pass: find maximum value (for normalization)
    double max_magnitude = 0.0;
    for (py::ssize_t i = 0; i < rows; ++i) {
        for (py::ssize_t j = 0; j < cols; ++j) {
            if (gradient_mat(i, j) > max_magnitude) {
                max_magnitude = gradient_mat(i, j);
            }
        }
    }
    
    // Third pass: normalize to 0-255 range
    for (py::ssize_t i = 0; i < rows; ++i) {
        for (py::ssize_t j = 0; j < cols; ++j) {
            // Avoid division by zero
            if (max_magnitude > 0.0) {
                result_mat(i, j) = static_cast<uint8_t>((gradient_mat(i, j) / max_magnitude) * 255.0);
            } else {
                result_mat(i, j) = 0;
            }
        }
    }
    
    return result;
}

PYBIND11_MODULE(sobel, m) {
    m.doc() = "Sobel edge detection";
    m.def("sobel", &sobel, "Sobel edge detection");
}
