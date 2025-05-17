# Pybind11 Vision Bench: Sobel Edge Detection

A high-performance implementation of the Sobel edge detection algorithm using C++ and pybind11 for Python bindings.

## Overview

This project demonstrates how to accelerate image processing algorithms by implementing them in C++ and exposing them to Python through pybind11. The Sobel edge detection algorithm is used as a benchmark to compare the performance of C++ vs. pure NumPy implementations.

## Results

The C++ implementation is dramatically faster than the NumPy version:

| Implementation | Execution Time | Relative Speed |
|----------------|---------------|---------------|
| C++ | ~0.0005 seconds | ~1,000x faster |
| NumPy | ~0.5 seconds | baseline |

![Sobel Comparison](sobel_comparison.png)

## Implementation Details

The project uses:

- **C++14** for the core algorithm implementation
- **pybind11** to create Python bindings for the C++ code
- **scikit-build** and **CMake** for the build system
- **NumPy** for the Python reference implementation and array handling
- **Docker** for consistent build environment

The Sobel operator uses two 3×3 kernels to approximate the gradient of an image in the x and y directions:

```
Gx = [[ 1  0 -1],
      [ 2  0 -2],
      [ 1  0 -1]]

Gy = [[ 1  2  1],
      [ 0  0  0],
      [-1 -2 -1]]
```

The gradient magnitude is calculated as: `sqrt(Gx² + Gy²)`

## Setup and Usage

### Prerequisites

- Docker
- Python 3.10+
- pip

### Build and Run with Docker

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/pybind11-vision-bench.git
   cd pybind11-vision-bench
   ```

2. Build and start the Docker container:
   ```
   docker-compose build
   docker-compose up -d
   ```

3. Enter the container:
   ```
   docker-compose exec pybind bash
   ```

4. Inside the container, install the package:
   ```
   cd /app
   pip install .
   ```

5. Run the comparison test:
   ```
   python tests/compare_sobel.py
   ```

### Manual Installation (without Docker)

1. Install development requirements:
   ```
   pip install scikit-build pybind11 numpy matplotlib Pillow
   ```

2. Install the package:
   ```
   pip install .
   ```

3. Run the comparison test:
   ```
   python tests/compare_sobel.py
   ```

## Project Structure

- `src/sobel.cpp` - C++ implementation of the Sobel algorithm
- `sobel/__init__.py` - Python package initialization
- `tests/compare_sobel.py` - Benchmark comparing C++ and NumPy implementations
- `CMakeLists.txt` - CMake configuration for building the C++ extension
- `pyproject.toml` - Python build system configuration
- `setup.py` - Python package setup

## License

[MIT License](LICENSE)
