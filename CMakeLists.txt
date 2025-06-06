# CMakeLists.txt

cmake_minimum_required(VERSION 3.10)

project(sobel)

set(CMAKE_CXX_STANDARD 14)

# Optimization options
option(USE_UNROLL "Enable loop unrolling" OFF)
option(USE_SIMD "Enable SIMD optimizations" OFF)

# Add optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -ffast-math")
set(CMAKE_BUILD_TYPE Release)

if (USE_UNROLL)
    add_compile_options(-funroll-loops)
endif()

if (USE_SIMD)
    add_compile_options(-ftree-vectorize -ffast-math)
endif()

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND python3 -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

list(APPEND CMAKE_PREFIX_PATH ${pybind11_dir})

find_package(pybind11 REQUIRED)

# Add the Python module
pybind11_add_module(sobel src/sobel.cpp)

install(TARGETS sobel LIBRARY DESTINATION .)
