# CMakeLists.txt

cmake_minimum_required(VERSION 3.10)

project(sobel)

set(CMAKE_CXX_STANDARD 14)

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
