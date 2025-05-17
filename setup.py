from skbuild import setup

setup(
    name="sobel",
    version="0.0.1",
    description="A Python wrapper for the Sobel edge detection algorithm",
    author="Daniel Leon",
    packages=["sobel"],
    cmake_install_dir="sobel"
)