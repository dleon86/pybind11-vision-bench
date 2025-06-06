#!/usr/bin/env python3
"""
Simple script to run the Sobel optimization benchmark.
Assumes the C++ module is already built.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Import the benchmark script
from tests.compare_sobel import main

if __name__ == "__main__":
    print("Running Sobel Filter Optimization Benchmark...")
    print("=" * 50)
    
    try:
        main()
        print("\nBenchmark completed successfully!")
        print("Check the generated PNG files for results:")
        
        # List generated files
        png_files = [f for f in os.listdir('.') if f.endswith('.png')]
        for file in sorted(png_files):
            if 'sobel' in file.lower() or 'optimization' in file.lower():
                print(f"  - {file}")
                
    except ImportError as e:
        print(f"Error importing sobel module: {e}")
        print("Please build the C++ module first:")
        print("  mkdir build && cd build")
        print("  cmake ..")
        print("  cmake --build . --config Release")
        print("  cd ..")
        sys.exit(1)
    except Exception as e:
        print(f"Benchmark failed: {e}")
        sys.exit(1) 