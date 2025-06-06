#!/bin/bash

echo "===================================="
echo "Sobel Filter Optimization Benchmark"
echo "===================================="

# Set up build environment
BUILD_DIR="build"
PYTHON_EXE="python3"

# Parse command line arguments
SKIP_BENCHMARKS=false
ITERATIONS=15
FORCE_REBUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-benchmarks)
            SKIP_BENCHMARKS=true
            shift
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --force-rebuild)
            FORCE_REBUILD=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --skip-benchmarks    Only create grouped analysis from existing data"
            echo "  --iterations N       Number of benchmark iterations (default: 15)"
            echo "  --force-rebuild      Force rebuild even if pickle data exists"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if we should skip benchmarks and just do analysis
if [ "$SKIP_BENCHMARKS" = true ]; then
    echo "Skipping benchmarks, creating grouped analysis from existing data..."
    $PYTHON_EXE tests/compare_sobel.py --load-only
    exit 0
fi

# Check if pickle files already exist and we're not forcing rebuild
existing_configs=()
if [ "$FORCE_REBUILD" = false ]; then
    for config in "basic" "unroll_only" "simd_only" "all_opts"; do
        if [ -f "benchmark_data_${config}.pkl" ]; then
            existing_configs+=("$config")
        fi
    done
    
    if [ ${#existing_configs[@]} -gt 0 ]; then
        echo "Found existing benchmark data for: ${existing_configs[*]}"
        echo "Use --force-rebuild to regenerate, or --skip-benchmarks to just create analysis"
        read -p "Continue with missing configurations only? [y/N]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Cancelled. Use --help for options."
            exit 0
        fi
    fi
fi

# Clean previous builds thoroughly (only if we're rebuilding)
echo "Cleaning previous builds..."
rm -rf "$BUILD_DIR"
rm -rf "_skbuild"
rm -rf "sobel.egg-info"
rm -rf "sobel/"
rm -f *.so sobel*.pyd sobel*.dll
rm -rf "__pycache__"
rm -rf "dist"

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Function to check if we should skip a configuration
should_skip_config() {
    local config=$1
    if [ "$FORCE_REBUILD" = false ] && [ -f "../benchmark_data_${config}.pkl" ]; then
        echo "Skipping $config (data exists, use --force-rebuild to regenerate)"
        return 0
    fi
    return 1
}

echo ""
echo "===================================="
echo "Building Basic Version (no optimizations)"
echo "===================================="
if ! should_skip_config "basic"; then
    cmake -DUSE_UNROLL=OFF -DUSE_SIMD=OFF ..
    cmake --build . --config Release
    if [ $? -ne 0 ]; then
        echo "Build failed!"
        exit 1
    fi

    # Copy built module
    cp sobel*.so .. 2>/dev/null || cp sobel*.pyd .. 2>/dev/null || true
    cd ..

    echo ""
    echo "===================================="
    echo "Testing Basic Build"
    echo "===================================="
    $PYTHON_EXE -c "
import sys
sys.path.insert(0, '.')
import sobel
print('Module imported successfully!')
print('Available methods:', [attr for attr in dir(sobel) if 'sobel' in attr])
print('Testing basic function...')
import numpy as np
test_img = np.random.randint(0, 256, (50, 50), dtype=np.int32)
result = sobel.sobel_basic(test_img)
print('Basic test passed, output shape:', result.shape)
"
    if [ $? -ne 0 ]; then
        echo "Import test failed!"
        exit 1
    fi

    echo ""
    echo "===================================="
    echo "Running Basic Configuration Benchmark"
    echo "===================================="
    $PYTHON_EXE tests/compare_sobel.py --config "basic:Basic C++ Implementation" --iterations $ITERATIONS
    cd "$BUILD_DIR"
fi

echo ""
echo "===================================="
echo "Building with Loop Unrolling Only"
echo "===================================="
if ! should_skip_config "unroll_only"; then
    cmake -DUSE_UNROLL=ON -DUSE_SIMD=OFF ..
    cmake --build . --config Release
    cp sobel*.so .. 2>/dev/null || cp sobel*.pyd .. 2>/dev/null || true
    cd ..
    echo "Testing unrolled version..."
    $PYTHON_EXE -c "
import sys
sys.path.insert(0, '.')
import sobel
import numpy as np
img = np.random.randint(0, 256, (100, 100), dtype=np.int32)
result = sobel.sobel_unrolled(img)
print('Unrolled test passed, output shape:', result.shape)
"

    echo ""
    echo "===================================="
    echo "Running Loop Unrolling Benchmark"
    echo "===================================="
    $PYTHON_EXE tests/compare_sobel.py --config "unroll_only:Loop Unrolling Only" --iterations $ITERATIONS
    cd "$BUILD_DIR"
fi

echo ""
echo "===================================="
echo "Building with SIMD Only"
echo "===================================="
if ! should_skip_config "simd_only"; then
    cmake -DUSE_UNROLL=OFF -DUSE_SIMD=ON ..
    cmake --build . --config Release
    cp sobel*.so .. 2>/dev/null || cp sobel*.pyd .. 2>/dev/null || true
    cd ..
    echo "Testing SIMD version..."
    $PYTHON_EXE -c "
import sys
sys.path.insert(0, '.')
import sobel
import numpy as np
img = np.random.randint(0, 256, (100, 100), dtype=np.int32)
result = sobel.sobel_simd(img)
print('SIMD test passed, output shape:', result.shape)
"

    echo ""
    echo "===================================="
    echo "Running SIMD Only Benchmark"
    echo "===================================="
    $PYTHON_EXE tests/compare_sobel.py --config "simd_only:SIMD Vectorization Only" --iterations $ITERATIONS
    cd "$BUILD_DIR"
fi

echo ""
echo "===================================="
echo "Building with All Optimizations"
echo "===================================="
if ! should_skip_config "all_opts"; then
    cmake -DUSE_UNROLL=ON -DUSE_SIMD=ON ..
    cmake --build . --config Release
    cp sobel*.so .. 2>/dev/null || cp sobel*.pyd .. 2>/dev/null || true
    cd ..

    echo ""
    echo "===================================="
    echo "Final All-Optimizations Benchmark"
    echo "===================================="
    $PYTHON_EXE tests/compare_sobel.py --config "all_opts:All Optimizations Enabled" --iterations $ITERATIONS
fi

# Return to main directory
cd "$(dirname "$0")"

echo ""
echo "===================================="
echo "Creating Comprehensive Grouped Analysis"
echo "===================================="
echo "Generating grouped bar charts from all available data..."
$PYTHON_EXE tests/compare_sobel.py --load-only

echo ""
echo "===================================="
echo "Creating Performance Summary"
echo "===================================="
$PYTHON_EXE -c "
import matplotlib.pyplot as plt
import os
import json

print('Generated files:')
print()

# Individual configuration files
config_files = {}
for file in sorted(os.listdir('.')):
    if file.endswith('.png'):
        if '_basic.png' in file or '_unroll_only.png' in file or '_simd_only.png' in file or '_all_opts.png' in file:
            # Extract config from filename
            for config in ['basic', 'unroll_only', 'simd_only', 'all_opts']:
                if f'_{config}.png' in file:
                    if config not in config_files:
                        config_files[config] = []
                    config_files[config].append(file)
                    break

# Print by configuration
for config, files in config_files.items():
    config_name = config.replace('_', ' ').title()
    print(f'{config_name} Configuration:')
    for file in sorted(files):
        file_type = file.replace(f'_{config}.png', '').replace('_', ' ').title()
        print(f'  - {file}')
    print()

# Global analysis files
print('Global Analysis:')
global_files = []
for file in sorted(os.listdir('.')):
    if file.endswith('.png') and not any(f'_{config}.png' in file for config in ['basic', 'unroll_only', 'simd_only', 'all_opts']):
        global_files.append(file)

for file in global_files:
    print(f'  - {file}')

# Data files
print()
print('Data Files:')
for file in sorted(os.listdir('.')):
    if file.endswith('.json') or file.endswith('.pkl'):
        print(f'  - {file}')

# Print global results summary
if os.path.exists('global_benchmark_results.json'):
    print()
    print('Performance Summary:')
    with open('global_benchmark_results.json', 'r') as f:
        results = json.load(f)
    
    for config, data in results.items():
        print(f'  {config}: {data["config_name"]}')
        if 'NumPy' in data['results'] and 'C++ Optimized' in data['results']:
            numpy_time = data['results']['NumPy']['mean']
            cpp_time = data['results']['C++ Optimized']['mean']
            speedup = numpy_time / cpp_time
            print(f'    Best speedup: {speedup:.1f}x vs NumPy')

print()
print('Benchmark complete! Check the generated PNG files for results.')
"

echo ""
echo "===================================="
echo "Benchmark Complete!"
echo "===================================="
echo "Generated files organized by type:"
echo ""
echo "📊 Configuration-specific analysis (per build config):"
echo "  - performance_comparison_<config>.png: Performance bars and speedup charts"
echo "  - time_distribution_<config>.png: Statistical distribution analysis"
echo "  - results_quality_comparison_<config>.png: Visual results comparison"
echo "  - performance_statistics_<config>.png: Detailed statistics table"
echo "  - sobel_before_after_comparison_<config>.png: Before/after comparison"
echo ""
echo "📈 Global comparison analysis:"
echo "  - global_performance_comparison.png: All configurations compared in one table"
echo "  - grouped_by_config_execution_times.png: Times grouped by build configuration"
echo "  - grouped_by_config_speedups.png: Speedups grouped by build configuration"
echo "  - grouped_by_method_execution_times.png: Times grouped by algorithm"
echo "  - grouped_by_method_speedups.png: Speedups grouped by algorithm"
echo "  - grouped_time_distributions.png: Statistical distributions for all methods"
echo "  - optimization_techniques.png: Explanation of optimization methods"
echo ""
echo "💾 Data files:"
echo "  - benchmark_data_<config>.pkl: Detailed benchmark data (for fast reloading)"
echo "  - global_benchmark_results.json: Summary results in JSON format"
echo ""
echo "🚀 Tip: Use './build_and_benchmark.sh --skip-benchmarks' to regenerate charts without re-running benchmarks!"
echo ""

# Don't pause in non-interactive environments
if [ -t 0 ]; then
    read -p "Press Enter to continue..."
fi 