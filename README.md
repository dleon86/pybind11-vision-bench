# Sobel Edge Detection Optimization Study

A comprehensive exploration of C++ optimization techniques for the Sobel edge detection algorithm, demonstrating how manual optimizations and compiler flags can achieve **2000× speedup** over NumPy.

## 🚀 Key Results

On an Intel i9-10885H @ 2.4GHz with 32GB RAM:

| Implementation | Execution Time | Speedup vs NumPy |
|----------------|---------------|------------------|
| **C++ Optimized** | ~0.00005s | **~2000×** |
| C++ SIMD | ~0.00008s | ~1250× |
| C++ Unrolled | ~0.00015s | ~670× |
| C++ Basic | ~0.0003s | ~330× |
| NumPy Baseline | ~0.10s | 1× |

## 🛠️ Optimization Techniques Explored

This project systematically implements and benchmarks multiple optimization approaches:

### Manual Code Optimizations
- **Loop Unrolling**: Manual 3×3 kernel expansion
- **SIMD Intrinsics**: SSE vectorization for 4-pixel parallel processing  
- **Cache Blocking**: Tiled processing for better memory locality
- **Memory Prefetching**: Explicit prefetch instructions
- **Combined Optimizations**: Best-of-all-worlds implementation

### Compiler Flag Analysis
- **Basic**: `-O2` standard optimization
- **Unroll Only**: `-O2 -funroll-loops`
- **SIMD Only**: `-O2 -march=native -mavx2`
- **All Optimizations**: `-O3 -march=native -ffast-math -funroll-loops`

## 📊 Comprehensive Benchmarking Framework

The project includes a sophisticated benchmarking and visualization system:

### Features
- **Statistical Analysis**: Multiple iterations with mean/std/min/max timing
- **Cross-Configuration Comparison**: Compare multiple build configurations
- **Visual Quality Verification**: Ensure optimizations don't break correctness
- **Performance Scaling Analysis**: Manual vs compiler optimization effectiveness
- **Persistent Results**: Pickle-based data storage for reproducible analysis

### Generated Visualizations
- Performance comparison charts (log scale with error bars)
- Speedup analysis vs NumPy baseline
- Results quality verification (bit-wise identical outputs)
- Optimization technique illustrations
- Cross-configuration grouped analysis
- Manual vs compiler effectiveness ratios

## 🔬 Key Insights

1. **Manual SIMD Still Matters**: Even with `-march=native`, hand-written SIMD provides 1.5-1.8× additional speedup
2. **Compiler Flags Leave Performance on Table**: Auto-vectorization captures ~80% of available SIMD performance
3. **Cache Blocking Scales**: Benefits increase with larger images and processors with bigger L3 caches
4. **Quality Preservation**: All optimizations maintain bit-wise identical results to NumPy reference

## 🖼️ Example Output

The framework generates comprehensive performance analysis including:
- Individual method timing distributions
- Cross-build configuration comparisons  
- Optimization technique effectiveness analysis
- System-specific performance projections

## 🏗️ Technical Stack

- **C++17**: Core algorithm implementations with modern features
- **pybind11**: Seamless Python-C++ integration
- **CMake**: Cross-platform build system with optimization flags
- **SSE SIMD**: 128-bit vector instructions for parallel processing
- **Comprehensive Benchmarking**: Statistical analysis with matplotlib/seaborn visualization

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- GCC/Clang with C++17 support  
- CMake 3.15+
- SSE-capable processor (Intel 2001+, AMD 2003+) - AVX2 recommended for compiler auto-vectorization

### Quick Start
```bash
# Clone and build
git clone https://github.com/dleon86/pybind11-vision-bench.git
cd pybind11-vision-bench
pip install .

# Run comprehensive benchmark
python tests/compare_sobel.py --config all_opts --iterations 15

# Generate cross-configuration analysis  
python tests/compare_sobel.py --load-only
python tests/compare_sobel.py --focused-analysis
```

### Docker Environment
```bash
docker-compose build && docker-compose up -d
docker-compose exec pybind bash
cd /app && pip install . && python tests/compare_sobel.py
```

## 📈 Performance Scaling

Expected performance on different systems:
- **Desktop i7/i9**: 2200-2600× (higher base clocks)
- **Server with AVX-512**: 4000-5000× (wider SIMD, if code upgraded to AVX-512)
- **Apple M1/M2**: 1800-2200× (NEON competitive with SSE)
- **Older CPUs (pre-AVX2)**: 1400-2000× (reduced SIMD benefits)

## 🎯 Use Cases

This project demonstrates optimization techniques applicable to:
- **Computer Vision**: Edge detection, convolution, filtering
- **Signal Processing**: 2D kernel operations, image transforms
- **Scientific Computing**: Stencil computations, numerical methods
- **Performance Engineering**: SIMD optimization, cache-aware algorithms

## 📂 Project Structure

```
├── src/sobel.cpp              # Multiple C++ implementations
├── tests/compare_sobel.py     # Comprehensive benchmarking framework  
├── figures/                   # Generated performance visualizations
├── CMakeLists.txt            # Build configuration with optimization flags
├── pyproject.toml            # Python packaging
└── slides.tex                # LaTeX presentation of results
```

## 🔍 Algorithm Details

The Sobel operator computes image gradients using 3×3 convolution kernels:

```cpp
// X-direction kernel          // Y-direction kernel
[[-1  0  1],                  [[-1 -2 -1],
 [-2  0  2],          and      [ 0  0  0],
 [-1  0  1]]                   [ 1  2  1]]
```

Gradient magnitude: `sqrt(Gx² + Gy²)`, normalized to [0, 255]

## 📄 License

MIT License - Feel free to use these optimization techniques in your own projects!

## 🤝 Contributing

PRs welcome for:
- AVX-512 implementations
- GPU (CUDA/OpenCL) ports  
- ARM NEON optimizations
- Additional compiler/architecture benchmarks
