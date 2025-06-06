import timeit
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import statistics
from typing import Dict, List, Tuple
import sys
import os
import json
import datetime
import argparse
import pickle
from collections import defaultdict

# Try to import seaborn, but make it optional
try:
    import seaborn as sns
    HAS_SEABORN = True
    # Set style for better plots
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not available, using basic matplotlib styling")
    plt.style.use('default')

# Ensure we can import the sobel module from the parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sobel

def get_build_config(config_override=None):
    """Detect current build configuration from available methods."""
    if config_override:
        # Parse from command line: "tag:description"
        if ':' in config_override:
            config_tag, config_name = config_override.split(':', 1)
        else:
            config_tag = config_override
            config_name = config_override.replace('_', ' ').title()
        return config_tag, config_name
    
    available_methods = [attr for attr in dir(sobel) if 'sobel' in attr and not attr.startswith('_')]
    
    # Try to determine build config by testing compilation flags
    config_tag = "unknown"
    config_name = "Unknown Configuration"
    
    # This is a simple heuristic - in practice you might want to pass this as an argument
    # or detect it more reliably through cmake variables
    if len(available_methods) >= 5:  # All methods available
        config_tag = "all_opts"
        config_name = "All Optimizations Enabled"
    else:
        config_tag = "basic"
        config_name = "Basic Configuration"
    
    return config_tag, config_name

def save_global_results(benchmark_results: Dict[str, Dict], config_tag: str, config_name: str, image_shape: Tuple[int, int]):
    """Save results to a global comparison file."""
    results_file = "global_benchmark_results.json"
    
    # Load existing results
    global_results = {}
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                global_results = json.load(f)
        except:
            global_results = {}
    
    # Add current results
    timestamp = datetime.datetime.now().isoformat()
    
    global_results[config_tag] = {
        'timestamp': timestamp,
        'config_name': config_name,
        'image_shape': image_shape,
        'results': {}
    }
    
    for method, stats in benchmark_results.items():
        global_results[config_tag]['results'][method] = {
            'mean': stats['mean'],
            'std': stats['std'],
            'min': stats['min'],
            'max': stats['max']
        }
    
    # Save updated results
    with open(results_file, 'w') as f:
        json.dump(global_results, f, indent=2)
    
    print(f"Global results saved to '{results_file}'")
    return global_results

def create_global_comparison_table(global_results: Dict):
    """Create a comprehensive comparison table across all build configurations."""
    if len(global_results) <= 1:
        print("Not enough configurations for global comparison")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Collect all unique methods across configurations
    all_methods = set()
    for config_data in global_results.values():
        all_methods.update(config_data['results'].keys())
    all_methods = sorted(list(all_methods))
    
    # Create table data
    headers = ['Method']
    configs = list(global_results.keys())
    
    for config in configs:
        config_name = global_results[config]['config_name']
        headers.append(f'{config_name}\n(Mean ± Std)')
        headers.append(f'{config}\nSpeedup vs NumPy')
    
    table_data = []
    
    for method in all_methods:
        row = [method]
        
        for config in configs:
            if method in global_results[config]['results']:
                stats = global_results[config]['results'][method]
                
                # Time column
                row.append(f"{stats['mean']:.4f}±{stats['std']:.4f}")
                
                # Speedup column
                if 'NumPy' in global_results[config]['results']:
                    numpy_time = global_results[config]['results']['NumPy']['mean']
                    speedup = numpy_time / stats['mean']
                    row.append(f"{speedup:.1f}x")
                else:
                    row.append("N/A")
            else:
                row.extend(["N/A", "N/A"])
        
        table_data.append(row)
    
    # Create table
    table = ax.table(cellText=table_data,
                     colLabels=headers,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Style the table
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header row
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white')
        elif j == 0:  # Method names
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#E6E6FA')
        else:
            cell.set_facecolor('#F2F2F2' if i % 2 == 0 else 'white')
    
    ax.set_title('Global Performance Comparison Across All Build Configurations', 
                fontweight='bold', fontsize=16, pad=20)
    
    plt.savefig('figures/global_performance_comparison.png', dpi=300, bbox_inches='tight')
    print("Global comparison table saved as 'global_performance_comparison.png'")
    plt.close(fig)

def create_optimization_illustration():
    """Create a figure illustrating different optimization techniques."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Sobel Filter Optimization Techniques', fontsize=16, fontweight='bold')
    
    # Basic implementation
    ax = axes[0, 0]
    ax.set_title('Basic Implementation', fontweight='bold')
    # Create a simple grid to show basic approach
    for i in range(5):
        for j in range(5):
            rect = patches.Rectangle((j, i), 1, 1, linewidth=1, edgecolor='black', facecolor='lightblue')
            ax.add_patch(rect)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_aspect('equal')
    ax.text(2.5, -0.5, 'Sequential pixel processing', ha='center', fontsize=10)
    ax.axis('off')
    
    # Loop unrolling
    ax = axes[0, 1]
    ax.set_title('Loop Unrolling', fontweight='bold')
    colors = ['red', 'green', 'blue'] * 3
    for i in range(3):
        for j in range(3):
            rect = patches.Rectangle((j, i), 1, 1, linewidth=2, edgecolor='black', facecolor=colors[i*3+j])
            ax.add_patch(rect)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_aspect('equal')
    ax.text(1.5, -0.5, 'Manually unrolled kernel', ha='center', fontsize=10)
    ax.axis('off')
    
    # SIMD
    ax = axes[0, 2]
    ax.set_title('SIMD Vectorization', fontweight='bold')
    # Show 4 pixels processed simultaneously
    for i in range(4):
        rect = patches.Rectangle((i, 0), 1, 1, linewidth=2, edgecolor='black', facecolor='orange')
        ax.add_patch(rect)
        ax.annotate('', xy=(i+0.5, 1.5), xytext=(i+0.5, 0.5), 
                   arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 2)
    ax.text(2, -0.3, '4 pixels processed simultaneously', ha='center', fontsize=10)
    ax.axis('off')
    
    # Cache blocking
    ax = axes[1, 0]
    ax.set_title('Cache Blocking/Tiling', fontweight='bold')
    # Show blocked processing
    colors = ['lightcoral', 'lightgreen', 'lightblue', 'lightyellow']
    block_size = 2
    for bi, color in enumerate(colors):
        start_i, start_j = (bi // 2) * block_size, (bi % 2) * block_size
        for i in range(block_size):
            for j in range(block_size):
                rect = patches.Rectangle((start_j + j, start_i + i), 1, 1, 
                                       linewidth=2, edgecolor='black', facecolor=color)
                ax.add_patch(rect)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_aspect('equal')
    ax.text(2, -0.5, 'Cache-friendly block processing', ha='center', fontsize=10)
    ax.axis('off')
    
    # Memory access pattern - FIXED ARROW PLACEMENT
    ax = axes[1, 1]
    ax.set_title('Memory Access Optimization', fontweight='bold')
    # Show prefetching pattern
    for i in range(3):
        for j in range(5):
            color = 'yellow' if j < 3 else 'lightgray'
            rect = patches.Rectangle((j, i), 1, 1, linewidth=1, edgecolor='black', facecolor=color)
            ax.add_patch(rect)
    
    # Fixed arrow placement - move it higher and adjust text position
    ax.arrow(3.2, 2.2, 0.6, 0, head_width=0.15, head_length=0.15, fc='red', ec='red')
    ax.text(4.2, 2.2, 'Prefetch', ha='left', va='center', fontsize=10, color='red', weight='bold')
    
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 3)
    ax.text(2.5, -0.5, 'Prefetching next cache lines', ha='center', fontsize=10)
    ax.axis('off')
    
    # Combined optimizations
    ax = axes[1, 2]
    ax.set_title('Combined Optimizations', fontweight='bold')
    ax.text(0.5, 0.8, '• Loop Unrolling', fontsize=12, transform=ax.transAxes)
    ax.text(0.5, 0.7, '• Cache Blocking', fontsize=12, transform=ax.transAxes)
    ax.text(0.5, 0.6, '• Memory Prefetching', fontsize=12, transform=ax.transAxes)
    ax.text(0.5, 0.5, '• Optimized Normalization', fontsize=12, transform=ax.transAxes)
    ax.text(0.5, 0.3, 'Best Performance', fontsize=14, fontweight='bold', 
           transform=ax.transAxes, ha='center', color='green')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('figures/optimization_techniques.png', dpi=300, bbox_inches='tight')
    print("Optimization techniques diagram saved as 'optimization_techniques.png'")
    plt.close(fig)
    return True

def create_individual_figures(benchmark_results: Dict[str, Dict], image_shape: Tuple[int, int], original_image, config_tag: str, config_name: str):
    """Create separate figures for better readability with unique filenames."""
    methods = list(benchmark_results.keys())
    means = [benchmark_results[method]['mean'] for method in methods]
    stds = [benchmark_results[method]['std'] for method in methods]
    
    # Get colors - use seaborn if available, otherwise matplotlib defaults
    if HAS_SEABORN:
        colors = sns.color_palette("husl", len(methods))
    else:
        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    
    # Figure 1: Performance comparison
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Performance bar chart
    bars = ax1.bar(methods, means, yerr=stds, capsize=5, alpha=0.8, color=colors)
    ax1.set_title('Average Execution Time Comparison (Log Scale)', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Time (seconds) - Log Scale', fontsize=12)
    ax1.set_yscale('log')  # Use logarithmic scale
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars (positioned better for log scale)
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        # Position label above the bar, accounting for log scale
        label_y = height * 1.5  # Multiplicative offset for log scale
        ax1.text(bar.get_x() + bar.get_width()/2., label_y,
                f'{mean:.4f}±{std:.4f}', ha='center', va='bottom', fontsize=8, rotation=0)
    
    # Add grid for better readability with log scale
    ax1.grid(True, alpha=0.3, which='major')
    ax1.grid(True, alpha=0.1, which='minor')
    
    # Speedup comparison (linear scale is fine here)
    numpy_time = benchmark_results['NumPy']['mean']
    speedups = [numpy_time / benchmark_results[method]['mean'] for method in methods]
    bars2 = ax2.bar(methods, speedups, alpha=0.8, color=colors)
    ax2.set_title('Speedup vs NumPy Implementation', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Speedup Factor', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='NumPy baseline')
    ax2.legend()
    
    # Add speedup labels (rounded to 1 decimal for clarity)
    for bar, speedup in zip(bars2, speedups):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(speedups)*0.02,
                f'{speedup:.1f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle(f'Sobel Filter Performance Analysis - {config_name}\nImage Size: {image_shape[0]}x{image_shape[1]} pixels | Intel i9-10885H @ 2.4GHz', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'figures/performance_comparison_{config_tag}.png', dpi=300, bbox_inches='tight')
    print(f"Performance comparison saved as 'performance_comparison_{config_tag}.png'")
    plt.close(fig1)
    
    # Figure 2: Statistical distribution
    fig2, ax = plt.subplots(1, 1, figsize=(12, 8))
    times_data = [benchmark_results[method]['times'] for method in methods]
    bp = ax.boxplot(times_data, tick_labels=methods, patch_artist=True)  # Fixed deprecated parameter
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_title(f'Execution Time Distribution Analysis - {config_name}', fontweight='bold', fontsize=16)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_yscale('log') 
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(f'figures/time_distribution_{config_tag}.png', dpi=300, bbox_inches='tight')
    print(f"Time distribution saved as 'time_distribution_{config_tag}.png'")
    plt.close(fig2)
    
    # Figure 3: Results quality comparison - Full size results
    # Calculate grid size for results display
    n_methods = len(methods)
    cols = 3
    rows = (n_methods + cols) // cols  # +1 for original, then ceil division
    
    fig3, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    fig3.suptitle(f'Sobel Filter Results Quality Comparison - {config_name}', fontsize=16, fontweight='bold')
    
    # Prepare display image - resize if too large for good display
    display_img = original_image
    display_results = {}
    
    if original_image.shape[0] > 800 or original_image.shape[1] > 800:
        # Resize for display
        from PIL import Image as PILImage
        scale_factor = min(800 / original_image.shape[0], 800 / original_image.shape[1])
        new_height = int(original_image.shape[0] * scale_factor)
        new_width = int(original_image.shape[1] * scale_factor)
        
        img_pil = PILImage.fromarray(original_image)
        img_pil = img_pil.resize((new_width, new_height), PILImage.LANCZOS)
        display_img = np.array(img_pil, dtype=np.uint8)
        print(f"Resized results for display: {display_img.shape}")
        
        # Also resize all results for consistent display
        for method in methods:
            result = benchmark_results[method]['result']
            result_pil = PILImage.fromarray(result)
            result_pil = result_pil.resize((new_width, new_height), PILImage.LANCZOS)
            display_results[method] = np.array(result_pil, dtype=np.uint8)
    else:
        for method in methods:
            display_results[method] = benchmark_results[method]['result']
    
    # Show original image first
    axes[0, 0].imshow(display_img, cmap='gray')
    axes[0, 0].set_title('Original Image', fontweight='bold', fontsize=12)
    axes[0, 0].axis('off')
    
    # Show results for each method
    for i, method in enumerate(methods):
        row = (i + 1) // cols
        col = (i + 1) % cols
        
        if row >= rows:  # Skip if we run out of subplot space
            break
            
        result = display_results[method]
        print(f"Displaying {method}: shape {result.shape}, dtype {result.dtype}, range [{result.min()}, {result.max()}]")
        
        # Debug: Check if result looks reasonable
        if result.max() == result.min():
            print(f"WARNING: {method} result has no variation (all values = {result.max()})")
        
        axes[row, col].imshow(result, cmap='gray', vmin=0, vmax=255)
        axes[row, col].set_title(f'{method}\n{means[i]:.4f}s', fontweight='bold', fontsize=12)
        axes[row, col].axis('off')
    
    # Hide unused subplots
    total_plots = (n_methods + 1)  # +1 for original
    for i in range(total_plots, rows * cols):
        row = i // cols
        col = i % cols
        if row < rows:
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'figures/results_quality_comparison_{config_tag}.png', dpi=300, bbox_inches='tight')
    print(f"Results quality comparison saved as 'results_quality_comparison_{config_tag}.png'")
    plt.close(fig3)
    
    # Figure 4: Side-by-side original vs best result
    fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    ax1.imshow(display_img, cmap='gray')
    ax1.set_title('Original Image', fontweight='bold', fontsize=14)
    ax1.axis('off')
    
    # Use NumPy result as reference
    numpy_result = display_results.get('NumPy', list(display_results.values())[0])
    ax2.imshow(numpy_result, cmap='gray', vmin=0, vmax=255)
    ax2.set_title('Sobel Edge Detection (NumPy Reference)', fontweight='bold', fontsize=14)
    ax2.axis('off')
    
    plt.suptitle(f'Sobel Filter: Before and After - {config_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'figures/sobel_before_after_comparison_{config_tag}.png', dpi=300, bbox_inches='tight')
    print(f"Before/after comparison saved as 'sobel_before_after_comparison_{config_tag}.png'")
    plt.close(fig4)
    
    # Figure 5: Performance statistics table
    fig5, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    for method in methods:
        stats = benchmark_results[method]
        table_data.append([
            method,
            f"{stats['mean']:.4f}",
            f"{stats['std']:.4f}",
            f"{stats['min']:.4f}",
            f"{stats['max']:.4f}",
            f"{numpy_time/stats['mean']:.1f}x"
        ])
    
    table = ax.table(cellText=table_data,
                     colLabels=['Method', 'Mean (s)', 'Std (s)', 'Min (s)', 'Max (s)', 'Speedup'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Style the table
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header row
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white')
        else:
            cell.set_facecolor('#F2F2F2' if i % 2 == 0 else 'white')
    
    ax.set_title(f'Detailed Performance Statistics - {config_name}', fontweight='bold', fontsize=16, pad=20)
    plt.savefig(f'figures/performance_statistics_{config_tag}.png', dpi=300, bbox_inches='tight')
    print(f"Performance statistics saved as 'performance_statistics_{config_tag}.png'")
    plt.close(fig5)

def sobel_numpy(image_array):
    """Sobel filter implementation using NumPy for comparison."""
    if image_array.ndim == 3:
        image_array = image_array.mean(axis=2) 

    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    Gx = np.zeros_like(image_array, dtype=np.float32)
    Gy = np.zeros_like(image_array, dtype=np.float32)

    for i in range(1, image_array.shape[0] - 1):
        for j in range(1, image_array.shape[1] - 1):
            patch = image_array[i-1:i+2, j-1:j+2]
            Gx[i, j] = np.sum(patch * Kx)
            Gy[i, j] = np.sum(patch * Ky)
    
    magnitude = np.sqrt(Gx * Gx + Gy * Gy)
    magnitude = (magnitude / np.max(magnitude) * 255).astype(np.uint8)
    return magnitude

def benchmark_method(method_func, image, n_iterations: int = 10) -> Dict[str, float]:
    """Benchmark a single method with statistical analysis."""
    times = []
    result = None
    
    for i in range(n_iterations):
        start_time = timeit.default_timer()
        result = method_func(image)
        end_time = timeit.default_timer()
        times.append(end_time - start_time)
    
    return {
        'mean': statistics.mean(times),
        'median': statistics.median(times),
        'std': statistics.stdev(times) if len(times) > 1 else 0.0,
        'min': min(times),
        'max': max(times),
        'times': times,
        'result': result  # Use the last result
    }

def save_benchmark_data(benchmark_results: Dict[str, Dict], config_tag: str, config_name: str, image_shape: Tuple[int, int]):
    """Save detailed benchmark data to pickle for faster reloading."""
    pickle_file = f"benchmark_data_{config_tag}.pkl"
    
    data_to_save = {
        'config_tag': config_tag,
        'config_name': config_name,
        'image_shape': image_shape,
        'timestamp': datetime.datetime.now().isoformat(),
        'benchmark_results': benchmark_results
    }
    
    with open(pickle_file, 'wb') as f:
        pickle.dump(data_to_save, f)
    
    print(f"Benchmark data saved to '{pickle_file}'")
    return pickle_file

def load_all_benchmark_data():
    """Load all available benchmark data from pickle files."""
    all_data = {}
    
    for file in os.listdir('.'):
        if file.startswith('benchmark_data_') and file.endswith('.pkl'):
            try:
                with open(file, 'rb') as f:
                    data = pickle.load(f)
                    config_tag = data['config_tag']
                    all_data[config_tag] = data
                    print(f"Loaded data for {config_tag}: {data['config_name']}")
            except Exception as e:
                print(f"Failed to load {file}: {e}")
    
    return all_data

def create_grouped_bar_charts(all_data: Dict):
    """Create comprehensive grouped bar charts from all benchmark data."""
    if len(all_data) < 2:
        print("Need at least 2 configurations for grouped analysis")
        return
    
    # Collect all methods and configs
    all_methods = set()
    all_configs = []
    
    for config_tag, data in all_data.items():
        all_configs.append(config_tag)
        all_methods.update(data['benchmark_results'].keys())
    
    all_methods = sorted(list(all_methods))
    all_configs = sorted(all_configs)
    
    # Prepare data matrices
    execution_times = defaultdict(dict)
    speedups = defaultdict(dict)
    
    for config_tag in all_configs:
        data = all_data[config_tag]
        results = data['benchmark_results']
        numpy_time = results.get('NumPy', {}).get('mean', 1.0)
        
        for method in all_methods:
            if method in results:
                execution_times[method][config_tag] = results[method]['mean']
                speedups[method][config_tag] = numpy_time / results[method]['mean']
            else:
                execution_times[method][config_tag] = None
                speedups[method][config_tag] = None
    
    # Create grouped charts
    create_grouped_by_config(all_data, all_methods, all_configs, execution_times, speedups)
    create_grouped_by_method(all_data, all_methods, all_configs, execution_times, speedups)
    create_grouped_time_distributions(all_data, all_methods, all_configs)

def create_grouped_by_config(all_data, all_methods, all_configs, execution_times, speedups):
    """Create charts grouped by configuration (each config shows all methods)."""
    
    # Colors for methods
    if HAS_SEABORN:
        method_colors = dict(zip(all_methods, sns.color_palette("husl", len(all_methods))))
    else:
        method_colors = dict(zip(all_methods, plt.cm.tab10(np.linspace(0, 1, len(all_methods)))))
    
    # 1. Execution times grouped by config
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    x = np.arange(len(all_configs))
    width = 0.8 / len(all_methods)
    
    for i, method in enumerate(all_methods):
        times = [execution_times[method].get(config) for config in all_configs]
        # Filter out None values
        valid_configs = [j for j, t in enumerate(times) if t is not None]
        valid_times = [t for t in times if t is not None]
        valid_x = [x[j] + i * width for j in valid_configs]
        
        if valid_times:
            bars = ax.bar(valid_x, valid_times, width, label=method, 
                         color=method_colors[method], alpha=0.8)
            
            # Add value labels
            for bar, time in zip(bars, valid_times):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                       f'{time:.4f}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    ax.set_xlabel('Build Configuration', fontsize=12)
    ax.set_ylabel('Execution Time (seconds) - Log Scale', fontsize=12)
    ax.set_title('Execution Times by Build Configuration', fontweight='bold', fontsize=14)
    ax.set_yscale('log')
    ax.set_xticks(x + width * (len(all_methods) - 1) / 2)
    ax.set_xticklabels([all_data[config]['config_name'] for config in all_configs], rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/grouped_by_config_execution_times.png', dpi=300, bbox_inches='tight')
    print("Saved 'grouped_by_config_execution_times.png'")
    plt.close()
    
    # 2. Speedups grouped by config
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    for i, method in enumerate(all_methods):
        speeds = [speedups[method].get(config) for config in all_configs]
        valid_configs = [j for j, s in enumerate(speeds) if s is not None]
        valid_speeds = [s for s in speeds if s is not None]
        valid_x = [x[j] + i * width for j in valid_configs]
        
        if valid_speeds:
            bars = ax.bar(valid_x, valid_speeds, width, label=method, 
                         color=method_colors[method], alpha=0.8)
            
            # Add value labels
            for bar, speed in zip(bars, valid_speeds):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(valid_speeds)*0.02,
                       f'{speed:.1f}x', ha='center', va='bottom', fontsize=8, rotation=90)
    
    ax.set_xlabel('Build Configuration', fontsize=12)
    ax.set_ylabel('Speedup vs NumPy', fontsize=12)
    ax.set_title('Speedup by Build Configuration', fontweight='bold', fontsize=14)
    ax.set_xticks(x + width * (len(all_methods) - 1) / 2)
    ax.set_xticklabels([all_data[config]['config_name'] for config in all_configs], rotation=45, ha='right')
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='NumPy baseline')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/grouped_by_config_speedups.png', dpi=300, bbox_inches='tight')
    print("Saved 'grouped_by_config_speedups.png'")
    plt.close()

def create_grouped_by_method(all_data, all_methods, all_configs, execution_times, speedups):
    """Create charts grouped by method (each method shows all configs)."""
    
    # Colors for configurations
    if HAS_SEABORN:
        config_colors = dict(zip(all_configs, sns.color_palette("Set2", len(all_configs))))
    else:
        config_colors = dict(zip(all_configs, plt.cm.Set2(np.linspace(0, 1, len(all_configs)))))
    
    # 1. Execution times grouped by method
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    x = np.arange(len(all_methods))
    width = 0.8 / len(all_configs)
    
    for i, config in enumerate(all_configs):
        times = [execution_times[method].get(config) for method in all_methods]
        valid_methods = [j for j, t in enumerate(times) if t is not None]
        valid_times = [t for t in times if t is not None]
        valid_x = [x[j] + i * width for j in valid_methods]
        
        if valid_times:
            bars = ax.bar(valid_x, valid_times, width, 
                         label=all_data[config]['config_name'], 
                         color=config_colors[config], alpha=0.8)
            
            # Add value labels
            for bar, time in zip(bars, valid_times):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                       f'{time:.4f}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    ax.set_xlabel('Algorithm Implementation', fontsize=12)
    ax.set_ylabel('Execution Time (seconds) - Log Scale', fontsize=12)
    ax.set_title('Execution Times by Algorithm Implementation', fontweight='bold', fontsize=14)
    ax.set_yscale('log')
    ax.set_xticks(x + width * (len(all_configs) - 1) / 2)
    ax.set_xticklabels(all_methods, rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/grouped_by_method_execution_times.png', dpi=300, bbox_inches='tight')
    print("Saved 'grouped_by_method_execution_times.png'")
    plt.close()
    
    # 2. Speedups grouped by method
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    for i, config in enumerate(all_configs):
        speeds = [speedups[method].get(config) for method in all_methods]
        valid_methods = [j for j, s in enumerate(speeds) if s is not None]
        valid_speeds = [s for s in speeds if s is not None]
        valid_x = [x[j] + i * width for j in valid_methods]
        
        if valid_speeds:
            bars = ax.bar(valid_x, valid_speeds, width, 
                         label=all_data[config]['config_name'], 
                         color=config_colors[config], alpha=0.8)
            
            # Add value labels
            for bar, speed in zip(bars, valid_speeds):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(valid_speeds)*0.02,
                       f'{speed:.1f}x', ha='center', va='bottom', fontsize=8, rotation=90)
    
    ax.set_xlabel('Algorithm Implementation', fontsize=12)
    ax.set_ylabel('Speedup vs NumPy', fontsize=12)
    ax.set_title('Speedup by Algorithm Implementation', fontweight='bold', fontsize=14)
    ax.set_xticks(x + width * (len(all_configs) - 1) / 2)
    ax.set_xticklabels(all_methods, rotation=45, ha='right')
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='NumPy baseline')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/grouped_by_method_speedups.png', dpi=300, bbox_inches='tight')
    print("Saved 'grouped_by_method_speedups.png'")
    plt.close()

def create_grouped_time_distributions(all_data, all_methods, all_configs):
    """Create grouped box plots showing time distributions."""
    
    # Create one large subplot grid
    n_methods = len(all_methods)
    cols = 3
    rows = (n_methods + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Time Distribution Analysis by Method and Configuration', fontsize=16, fontweight='bold')
    
    for idx, method in enumerate(all_methods):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        # Collect data for this method across all configs
        box_data = []
        box_labels = []
        
        for config in all_configs:
            if config in all_data and method in all_data[config]['benchmark_results']:
                times = all_data[config]['benchmark_results'][method]['times']
                box_data.append(times)
                box_labels.append(all_data[config]['config_name'])
        
        if box_data:
            bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
            
            # Color the boxes
            if HAS_SEABORN:
                colors = sns.color_palette("Set2", len(box_data))
            else:
                colors = plt.cm.Set2(np.linspace(0, 1, len(box_data)))
            
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.8)
            
            ax.set_title(f'{method}', fontweight='bold')
            ax.set_ylabel('Time (seconds)', fontsize=10)
            ax.set_yscale('log')
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.set_title(f'{method} (No Data)', fontweight='bold')
            ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n_methods, rows * cols):
        row = idx // cols
        col = idx % cols
        if row < rows and col < cols:
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('figures/grouped_time_distributions.png', dpi=300, bbox_inches='tight')
    print("Saved 'grouped_time_distributions.png'")
    plt.close()

def create_focused_analysis_charts(all_data: Dict):
    """Create focused analysis charts for better interpretability."""
    if len(all_data) < 2:
        print("Need at least 2 configurations for focused analysis")
        return
    
    print("Creating focused analysis charts...")
    
    # Create all the focused analysis
    create_flag_effectiveness_analysis(all_data)
    create_method_effectiveness_analysis(all_data)
    create_improvement_percentage_tables(all_data)
    create_efficiency_ratio_analysis(all_data)
    create_interaction_effects_analysis(all_data)

def create_flag_effectiveness_analysis(all_data: Dict):
    """Analyze how each compiler flag affects each method."""
    
    # Extract data
    configs = ['basic', 'unroll_only', 'simd_only', 'all_opts']
    available_configs = [c for c in configs if c in all_data]
    
    if len(available_configs) < 2:
        print("Not enough configurations for flag effectiveness analysis")
        return
    
    methods = ['C++ Basic', 'C++ Unrolled', 'C++ SIMD', 'C++ Blocked', 'C++ Optimized']
    
    # Create improvement matrix: method vs config progression
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # 1. Execution time progression
    for method in methods:
        times = []
        config_names = []
        
        for config in available_configs:
            if config in all_data and method in all_data[config]['benchmark_results']:
                times.append(all_data[config]['benchmark_results'][method]['mean'])
                config_names.append(all_data[config]['config_name'])
        
        if len(times) >= 2:
            ax1.plot(range(len(times)), times, 'o-', label=method, linewidth=2, markersize=6)
            
            # Add value labels
            for i, time in enumerate(times):
                ax1.text(i, time * 1.05, f'{time:.4f}', ha='center', va='bottom', fontsize=8)
    
    ax1.set_xlabel('Compiler Configuration Progression', fontsize=12)
    ax1.set_ylabel('Execution Time (seconds) - Log Scale', fontsize=12)
    ax1.set_title('Flag Effectiveness: How Each Method Responds to Compiler Flags', fontweight='bold', fontsize=14)
    ax1.set_yscale('log')
    ax1.set_xticks(range(len(config_names)))
    ax1.set_xticklabels(config_names, rotation=45, ha='right')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Relative improvement from basic config
    if 'basic' in available_configs:
        for method in methods:
            improvements = []
            config_names_subset = []
            
            basic_time = None
            if 'basic' in all_data and method in all_data['basic']['benchmark_results']:
                basic_time = all_data['basic']['benchmark_results'][method]['mean']
            
            if basic_time:
                for config in available_configs:
                    if config in all_data and method in all_data[config]['benchmark_results']:
                        current_time = all_data[config]['benchmark_results'][method]['mean']
                        improvement = ((basic_time - current_time) / basic_time) * 100
                        improvements.append(improvement)
                        config_names_subset.append(all_data[config]['config_name'])
                
                if improvements:
                    ax2.plot(range(len(improvements)), improvements, 'o-', label=method, linewidth=2, markersize=6)
                    
                    # Add value labels
                    for i, imp in enumerate(improvements):
                        ax2.text(i, imp + max(0, max(improvements)*0.02), f'{imp:.1f}%', 
                                ha='center', va='bottom', fontsize=8)
        
        ax2.set_xlabel('Compiler Configuration Progression', fontsize=12)
        ax2.set_ylabel('Performance Improvement vs Basic Config (%)', fontsize=12)
        ax2.set_title('Relative Performance Gains from Compiler Flags', fontweight='bold', fontsize=14)
        ax2.set_xticks(range(len(config_names)))
        ax2.set_xticklabels(config_names, rotation=45, ha='right')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No improvement')
    
    plt.tight_layout()
    plt.savefig('figures/flag_effectiveness_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved 'flag_effectiveness_analysis.png'")
    plt.close()

def create_method_effectiveness_analysis(all_data: Dict):
    """Analyze manual optimization effectiveness vs compiler flags."""
    
    # For each config, show the progression from basic to optimized methods
    configs = ['basic', 'unroll_only', 'simd_only', 'all_opts']
    available_configs = [c for c in configs if c in all_data]
    
    if not available_configs:
        return
    
    methods = ['C++ Basic', 'C++ Unrolled', 'C++ SIMD', 'C++ Blocked', 'C++ Optimized']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Create one subplot for each configuration
    for idx, config in enumerate(available_configs[:4]):  # Max 4 configs
        ax = axes[idx]
        
        times = []
        method_names = []
        
        for method in methods:
            if method in all_data[config]['benchmark_results']:
                times.append(all_data[config]['benchmark_results'][method]['mean'])
                method_names.append(method.replace('C++ ', ''))
        
        if times:
            bars = ax.bar(range(len(times)), times, alpha=0.8, 
                         color=plt.cm.viridis(np.linspace(0, 1, len(times))))
            
            # Add value labels
            for bar, time in zip(bars, times):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height * 1.05,
                       f'{time:.4f}', ha='center', va='bottom', fontsize=9, rotation=90)
            
            ax.set_title(f'{all_data[config]["config_name"]}', fontweight='bold', fontsize=12)
            ax.set_ylabel('Execution Time (seconds)', fontsize=10)
            ax.set_yscale('log')
            ax.set_xticks(range(len(method_names)))
            ax.set_xticklabels(method_names, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(available_configs), 4):
        axes[idx].axis('off')
    
    plt.suptitle('Method Effectiveness: Manual Optimizations vs Compiler Flags', 
                fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.savefig('figures/method_effectiveness_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved 'method_effectiveness_analysis.png'")
    plt.close()

def create_improvement_percentage_tables(all_data: Dict):
    """Create detailed improvement percentage tables."""
    
    configs = ['basic', 'unroll_only', 'simd_only', 'all_opts']
    available_configs = [c for c in configs if c in all_data]
    methods = ['C++ Basic', 'C++ Unrolled', 'C++ SIMD', 'C++ Blocked', 'C++ Optimized']
    
    if len(available_configs) < 2:
        return
    
    # Create improvement matrix
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # 1. Absolute improvement vs NumPy baseline
    improvement_data = []
    row_labels = []
    col_labels = [all_data[config]['config_name'] for config in available_configs]
    
    for method in methods:
        row = []
        for config in available_configs:
            if (config in all_data and method in all_data[config]['benchmark_results'] and 
                'NumPy' in all_data[config]['benchmark_results']):
                
                numpy_time = all_data[config]['benchmark_results']['NumPy']['mean']
                method_time = all_data[config]['benchmark_results'][method]['mean']
                speedup = numpy_time / method_time
                row.append(f"{speedup:.0f}x")
            else:
                row.append("N/A")
        improvement_data.append(row)
        row_labels.append(method.replace('C++ ', ''))
    
    # Create speedup table
    table1 = ax1.table(cellText=improvement_data,
                      rowLabels=row_labels,
                      colLabels=col_labels,
                      cellLoc='center',
                      loc='center')
    table1.auto_set_font_size(False)
    table1.set_fontsize(11)
    table1.scale(1.2, 2)
    
    # Style the table
    for (i, j), cell in table1.get_celld().items():
        if i == 0:  # Header row
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white')
        elif j == -1:  # Row labels
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#E6E6FA')
        else:
            cell.set_facecolor('#F2F2F2' if i % 2 == 0 else 'white')
    
    ax1.set_title('Speedup vs NumPy by Method and Configuration', fontweight='bold', fontsize=14, pad=20)
    ax1.axis('off')
    
    # 2. Relative improvement between configurations (if basic config exists)
    if 'basic' in available_configs:
        relative_data = []
        
        for method in methods:
            row = []
            basic_time = None
            if method in all_data['basic']['benchmark_results']:
                basic_time = all_data['basic']['benchmark_results'][method]['mean']
            
            for config in available_configs:
                if basic_time and config in all_data and method in all_data[config]['benchmark_results']:
                    current_time = all_data[config]['benchmark_results'][method]['mean']
                    improvement = ((basic_time - current_time) / basic_time) * 100
                    row.append(f"{improvement:+.1f}%")
                else:
                    row.append("N/A")
            relative_data.append(row)
        
        # Create relative improvement table
        table2 = ax2.table(cellText=relative_data,
                          rowLabels=row_labels,
                          colLabels=col_labels,
                          cellLoc='center',
                          loc='center')
        table2.auto_set_font_size(False)
        table2.set_fontsize(11)
        table2.scale(1.2, 2)
        
        # Style the table with color coding for improvements
        for (i, j), cell in table2.get_celld().items():
            if i == 0:  # Header row
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#4472C4')
                cell.set_text_props(color='white')
            elif j == -1:  # Row labels
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#E6E6FA')
            else:
                # Color code improvements
                text = cell.get_text().get_text()
                if text != "N/A" and "%" in text:
                    try:
                        value = float(text.replace('%', '').replace('+', ''))
                        if value > 5:
                            cell.set_facecolor('#90EE90')  # Light green for good improvements
                        elif value > 0:
                            cell.set_facecolor('#FFFFE0')  # Light yellow for small improvements
                        elif value < -5:
                            cell.set_facecolor('#FFB6C1')  # Light red for regressions
                        else:
                            cell.set_facecolor('white')
                    except:
                        cell.set_facecolor('white')
                else:
                    cell.set_facecolor('#F2F2F2' if i % 2 == 0 else 'white')
        
        ax2.set_title('Performance Change vs Basic Configuration (%)', fontweight='bold', fontsize=14, pad=20)
        ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('figures/improvement_percentage_tables.png', dpi=300, bbox_inches='tight')
    print("Saved 'improvement_percentage_tables.png'")
    plt.close()

def create_efficiency_ratio_analysis(all_data: Dict):
    """Analyze the efficiency ratio between manual and compiler optimizations."""
    
    configs = ['basic', 'all_opts']  # Compare basic vs full optimization
    if not all(c in all_data for c in configs):
        print("Need both basic and all_opts configurations for efficiency analysis")
        return
    
    methods = ['C++ Basic', 'C++ Unrolled', 'C++ SIMD', 'C++ Blocked', 'C++ Optimized']
    
    # Calculate efficiency metrics
    manual_benefits = []  # Improvement from basic method to optimized methods
    compiler_benefits = []  # Improvement from basic flags to full flags
    method_names = []
    
    for method in methods:
        if (method in all_data['basic']['benchmark_results'] and 
            method in all_data['all_opts']['benchmark_results']):
            
            basic_config_time = all_data['basic']['benchmark_results'][method]['mean']
            full_config_time = all_data['all_opts']['benchmark_results'][method]['mean']
            
            # Get C++ Basic time in both configurations for comparison
            cpp_basic_time_basic = all_data['basic']['benchmark_results']['C++ Basic']['mean']
            cpp_basic_time_full = all_data['all_opts']['benchmark_results']['C++ Basic']['mean']
            
            # Manual optimization benefit (vs C++ Basic in same config)
            manual_benefit_basic = (cpp_basic_time_basic - basic_config_time) / cpp_basic_time_basic * 100
            manual_benefit_full = (cpp_basic_time_full - full_config_time) / cpp_basic_time_full * 100
            
            # Compiler benefit (same method, different configs)
            compiler_benefit = (basic_config_time - full_config_time) / basic_config_time * 100
            
            manual_benefits.append(max(manual_benefit_basic, manual_benefit_full))
            compiler_benefits.append(compiler_benefit)
            method_names.append(method.replace('C++ ', ''))
    
    # Create the efficiency analysis chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Manual vs Compiler optimization contributions
    x = np.arange(len(method_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, manual_benefits, width, label='Manual Optimizations', 
                    color='skyblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, compiler_benefits, width, label='Compiler Flags', 
                    color='lightcoral', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(manual_benefits + compiler_benefits)*0.01,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('Optimization Method', fontsize=12)
    ax1.set_ylabel('Performance Improvement (%)', fontsize=12)
    ax1.set_title('Manual vs Compiler Optimization Contributions', fontweight='bold', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(method_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Efficiency ratio (Manual improvement / Compiler improvement)
    efficiency_ratios = []
    for i in range(len(manual_benefits)):
        if compiler_benefits[i] > 0:
            ratio = manual_benefits[i] / compiler_benefits[i]
            efficiency_ratios.append(ratio)
        else:
            efficiency_ratios.append(0)
    
    bars = ax2.bar(method_names, efficiency_ratios, color='gold', alpha=0.8)
    
    # Add value labels
    for bar, ratio in zip(bars, efficiency_ratios):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(efficiency_ratios)*0.02,
                f'{ratio:.1f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('Optimization Method', fontsize=12)
    ax2.set_ylabel('Efficiency Ratio (Manual/Compiler)', fontsize=12)
    ax2.set_title('Manual Optimization Efficiency vs Compiler Flags', fontweight='bold', fontsize=14)
    ax2.set_xticklabels(method_names, rotation=45, ha='right')
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Equal effectiveness')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/efficiency_ratio_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved 'efficiency_ratio_analysis.png'")
    plt.close()

def create_interaction_effects_analysis(all_data: Dict):
    """Analyze interaction effects between manual optimizations and compiler flags."""
    
    # Focus on SIMD as it's the most impactful
    configs_to_compare = [
        ('basic', 'Basic Config'),
        ('simd_only', 'SIMD Only'),
    ]
    
    available_comparisons = [(config, name) for config, name in configs_to_compare if config in all_data]
    
    if len(available_comparisons) < 2:
        print("Not enough configurations for interaction analysis")
        return
    
    methods = ['C++ Basic', 'C++ SIMD', 'C++ Optimized']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Direct comparison of SIMD methods with/without SIMD flags
    for method in methods:
        times = []
        config_names = []
        
        for config, name in available_comparisons:
            if method in all_data[config]['benchmark_results']:
                times.append(all_data[config]['benchmark_results'][method]['mean'])
                config_names.append(name)
        
        if len(times) == 2:
            ax1.plot([0, 1], times, 'o-', label=method, linewidth=3, markersize=8)
            
            # Add value labels
            for i, time in enumerate(times):
                ax1.text(i, time * 1.05, f'{time:.4f}s', ha='center', va='bottom', fontsize=10)
    
    ax1.set_xlabel('Configuration', fontsize=12)
    ax1.set_ylabel('Execution Time (seconds) - Log Scale', fontsize=12)
    ax1.set_title('Interaction Effects: Manual SIMD + SIMD Flags', fontweight='bold', fontsize=14)
    ax1.set_yscale('log')
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(config_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Synergy analysis - does manual + compiler > sum of parts?
    if len(available_comparisons) >= 2:
        basic_config, _ = available_comparisons[0]
        simd_config, _ = available_comparisons[1]
        
        synergy_data = []
        method_labels = []
        
        for method in methods:
            if (method in all_data[basic_config]['benchmark_results'] and 
                method in all_data[simd_config]['benchmark_results']):
                
                basic_time = all_data[basic_config]['benchmark_results'][method]['mean']
                simd_time = all_data[simd_config]['benchmark_results'][method]['mean']
                
                # Calculate actual improvement
                actual_improvement = (basic_time - simd_time) / basic_time * 100
                
                synergy_data.append(actual_improvement)
                method_labels.append(method.replace('C++ ', ''))
        
        bars = ax2.bar(method_labels, synergy_data, color=['lightblue', 'orange', 'green'], alpha=0.8)
        
        # Add value labels
        for bar, improvement in zip(bars, synergy_data):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(synergy_data)*0.02,
                    f'{improvement:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax2.set_xlabel('Method', fontsize=12)
        ax2.set_ylabel('Performance Improvement with SIMD Flags (%)', fontsize=12)
        ax2.set_title('SIMD Flag Effectiveness by Method', fontweight='bold', fontsize=14)
        ax2.set_xticklabels(method_labels, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/interaction_effects_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved 'interaction_effects_analysis.png'")
    plt.close()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Benchmark Sobel filter implementations')
    parser.add_argument('--config', '-c', type=str, 
                       help='Build configuration tag (e.g., "basic", "unroll_only:Loop Unrolling Only", "simd_only", "all_opts")')
    parser.add_argument('--iterations', '-i', type=int, default=15,
                       help='Number of benchmark iterations (default: 15)')
    parser.add_argument('--load-only', action='store_true',
                       help='Only load existing data and create grouped charts (skip benchmarking)')
    parser.add_argument('--skip-save', action='store_true',
                       help='Skip saving benchmark data to pickle')
    parser.add_argument('--focused-analysis', action='store_true',
                       help='Create focused interpretability analysis from existing data')
    
    args = parser.parse_args()
    
    # If focused analysis mode, create interpretability charts
    if args.focused_analysis:
        print("Creating focused interpretability analysis...")
        all_data = load_all_benchmark_data()
        if all_data:
            create_focused_analysis_charts(all_data)
            print("Focused analysis complete!")
        else:
            print("No existing benchmark data found!")
        return
    
    # If load-only mode, just create grouped charts from existing data
    if args.load_only:
        print("Loading existing benchmark data for grouped analysis...")
        all_data = load_all_benchmark_data()
        if all_data:
            create_grouped_bar_charts(all_data)
            print("Grouped analysis complete!")
        else:
            print("No existing benchmark data found!")
        return
    
    # Get build configuration
    config_tag, config_name = get_build_config(args.config)
    print(f"Build configuration: {config_name} ({config_tag})")
    
    # Image loading
    image_path = "pair_of_boobies.jpg"
    try:
        img_pil = Image.open(image_path).convert('L')
        original_image_np = np.array(img_pil, dtype=np.uint8)
        print(f"Image loaded: {original_image_np.shape} pixels")
        print(f"Image data type: {original_image_np.dtype}, range: [{original_image_np.min()}, {original_image_np.max()}]")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        print("Using synthetic test image...")
        original_image_np = np.random.randint(0, 256, (500, 500), dtype=np.uint8)
    
    # Create optimization techniques illustration (only once)
    if not os.path.exists('figures/optimization_techniques.png'):
        create_optimization_illustration()
    
    # Define optimization methods to benchmark
    methods = {
        'NumPy': lambda img: sobel_numpy(img.astype(np.float32)),
        'C++ Basic': lambda img: sobel.sobel_basic(img.astype(np.int32)),
        'C++ Unrolled': lambda img: sobel.sobel_unrolled(img.astype(np.int32)),
        'C++ SIMD': lambda img: sobel.sobel_simd(img.astype(np.int32)),
        'C++ Blocked': lambda img: sobel.sobel_blocked(img.astype(np.int32)),
        'C++ Optimized': lambda img: sobel.sobel_optimized(img.astype(np.int32))
    }
    
    n_iterations = args.iterations
    print(f"\nBenchmarking {len(methods)} methods with {n_iterations} iterations each...")
    print(f"System: Intel i9-10885H @ 2.4GHz, 32GB RAM, Windows 64-bit")
    print(f"SIMD: Using SSE (128-bit vectors), CPU supports AVX2, 8 cores/16 threads")
    
    # Benchmark all methods
    benchmark_results = {}
    for method_name, method_func in methods.items():
        print(f"Benchmarking {method_name}...")
        try:
            results = benchmark_method(method_func, original_image_np, n_iterations)
            benchmark_results[method_name] = results
            print(f"  Mean: {results['mean']:.4f}s ± {results['std']:.4f}s")
            
            # Quick result check
            result = results['result']
            print(f"  Result: shape={result.shape}, dtype={result.dtype}, range=[{result.min():.1f}, {result.max():.1f}]")
            
        except Exception as e:
            print(f"  Failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not benchmark_results:
        print("No successful benchmarks!")
        return
    
    # Save detailed benchmark data to pickle
    if not args.skip_save:
        save_benchmark_data(benchmark_results, config_tag, config_name, original_image_np.shape)
    
    # Save to global results and create individual figures
    global_results = save_global_results(benchmark_results, config_tag, config_name, original_image_np.shape)
    create_individual_figures(benchmark_results, original_image_np.shape, original_image_np, config_tag, config_name)
    
    # Create global comparison if we have multiple configurations
    create_global_comparison_table(global_results)
    
    # Load all available data and create grouped charts
    print("\nCreating comprehensive grouped analysis...")
    all_data = load_all_benchmark_data()
    if len(all_data) >= 2:
        create_grouped_bar_charts(all_data)
    
    # Print detailed summary
    print("\n" + "="*80)
    print(f"PERFORMANCE SUMMARY - {config_name}")
    print("="*80)
    
    # Sort by performance
    sorted_methods = sorted(benchmark_results.keys(), 
                          key=lambda x: benchmark_results[x]['mean'])
    
    numpy_time = benchmark_results.get('NumPy', {}).get('mean', 1.0)
    
    for i, method in enumerate(sorted_methods):
        stats = benchmark_results[method]
        speedup = numpy_time / stats['mean']
        print(f"{i+1:2d}. {method:<15} | "
              f"Time: {stats['mean']:.4f}±{stats['std']:.4f}s | "
              f"Speedup: {speedup:5.1f}x | "
              f"Range: {stats['min']:.4f}-{stats['max']:.4f}s")
    
    print("\nOptimization effectiveness:")
    if 'C++ Basic' in benchmark_results and 'C++ Optimized' in benchmark_results:
        basic_time = benchmark_results['C++ Basic']['mean']
        optimized_time = benchmark_results['C++ Optimized']['mean']
        improvement = basic_time / optimized_time
        print(f"C++ Optimized vs C++ Basic: {improvement:.1f}x faster")

if __name__ == "__main__":
    main()