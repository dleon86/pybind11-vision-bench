import timeit
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sobel

def sobel_numpy(image_array):
    """Sobel filter implementation using NumPy for comparison."""
    if image_array.ndim == 3:
        # Convert to grayscale if RGB
        image_array = image_array.mean(axis=2) 

    # Sobel kernels
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Apply kernels
    # Using a simplified approach for demonstration, might need padding for edges
    Gx = np.zeros_like(image_array, dtype=np.float32)
    Gy = np.zeros_like(image_array, dtype=np.float32)

    for i in range(1, image_array.shape[0] - 1):
        for j in range(1, image_array.shape[1] - 1):
            patch = image_array[i-1:i+2, j-1:j+2]
            Gx[i, j] = np.sum(patch * Kx)
            Gy[i, j] = np.sum(patch * Ky)
    
    magnitude = np.sqrt(Gx * Gx + Gy * Gy)
    magnitude = (magnitude / np.max(magnitude) * 255).astype(np.uint8) # Normalize
    return magnitude


def main():
    image_path = "lena.png" 
    try:
        img_pil = Image.open(image_path).convert('L')
        original_image_np = np.array(img_pil, dtype=np.uint8)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")

    # --- Run C++ Sobel filter ---
    start_time_cpp = timeit.default_timer()
    sobel_cpp_output = sobel.sobel(original_image_np) 
    end_time_cpp = timeit.default_timer()
    print(f"C++ Sobel execution time: {end_time_cpp - start_time_cpp:.6f} seconds")

    # --- Run NumPy Sobel filter ---
    start_time_np = timeit.default_timer()
    sobel_np_output = sobel_numpy(original_image_np.astype(np.float32)) 
    end_time_np = timeit.default_timer()
    print(f"NumPy Sobel execution time: {end_time_np - start_time_np:.6f} seconds")

    # --- Display results using Matplotlib ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 8))
    
    axes[0].imshow(original_image_np, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(sobel_cpp_output, cmap='gray')
    axes[1].set_title(f'C++ Sobel Output\nExecution Time: {end_time_cpp - start_time_cpp:.6f} seconds')
    axes[1].axis('off')

    axes[2].imshow(sobel_np_output, cmap='gray')
    axes[2].set_title(f'NumPy Sobel Output\nExecution Time: {end_time_np - start_time_np:.6f} seconds')
    axes[2].axis('off')

    plt.tight_layout()
    # Trying to show, but not working in Docker
    try:
        plt.show()
    except Exception as e:
        print(f"Matplotlib plt.show() failed: {e}.")
    
    output_filename = "sobel_comparison.png"
    plt.savefig(output_filename)
    print(f"Comparison saved to {output_filename}")

if __name__ == "__main__":
    main()