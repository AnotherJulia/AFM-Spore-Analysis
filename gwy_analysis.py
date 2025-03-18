import gwyfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, laplace
from skimage.exposure import equalize_hist


# Load the Gwyddion AFM file
def load_gwy_image(file_path):
    gwy_container = gwyfile.load(file_path)
    
    # Extract the first dataset (Assuming it's a single-layer AFM image)
    for key, value in gwy_container.items():
        if key.startswith('/0/data'):  # The key pattern for Gwyddion height data
            afm_data = value['data']
            xres, yres = value['xres'], value['yres']
            return np.array(afm_data).reshape((yres, xres))  # Reshape to 2D array

    raise ValueError("No valid AFM image found in the Gwyddion file.")

# Function to clean the image
def process_afm_image(image):
    # Apply Gaussian filtering for smoothing
    cleaned_image = gaussian_filter(image, sigma=1)
    
    # Normalize the image (scaling between 0 and 1)
    cleaned_image -= np.min(cleaned_image)
    cleaned_image /= np.max(cleaned_image)
    
    return cleaned_image

def smooth_image(image, sigma=1):
    """Apply Gaussian blur to reduce noise."""
    return gaussian_filter(image, sigma=sigma)

def enhance_edges(image):
    """Enhance edges using Laplacian filtering."""
    return laplace(image)

def percentile_normalization(image, lower=2, upper=98):
    """Normalize image intensity using percentiles."""
    p_low, p_high = np.percentile(image, (lower, upper))
    normalized = np.clip((image - p_low) / (p_high - p_low), 0, 1)
    return normalized

# Function to display the image with a scientific colormap
def plot_afm_image(image, cmap='viridis'):
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap=cmap, origin='lower')
    plt.colorbar(label="Height (Normalized)")
    plt.title("AFM Image of Spores")
    plt.axis("off")
    plt.show()

# Load the AFM file (Replace with your actual file path)
file_path = "30small.gwy"

def enhance_contrast(image):
    """Enhance contrast using histogram equalization."""
    return equalize_hist(image)

# Process and visualize
afm_image = load_gwy_image(file_path)
afm_image = process_afm_image(afm_image)
# afm_image = percentile_normalization(afm_image)
# afm_image = smooth_image(afm_image)

plot_afm_image(afm_image)
