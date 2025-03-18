# Imports
import gwyfile as gwy
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import scipy.ndimage as ndimage
from scipy.ndimage import gaussian_filter, median_filter, label, find_objects, maximum_position   
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import zscore
from scipy.optimize import leastsq
import cv2
import numpy.fft as fft
import seaborn as sns


# Variables
filenames = ["raw/30small.gwy", "raw/30big.gwy", "raw/80small.gwy", "raw/80big.gwy"]
prominence_zoomed = 0.3
prominence_normal = 0.05

def load_files():

    # Load the Gwyddion AFM files
    afm_data = {}

    for file_path in filenames:
        try:
            gwy_data = gwy.load(file_path)

            # extract the first datafield (topography is first dataset)
            gwy_container = gwy_data["/0/data"]
                
            # Get resolution & real-world size
            xres, yres = gwy_container["xres"], gwy_container["yres"]
            xreal, yreal = gwy_container["xreal"], gwy_container["yreal"]
            unit_z = gwy_container["si_unit_z"]

            # Extract height map as NumPy array
            height_map = np.array(gwy_container["data"]).reshape((yres, xres))  # Extract actual height values

            # Store extracted data
            afm_data[file_path] = {
                "file_path": file_path,
                "height_map": height_map,
                "xres": xres,
                "yres": yres,
                "xreal": xreal,
                "yreal": yreal,
                "unit_z": unit_z
            }

            print(f"Loaded {file_path} with shape {height_map.shape}")

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return afm_data

def plot_AFM(AFM_data, title="AFM Image", cmap="viridis"):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    for i, (file_path, data) in enumerate(AFM_data.items()):
        ax = axs[i // 2, i % 2]
        ax.imshow(data["height_map"], cmap=cmap, origin="lower")
        ax.set_title(f"AFM Image: {data['file_path']}")
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def calculate_roughness(height_map):
    Rq = np.sqrt(np.mean(height_map**2)) #RMS roughness
    Ra = np.mean(np.abs(height_map-np.mean(height_map))) #Mean roughness
    skewness = np.mean(((height_map - np.mean(height_map)) / np.std(height_map))**3)  # Skewness
    kurtosis = np.mean(((height_map - np.mean(height_map)) / np.std(height_map))**4)  # Kurtosis

    return {"Rq (RMS Roughness)": Rq, "Ra (Mean Roughness)": Ra, "Skewness": skewness, "Kurtosis": kurtosis}

def plot_height_distribution(height_map, title="Height Distribution"):
    plt.figure(figsize=(8, 6))
    plt.hist(height_map.flatten(), bins=100, color='dodgerblue', alpha=0.7)
    plt.title(title)
    plt.xlabel("Height (Z, nm)")
    plt.ylabel("Frequency")
    plt.show()

def plot_height_comparision(afm_data, title="Height Distribution Comparison"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)  # Side-by-side plots

    # Define categories
    categories = {
        "30small.gwy": ("blue", "30° Small"),
        "80small.gwy": ("red", "80° Small"),
        "30big.gwy": ("cyan", "30° Big"),
        "80big.gwy": ("orange", "80° Big")
    }

    # Separate files into "Small" and "Big" groups
    for filename, data in afm_data.items():
        height_values = data["height_map"].flatten()  # Convert to 1D
        color, label = categories.get(filename.split("/")[-1], ("black", filename))

        if "small" in filename:  # Left plot for small samples
            axes[0].hist(height_values, bins=100, density=True, alpha=0.5, color=color, label=label)
        elif "big" in filename:  # Right plot for big samples
            axes[1].hist(height_values, bins=100, density=True, alpha=0.5, color=color, label=label)

    # Set titles and labels
    axes[0].set_title("Height Distribution (Small Samples: 30° vs. 80°)")
    axes[1].set_title("Height Distribution (Big Samples: 30° vs. 80°)")

    for ax in axes:
        ax.set_xlabel("Height (nm)")
        ax.set_ylabel("Density")
        ax.set_xlim(-0.05, 0.5)
        ax.set_xticks(np.arange(0, 0.55, 0.05))
        ax.legend()
        ax.grid(True)

    plt.suptitle(title)
    plt.show()

def plot_3D_topography(height_map, xreal, yreal, title="3D AFM Surface Plot"):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Create X, Y coordinate grids
    x = np.linspace(0, xreal, height_map.shape[1])
    y = np.linspace(0, yreal, height_map.shape[0])
    X, Y = np.meshgrid(x, y)

    # Plot surface
    surf = ax.plot_surface(X, Y, height_map, cmap="inferno", edgecolor='none')

    # Labels and title
    ax.set_xlabel("X-axis (µm)")
    ax.set_ylabel("Y-axis (µm)")
    ax.set_zlabel("Height (nm)")
    ax.set_title(title)

    # Color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    plt.show()

def calculate_peak_to_valley(height_map):
    min_height=np.min(height_map)
    max_height = np.max(height_map)
    pvh = max_height - min_height
    return pvh

def detect_peaks(height_map, prominence=0.02, zoomed=False):
    # if zoomed: prominence = prominence_zoomed
    # else: prominence= prominence_normal

   # Flatten the height map for peak detection
    height_values = height_map.flatten()

    # Detect peaks with prominence filtering
    peaks, properties = find_peaks(height_values, prominence=prominence)

    # Extract peak heights
    peak_heights = height_values[peaks]

    return peaks, peak_heights, properties

def measure_peak_diameter(height_map, zoomed=False):
    if zoomed: prominence = prominence_zoomed
    else: prominence= prominence_normal

    height_values = height_map.flatten()

    # Detect peaks with prominence filtering
    peaks, properties = find_peaks(height_values, prominence=prominence, width=5)

    # Extract peak widths (diameter estimation)
    diameters = properties["widths"]

    return diameters

def measure_peak_diameter_nm(height_map, xreal, xres, prominence=0.01):
    """
    Measures peak diameters in nanometers.
    
    :param height_map: 2D NumPy array (AFM height data)
    :param xreal: Real-world scan width (nm)
    :param xres: Number of pixels along X-axis
    :param prominence: Minimum prominence to filter peaks
    :return: List of peak diameters in nm
    """
    height_values = height_map.flatten()

    # Detect peaks with prominence filtering
    peaks, properties = find_peaks(height_values, prominence=prominence, width=5)

    # Convert peak widths (originally in pixels) to nanometers
    diameters_nm = [convert_pixels_to_nm(w, xreal, xres) for w in properties["widths"]]

    return diameters_nm

def plot_peak_detection(height_map, y_index=128, prominence=0.01):
    """
    Plots an AFM height profile with detected peaks.
    
    :param height_map: 2D NumPy array of AFM height values
    :param y_index: Row index for the profile
    :param prominence: Minimum prominence for peak detection
    """
    # Extract height values for a cross-section
    height_values = height_map[y_index, :]

    # Detect peaks with prominence filtering
    peaks, properties = find_peaks(height_values, prominence=prominence)

    # Plot the height profile
    plt.figure(figsize=(8, 4))
    plt.plot(height_values, label="Height Profile", color="blue", linewidth=1)
    plt.scatter(peaks, height_values[peaks], color="red", label="Detected Peaks", marker="o", zorder=3)

    plt.xlabel("X-axis (pixels)")
    plt.ylabel("Height (nm)")
    plt.title(f"Peak Detection with Prominence ≥ {prominence} nm")
    plt.legend()
    plt.grid(True)
    plt.show()

def convert_pixels_to_nm(pixels, xreal, xres):
    """
    Converts pixel measurements to nanometers (nm).

    :param pixels: Pixel value to convert
    :param xreal: Real-world scan width (nm)
    :param xres: Number of pixels along X-axis
    :return: Value in nanometers
    """
    nm_per_pixel = xreal / xres
    return pixels * nm_per_pixel

def measure_peak_diameters_over_rows(height_map, xreal, xres, row_step=10, prominence=0.01):
    """
    Measures peak diameters across multiple rows in an AFM height map.

    :param height_map: 2D NumPy array (AFM height data)
    :param xreal: Real-world scan width (nm)
    :param xres: Number of pixels along X-axis
    :param row_step: Step size for selecting rows (e.g., every 10 rows)
    :param prominence: Minimum prominence to filter peaks
    :return: List of peak diameters in nm
    """
    all_diameters_nm = []

    # Loop through multiple rows
    for y_index in range(0, height_map.shape[0], row_step):
        height_values = height_map[y_index, :]  # Extract horizontal profile

        # Detect peaks in the row
        peaks, properties = find_peaks(height_values, prominence=prominence, width=5)

        # Convert peak widths (originally in pixels) to nanometers
        diameters_nm = [convert_pixels_to_nm(w, xreal, xres) for w in properties["widths"]]

        # Store results
        all_diameters_nm.extend(diameters_nm)

    return all_diameters_nm

def detect_3d_peaks(height_map, prominence_factor=0.1, min_distance=3):
    # Detects peaks in a 3D AFM height map without flattening.


    # Calculate adaptive prominence based on dataset range
    prominence = prominence_factor * (np.max(height_map) - np.min(height_map))

    # Apply maximum filer to find local maximas
    neighborhood = ndimage.generate_binary_structure(2, 2)
    local_max = ndimage.maximum_filter(height_map, footprint=neighborhood) == height_map

    # Extract peak positions
    peak_positions = np.argwhere(local_max)
    peak_heights = height_map[local_max]

    peaks_filtered, properties = find_peaks(peak_heights, prominence=prominence, distance=min_distance)
    peak_mask = np.zeros_like(height_map, dtype=bool)
    for i in peaks_filtered:
        y, x = peak_positions[i]
        peak_mask[y, x] = True

    return peak_mask

def plot_peak_mask(filename, data, peak_mask, title="Peak Detection Mask"):
    plt.figure(figsize=(6, 6))
    plt.imshow(data["height_map"], cmap="gray", origin="lower")
    plt.imshow(peak_mask, cmap="Reds", alpha=0.6)  # Overlay detected peaks
    plt.colorbar(label="Height (nm)")
    plt.title(f"Detected Peaks: {filename.split('/')[-1]}")
    plt.show()

def enhance_peak_contrast(height_map, peak_mask, reduction_factor=0.2):
    """
    Reduces non-peak height values to make peaks more prominent.

    :param height_map: 2D NumPy array of AFM height values
    :param peak_mask: Boolean mask with detected peaks
    :param reduction_factor: Factor to reduce non-peak areas (0-1)
    :return: Contrast-enhanced height map
    """
    enhanced_map = height_map.copy()
    
    # Reduce non-peak areas
    enhanced_map[~peak_mask] *= reduction_factor
    
    return enhanced_map

def plot_peak_contrast(filename, enhanced_map):
    # Visualization
    plt.figure(figsize=(6, 6))
    plt.imshow(enhanced_map, cmap="inferno", origin="lower")
    plt.colorbar(label="Height (nm)")
    plt.title(f"Enhanced Peaks: {filename.split('/')[-1]}")
    plt.show()

def plot_3d_contrast_map(enhanced_map, xreal, yreal, title="3D Enhanced AFM Surface"):
    """
    Generates a 3D surface plot of an AFM height map with contrast-enhanced peaks.

    :param enhanced_map: 2D NumPy array (contrast-enhanced AFM height data)
    :param xreal: Real-world scan width (nm or µm)
    :param yreal: Real-world scan height (nm or µm)
    :param title: Title of the plot
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Create X, Y coordinate grids
    x = np.linspace(0, xreal, enhanced_map.shape[1])
    y = np.linspace(0, yreal, enhanced_map.shape[0])
    X, Y = np.meshgrid(x, y)

    # Plot surface with color map
    surf = ax.plot_surface(X, Y, enhanced_map, cmap="inferno", edgecolor='none')

    # Labels and title
    ax.set_xlabel("X-axis (µm)")
    ax.set_ylabel("Y-axis (µm)")
    ax.set_zlabel("Height (nm)")
    ax.set_title(title)

    # Color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    plt.show()

def remove_outliers(height_map, threshold=5):
    """
    Removes outliers in the AFM height data using Z-score filtering.

    :param height_map: 2D NumPy array (AFM height data)
    :param threshold: Z-score threshold for identifying outliers
    :return: Cleaned height map
    """
    # Flatten and compute Z-scores
    z_scores = zscore(height_map.flatten())

    # Create mask for non-outliers
    non_outlier_mask = np.abs(z_scores) < threshold

    # Replace outliers with median value
    cleaned_data = height_map.flatten()
    cleaned_data[~non_outlier_mask] = np.median(cleaned_data)

    return cleaned_data.reshape(height_map.shape)

def apply_denoising(height_map, gaussian_sigma=1, median_size=3):
    """
    Applies Gaussian and Median filtering to reduce noise.

    :param height_map: 2D NumPy array (AFM height data)
    :param gaussian_sigma: Standard deviation for Gaussian filter
    :param median_size: Kernel size for median filter
    :return: Denoised height map
    """
    # Apply Gaussian filter (smooth small noise)
    smoothed = gaussian_filter(height_map, sigma=gaussian_sigma)

    # Apply Median filter (remove impulse noise)
    denoised = median_filter(smoothed, size=median_size)

    return denoised


def correct_background_tilt(height_map):
    """
    Removes large-scale tilt by fitting and subtracting a plane.

    :param height_map: 2D NumPy array (AFM height data)
    :return: Tilt-corrected height map
    """
    # Get the X, Y grid
    x = np.arange(height_map.shape[1])
    y = np.arange(height_map.shape[0])
    X, Y = np.meshgrid(x, y)
    
    # Flatten the data
    Z = height_map.flatten()
    A = np.c_[X.flatten(), Y.flatten(), np.ones_like(X.flatten())]

    # Fit plane
    plane_params, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
    fitted_plane = (plane_params[0] * X + plane_params[1] * Y + plane_params[2])

    # Subtract plane from height map
    corrected_map = height_map - fitted_plane

    return corrected_map

def correct_background_softly(height_map, smoothing_factor=5):
    """
    Applies a mild tilt correction to preserve fine details.
    
    :param height_map: 2D NumPy array (AFM height data)
    :param smoothing_factor: Controls how strong the background subtraction is
    :return: Softly tilt-corrected height map
    """
    background = gaussian_filter(height_map, sigma=smoothing_factor)  # Large-scale features
    corrected_map = height_map - background  # Subtract background trend

    return corrected_map

def clean_map(filename, height_map, sigma=1, threshold=3):

    if "big" in filename:
        cleaned_map = apply_denoising(height_map, gaussian_sigma=sigma)
        cleaned_map = correct_background_tilt(cleaned_map)
        # cleaned_map = contrast_stretch(cleaned_map)
        cleaned_map = apply_high_pass_filter(cleaned_map)
        return cleaned_map
    else:
        cleaned_map = correct_background_tilt(height_map)
        # cleaned_map = contrast_stretch(cleaned_map)
        cleaned_map = apply_high_pass_filter(cleaned_map)
        return cleaned_map


def contrast_stretch(height_map, low_percentile=1, high_percentile=99):
    """
    Stretches the contrast of an AFM height map based on percentiles.
    
    :param height_map: 2D NumPy array (AFM height data)
    :param low_percentile: Lower percentile for contrast adjustment
    :param high_percentile: Upper percentile for contrast adjustment
    :return: Contrast-enhanced height map
    """
    min_val = np.percentile(height_map, low_percentile)
    max_val = np.percentile(height_map, high_percentile)

    # Scale values to 0-1 range
    stretched_map = (height_map - min_val) / (max_val - min_val)
    stretched_map = np.clip(stretched_map, 0, 1)  # Ensure valid range

    return stretched_map

def remove_high_frequency_noise(height_map, threshold=0.1):
    """
    Removes high-frequency noise from an AFM height map using Fourier Transform (FFT).

    :param height_map: 2D NumPy array (AFM height data)
    :param threshold: Fraction of high-frequency components to remove
    :return: Denoised height map
    """
    # Apply FFT
    f_transform = fft.fft2(height_map)
    f_shift = fft.fftshift(f_transform)

    # Create a low-pass filter mask
    rows, cols = height_map.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), dtype=np.uint8)
    radius = int(min(rows, cols) * threshold)
    mask[crow - radius:crow + radius, ccol - radius:ccol + radius] = 1

    # Apply mask and inverse FFT
    f_shift_filtered = f_shift * mask
    f_ishift = fft.ifftshift(f_shift_filtered)
    height_map_filtered = np.real(fft.ifft2(f_ishift))

    return height_map_filtered

def smooth_height_map(filename, height_map, xreal, yreal, sigma=0.5, interpolation_factor=2, show=True):
    """
    Smooths AFM data while preserving important features.

    :param height_map: 2D NumPy array (AFM height data)
    :param sigma: Strength of Gaussian filter (controls noise removal)
    :param interpolation_factor: Upscaling factor for smoother 3D rendering
    :return: Smoothed, upscaled height map
    """
    # Apply Gaussian smoothing (mild to remove noise but keep spores)
    fft_filtered_map = remove_high_frequency_noise(height_map, threshold=0.1)

    smoothed_map = gaussian_filter(fft_filtered_map, sigma=sigma)

    # Upscale with bicubic interpolation for smoother 3D surface
    upscaled_map = ndimage.zoom(smoothed_map, interpolation_factor, order=3)

    if show:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Generate X, Y grids with the new smooth map
        x = np.linspace(0, xreal, height_map.shape[1]*interpolation_factor)
        y = np.linspace(0, yreal, height_map.shape[0]*interpolation_factor)
        X, Y = np.meshgrid(x, y)

        # Create the smooth surface plot
        surf = ax.plot_surface(X, Y, upscaled_map, cmap="inferno", edgecolor='none', rstride=1, cstride=1, antialiased=True)

        # Labels and title
        ax.set_xlabel("X-axis (µm)")
        ax.set_ylabel("Y-axis (µm)")
        ax.set_zlabel("Height (nm)")
        ax.set_title(f"3D Topography Map of {filename}")

        # Color bar for better visualization
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

        plt.show()
    return upscaled_map


def histogram_equalization(height_map):
    """
    Applies histogram equalization to improve AFM contrast.
    
    :param height_map: 2D NumPy array (AFM height data)
    :return: Contrast-enhanced height map
    """
    # Normalize to 0-255 range
    height_map_scaled = ((height_map - np.min(height_map)) / (np.max(height_map) - np.min(height_map))) * 255
    height_map_scaled = height_map_scaled.astype(np.uint8)

    # Apply histogram equalization
    equalized = cv2.equalizeHist(height_map_scaled)

    # Scale back to original range
    equalized_map = (equalized / 255) * (np.max(height_map) - np.min(height_map)) + np.min(height_map)

    return equalized_map

def apply_high_pass_filter(height_map, alpha=1.5):
    """
    Applies a high-pass filter to enhance fine details in AFM height data.
    
    :param height_map: 2D NumPy array (AFM height data)
    :param alpha: Strength of the sharpening effect
    :return: Sharpened height map
    """
    blurred = gaussian_filter(height_map, sigma=1)
    high_pass = height_map - blurred  # Extract high-frequency details
    enhanced_map = height_map + alpha * high_pass  # Add details back

    return enhanced_map

def smooth_height_map_robust(height_map, rolling_window=3):
    """
    Smooths AFM height map using a rolling average to reduce noise without blurring spores.
    
    :param height_map: 2D NumPy array (AFM height data)
    :param rolling_window: Window size for smoothing
    :return: Smoothed height map
    """
    # Apply a uniform rolling filter
    smoothed_map = ndimage.uniform_filter(height_map, size=rolling_window)

    return smoothed_map

def normalize_height_map(height_map):
    """
    Normalizes the AFM height map to a 0-1 range for improved contrast.

    :param height_map: 2D NumPy array (AFM height data)
    :return: Normalized height map
    """
    min_val, max_val = np.min(height_map), np.max(height_map)
    return (height_map - min_val) / (max_val - min_val)




def compare_cleaning_steps(filename, raw_map, fig_title="AFM Cleaning Steps"):
    """
    Plots the AFM surface before and after cleaning.

    :param raw_map: Original height data
    :param cleaned_map: After outlier removal
    :param denoised_map: After noise filtering
    :param tilt_corrected_map: After tilt correction
    """
    fig, axes = plt.subplots(1, 5, figsize=(16, 5))

    if "big" in filename:
        denoised_map = apply_denoising(raw_map, gaussian_sigma=1)
        tilt_corrected_map = correct_background_tilt(denoised_map)
    else:
        denoised_map = raw_map
        tilt_corrected_map = correct_background_tilt(raw_map)
    
    # contrast_map = contrast_stretch(tilt_corrected_map)
    # equalized_map = histogram_equalization(contrast_map)
    # sharpened_map = apply_high_pass_filter(contrast_map)

    smoothed_map = smooth_height_map_robust(tilt_corrected_map)
    normalized_map = normalize_height_map(tilt_corrected_map)

    
    datasets = [raw_map, denoised_map, tilt_corrected_map, smoothed_map, normalized_map]
    titles = ["Raw Data", "Denoised", "Tilt-Corrected", "Smoothed", "Normalized"]

    for i, (data, title) in enumerate(zip(datasets, titles)):
        axes[i].imshow(data, cmap="inferno", origin="lower")
        axes[i].set_title(title)
        axes[i].axis("off")

    plt.suptitle(f"{fig_title}: {filename}")
    plt.show()


def detect_peaks_smooth(height_map, prominence=0.02, min_distance=5):
    """
    Detects peaks in a 2D AFM height map.
    
    :param height_map: 2D NumPy array (smoothed AFM height data)
    :param prominence: Minimum prominence of peaks to consider
    :param min_distance: Minimum pixel distance between peaks
    :return: Peak mask (boolean array), peak coordinates
    """
    peak_mask = np.zeros_like(height_map, dtype=bool)

    for y in range(height_map.shape[0]):
        # Find peaks along each row
        peaks, properties = find_peaks(height_map[y, :], prominence=prominence, distance=min_distance)
        
        # Mark peaks in the mask
        peak_mask[y, peaks] = True

    return peak_mask, np.argwhere(peak_mask)

def detect_spores(height_map, prominence=0.02):
    """
    Detects spores (bubbles) in the AFM height map as single objects instead of multiple peaks.
    
    :param height_map: 2D NumPy array (AFM height data)
    :param prominence: Minimum prominence of peaks to be considered
    :return: List of spore properties (peak position, bounding box, diameter)
    """
    # Step 1: Threshold the height map to get high regions
    threshold = np.percentile(height_map, 95)  # Top 5% of height values
    spore_mask = height_map > threshold

    # Step 2: Label connected regions (group peaks into one bubble)
    labeled_spores, num_spores = label(spore_mask)

    # Step 3: Get bounding boxes for each detected spore
    spore_data = []
    for i in range(1, num_spores + 1):  # Skip background (label 0)
        # Get spore bounding box
        slice_x, slice_y = find_objects(labeled_spores == i)[0]
        min_x, max_x = slice_x.start, slice_x.stop
        min_y, max_y = slice_y.start, slice_y.stop

        # Find the true highest peak inside this spore
        peak_y, peak_x = maximum_position(height_map * (labeled_spores == i))

        # Calculate spore diameter (max distance across X or Y)
        diameter_nm = max((max_x - min_x), (max_y - min_y))  # In pixels
        
        spore_data.append({
            "peak_x": peak_x,
            "peak_y": peak_y,
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
            "diameter_pixels": diameter_nm
        })

    return spore_data, labeled_spores




def plot_peak_analysis(height_map, peak_mask, peak_diameters, filename):
    """
    Plots peak analysis in three panels:
    1. AFM height map with detected peaks overlayed
    2. Histogram of peak diameters
    3. Boxplot showing peak diameter statistics
    
    :param height_map: 2D NumPy array (AFM height data)
    :param peak_mask: Boolean mask of detected peaks
    :param peak_diameters: List of peak diameters in nanometers
    :param filename: Name of the sample file (for labeling)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # **Left: Height map with detected peaks**
    axes[0].imshow(height_map, cmap="gray", origin="lower")
    axes[0].imshow(peak_mask, cmap="Reds", alpha=0.6)  # Overlay detected peaks
    axes[0].set_title(f"Detected Peaks: {filename}")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # **Middle: Histogram of peak diameters**
    axes[1].hist(peak_diameters, bins=30, color="blue", alpha=0.7, edgecolor="black")
    axes[1].set_xlabel("Peak Diameter (nm)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"Histogram of Peak Diameters ({filename})")

    # **Right: Boxplot of peak diameters**
    sns.boxplot(data=peak_diameters, ax=axes[2], color="blue")
    axes[2].set_xlabel("Peak Diameters (nm)")
    axes[2].set_title(f"Boxplot of Peak Diameters ({filename})")

    plt.tight_layout()
    plt.show()



# -------
AFM_data = load_files()
# plot_AFM(AFM_data=AFM_data, title="Raw AFM Images")

def analyze_spore_data(detect_spores, filename, data, smooth_map):
    spore_data, spore_mask = detect_spores(smooth_map)
    nm_per_pixel = data['xreal'] / data['xres']  # Convert pixels to nm

    for spore in spore_data:
        spore["diameter_nm"] = spore["diameter_pixels"] * nm_per_pixel  # Convert to nm

    
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # **Left: Height map with spore boundaries & peaks**
    axes[0].imshow(smooth_map, cmap="gray", origin="lower")
    axes[0].imshow(spore_mask, cmap="Reds", alpha=0.4)  # Overlay detected spores

    # Mark peak positions
    for spore in spore_data:
        axes[0].scatter(spore["peak_x"], spore["peak_y"], color="black", s=20, marker="x")

    axes[0].set_title(f"Detected Spores: {filename}")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # **Middle: Histogram of spore diameters**
    diameters = [spore["diameter_nm"] for spore in spore_data]
    axes[1].hist(diameters, bins=20, color="blue", alpha=0.7, edgecolor="black")
    axes[1].set_xlabel("Spore Diameter (nm)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"Histogram of Spore Diameters ({filename})")

    # **Right: Boxplot of spore diameters**
    sns.boxplot(data=diameters, ax=axes[2], color="blue")
    axes[2].set_xlabel("Spore Diameters (nm)")
    axes[2].set_title(f"Boxplot of Spore Diameters ({filename})")

    plt.tight_layout()
    plt.show()

def print_spore_statistics(detect_spores, filename, data, smooth_map):
    """
    Prints statistical analysis of spore diameters.

    :param spore_data: List of detected spore properties with 'diameter_nm' key
    :param filename: Name of the sample file for reference
    """
    
    spore_data, spore_mask = detect_spores(smooth_map)
    nm_per_pixel = data['xreal'] / data['xres']  # Convert pixels to nm

    for spore in spore_data:
        spore["diameter_nm"] = spore["diameter_pixels"] * nm_per_pixel *1E6 # Convert to nm

    diameters = np.array([spore["diameter_nm"] for spore in spore_data])

    if len(diameters) == 0:
        print(f"📌 No spores detected in {filename}.")
        return

    # Calculate statistics
    mean_diameter = np.mean(diameters)
    median_diameter = np.median(diameters)
    std_dev = np.std(diameters)
    min_diameter = np.min(diameters)
    max_diameter = np.max(diameters)
    iqr = np.percentile(diameters, 75) - np.percentile(diameters, 25)  # Interquartile Range
    count = len(diameters)

    # Print results
    print("\n" + "="*50)
    print(f"Spore Diameter Statistics for {filename}")
    print("="*50)
    print(f"* Total Spores Detected: {count}")
    print(f"* Mean Diameter: {mean_diameter:.2f} nm")
    print(f"* Median Diameter: {median_diameter:.2f} nm")
    print(f"* Standard Deviation: {std_dev:.2f} nm")
    print(f"* Minimum Diameter: {min_diameter:.2f} nm")
    print(f"* Maximum Diameter: {max_diameter:.2f} nm")
    print(f"* Interquartile Range (IQR): {iqr:.2f} nm")
    print("="*50 + "\n")


def print_peak_statistics(detect_spores, filename, data, smooth_map):
    """
    Prints statistical analysis of peak heights.

    :param peak_heights: List of detected peak heights
    :param filename: Name of the sample file for reference
    """
    
    spore_data, spore_mask = detect_spores(smooth_map)
    nm_per_pixel = data['xreal'] / data['xres']  # Convert pixels to nm

    for spore in spore_data:
        # Extract bounding box
        min_x, max_x = spore["min_x"], spore["max_x"]
        min_y, max_y = spore["min_y"], spore["max_y"]

        # Get height values within the spore region
        spore_region = smooth_map[min_x:max_x, min_y:max_y]

        # Find peak height
        spore["peak_height"] = np.max(spore_region)

    
    peak_heights = np.array([spore["peak_height"] for spore in spore_data])
    peak_heights = np.array(peak_heights)

    if len(peak_heights) == 0:
        print(f"📌 No peaks detected in {filename}.")
        return

    # Calculate statistics
    mean_height = np.mean(peak_heights)
    median_height = np.median(peak_heights)
    std_dev = np.std(peak_heights)
    min_height = np.min(peak_heights)
    max_height = np.max(peak_heights)
    iqr = np.percentile(peak_heights, 75) - np.percentile(peak_heights, 25)  # Interquartile Range
    count = len(peak_heights)

    # Print results
    print("\n" + "="*50)
    print(f"Peak Height Statistics for {filename}")
    print("="*50)
    print(f"* Total Peaks Detected: {count}")
    print(f"* Mean Peak Height: {mean_height:.2f} nm")
    print(f"* Median Peak Height: {median_height:.2f} nm")
    print(f"* Standard Deviation: {std_dev:.2f} nm")
    print(f"* Minimum Peak Height: {min_height:.2f} nm")
    print(f"* Maximum Peak Height: {max_height:.2f} nm")
    print(f"* Interquartile Range (IQR): {iqr:.2f} nm")
    print("="*50 + "\n")


for filename, data in AFM_data.items():
    if "small" in filename: zoomed=True
    else: zoomed=False

    data['height_map'] = clean_map(filename, data['height_map'], sigma=1, threshold=5)
    # roughness = calculate_roughness(data["height_map"])
    # print(f"Roughness Metrics for {filename}: {roughness}")

    # plot_height_distribution(data["height_map"], title=f"Height Distribution: {filename}")

    # Plot 3D Map
    # plot_3D_topography(data["height_map"], data["xreal"], data["yreal"], title=f"3D Surface Plot: {filename}")

    # # Calculate Peak-to-Valley Height
    # prominence=0.1
    # peaks, peak_heights, properties = detect_peaks(data["height_map"], prominence=prominence)

    # print(f"🔍 {filename}: {len(peaks)} peaks detected with prominence ≥ {prominence} nm")

    # plot_peak_detection(data["height_map"], y_index=128, prominence=prominence)

    # # Calculate Peak Diameter
    # diameters = measure_peak_diameter_nm(data["height_map"], data["xreal"], data["xres"], prominence)
    # print(f"🔍 {filename}: Average Peak Diameter: {np.mean(diameters):.3f} nm")

    # diameters_nm = measure_peak_diameters_over_rows(data["height_map"], data["xreal"], data["xres"], row_step=10, prominence=prominence)
    # print(f"🔍 {filename}: Average Peak Diameter (over multiple rows): {np.mean(diameters_nm):.2f} nm")

    smooth_map = smooth_height_map(filename, data["height_map"], data["xreal"], data["yreal"], sigma=0.5, interpolation_factor=1, show=False)
    peak_mask, peak_positions = detect_peaks_smooth(smooth_map, prominence=0.02, min_distance=5)

    # analyze_spore_data(detect_spores, filename, data, smooth_map)
    print_spore_statistics(detect_spores, filename, data, smooth_map)
    print_peak_statistics(detect_spores, filename, data, smooth_map)



    # if "small" in filename:
    #     peak_mask = detect_3d_peaks(data["height_map"], prominence_factor=0.05, min_distance=3)
    #     plot_peak_mask(filename, data, peak_mask, title="Peak Detection Mask")

    #     enhanced_map = enhance_peak_contrast(data["height_map"], peak_mask, reduction_factor=0.2)
    #     plot_peak_contrast(filename, enhanced_map)

    #     plot_3d_contrast_map(enhanced_map, data["xreal"], data["yreal"], title="3D Enhanced AFM Surface")
    # else:
    #     peak_mask = detect_3d_peaks(data["height_map"], prominence_factor=0.05, min_distance=50)
    #     plot_peak_mask(filename, data, peak_mask, title="Peak Detection Mask")
    #     plot_3D_topography(data["height_map"], data["xreal"], data["yreal"], title=f"3D Surface Plot: {filename}")

    # compare_cleaning_steps(filename, data["height_map"], fig_title="AFM Cleaning Steps")

# plot_height_comparision(AFM_data, title="Height Distribution Comparison")
