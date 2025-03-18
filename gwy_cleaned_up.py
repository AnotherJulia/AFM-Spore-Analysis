# Imports
import gwyfile as gwy
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import scipy.ndimage as ndimage
from scipy.ndimage import gaussian_filter, median_filter, label, find_objects, maximum_position
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import zscore
import seaborn as sns

# Variables
filenames = ["raw/30small.gwy", "raw/30big.gwy", "raw/80small.gwy", "raw/80big.gwy"]
prominence_zoomed = 0.3
prominence_normal = 0.05

def load_files():
    afm_data = {}
    for file_path in filenames:
        try:
            gwy_data = gwy.load(file_path)
            gwy_container = gwy_data["/0/data"]
            xres, yres = gwy_container["xres"], gwy_container["yres"]
            xreal, yreal = gwy_container["xreal"], gwy_container["yreal"]
            unit_z = gwy_container["si_unit_z"]
            height_map = np.array(gwy_container["data"]).reshape((yres, xres))
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

def correct_background_tilt(height_map):
    x = np.arange(height_map.shape[1])
    y = np.arange(height_map.shape[0])
    X, Y = np.meshgrid(x, y)
    Z = height_map.flatten()
    A = np.c_[X.flatten(), Y.flatten(), np.ones_like(X.flatten())]
    plane_params, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
    fitted_plane = (plane_params[0] * X + plane_params[1] * Y + plane_params[2])
    corrected_map = height_map - fitted_plane
    return corrected_map

def apply_high_pass_filter(height_map, alpha=1.5):
    blurred = gaussian_filter(height_map, sigma=1)
    high_pass = height_map - blurred
    enhanced_map = height_map + alpha * high_pass
    return enhanced_map

def clean_map(filename, height_map, sigma=1, threshold=3):
    if "big" in filename:
        cleaned_map = gaussian_filter(height_map, sigma=sigma)
        cleaned_map = correct_background_tilt(cleaned_map)
        cleaned_map = apply_high_pass_filter(cleaned_map)
        return cleaned_map
    else:
        cleaned_map = correct_background_tilt(height_map)
        cleaned_map = apply_high_pass_filter(cleaned_map)
        return cleaned_map

def remove_high_frequency_noise(height_map, threshold=0.1):
    f_transform = np.fft.fft2(height_map)
    f_shift = np.fft.fftshift(f_transform)
    rows, cols = height_map.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), dtype=np.uint8)
    radius = int(min(rows, cols) * threshold)
    mask[crow - radius:crow + radius, ccol - radius:ccol + radius] = 1
    f_shift_filtered = f_shift * mask
    f_ishift = np.fft.ifftshift(f_shift_filtered)
    height_map_filtered = np.real(np.fft.ifft2(f_ishift))
    return height_map_filtered

def smooth_height_map(filename, height_map, xreal, yreal, sigma=0.5, interpolation_factor=2, show=True):
    fft_filtered_map = remove_high_frequency_noise(height_map, threshold=0.1)
    smoothed_map = gaussian_filter(fft_filtered_map, sigma=sigma)
    upscaled_map = ndimage.zoom(smoothed_map, interpolation_factor, order=3)
    if show:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        x = np.linspace(0, xreal, height_map.shape[1] * interpolation_factor)
        y = np.linspace(0, yreal, height_map.shape[0] * interpolation_factor)
        X, Y = np.meshgrid(x, y)
        surf = ax.plot_surface(X, Y, upscaled_map, cmap="inferno", edgecolor='none', rstride=1, cstride=1, antialiased=True)
        ax.set_xlabel("X-axis (µm)")
        ax.set_ylabel("Y-axis (µm)")
        ax.set_zlabel("Height (nm)")
        ax.set_title(f"3D Topography Map of {filename}")
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        plt.show()
    return upscaled_map

def detect_peaks_smooth(height_map, prominence=0.02, min_distance=5):
    peak_mask = np.zeros_like(height_map, dtype=bool)
    for y in range(height_map.shape[0]):
        peaks, properties = find_peaks(height_map[y, :], prominence=prominence, distance=min_distance)
        peak_mask[y, peaks] = True
    return peak_mask, np.argwhere(peak_mask)

def detect_spores(height_map, prominence=0.02):
    threshold = np.percentile(height_map, 95)
    spore_mask = height_map > threshold
    labeled_spores, num_spores = label(spore_mask)
    spore_data = []
    for i in range(1, num_spores + 1):
        slice_x, slice_y = find_objects(labeled_spores == i)[0]
        min_x, max_x = slice_x.start, slice_x.stop
        min_y, max_y = slice_y.start, slice_y.stop
        peak_y, peak_x = maximum_position(height_map * (labeled_spores == i))
        diameter_nm = max((max_x - min_x), (max_y - min_y))
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

def print_spore_statistics(detect_spores, filename, data, smooth_map):
    spore_data, spore_mask = detect_spores(smooth_map)
    nm_per_pixel = data['xreal'] / data['xres']
    for spore in spore_data:
        spore["diameter_nm"] = spore["diameter_pixels"] * nm_per_pixel * 1E6
    diameters = np.array([spore["diameter_nm"] for spore in spore_data])
    if len(diameters) == 0:
        print(f"📌 No spores detected in {filename}.")
        return
    mean_diameter = np.mean(diameters)
    median_diameter = np.median(diameters)
    std_dev = np.std(diameters)
    min_diameter = np.min(diameters)
    max_diameter = np.max(diameters)
    iqr = np.percentile(diameters, 75) - np.percentile(diameters, 25)
    count = len(diameters)
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
    spore_data, spore_mask = detect_spores(smooth_map)
    nm_per_pixel = data['xreal'] / data['xres']
    for spore in spore_data:
        min_x, max_x = spore["min_x"], spore["max_x"]
        min_y, max_y = spore["min_y"], spore["max_y"]
        spore_region = smooth_map[min_x:max_x, min_y:max_y]
        spore["peak_height"] = np.max(spore_region)
    peak_heights = np.array([spore["peak_height"] for spore in spore_data])
    if len(peak_heights) == 0:
        print(f"📌 No peaks detected in {filename}.")
        return
    mean_height = np.mean(peak_heights)
    median_height = np.median(peak_heights)
    std_dev = np.std(peak_heights)
    min_height = np.min(peak_heights)
    max_height = np.max(peak_heights)
    iqr = np.percentile(peak_heights, 75) - np.percentile(peak_heights, 25)
    count = len(peak_heights)
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

AFM_data = load_files()

for filename, data in AFM_data.items():
    if "small" in filename:
        zoomed = True
    else:
        zoomed = False
    data['height_map'] = clean_map(filename, data['height_map'], sigma=1, threshold=5)
    smooth_map = smooth_height_map(filename, data["height_map"], data["xreal"], data["yreal"], sigma=0.5, interpolation_factor=1, show=False)
    peak_mask, peak_positions = detect_peaks_smooth(smooth_map, prominence=0.02, min_distance=5)
    print_spore_statistics(detect_spores, filename, data, smooth_map)
    print_peak_statistics(detect_spores, filename, data, smooth_map)