import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from plot_images import plot_images

plot_images()

# Define the path and read JSON data from file
path = '/Users/u5500483/Documents/GitHub/Paper_I/Results/'
json_filename = path + 'PTC_HDR_12bit_temp_-25.json'  # Change this to your JSON file path
with open(json_filename, 'r') as file:
    data = json.load(file)

json_filename_2 = path + 'PTC_CSC-1012.json'  # Change this to your JSON file path
with open(json_filename_2, 'r') as file:
    data_2 = json.load(file)

corr_cts_1012 = np.array(data_2["corrected_counts"])
corr_var_1012 = np.array(data_2["corrected_variance_sqr"])
# Extract corrected_counts and corrected_variance_sqr
corrected_counts = np.array(data["corrected_counts"])
corrected_variance_sqr = np.array(data["corrected_variance_sqr"])

# Remove negative variance values
positive_mask = corrected_variance_sqr > 0
corrected_counts = corrected_counts[positive_mask]
corrected_variance_sqr = corrected_variance_sqr[positive_mask]

# Check if there are duplicate x values and remove them
_, unique_indices = np.unique(corrected_counts, return_index=True)
corrected_counts = corrected_counts[unique_indices]
corrected_variance_sqr = corrected_variance_sqr[unique_indices]

# Print some diagnostics after filtering
print(f"Filtered Data Statistics:")
print(f"  Corrected Counts Range: {corrected_counts.min()} to {corrected_counts.max()}")
print(f"  Corrected Variance Squared Range: {corrected_variance_sqr.min()} to {corrected_variance_sqr.max()}")

# Sort data to ensure x is increasing
sorted_indices = np.argsort(corrected_counts)
corrected_counts_sorted = corrected_counts[sorted_indices]
corrected_variance_sqr_sorted = corrected_variance_sqr[sorted_indices]

# Define the range of interest
range_min, range_max = 20000, 62000
range_mask = (corrected_counts_sorted >= range_min) & (corrected_counts_sorted <= range_max)
corrected_counts_range = corrected_counts_sorted[range_mask]
corrected_variance_sqr_range = corrected_variance_sqr_sorted[range_mask]

# Print the size of the filtered data
print(f"Number of points in the filtered range: {len(corrected_counts_range)}")

# Adjust the window length based on the number of data points
window_length = min(51, len(corrected_counts_range))  # Ensure window_length <= number of points
polyorder = min(5, window_length - 1)  # Polynomial order must be less than window_length

# Apply Savitzky-Golay filter for smoothing within the specified range
smooth_var_range = savgol_filter(corrected_variance_sqr_range, window_length, polyorder)

# Replace original variance values within the range with the smoothed values
corrected_variance_sqr_sorted[range_mask] = smooth_var_range

# Plot original data points and the updated smoothed data points
plt.plot(corrected_counts_sorted, corrected_variance_sqr_sorted, 'ro', label='Updated Data with Smoothed Values')
plt.plot(corr_cts_1012, corr_var_1012, 'bo', label='CSC-1012')
# Labels, legend, and grid
plt.xlabel('Corrected Counts')
plt.ylabel('Corrected Variance Squared')
plt.title('Updated Data with Smoothed Values')
plt.legend()
plt.grid(True)

# Show plot
plt.show()