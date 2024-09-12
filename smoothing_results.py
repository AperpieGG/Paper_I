import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
path = '/Users/u5500483/Documents/GitHub/Paper_I/Results/'
# Read JSON data from file
json_filename = path + 'PTC_HDR_12bit_temp_-25.json'  # Change this to your JSON file path
with open(json_filename, 'r') as file:
    data = json.load(file)

# Extract corrected_counts and corrected_variance_sqr
corrected_counts = np.array(data["corrected_counts"])
corrected_variance_sqr = np.array(data["corrected_variance_sqr"])

# Filter data points where corrected_counts > 20000
mask = corrected_counts > 20000
corrected_counts_filtered = corrected_counts[mask]
corrected_variance_sqr_filtered = corrected_variance_sqr[mask]

# Sort filtered data by corrected_counts to ensure x-values are increasing
sorted_indices = np.argsort(corrected_counts_filtered)
corrected_counts_sorted = corrected_counts_filtered[sorted_indices]
corrected_variance_sqr_sorted = corrected_variance_sqr_filtered[sorted_indices]

# Apply smoothing spline
smoothing_factor = 180  # Adjust this value as needed
spline = UnivariateSpline(corrected_counts_sorted, corrected_variance_sqr_sorted, s=smoothing_factor)

# Generate new x-values (corrected_counts) and compute smoothed y-values (corrected_variance_sqr)
new_counts = np.linspace(corrected_counts_sorted.min(), corrected_counts_sorted.max(), 100)
smoothed_variance_sqr = spline(new_counts)

# Plot original data points
plt.scatter(corrected_counts_sorted, corrected_variance_sqr_sorted, color='red', label='Original Data Points')

# Convert the smoothing spline into data points and plot
plt.scatter(new_counts, smoothed_variance_sqr, color='blue', label='Smoothed Data Points', s=10)

# Labels and Title
plt.xlabel('Corrected Counts')
plt.ylabel('Corrected Variance Squared')
plt.title('Smoothed Data Points Plot for Corrected Variance vs. Corrected Counts')
plt.legend()

# Show plot
plt.show()