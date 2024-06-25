import matplotlib.pyplot as plt
import numpy as np
import json
import matplotlib.patches as mpatches
from scipy.optimize import curve_fit
from plot_images import plot_images

plot_images()


def read_json():
    print("Reading the json file")
    path = '/Users/u5500483/Downloads/home/ops/cmos_marana/Marana_updated_compare/Testing_CMOS_Linux/Images/json/'
    with open(path + 'PTC_HDR_12bit_temp_-25.json') as json_file:
        data = json.load(json_file)
    exposure_times = data["exposure_times"]
    corrected_counts = data["corrected_counts"]
    corrected_variance_sqr = data["corrected_variance_sqr"]
    saturation_grey_value = data["saturation_grey_value"]
    variance_sqr_saturation_grey = data["variance_sqr_saturation_grey"]
    CameraSensitivity = data["CameraSensitivity"]
    return exposure_times, corrected_counts, corrected_variance_sqr, saturation_grey_value


def filter_data(signal, variance, threshold=20000):
    filtered_signal = []
    filtered_variance = []
    for i in range(len(signal)):
        if signal[i] <= threshold:
            filtered_signal.append(signal[i])
            filtered_variance.append(variance[i])
    return np.array(filtered_signal), np.array(filtered_variance)


def linear_fit(x, a, b):
    return a * x + b


# Read data from JSON file
exposure_times, corrected_counts, corrected_variance_sqr, saturation_grey_value = read_json()

# Filter data based on signal <= 10000
filtered_counts, filtered_variance = filter_data(corrected_counts, corrected_variance_sqr)

# Fit linear models for two ranges
x1 = filtered_counts[filtered_counts <= 1260]
y1 = filtered_variance[filtered_counts <= 1260]
popt1, pcov1 = curve_fit(linear_fit, x1, y1)

x2 = filtered_counts[filtered_counts >= 2000]
y2 = filtered_variance[filtered_counts >= 2000]
popt2, pcov2 = curve_fit(linear_fit, x2, y2)

# Plot the filtered data
plot_images()
fig, ax = plt.subplots()
ax.plot(filtered_counts, filtered_variance, 'ro', label='HDR mode')
plt.plot(x1, linear_fit(x1, *popt1), 'b-')
plt.plot(x2, linear_fit(x2, *popt2), 'b--')
plt.xlabel('Signal (ADU)')
plt.ylabel('Variance (ADU$^2$)')
plt.legend()
plt.tight_layout()
# plt.savefig('PTC_HDR_-25.pdf', dpi=300)

plt.show()

# save the plot

print('Fit parameters for the range 0-1800:')
print('a:', popt1[0], 'b:', popt1[1], 'and Gain:', 1/popt1[0])
print('Fit parameters for the range 2000-end:')
print('a:', popt2[0], 'b:', popt2[1], 'and Gain:', 1/popt2[0])

# Print errors for the slopes
print('Errors for the slopes:')
print('Error for the slope in the range 0-1800:', np.sqrt(pcov1[0, 0]))
print('Error for the slope in the range 2000-end:', np.sqrt(pcov2[0, 0]))

gain = filtered_counts / filtered_variance

# Plot the gain vs signal
plt.plot(filtered_counts, gain, 'ro', label='HDR mode')
plt.xlabel('Signal (ADU)')
plt.ylabel('Sensitivity (e$^-$/ADU)')

# add a line of the gain from the performance sheet which is 1.12 e-/ADU
plt.axhline(y=1.131, color='b', linestyle='-')
plt.axhline(y=1.120, color='b', linestyle='--')
plt.ylim(0.4, 1.6)
plt.legend()
plt.tight_layout()

# save the plot
# plt.savefig('Gain_HDR_-25.pdf', dpi=300)
plt.show()

