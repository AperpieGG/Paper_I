"""
This script takes json files where the exposure and dark current data it.
It plots them along with linear fits on the same plot.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import json
from plot_images import plot_images
from matplotlib import colors as mcolors

default_blue_color = mcolors.to_rgba('#1f77b4')
default_orange_color = mcolors.to_rgba('#ff7f0e')
path = '/Users/u5500483/Documents/GitHub/Paper_I/Results/'


def read_json_HDR():
    print("Reading the HDR json file")
    with open(path + 'Dark_Current_HDR_12bit_temp_-25.json') as json_file:
        data = json.load(json_file)
    exposure_times = np.array(data["exposure_times"])
    mean_dark_current = np.array(data["mean_dark_current"])
    gradient = data["gradient"]
    offset = data["offset"]
    return exposure_times, mean_dark_current, gradient, offset


def read_json_FFR():
    print("Reading the FFR json file")
    with open(path + 'Dark_Current_FFR_12bit_temp_-25.json') as json_file:
        data = json.load(json_file)
    exposure_times = np.array(data["exposure_times"])
    mean_dark_current = np.array(data["mean_dark_current"])
    gradient = data["gradient"]
    offset = data["offset"]
    return exposure_times, mean_dark_current, gradient, offset


def linear_fit(x, a, b):
    return a * x + b


def fit_in_data(exposure_times, mean_dark_current):
    popt, pcov = curve_fit(linear_fit, exposure_times, mean_dark_current)
    print("Gradient and Offset: ", popt)
    return popt, pcov


def plot_dc(exposure_times, mean_dark_current, gradient, offset):
    # Create a grid for the main plot and residuals subplot
    figure, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # Main plot on the left
    ax1.plot(exposure_times, linear_fit(exposure_times, gradient, offset), 'g-')

    ax1.plot(exposure_times, mean_dark_current, 's', color=default_blue_color, label='HDR')  # Plot HDR data
    ax1.set_ylabel('Dark signal (ADU)')

    # Read and plot FFR data
    exposure_times_ffr, mean_dark_current_ffr, gradient_ffr, offset_ffr = read_json_FFR()
    ax1.plot(exposure_times_ffr, linear_fit(exposure_times_ffr, gradient_ffr, offset_ffr), 'g-')
    ax1.plot(exposure_times_ffr, mean_dark_current_ffr, 'o', color=default_orange_color, label='FFR')  # Plot FFR data
    ax1.legend(loc='best')

    # Calculate residuals for HDR data
    residuals_hdr = mean_dark_current - linear_fit(exposure_times, gradient, offset)

    # Calculate residuals for FFR data
    residuals_ffr = mean_dark_current_ffr - linear_fit(exposure_times_ffr, gradient_ffr, offset_ffr)

    # Plot residuals for HDR and FFR data
    ax2.plot([exposure_times.min(), exposure_times.max()], [0, 0], 'g-')
    ax2.plot(exposure_times, residuals_hdr, 's', color=default_blue_color)
    ax2.plot(exposure_times_ffr, residuals_ffr, 'o', color=default_orange_color)
    ax2.set_ylim([3 * min(residuals_hdr.min(), residuals_ffr.min()), 3 * max(residuals_hdr.max(), residuals_ffr.max())])
    ax2.set_xlabel('Exposure time (s)')
    ax2.set_ylabel('Residuals (ADU)')

    # Set exposure time ticks
    ax1.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ax2.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    plt.tight_layout()
    plt.savefig('Dark_Current_HDR_FFR.pdf', bbox_inches='tight', dpi=300)
    plt.show()


def run():
    exposure_times, mean_dark_current, gradient, offset = read_json_HDR()
    plot_images()
    plot_dc(exposure_times, mean_dark_current, gradient, offset)


run()
