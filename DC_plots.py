"""
Description: This code is used to import a json file and plots the results
The json files contains the dc current for the Marana.

It meaaures the offset and gradient and estimates the covariance error from the fit.
It plots dark signal vs exposure and the residuals
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import json
from plot_images import plot_images


def read_json():
    print("Reading the json file")
    path = '/Users/u5500483/Downloads/Paper_I/Marana_updated_compare/Testing_CMOS_Linux/Images/json/'
    with open(path + 'Dark_Current_HDR_12bit_temp_-25.json') as json_file:
        data = json.load(json_file)
    temperature = data["temperature"]
    exposure_times = data["exposure_times"]
    mean_dark_current = data["mean_dark_current"]
    gradient = data["gradient"]
    offset = data["offset"]
    exposure_times = np.array(exposure_times)
    mean_dark_current = np.array(mean_dark_current)
    return exposure_times, mean_dark_current, gradient, offset, temperature


def linear_fit(x, a, b):
    return a * x + b


def fit_in_data(exposure_times, mean_dark_current):
    popt, pcov = curve_fit(linear_fit, exposure_times, mean_dark_current)
    print("Gradient and Offset: ", popt)
    return popt, pcov


def estimate_errors_from_covariance(pcov):
    errors = np.sqrt(np.diag(pcov))
    print("Errors: ", errors)
    return errors


def plot_dc(exposure_times, mean_dark_current, gradient, offset, temperature):
    # Create a grid for the main plot and residuals subplot
    figure, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # Main plot on the left
    ax1.plot(exposure_times, linear_fit(exposure_times, gradient, offset), 'b-',
             label='%5.3f (cts/s) $\cdot$ $t_{\mathrm{exp}}\mathrm{(s)}$ + %5.3f (cts)' % (gradient, offset))

    ax1.plot(exposure_times, mean_dark_current, 'ro',
             label='DC = %5.3f e$^-$/pix/sec @%5.0f$^ \degree$C' % (
                 gradient * get_sensitivity(mean_dark_current), temperature))
    ax1.set_ylabel('Dark signal [cts]')
    ax1.legend(loc='best')

    startx = 1
    endx = 11
    step = 1
    residuals = []
    for i in range(len(exposure_times)):
        y_pred = exposure_times[i] * gradient + offset
        residuals.append(mean_dark_current[i] - y_pred)

    max_residual = np.max(residuals)
    min_residual = np.min(residuals)
    ax2.plot(exposure_times, residuals, 'ro')
    ax2.set_ylim([3 * min_residual, 3 * max_residual])
    ax2.plot([startx, endx - 1], [0, 0], 'b-')
    ax2.set_xlabel('Exposure time [s]')
    ax2.set_ylabel('Residuals [cts]')

    plt.tight_layout()
    plt.show()
    if mean_dark_current[0] > 100:
        figure.savefig('Dark_Current_HDR_12bit_temp_-25.pdf', bbox_inches='tight')
    else:
        figure.savefig('Dark_Current_FFR_12bit_temp_-25.pdf', bbox_inches='tight')


def get_sensitivity(mean_dark_current):
    if mean_dark_current[0] > 100:
        return 1.12
    else:
        return 0.64


def run():
    exposure_times, mean_dark_current, gradient, offset, temperature = read_json()
    plot_images()
    plot_dc(exposure_times, mean_dark_current, gradient, offset, temperature)
    popt, pcov = fit_in_data(exposure_times, mean_dark_current)
    errors = estimate_errors_from_covariance(pcov)
    gradient_error = errors[0]  # Extracting error for the slope (gradient)
    print("Error for gradient (slope):", gradient_error)


run()
