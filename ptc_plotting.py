import matplotlib.pyplot as plt
import numpy as np
import json
import matplotlib.patches as mpatches
from scipy.optimize import curve_fit
from plot_images import plot_images


def read_json():
    print("Reading the json file")
    path = '/Users/u5500483/Documents/GitHub/Paper_I/Results/'
    with open(path + 'PTC_HDR_12bit_temp_-25.json') as json_file:
        data = json.load(json_file)
    exposure_times = data["exposure_times"]
    corrected_counts = data["corrected_counts"]
    corrected_variance_sqr = data["corrected_variance_sqr"]
    saturation_grey_value = data["saturation_grey_value"]
    variance_sqr_saturation_grey = data["variance_sqr_saturation_grey"]
    CameraSensitivity = data["CameraSensitivity"]
    return exposure_times, corrected_counts, corrected_variance_sqr, saturation_grey_value, variance_sqr_saturation_grey, CameraSensitivity


def linear_fit(x, a, b):
    return a * x + b


def fit_in_data(corrected_counts, corrected_variance_sqr):
    popt, pcov = curve_fit(linear_fit, corrected_counts, corrected_variance_sqr)
    print("Gradient and Offset: ", popt, 'and Gain:', 1 / popt[0])
    return popt, pcov


def plot_ptc(corrected_counts, corrected_variance_sqr, saturation_grey_value, variance_sqr_saturation_grey,
             CameraSensitivity):
    plot_images()
    fig, ax = plt.subplots()
    ax.set_xlabel('Signal (ADU)')
    ax.set_ylabel('Variance (ADU$^2$)')
    ax.plot(corrected_counts, corrected_variance_sqr, 'ro')

    corrected_counts = np.array(corrected_counts)
    corrected_variance_sqr = np.array(corrected_variance_sqr)
    # reject = corrected_counts > 2625  # For FFR channel only
    reject = corrected_counts > 1260
    x_1 = (corrected_counts[~reject])
    y_1 = (corrected_variance_sqr[~reject])
    plt.plot(x_1, fit_in_data(x_1, y_1)[0][0] * x_1 + fit_in_data(x_1, y_1)[0][1], 'b-',
             label='%5.3f x + %5.3f' % (fit_in_data(x_1, y_1)[0][0], fit_in_data(x_1, y_1)[0][1]))
    popt, pcov = fit_in_data(x_1, y_1)
    slope_error = np.sqrt(pcov[0, 0])  # Extract error for the slope from the covariance matrix
    print("Error from the slope:", slope_error)
    # # for HDR channel only
    # axins = ax.inset_axes([0.5, 0.1, 0.3, 0.3])
    # axins.plot([corrected_counts], [corrected_variance_sqr], 'ro')
    # axins.plot(x_1, fit_in_data(x_1, y_1)[0][0] * x_1 + fit_in_data(x_1, y_1)[0][1], 'b-')
    #
    # axins.set_xlim(-100, 2990)
    # axins.set_ylim(-100, 2990)
    # ax.indicate_inset_zoom(axins)
    # ax.set_xlabel('Signal [cts]')
    # ax.set_ylabel('Variance [cts$^2$]')

    ax.plot(saturation_grey_value, variance_sqr_saturation_grey, 'b*', label='FWC = %.1f cts // %.1f e$^-$' % (
        saturation_grey_value, saturation_grey_value * CameraSensitivity))
    extra_label = 'Sensitivity = %.2f e$^-$/cts' % CameraSensitivity
    extra_handle = mpatches.Patch(facecolor='white', edgecolor='red', label=extra_label)
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(extra_handle)
    ax.legend(handles=handles, labels=labels + [extra_label], loc='best')
    plt.tight_layout()
    plt.show()
    # fig.savefig('PTC_FFR_12bit_temp_-25.pdf', bbox_inches='tight')


(exposure_times, corrected_counts, corrected_variance_sqr, saturation_grey_value,
 variance_sqr_saturation_grey, CameraSensitivity) = read_json()
plot_ptc(corrected_counts, corrected_variance_sqr, saturation_grey_value, variance_sqr_saturation_grey,
         CameraSensitivity)
