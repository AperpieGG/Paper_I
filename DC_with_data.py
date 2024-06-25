"""
This script is used to plot the dark current data and fit a linear model to it.
The data is taken from NGTS. It will also estimate the error in the slope of the linear model.
"""
import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from astropy.io import fits
from plot_images import plot_images


def get_values_HDR(path):
    pixels = 2048 * 2048
    dark_list = sorted(glob.glob(path + '/dark*.fits'))
    dark_list_HDR = [f for f in dark_list if 'HDR' in fits.getheader(f)['READMODE']]
    exposure = [fits.getheader(dark)['EXPTIME'] for dark in dark_list_HDR]
    exposure = np.array(exposure)
    print(len(dark_list_HDR))

    dark_data = [fits.getdata(dark) for dark in dark_list_HDR]
    mean = [np.mean(dark) for dark in dark_data]
    std = [np.std(dark) for dark in dark_data] / np.sqrt(pixels)
    print("Exposure times: ", exposure)
    print("Mean dark current: ", mean)
    print("Standard deviation: ", std)
    return exposure, mean, std


def get_values_FFR(path):
    pixels = 2048 * 2048
    dark_list = sorted(glob.glob(path + '/dark*.fits'))
    dark_list_HDR = [f for f in dark_list if 'FFR' in fits.getheader(f)['READMODE']]
    exposure = [fits.getheader(dark)['EXPTIME'] for dark in dark_list_HDR]
    exposure_FFR = np.array(exposure)
    print(len(dark_list_HDR))

    dark_data = [fits.getdata(dark)[1250:1650, 200:1200] for dark in dark_list_HDR]
    mean_FFR = [np.mean(dark) for dark in dark_data]
    std_FFR = [np.std(dark) for dark in dark_data] / np.sqrt(pixels)
    print("Exposure times: ", exposure)
    print("Mean dark current: ", mean_FFR)
    print("Standard deviation: ", std_FFR)
    return exposure_FFR, mean_FFR, std_FFR


def linear_fit(x, a, b):
    """Linear fit function."""
    return a * x + b


def fit_in_data(exposure_times, mean_dark_current):
    """
    Fit the data to a linear model.

    Parameters:
        exposure_times (array): Array of exposure times.
        mean_dark_current (array): Array of mean dark current values.

    Returns:
        tuple: Tuple containing the optimized parameters and the covariance matrix.
    """
    popt, pcov = curve_fit(linear_fit, exposure_times, mean_dark_current)
    print("Gradient and Offset: ", popt)
    return popt, pcov


def estimate_errors_from_covariance(pcov):
    """
    Estimate errors from the covariance matrix.

    Parameters:
        pcov (array): Covariance matrix.

    Returns:
        array: Array of errors.
    """
    errors = np.sqrt(np.diag(pcov))
    print("Errors: ", errors)
    return errors


def plot_dc(exposure, mean_dark_MID, std_dark_MID, popt, exposure_FFR, mean_dark_FFR, std_dark_FFR, popt_FFR):
    figure, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    ax1.errorbar(exposure, mean_dark_MID, yerr=std_dark_MID, fmt='o', label='Data HDR')
    ax1.errorbar(exposure_FFR, mean_dark_FFR, yerr=std_dark_FFR, fmt='o', label='Data FFR')
    ax1.set_ylabel('Mean Dark Signal')
    ax1.legend(loc='best')

    # Overlay linear fit
    ax1.plot(exposure, linear_fit(exposure, *popt), 'g-', label='Fit HDR')
    ax1.plot(exposure_FFR, linear_fit(exposure_FFR, *popt_FFR), 'g-', label='Fit FFR')

    ax2.plot([exposure.min(), exposure.max()], [0, 0], 'g-')
    ax2.errorbar(exposure, mean_dark_MID - linear_fit(exposure, *popt), yerr=std_dark_MID, fmt='o')
    ax2.errorbar(exposure_FFR, mean_dark_FFR - linear_fit(exposure_FFR, *popt_FFR), yerr=std_dark_FFR, fmt='o')
    ax2.set_xlabel('Exposure time (s)')
    ax2.set_ylabel('Residuals (ADU)')
    # ax2.set_ylim([3 * min(residuals_hdr.min(), residuals_ffr.min()), 3 * max(residuals_hdr.max(), residuals_ffr.max())])

    plt.tight_layout()
    plt.show()


def run():
    plot_images()
    path = (f'/Users/u5500483/Downloads/home/ops/cmos_marana/Marana_updated_compare/'
            f'Testing_CMOS_Linux/Images/json/NGTS_DARK_DATA/')
    exposure, mean_dark_MID, std_dark_MID = get_values_HDR(path)
    exposure_FFR, mean_dark_FFR, std_dark_FFR = get_values_FFR(path)

    popt, pcov = fit_in_data(exposure, mean_dark_MID)
    popt_FFR, pcov_FFR = fit_in_data(exposure_FFR, mean_dark_FFR)
    plot_dc(exposure, mean_dark_MID, std_dark_MID, popt, exposure_FFR, mean_dark_FFR, std_dark_FFR, popt_FFR)
    errors = estimate_errors_from_covariance(pcov)
    errors_FFR = estimate_errors_from_covariance(pcov_FFR)
    gradient_error = errors[0]  # Extracting error for the slope (gradient)
    print("Error for gradient (slope):", gradient_error)
    gradient_error_FFR = errors_FFR[0]  # Extracting error for the slope (gradient)
    print("Error for gradient (slope):", gradient_error_FFR)


run()
