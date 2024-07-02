"""
Description: This script is used take image data and plot the results.
The image data is recorded with the Marana.

It measures the mean dark signal from the FFI for each image exposure at different temperatures.
It then finds the gradient/slope for each diff temperature and estimate the dc.
It plots the dc as function of temperature.
The errors for each slope for the dc vs temp are estimated using the covariance matrix.
"""

from astropy.io import fits
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def plot_images():
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['xtick.top'] = True
    plt.rcParams['xtick.labeltop'] = False
    plt.rcParams['xtick.labelbottom'] = True
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['xtick.major.top'] = True
    plt.rcParams['xtick.minor.top'] = True
    plt.rcParams['xtick.minor.bottom'] = True
    plt.rcParams['xtick.alignment'] = 'center'

    plt.rcParams['ytick.left'] = True
    plt.rcParams['ytick.labelleft'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['ytick.major.right'] = True
    plt.rcParams['ytick.major.left'] = True
    plt.rcParams['ytick.minor.right'] = True
    plt.rcParams['ytick.minor.left'] = True

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14

    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.8
    plt.rcParams['legend.loc'] = 'best'
    plt.rcParams['legend.fancybox'] = True
    plt.rcParams['legend.fontsize'] = 14


def get_dark_data(path, suffix):
    dark_list = glob.glob(path + f'/DATA_{suffix}/dark*.fits')
    print("The shape of the dark_list is: ", np.shape(dark_list))
    dark_data = [fits.getdata(dark)[0] for dark in dark_list]
    dark_data = dark_data + np.array(99.26)
    return np.array(dark_data)


def get_exposure_values(path, suffix):
    dark_list = glob.glob(path + f'/DATA_{suffix}/dark*.fits')
    exposure = [fits.getheader(dark)['EXPOSURE'] for dark in dark_list]
    exposure = np.array(exposure)
    return exposure


def get_mean_dark_MID(dark_data):
    return [np.mean(dark) for dark in dark_data]


def get_std_dark_MID(dark_data):
    return [np.std(dark) for dark in dark_data]


def linear_fit(x, a, b):
    return a * x + b


def fit_in_data(exposure_times, mean_dark_current):
    popt, pcov = curve_fit(linear_fit, exposure_times, mean_dark_current)
    return popt, pcov


def estimate_errors_from_covariance(pcov):
    errors = np.sqrt(np.diag(pcov))
    print("Errors: ", errors)
    return errors


def plot_ds_exp(exposure, mean_dark_MID_list, std_dark_MID_list, temperature):
    fig, ax1 = plt.subplots(1, 1)
    cmap = plt.get_cmap('cividis')
    colors = cmap(np.linspace(0, 1, len(temperature)))

    print('The standard deviation for the suffix {} is {}'.format(temperature, std_dark_MID_list))

    for temp, mean_dark_MID, std_dark_MID, color in zip(temperature, mean_dark_MID_list, std_dark_MID_list, colors):
        ax1.errorbar(exposure, mean_dark_MID, yerr=std_dark_MID, fmt='o', color=color,
                     label=f'{temp} \N{DEGREE SIGN}C')

    ax1.set_xlabel('Exposure Time (s)')
    ax1.set_ylabel('Dark signal (ADU)')
    ax1.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    ticks = np.linspace(-55, 15, 5)
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-55, vmax=15)
                                              ), ax=ax1, pad=0.01, fraction=0.057, shrink=1, aspect=20,
                        extend='both', extendrect=True, extendfrac='auto')

    cbar.set_label('Temperature (\N{DEGREE SIGN}C)')

    plt.tight_layout()
    plt.show()


def plot_dc_temp(temperature, list_m, error_list):
    list_m = [m for m in list_m]
    cmap = plt.get_cmap('cividis')
    colors = cmap(np.linspace(0, 1, len(temperature)))
    fig, ax1 = plt.subplots(1, 1)

    for temp, m, error, color in zip(temperature, list_m, error_list, colors):
        ax1.errorbar(temp, m[0], yerr=error[0], fmt='o', color=color)

    ax1.set_xlabel('Temperature (\N{DEGREE SIGN}C)')
    ax1.set_ylabel('Dark Current (ADU/pix/s)')
    ax1.set_xlim(-60, 20)

    ax1.set_xticks([-60, -50, -40, -30, -20, -10, 0, 10, 20])

    cbar2 = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-55, vmax=15)
                                               ), ax=ax1, pad=0.01, fraction=0.057, shrink=1, aspect=20,
                         extend='both', extendrect=True, extendfrac='auto')
    cbar2.set_label('Temperature (\N{DEGREE SIGN}C)')

    plt.tight_layout()
    plt.show()


def playing_with_plot_dark_current(exposure, mean_dark_MID_list, temperature, list_m, std_dark_MID_list, error_list):
    list_m = [m for m in list_m]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    cmap = plt.get_cmap('cividis')
    colors = cmap(np.linspace(0, 1, len(temperature)))

    for temp, mean_dark_MID, std_dark_MID, color in zip(temperature, mean_dark_MID_list, std_dark_MID_list, colors):
        ax1.errorbar(exposure, mean_dark_MID, yerr=std_dark_MID, fmt='o', color=color,
                     label=f'{temp} \N{DEGREE SIGN}C')

    ax1.set_xlabel('Exposure Time (s)')
    ax1.set_ylabel('Dark signal (ADU)')

    ticks = np.linspace(-55, 15, 5)
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-55, vmax=15)
                                              ), ax=ax1, pad=0.01, fraction=0.057, shrink=1, aspect=20,
                        extend='both', extendrect=True, extendfrac='auto')

    cbar.set_label('Temperature (\N{DEGREE SIGN}C)')

    for temp, m, error, color in zip(temperature, list_m, error_list, colors):
        ax2.errorbar(temp, m[0], yerr=error[0], fmt='o', color=color)

    ax2.set_xlabel('Temperature (\N{DEGREE SIGN}C)')
    ax2.set_ylabel('Dark Current (ADU/pix/s)')
    ax2.set_xlim(-60, 20)

    cbar2 = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-55, vmax=15)
                                               ), ax=ax2, pad=0.01, fraction=0.057, shrink=1, aspect=20,
                         extend='both', extendrect=True, extendfrac='auto')
    cbar2.set_label('Temperature (\N{DEGREE SIGN}C)')

    plt.tight_layout()
    plt.show()
    # plt.savefig('Dark_FFR_T.pdf', bbox_inches='tight')


def main():
    plot_images()
    path = '/Users/u5500483/Documents/GitHub/Paper_I/Results/Images/DARK_DATA/'
    temperature = [-55, -50, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15]
    number_of_pixels = 4194304
    dark_data_list = [get_dark_data(path, suffix) for suffix in temperature]
    exposure_list = [get_exposure_values(path, suffix) for suffix in temperature]
    mean_dark_MID_list = [get_mean_dark_MID(dark_data) for dark_data in dark_data_list]
    std_dark_MID_list = [get_std_dark_MID(dark_data) for dark_data in dark_data_list] / np.sqrt(number_of_pixels)
    print('The standard deviation for the suffix {} is {}'.format(temperature, std_dark_MID_list))
    print('The mean for the suffix {} is {}'.format(temperature, mean_dark_MID_list))

    # Curve fitting
    popts, pcovs = zip(*[fit_in_data(exposure, mean_dark_MID) for exposure, mean_dark_MID
                         in zip(exposure_list, mean_dark_MID_list)])

    # Calculate errors from covariance matrices
    errors_list = [estimate_errors_from_covariance(pcov) for pcov in pcovs]

    plot_ds_exp(exposure_list[0], mean_dark_MID_list, std_dark_MID_list, temperature)
    plot_dc_temp(temperature, popts, errors_list)

    # # Plotting playing_with_plot_dark_current(exposure_list[0], mean_dark_MID_list, temperature, popts,
    # std_dark_MID_list, errors_list)


if __name__ == '__main__':
    main()
