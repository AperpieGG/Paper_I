#! /usr/bin/env python

"""
This script reads the bias images from the given directory. It then plots
the read noise 2-D histogram and the histogram of the read noise values.
It will print the median, mean, and RMS of the read noise values.
"""

import argparse
import glob
import os
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch


def plot_images():
    """
    Configures the plot settings to ensure a consistent style and appearance for all plots.
    """
    plt.rcParams['figure.dpi'] = 100
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


def get_sensitivity(mode):
    """
    Returns the sensitivity value based on the given mode.

    Parameters:
    mode (str): The mode of operation, either 'HDR' or 'FFR'.

    Returns:
    float: The sensitivity value.

    Raises:
    ValueError: If an invalid mode is provided.
    """
    if mode == 'HDR':
        return 1.12
    elif mode == 'FFR':
        return 0.63
    else:
        raise ValueError(f"Invalid mode: {mode}")


def read_bias_data(directory):
    """
    Reads the bias images from the specified directory, calculates the mean and standard deviation
    of the bias values, and returns these along with the directory mode.

    Parameters:
    directory (str): The mode of operation, either 'HDR' or 'FFR'.

    Returns:
    tuple: A tuple containing the mean values, standard deviation values, and the directory mode.
    """
    sensitivity = get_sensitivity(directory)
    print(f'Calculating the read noise with sensitivity of {sensitivity}')

    path = (f'/Users/u5500483/Documents/GitHub/Paper_I/Results/'
            f'Images/Bias_Dark_Frames/Bias_{directory}/')

    list_images = glob.glob(path + '*.fits')
    bias_values = []

    for image_path in list_images:
        hdulist = fits.open(image_path)
        image_data = hdulist[0].data  # * sensitivity
        hdulist.close()
        bias_values.append(image_data)

    bias_values = np.array(bias_values)

    value_mean = np.mean(bias_values, axis=0).flatten()
    value_std = np.std(bias_values, axis=0).flatten()
    print('The value_mean and value_std are [{}] and [{}]'.format(value_mean, value_std))
    return value_mean, value_std, directory


def plot_read_noise(value_mean, value_std, directory):
    """
    Plots the 2-D histogram of the mean and standard deviation of the read noise values,
    and also plots a histogram of the standard deviation values.

    Parameters:
    value_mean (numpy.ndarray): The mean values of the read noise.
    value_std (numpy.ndarray): The standard deviation values of the read noise.
    directory (str): The mode of operation, either 'HDR' or 'FFR'.
    """
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0, hspace=0)

    ax1 = plt.subplot(gs[0])
    print(f'Type of value_mean is {type(value_mean)}, shape is {value_mean.shape}')
    print(f'Type of value_std is {type(value_std)}, shape is {value_std.shape}')
    hb = ax1.hist2d(value_mean, value_std, bins=100, cmap='cividis', norm=LogNorm())
    ax1.set_xlabel('Mean (ADU)')
    ax1.set_ylabel('RMS (ADU)')
    ax1.legend(loc='best')
    legend_elements = [Patch(facecolor='none', edgecolor='darkblue', label=f'{directory} Mode')]
    ax1.legend(handles=legend_elements, loc='upper left')

    ax2 = plt.subplot(gs[1])
    ax2.hist(value_std, bins=100, orientation='horizontal', color='b', histtype='step')
    ax2.set_xlabel('Number of pixels')
    ax2.set_xticks([1e0, 1e3, 1e5])
    ax2.set_xticks([1e1, 1e2, 1e4, 1e5, 1e7], minor=True)
    ax2.set_xticklabels([], minor=True)
    ax2.set_xscale('log')

    value_median_hist = np.median(value_std)
    print('Value Median = ', value_median_hist)
    value_mean_hist = np.mean(value_std)
    print('Value Mean = ', value_mean_hist)
    RMS = np.sqrt(np.mean(value_std ** 2))
    print('RMS = ', RMS)
    ax2.axhline(value_median_hist, color='g', linestyle=':')
    ax2.yaxis.set_ticklabels([])

    y_min, y_max = ax1.get_ylim()
    ax2.set_ylim(y_min, y_max)

    fig.colorbar(ax1.get_children()[0], ax=ax2, label='Number of pixels')
    fig.tight_layout()

    if directory == "HDR":  # pass argument to save
        plt.savefig('RN_HDR.pdf', bbox_inches='tight')
    elif directory == "FFR":
        plt.savefig('RN_FFR.pdf', bbox_inches='tight')
    plt.show()


def main():
    """
    Main function to parse command-line arguments, read bias data, and plot read noise.
    """
    plot_images()
    parser = argparse.ArgumentParser(description='Plot read noise from FITS files.')
    parser.add_argument('directory', choices=['HDR', 'FFR'], help='Specify the directory mode (HDR or FFR).')
    args = parser.parse_args()

    value_mean, value_std, directory = read_bias_data(args.directory)
    plot_read_noise(value_mean, value_std, directory)
    print('Saved in path: ', os.getcwd())


if __name__ == "__main__":
    main()
