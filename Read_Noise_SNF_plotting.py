#!/usr/bin/env python

"""
This script reads the bias data from the given directory. It will then plot
the read noise 2-D histogram and the histogram of the read noise values.
Sensitivities are set for Marana 1041
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
    plt.rcParams['font.size'] = 10

    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.8
    plt.rcParams['legend.loc'] = 'best'
    plt.rcParams['legend.fancybox'] = True
    plt.rcParams['legend.fontsize'] = 10


def get_sensitivity(mode):
    if mode == 'HDR':
        return 1.18
    elif mode == 'FFR':
        return 0.6
    else:
        raise ValueError(f"Invalid mode: {mode}")


def read_bias_data(directory, snf_mode):
    sensitivity = get_sensitivity(directory)
    print(f'Calculating the read noise with sensitivity of {sensitivity}')

    path = (f'/Users/u5500483/Documents/GitHub/Paper_I/Results/'
            f'Images/Bias_Dark_Frames/Bias_{directory}_{snf_mode}/')

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
    print(f'The value_mean and value_std for {directory} {snf_mode} are [{value_mean}] and [{value_std}]')
    return value_mean, value_std


def plot_read_noise(value_mean_on, value_std_on, value_mean_off, value_std_off, directory):
    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(2, 4, width_ratios=[2, 1, 0.2, 0.05], height_ratios=[1, 1], wspace=0.0, hspace=0.3)

    # HDR SNF ON
    ax1 = plt.subplot(gs[0, 0])
    hb_on = ax1.hist2d(value_mean_on, value_std_on, bins=100, cmap='cividis', norm=LogNorm())
    print('The read noise for SNF ON is:', np.median(value_std_on))
    ax1.set_xlabel('Mean (ADU)')
    ax1.set_ylabel('RMS (ADU)')
    ax1.set_ylim(0, 25)
    ax1.set_xlim(90, 110)
    legend_elements = [Patch(facecolor='none', edgecolor='darkblue', label=f'{directory} Mode SNF ON')]
    ax1.legend(handles=legend_elements, loc='upper left')

    ax2 = plt.subplot(gs[0, 1])
    ax2.hist(value_std_on, bins=100, orientation='horizontal', color='b', histtype='step')
    ax2.set_xlabel('Number of pixels')
    ax2.set_xticks([1e0, 1e3, 1e5])
    ax2.set_xscale('log')
    ax2.axhline(np.median(value_std_on), color='g', linestyle=':')

    y_min, y_max = ax1.get_ylim()
    ax2.set_ylim(y_min, y_max)
    ax2.yaxis.set_ticklabels([])

    # HDR SNF OFF
    ax3 = plt.subplot(gs[1, 0])
    hb_off = ax3.hist2d(value_mean_off, value_std_off, bins=100, cmap='cividis', norm=LogNorm())
    print('The read noise for SNF 0FF is:', np.median(value_std_off))
    ax3.set_xlabel('Mean (ADU)')
    ax3.set_ylabel('RMS (ADU)')
    ax3.set_ylim(0, 25)
    ax3.set_xlim(90, 110)
    legend_elements = [Patch(facecolor='none', edgecolor='darkblue', label=f'{directory} Mode SNF OFF')]
    ax3.legend(handles=legend_elements, loc='upper left')

    ax4 = plt.subplot(gs[1, 1])
    ax4.hist(value_std_off, bins=100, orientation='horizontal', color='b', histtype='step')
    ax4.set_xlabel('Number of pixels')
    ax4.set_xticks([1e0, 1e3, 1e5])
    ax4.set_xscale('log')
    ax4.axhline(np.median(value_std_off), color='g', linestyle=':')

    y_min, y_max = ax3.get_ylim()
    ax4.set_ylim(y_min, y_max)
    ax4.yaxis.set_ticklabels([])

    # Empty subplot to create space
    ax_empty1 = plt.subplot(gs[0, 2])
    ax_empty2 = plt.subplot(gs[1, 2])
    ax_empty1.axis('off')
    ax_empty2.axis('off')

    cbar = fig.colorbar(ax1.get_children()[0], ax=ax2, label='Number of pixels')
    cbar = fig.colorbar(ax3.get_children()[0], ax=ax4, label='Number of pixels')

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
    plt.savefig(f'RN_{directory}_SNF_ON_OFF.pdf', bbox_inches='tight')
    plt.show()


def main():
    plot_images()
    parser = argparse.ArgumentParser(description='Plot read noise from FITS files.')
    parser.add_argument('directory', choices=['HDR', 'FFR'], help='Specify the directory mode (HDR or FFR).')
    args = parser.parse_args()

    value_mean_on, value_std_on = read_bias_data(args.directory, 'SNF_ON')
    value_mean_off, value_std_off = read_bias_data(args.directory, 'SNF_OFF')
    plot_read_noise(value_mean_on, value_std_on, value_mean_off, value_std_off, args.directory)
    print('Saved in path: ', os.getcwd())


if __name__ == "__main__":
    main()
