"""
This script takes bias images and does some statistics on them.
Measuring the row to row or column to column variation and prints the results
"""

import matplotlib.pyplot as plt
import numpy as np
from plot_images import plot_images
import astropy.io.fits as fits
import glob

# Define paths for different modes
MODE_HDR = 'HDR'
MODE_FFR = 'FFR'
path = '/Users/u5500483/Documents/GitHub/Paper_I/Results/'
Bias_images_path_hdr = f'/Users/u5500483/Documents/GitHub/Paper_I/Results/Images/Bias_Dark_Frames/Bias_{MODE_HDR}/'
Bias_images_path_ffr = f'/Users/u5500483/Documents/GitHub/Paper_I/Results/Images/Bias_Dark_Frames/Bias_{MODE_FFR}/'


def get_images(path, sense):
    bias_values = []
    list_images = glob.glob(path + '*.fits')
    for filename in list_images:
        bias_values.append(fits.getdata(filename))
    bias_values = np.array(bias_values)
    print(f'The shape of the bias_values is: {bias_values.shape}')
    bias_values = bias_values * sense
    return bias_values


def row_to_row(bias_values):
    row_means = np.mean(bias_values[:, :, :], axis=2)  # Mean along columns for each row
    std_val = np.std(row_means, axis=1)  # 0 for 2048x100, 1 for 100
    return std_val


def column_to_column(bias_values):
    col_means = np.mean(bias_values[:, :, :], axis=1)  # Mean along rows for each column
    std_val = np.std(col_means, axis=1)  # 0 for 2048x100, 1 for 100
    return std_val


def plot_histograms(row_std_hdr, row_std_ffr, col_std_hdr, col_std_ffr):

    plt.subplot(2, 1, 1)
    plt.hist(row_std_hdr, bins=15, alpha=0.8, label='HDR')
    plt.hist(row_std_ffr, bins=15, alpha=0.8, label='FFR')
    # plt.title('Row-to-Row Variation')
    plt.xlabel('Standard Deviation (e$^{-}$)')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.hist(col_std_hdr, bins=15, alpha=0.8, label='HDR')
    plt.hist(col_std_ffr, bins=15, alpha=0.8, label='FFR')
    # plt.title('Column-to-Column Variation')
    plt.xlabel('Standard Deviation (e$^{-}$)')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{path}Row_Column_Variation_2048.png')
    plt.show()


def main():
    plot_images()
    # Load and process HDR images
    bias_values_hdr = get_images(Bias_images_path_hdr, sense=1.131)
    row_std_hdr = row_to_row(bias_values_hdr)
    col_std_hdr = column_to_column(bias_values_hdr)

    print(f'HDR Row-to-Row std mean: {np.mean(row_std_hdr)}')
    print(f'HDR Column-to-Column std mean: {np.mean(col_std_hdr)}')

    # Load and process FFR images
    bias_values_ffr = get_images(Bias_images_path_ffr, sense=0.632)
    row_std_ffr = row_to_row(bias_values_ffr)
    col_std_ffr = column_to_column(bias_values_ffr)

    print(f'FFR Row-to-Row std mean: {np.mean(row_std_ffr)}')
    print(f'FFR Column-to-Column std mean: {np.mean(col_std_ffr)}')

    # Plot histograms
    plot_histograms(row_std_hdr, row_std_ffr, col_std_hdr, col_std_ffr)


if __name__ == '__main__':
    main()


