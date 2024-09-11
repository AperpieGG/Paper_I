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
Bias_images_path_ffr = f'/Users/u5500483/Documents/GitHub/Paper_I/Results/Images/Bias_Dark_Frames/Bias_{MODE_FFR}/'
Bias_images_path_hdr = f'/Users/u5500483/Documents/GitHub/Paper_I/Results/Images/Bias_Dark_Frames/Bias_{MODE_HDR}/'


def get_images(path, sense):
    bias_values = []
    list_images = glob.glob(path + '*.fits')
    for filename in list_images:
        bias_values.append(fits.getdata(filename))
    bias_values = np.array(bias_values)
    print(f'The shape of the bias_values is: {bias_values.shape}')
    bias_values = bias_values * sense

    return bias_values


def row_to_row_odd(bias_values):
    row_means = np.mean(bias_values[:, 1::2, :], axis=2)  # Mean along columns for each row
    std_val = np.std(row_means, axis=0)  # 0 for 2048x100, 1 for 100
    return std_val


def row_to_row_even(bias_values):
    row_means = np.mean(bias_values[:, ::2, :], axis=2)  # Mean along rows for each column
    std_val = np.std(row_means, axis=0)  # 0 for 2048x100, 1 for 100
    return std_val


def row_to_row(bias_values):
    row_means = np.mean(bias_values[:, :, :], axis=2)  # Mean along rows for each column
    std_val = np.std(row_means, axis=0)  # 0 for 2048x100, 1 for 100
    return std_val


def row_to_row_hdr(bias_values):
    row_means = np.mean(bias_values[:, :, :], axis=2)  # Mean along rows for each column
    std_val = np.std(row_means, axis=0)  # 0 for 2048x100, 1 for 100
    return std_val


def plot_histograms(row_std_odd, row_std_even, row_row_ffr, row_row_hdr):

    # plt.hist(row_row_hdr, bins=50, alpha=0.5, label='HDR')
    plt.hist(row_row_ffr, bins=50, alpha=0.5, label='FFR')
    plt.hist(row_std_odd, bins=50, alpha=0.5, label='FFR ODD')
    plt.hist(row_std_even, bins=50, alpha=0.5, label='FFR EVEN')
    # plt.title('Row-to-Row Variation')
    plt.xlabel('Standard Deviation (e$^{-}$)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{path}FFR_Row_to_Row_Histogram.png')
    plt.show()


def save_fits_image(image, filename):
    hdu = fits.PrimaryHDU(image)
    hdu.writeto(filename, overwrite=True)


def reshape_and_save_to_fits(array, filename):
    """
    Reshape a 1D numpy array to a 2D array where each row is the original 1D array,
    rotate it by 90 degrees, and save it to a FITS file.

    Parameters:
    array (numpy.ndarray): The 1D array to reshape and rotate.
    filename (str): The filename for the FITS file.
    """
    if array.ndim != 1:
        raise ValueError("Input array must be 1D.")

    # Reshape the 1D array to a 2D array
    reshaped_array = np.tile(array, (array.shape[0], 1))

    # Rotate the 2D array by 90 degrees counterclockwise
    rotated_array = np.rot90(reshaped_array)

    # Create a Primary HDU object
    hdu = fits.PrimaryHDU(rotated_array)

    # Create an HDU list and write to a file
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(filename, overwrite=True)

    print(f'Successfully saved rotated array to {filename}')

    return filename


def plot_fits_image(filename):
    """
    Open a FITS file and plot the image using imshow with a colorbar.

    Parameters:
    filename (str): The filename of the FITS file to plot.
    """
    # Open the FITS file
    with fits.open(filename) as hdul:
        # Get the data from the primary HDU
        data = hdul[0].data

    # Set the colorbar limits
    vmin = 0.25
    vmax = 0.55

    # Plot the image
    plt.figure(figsize=(10, 8))
    plt.imshow(data, cmap='hot', origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Pixel Value (e$^{-}$)')
    plt.title('FFR Row-to-Row Map for Odd and Even Rows')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.savefig(f'{path}FFR_Row_to_Row_odd_even_Map.png')
    plt.show()


def main():
    plot_images()
    # Load and process FFR images
    bias_values_ffr = get_images(Bias_images_path_ffr, sense=0.632)
    bias_values_hdr = get_images(Bias_images_path_hdr, sense=1.131)
    row_std_odd = row_to_row_odd(bias_values_ffr)
    row_std_even = row_to_row_even(bias_values_ffr)
    row_std_ffr = row_to_row(bias_values_ffr)
    row_std_hdr = row_to_row_hdr(bias_values_hdr)

    print(f'FFR Row-to-Row ffr odd std mean: {np.mean(row_std_odd)}')
    print(f'FFR Row-to-Row ffr even std mean: {np.mean(row_std_even)}')
    print(f'FFR Row-to-Row ffr std mean: {np.mean(row_std_ffr)}')
    print(f'FFR Row-to-Row hdr std mean: {np.mean(row_std_hdr)}')

    # Plot histograms
    plot_histograms(row_std_odd, row_std_even, row_std_ffr, row_std_hdr)

    odd_even = np.concatenate((row_std_odd, row_std_even), axis=0)
    # # Save the row to row map
    filename = reshape_and_save_to_fits(odd_even, 'ODD_EVEN_ROW_map.fits')

    plot_fits_image(filename)


if __name__ == '__main__':
    main()


