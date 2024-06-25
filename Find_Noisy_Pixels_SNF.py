#!/usr/bin/env python

"""
This script reads the bias data from the given directory.
It then plots the binary image of the noise pixels and the time series of the most noisy pixel.
The noise pixels will be the ones that are equal or above the threshold value.

Arguments:
    camera mode: str
        mode to work with (HDR or FFR)
    threshold: float
        Threshold value for noise pixels
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
import argparse


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
    plt.rcParams['font.size'] = 12

    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.8
    plt.rcParams['legend.loc'] = 'best'
    plt.rcParams['legend.fancybox'] = True
    plt.rcParams['legend.fontsize'] = 12


def read_bias_data(directory):
    base_path = '/Users/u5500483/Documents/GitHub/Paper_I/Results/Images/Bias_Dark_Frames/'
    path_off = f'{base_path}Bias_{directory}_SNF_OFF/'
    path_on = f'{base_path}Bias_{directory}_SNF_ON/'

    list_images_off = glob.glob(path_off + '*.fits')
    list_images_on = glob.glob(path_on + '*.fits')

    bias_data_off = [fits.getdata(image_path) for image_path in list_images_off]
    bias_data_on = [fits.getdata(image_path) for image_path in list_images_on]

    bias_data_off = np.array(bias_data_off)
    bias_data_on = np.array(bias_data_on)

    return bias_data_off, bias_data_on


def find_pixel_coordinates(bias_data, threshold, snf_state):
    stds_1 = np.std(bias_data, axis=0).reshape(2048, 2048)
    find = np.where(stds_1 > threshold)
    pixel_coordinates = np.array(find).T
    print(f'Number of noise pixels with SNF {snf_state}:', len(pixel_coordinates))
    return pixel_coordinates


def create_binary_image(bias_data, threshold):
    stds = np.std(bias_data, axis=0).reshape(2048, 2048)
    binary_image = np.zeros_like(stds)
    binary_image[stds > threshold] = 1
    coordinates = np.array(np.where(binary_image == 1)).T
    return binary_image


def plot_binary_images(binary_image_off, binary_image_on):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # SNF OFF
    im1 = axes[0].imshow(binary_image_off, cmap='Reds', origin='lower', extent=[0, 2048, 0, 2048], vmin=0, vmax=1)
    axes[0].set_title(
        'Noise pixels with SNF OFF: ' + str(round(np.sum(binary_image_off) / (2048 * 2048) * 100, 5)) + '%',
        fontsize=12)
    fig.colorbar(im1, ax=axes[0], label='Threshold')

    # SNF ON
    im2 = axes[1].imshow(binary_image_on, cmap='Reds', origin='lower', extent=[0, 2048, 0, 2048], vmin=0, vmax=1)
    axes[1].set_title('Noise pixels with SNF ON: ' + str(round(np.sum(binary_image_on) / (2048 * 2048) * 100, 5)) + '%',
                      fontsize=12)
    fig.colorbar(im2, ax=axes[1], label='Threshold')

    plt.tight_layout()
    if args.directory == "HDR":
        plt.savefig('Binary_image_HDR_SNF.pdf', bbox_inches='tight')
    elif args.directory == "FFR":
        plt.savefig('Binary_image_FFR_SNF.pdf', bbox_inches='tight')
    plt.show()


def extract_pixel_time_series(bias_data, coordinates):
    frame_numbers = np.arange(bias_data.shape[0])
    pixel_values = bias_data[:, 0, coordinates[0], coordinates[1]]
    return frame_numbers, pixel_values


def plot_pixel_time_series(frame_numbers_off, pixel_values_off, frame_numbers_on, pixel_values_on, max_std_index_off,
                           max_std_index_on):
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))

    # SNF OFF
    axes[0].plot(frame_numbers_off, pixel_values_off, 'o',
                 label="SNF OFF: RMS = " + str(round(np.std(pixel_values_off), 2)) + ' ADU')
    axes[0].plot(frame_numbers_off, pixel_values_off, '-', alpha=0.2)
    axes[0].set_xlabel('Frame Number')
    axes[0].set_ylabel('Pixel Value (ADU)')
    axes[0].set_title(f'Time Series for Pixel {max_std_index_off}) with SNF OFF')
    axes[0].legend(loc='best')

    # SNF ON
    axes[1].plot(frame_numbers_on, pixel_values_on, 'o',
                 label="SNF ON: RMS = " + str(round(np.std(pixel_values_on), 2)) + ' ADU')
    axes[1].plot(frame_numbers_on, pixel_values_on, '-', alpha=0.2)
    axes[1].set_xlabel('Frame Number')
    axes[1].set_ylabel('Pixel Value (ADU)')
    axes[1].set_title(f'Time Series for Pixel {max_std_index_on} with SNF ON')
    axes[1].legend(loc='best')

    plt.tight_layout()
    if args.directory == "HDR":  # pass argument to save
        plt.savefig('Time_series_HDR_SNF.pdf', bbox_inches='tight')
    elif args.directory == "FFR":
        plt.savefig('Time_series_FFR_SNF.pdf', bbox_inches='tight')
    plt.show()


def main(args):
    plot_images()

    bias_values_off, bias_values_on = read_bias_data(args.directory)

    binary_image_off = create_binary_image(bias_values_off, args.threshold)
    binary_image_on = create_binary_image(bias_values_on, args.threshold)
    plot_binary_images(binary_image_off, binary_image_on)

    # Find coordinates of pixel with maximum standard deviation
    stds_off = np.std(bias_values_off, axis=0).reshape(2048, 2048)
    stds_on = np.std(bias_values_on, axis=0).reshape(2048, 2048)

    max_std_index_off = np.unravel_index(np.argmax(stds_off), stds_off.shape)
    max_std_index_on = np.unravel_index(np.argmax(stds_on), stds_on.shape)

    print(f"Coordinates of pixel with maximum standard deviation with SNF OFF: {max_std_index_off}")
    print(f"Coordinates of pixel with maximum standard deviation with SNF ON: {max_std_index_on}")

    find_pixel_coordinates(bias_values_off, args.threshold, 'OFF')
    find_pixel_coordinates(bias_values_on, args.threshold, 'ON')

    # Extract time series for the pixel with maximum standard deviation
    frame_numbers_off, pixel_values_off = extract_pixel_time_series(bias_values_off, max_std_index_off)
    frame_numbers_on, pixel_values_on = extract_pixel_time_series(bias_values_on, max_std_index_on)
    plot_pixel_time_series(frame_numbers_off, pixel_values_off, frame_numbers_on, pixel_values_on, max_std_index_off,
                           max_std_index_on)

    # Print the total number of pixels that exceed the threshold value
    num_noise_pixels_off = np.sum(stds_off > args.threshold)
    num_noise_pixels_on = np.sum(stds_on > args.threshold)

    print(f'Total number of pixels exceeding the threshold value with SNF OFF: {num_noise_pixels_off}')
    print(f'Total number of pixels exceeding the threshold value with SNF ON: {num_noise_pixels_on}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process bias data.')
    parser.add_argument('directory', choices=['HDR', 'FFR'], help='Directory to work with (HDR or FFR)')
    parser.add_argument('threshold', type=float, help='Threshold value for noise pixels')

    args = parser.parse_args()
    main(args)
