#! /usr/bin/env python
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


def read_bias_data():
    path_FFR = '/Users/u5500483/Documents/GitHub/Paper_I/Results/Images/Bias_Dark_Frames/Bias_FFR/'
    path_HDR = '/Users/u5500483/Documents/GitHub/Paper_I/Results/Images/Bias_Dark_Frames/Bias_HDR/'
    list_images_ffr = glob.glob(path_FFR + '*.fits')
    list_images_hdr = glob.glob(path_HDR + '*.fits')
    bias_data_ffr = [fits.getdata(image_path) for image_path in list_images_ffr]
    bias_data_hdr = [fits.getdata(image_path) for image_path in list_images_hdr]
    bias_data_ffr = np.array(bias_data_ffr)
    bias_data_hdr = np.array(bias_data_hdr)
    return bias_data_ffr, bias_data_hdr


def find_pixel_coordinates(bias_data, threshold):
    stds_1 = np.std(bias_data, axis=0).reshape(2048, 2048)
    find = np.where(stds_1 > threshold)
    pixel_coordinates = np.array(find).T
    print('Number of noise pixels:', len(pixel_coordinates))

    fig_3 = plt.figure(figsize=(8, 8))
    plt.scatter(pixel_coordinates[:, 1], pixel_coordinates[:, 0], s=1, c='red', label='Pixel coordinates')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    percent = len(pixel_coordinates) / (2048 * 2048) * 100
    print("The pixels with coordinates {} are noise pixels".format(pixel_coordinates))
    plt.ylim(0, 2048)
    plt.xlim(0, 2048)
    plt.title('Noise pixels: ' + str(round(percent, 2)) + '%')
    fig_3.tight_layout()
    plt.show()

    print('Noise pixels:', percent, '%')
    return pixel_coordinates


def create_binary_image(bias_data, threshold):
    stds = np.std(bias_data, axis=0).reshape(2048, 2048)
    binary_image = np.zeros_like(stds)
    binary_image[stds > threshold] = 1
    coordinates = np.array(np.where(binary_image == 1)).T
    print("Number of noise pixels:", len(coordinates))
    for coord in coordinates:
        x, y = coord[0], coord[1]
        value = stds[x, y]
        # print("Pixel at coordinates ({}, {}) has value {:.2f}.".format(x, y, value))
    return binary_image


def plot_binary_image(binary_image):
    fig, ax = plt.subplots()
    im = ax.imshow(binary_image, cmap='Reds', origin='lower', extent=[0, 2048, 0, 2048], vmin=0, vmax=1)
    ax.set_title('Noise pixels: ' + str(round(np.sum(binary_image) / (2048 * 2048) * 100, 5)) + '%', fontsize=12)
    fig.colorbar(im, ax=ax, label='Threshold')
    plt.tight_layout()
    # if args.directory == "HDR":
    #     plt.savefig('Binary_image_HDR.pdf', bbox_inches='tight')
    # elif args.directory == "FFR":
    #     plt.savefig('Binary_image_FFR.pdf', bbox_inches='tight')
    plt.show()


def extract_pixel_time_series(bias_data, coordinates):
    frame_numbers = np.arange(bias_data.shape[0])
    pixel_values = bias_data[:, coordinates[0], coordinates[1]]
    return frame_numbers, pixel_values


def plot_pixel_time_series(frame_numbers, pixel_values, coordinates):
    plt.figure()

    plt.plot(frame_numbers, pixel_values, 'o',
             label="RMS = " + str(round(np.std(pixel_values), 2)) + ' ADU')
    plt.plot(frame_numbers, pixel_values, '-', alpha=0.2)
    plt.xlabel('Frame Number')
    plt.ylabel('Pixel Value (ADU)')
    if args.directory == "HDR":
        plt.title(f'Time Series for Pixel in HDR ({coordinates[0]}, {coordinates[1]})')
    elif args.directory == "FFR":
        plt.title(f'Time Series for Pixel in FFR ({coordinates[0]}, {coordinates[1]})')
    plt.legend(loc='best')
    plt.tight_layout()
    if args.directory == "HDR":  # pass argument to save
        plt.savefig('Time_series_HDR.pdf', bbox_inches='tight')
    elif args.directory == "FFR":
        plt.savefig('Time_series_FFR.pdf', bbox_inches='tight')
    plt.show()


def main(args):
    plot_images()

    if args.directory == "HDR":
        bias_values = read_bias_data()[1]
    elif args.directory == "FFR":
        bias_values = read_bias_data()[0]
    else:
        raise ValueError("Unknown directory. Please use 'HDR' or 'FFR'.")

    binary_image = create_binary_image(bias_values, args.threshold)
    plot_binary_image(binary_image)

    # Find coordinates of pixel with maximum standard deviation
    stds = np.std(bias_values, axis=0).reshape(2048, 2048)
    max_std_index = np.unravel_index(np.argmax(stds), stds.shape)
    print("Coordinates of pixel with maximum standard deviation:", max_std_index)

    # Extract time series for the pixel with maximum standard deviation
    frame_numbers, pixel_values = extract_pixel_time_series(bias_values, max_std_index)
    plot_pixel_time_series(frame_numbers, pixel_values, max_std_index)

    # Print the total number of pixels that exceed the threshold value
    num_noise_pixels = np.sum(stds > args.threshold)
    print(f'Total number of pixels exceeding the threshold value: {num_noise_pixels}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process bias data.')
    parser.add_argument('directory', choices=['HDR', 'FFR'], help='Directory to work with (HDR or FFR)')
    parser.add_argument('threshold', type=float, help='Threshold value for noise pixels')

    args = parser.parse_args()
    main(args)
