import matplotlib.pyplot as plt
from astropy.io import fits
import os
from matplotlib import patches
import matplotlib.colors as mcolors
import numpy as np
from plot_images import plot_images

default_blue_color = mcolors.to_rgba('#1f77b4')
default_orange_color = mcolors.to_rgba('#ff7f0e')


def Calculate_prnu():
    PRNU_FFR_imagects = read_fits_prnu_FFR()
    PRNU_HDR_imagects = read_fits_prnu_HDR()
    plot_prnu(PRNU_FFR_imagects, PRNU_HDR_imagects)


def read_fits_prnu_FFR():
    prnu_FFR_path = '/Users/u5500483/Downloads/home/ops/cmos_marana/Marana_updated_compare/Testing_CMOS_Linux/Images/json/PRNU_FFR_imagemapcts.fits'
    prnu_FFR = fits.getdata(prnu_FFR_path)
    return prnu_FFR


def read_fits_prnu_HDR():
    prnu_HDR_path = '/Users/u5500483/Downloads/home/ops/cmos_marana/Marana_updated_compare/Testing_CMOS_Linux/Images/json/PRNU_HDR_imagemapcts.fits'
    prnu_HDR = fits.getdata(prnu_HDR_path)
    return prnu_HDR


def plot_prnu(prnu_FFR, prnu_HDR):
    plot_images()
    binwidth = 10
    figure1 = 1
    HistList1, bins1, MaxNoOccurs1, x = create_histogram(prnu_HDR, figure1, binwidth)
    HistList1, bins1, MaxNoOccurs1, x = create_histogram_2(prnu_FFR, figure1, binwidth)

    plot_filename = os.path.join('prnu_FFR_HDR.pdf')

    if os.path.exists(plot_filename):
        os.remove(plot_filename)

    plt.figure(1)
    plt.hist(HistList1, bins1, log=True, alpha=0.1)
    plt.hist(HistList1, bins1, log=True, alpha=0.1)
    plt.xlabel('Deviation from the mean (ADU)')
    plt.ylabel('# of Occurrences (pixels/bin)')

    extra_label_log = 'FFR mode'  # extra_label_log = 'PRNU FFR = 0.294 %'
    extra_label_FFR = 'HDR mode'  # extra_label_log = 'PRNU HDR = 0.31 %'

    extra_handle_FFR = patches.Patch(facecolor=default_blue_color, edgecolor='none', label=extra_label_FFR)
    extra_handle_log = patches.Patch(facecolor=default_orange_color, edgecolor='none', label=extra_label_log)

    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(extra_handle_log)
    handles.append(extra_handle_FFR)

    plt.legend(handles=handles, loc='best')
    plt.tight_layout()

    plt.savefig(plot_filename)
    plt.close(1)


def Calculate_dsnu():
    DSNUimage_FFR_mean, DSNUstd_FFR_value = read_fits_dsnu_FFR()
    DSNUimage_HDR_mean, DSNUstd_HDR_value = read_fits_dsnu_HDR()
    plot_dsnu(DSNUimage_HDR_mean, DSNUimage_FFR_mean, DSNUstd_FFR_value, DSNUstd_HDR_value)


def read_fits_dsnu_FFR():
    dsnu_FFR_path = '/Users/u5500483/Downloads/home/ops/cmos_marana/Marana_updated_compare/Testing_CMOS_Linux/Images/json/DSNU_FFR_imagects.fits'
    dsnu_FFR = fits.getdata(dsnu_FFR_path)
    DSNUimage_FFR_mean = dsnu_FFR
    DSNUstd_FFR_value = np.std(dsnu_FFR)
    return DSNUimage_FFR_mean, DSNUstd_FFR_value


def read_fits_dsnu_HDR():
    dsnu_HDR_path = '/Users/u5500483/Downloads/home/ops/cmos_marana/Marana_updated_compare/Testing_CMOS_Linux/Images/json/DSNU_HDR_imagects.fits'
    dsnu_HDR = fits.getdata(dsnu_HDR_path)
    DSNUimage_HDR_mean = dsnu_HDR
    DSNUstd_HDR_value = np.std(dsnu_HDR)
    return DSNUimage_HDR_mean, DSNUstd_HDR_value


def plot_dsnu(DSNUimage_HDR_mean, DSNUimage_FFR_mean, DSNUstd_FFR_value, DSNUstd_HDR_value):
    plot_images()

    binwidth = 1
    figure2 = 1

    HistList1, bins1, MaxNoOccurs1, x = create_histogram_2(DSNUimage_FFR_mean, figure2, binwidth)
    HistList1, bins1, MaxNoOccurs1, x = create_histogram(DSNUimage_HDR_mean, figure2, binwidth)

    plot_filename = os.path.join('dsnu_FFR_HDR.pdf')

    if os.path.exists(plot_filename):
        os.remove(plot_filename)

    plt.figure(1)
    plt.hist(HistList1, bins1, log=True, alpha=0.01)
    plt.hist(HistList1, bins1, log=True, alpha=0.01)

    plt.xlabel('Deviation from the mean (ADU)')
    plt.ylabel('# of Occurrences (pixels/bin)')
    plt.xlim(-10, 10)

    extra_label_log = 'HDR mode'
    extra_label_FFR = 'FFR mode'

    # extra_label_log = 'DSNU HDR = %5.3f' % (DSNUstd_HDR_value) + ' cts' + ' // ' + '%5.3f' % (
    #         DSNUstd_HDR_value * 1.12) + ' e$^-$'
    # extra_label_FFR = 'DSNU FFR = %5.3f' % (DSNUstd_FFR_value) + ' cts' + ' // ' + '%5.3f' % (
    #         DSNUstd_FFR_value * 0.63) + ' e$^-$'

    extra_handle_log = patches.Patch(facecolor=default_blue_color, edgecolor='none', label=extra_label_log)
    extra_handle_FFR = patches.Patch(facecolor=default_orange_color, edgecolor='none', label=extra_label_FFR)

    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(extra_handle_FFR)
    handles.append(extra_handle_log)

    plt.legend(handles=handles, loc='best')
    plt.tight_layout()

    plt.savefig(plot_filename)
    plt.close(1)


def create_histogram_2(image, figure1, binwidth):
    yaxis = image.shape[0]
    xaxis = image.shape[1]

    HistList = []
    for x in range(0, xaxis):
        for y in range(0, yaxis):
            pixvalue = image[y, x]
            HistList.append(image[y, x])

    _max = np.max(image)
    _min = np.min(image)

    if _max >= 1000 or _min <= -1000:
        binwidth = 100

    bins = np.arange(_min, _max + binwidth, binwidth)

    plt.figure(figure1)

    MaxNoOccurs, x, _ = plt.hist(HistList, bins, log=True, color=default_orange_color)

    return (HistList, bins, MaxNoOccurs, x)


def create_histogram(image, figure1, binwidth):
    yaxis = image.shape[0]
    xaxis = image.shape[1]

    HistList = []
    for x in range(0, xaxis):
        for y in range(0, yaxis):
            pixvalue = image[y, x]
            HistList.append(image[y, x])

    _max = np.max(image)
    _min = np.min(image)

    if _max >= 1000 or _min <= -1000:
        binwidth = 100

    bins = np.arange(_min, _max + binwidth, binwidth)

    plt.figure(figure1)

    MaxNoOccurs, x, _ = plt.hist(HistList, bins, log=True, color=default_blue_color)

    return (HistList, bins, MaxNoOccurs, x)


# Calculate_prnu()
Calculate_dsnu()
