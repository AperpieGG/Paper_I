from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LogNorm
import warnings
import os
import glob


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


plot_images()


def get_images():
    path_bias_ffr = '/Users/u5500483/Downloads/Paper_I/Images/json/Bias_Dark_Frames/Bias_FFR/'
    path_bias_hdr = '/Users/u5500483/Downloads/Paper_I/Images/json/Bias_Dark_Frames/Bias_HDR/'
    path_dark_ffr = '/Users/u5500483/Downloads/Paper_I/Images/json/Bias_Dark_Frames/Dark_FFR/'
    path_dark_hdr = '/Users/u5500483/Downloads/Paper_I/Images/json/Bias_Dark_Frames/Dark_HDR/'
    path_dark_hdr_disable = '/Users/u5500483/Downloads/Paper_I/Images/json/Bias_Dark_Frames/Dark_HDR/'

    bias_ffr = fits.getdata(path_bias_ffr + 'image_1.fits')
    bias_hdr = fits.getdata(path_bias_hdr + 'image_2.fits')
    dark_ffr = fits.getdata(path_dark_ffr + 'Dark_01.fits')[0]
    dark_hdr = fits.getdata(path_dark_hdr + 'Dark_01.fits')[0]
    dark_hdr_disable = fits.getdata(path_dark_hdr_disable + 'Dark_01_disable.fits')[0]

    return bias_ffr, bias_hdr, dark_ffr, dark_hdr, dark_hdr_disable


def plot_bias_images(bias_ffr, bias_hdr):
    mean_bias_ffr = np.mean(bias_ffr)
    mean_bias_hdr = np.mean(bias_hdr)
    std_bias_ffr = np.std(bias_ffr)
    std_bias_hdr = np.std(bias_hdr)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    im1 = ax1.imshow(bias_ffr, cmap='hot', origin='lower', vmin=mean_bias_ffr - 2 * std_bias_ffr,
                   vmax=mean_bias_ffr + 2 * std_bias_ffr)
    ax1.set_title('FFR, mean = %5.2f [cts]' % mean_bias_ffr, fontsize=14)
    fig.colorbar(im1, ax=ax1, pad=0.01, fraction=0.047, shrink=1, aspect=10, extend='both',
                 extendfrac='auto')
    im2 = ax2.imshow(bias_hdr, cmap='hot', origin='lower', vmin=mean_bias_hdr - 2 * std_bias_hdr,
                   vmax=mean_bias_hdr + 2 * std_bias_hdr)
    ax2.set_title('HDR, mean = %5.2f [cts]' % mean_bias_hdr, fontsize=14)
    fig.colorbar(im2, ax=ax2, pad=0.01, fraction=0.047, shrink=1, aspect=10, extend='both',
                 extendfrac='auto')

    plt.tight_layout()
    plt.show()
    # fig.savefig('Bias_FFR_HDR.pdf', bbox_inches='tight', dpi=300)


def plot_dark_images(dark_ffr, dark_hdr):
    mean_dark_ffr = np.mean(dark_ffr)
    mean_dark_hdr = np.mean(dark_hdr)
    std_dark_ffr = np.std(dark_ffr)
    std_dark_hdr = np.std(dark_hdr)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    im1 = ax1.imshow(dark_ffr, cmap='hot', origin='lower', vmin=mean_dark_ffr - 1 * std_dark_ffr,
                   vmax=mean_dark_ffr + 1 * std_dark_ffr)
    ax1.set_title('FFR, mean = %5.2f [cts]' % mean_dark_ffr, fontsize=14)
    fig.colorbar(im1, ax=ax1, pad=0.01, fraction=0.047, shrink=1, aspect=10, extend='both',
                 extendfrac='auto')
    im2 = ax2.imshow(dark_hdr, cmap='hot', origin='lower', vmin=mean_dark_hdr - 1 * std_dark_hdr,
                   vmax=mean_dark_hdr + 1 * std_dark_hdr)
    ax2.set_title('HDR, mean = %5.2f [cts]' % mean_dark_hdr, fontsize=14)
    fig.colorbar(im2, ax=ax2, pad=0.01, fraction=0.047, shrink=1, aspect=10, extend='both',
                 extendfrac='auto')

    plt.tight_layout()
    plt.show()
    # fig.savefig('Dark_FFR_HDR.pdf', bbox_inches='tight', dpi=300)


plot_dark_images(get_images()[2], get_images()[3])
plot_bias_images(get_images()[0], get_images()[1])


def plot_dark_histogram(dark_hdr, dark_hdr_disable):
    mean_dark_hdr = np.mean(dark_hdr)
    mean_dark_hdr_disable = np.mean(dark_hdr_disable)
    std_dark_hdr = np.std(dark_hdr)
    std_dark_hdr_disable = np.std(dark_hdr_disable)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharey='row',
                            gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [4, 1]})

    axs[0, 0].imshow(dark_hdr, cmap='hot', origin='lower', vmin=mean_dark_hdr - 1 * std_dark_hdr,
                     vmax=mean_dark_hdr + 1 * std_dark_hdr)
    axs[0, 0].set_title('HDR, Anti-Glow [ON], mean = %5.2f [cts/pix/s]' % mean_dark_hdr, fontsize=14)
    axs[0, 1].imshow(dark_hdr_disable, cmap='hot', origin='lower',
                     vmin=mean_dark_hdr_disable - 1 * std_dark_hdr_disable,
                     vmax=mean_dark_hdr_disable + 1 * std_dark_hdr_disable)
    axs[0, 1].set_title('HDR, Anti-Glow [OFF], mean = %5.2f [cts/pix/s]' % mean_dark_hdr_disable, fontsize=14)

    cbar = fig.colorbar(axs[0, 0].imshow(dark_hdr, cmap='hot', origin='lower', vmin=mean_dark_hdr - 1 * std_dark_hdr,
                                         vmax=mean_dark_hdr + 1 * std_dark_hdr), ax=axs[0, 0], pad=0.01, fraction=0.047,
                        shrink=1, aspect=10, extend='both', extendfrac='auto')
    cbar = fig.colorbar(axs[0, 1].imshow(dark_hdr_disable, cmap='hot', origin='lower',
                                         vmin=mean_dark_hdr_disable - 1 * std_dark_hdr_disable,
                                         vmax=mean_dark_hdr_disable + 1 * std_dark_hdr_disable), ax=axs[0, 1], pad=0.01,
                        fraction=0.047, shrink=1, aspect=10, extend='both', extendfrac='auto')

    axs[1, 0].hist(dark_hdr.flatten(), bins=100, log=True)
    axs[1, 0].set_xlabel('Dark Current [cts/pix/s]')
    axs[1, 0].set_ylabel('# of Occurrences [Pixels/bin]')
    axs[1, 0].axvline(x=mean_dark_hdr, color='r', linestyle='--', label='mean = %5.2f [cts/pix/s]' % mean_dark_hdr,
                      linewidth=0.8)
    axs[1, 0].legend(loc='best')
    axs[1, 0].set_xlim(-10, 60)

    axs[1, 1].hist(dark_hdr_disable.flatten(), bins=100, log=True)
    axs[1, 1].set_xlabel('Dark Current [cts/pix/s]')
    axs[1, 1].set_ylabel('# of Occurrences [Pixels/bin]')
    axs[1, 1].axvline(x=mean_dark_hdr_disable, color='r', linestyle='--',
                      label='mean = %5.2f [cts/pix/s]' % mean_dark_hdr_disable, linewidth=0.8)
    axs[1, 1].legend(loc='best')
    axs[1, 1].set_xlim(-10, 60)

    fig.tight_layout()
    plt.show()
    # fig.savefig('Dark_HDR_Glow.pdf', bbox_inches='tight', dpi=300)


plot_dark_histogram(get_images()[3], get_images()[4])
