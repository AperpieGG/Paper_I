import matplotlib.pyplot as plt
import numpy as np
import json
from scipy import optimize as optimization


def plot_images():
    # plt.rcParams['figure.figsize'] = (10, 8)
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


path = '/Users/u5500483/Documents/GitHub/Paper_I/Results/'
with open(path + 'PTC_FFR_12bit_temp_-25.json') as json_file:
    data = json.load(json_file)
temperature = data["temperature"]
exposure_times = data["exposure_times"]
corrected_counts = data["corrected_counts"]
corrected_variance_sqr = data["corrected_variance_sqr"]
saturation_grey_value = data["saturation_grey_value"]
variance_sqr_saturation_grey = data["variance_sqr_saturation_grey"]
CameraSensitivity = data["CameraSensitivity"]
linearity_gradient = data["Linearitygradient"]
linearity_offset = data["LinearityOffset"]
linearity_error = data["LinearityError"]


def plot_linearity_line(gradient, offset, startx, endx, step, figure, ax1):
    plt.figure(figure)
    x_values = []
    y_values = []

    for x in np.arange(startx, endx, step):
        y = x * gradient + offset  # y = mx + c
        x_values.append(x)
        y_values.append(y)
    ax1.plot(x_values, y_values, 'b-', label='%5.3f $x$ + %5.3f' % (gradient, offset))


def plot_linearity(exposure_times, ExposureTimeList_5_95, Linearitygradient, LinearityOffset, CorrectedCtsList_5_95,
                   ResidualsList_5_95, LinearityError, ResidualsList, corrected_counts, figure):
    startx = (min(ExposureTimeList_5_95))
    endx = (max(ExposureTimeList_5_95))
    step = 0.0001

    figure, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    ax1.set_ylabel('Signal [cts]')
    ax1.plot([exposure_times], [corrected_counts], 'ro')
    plot_linearity_line(Linearitygradient, LinearityOffset, startx, endx, step, figure, ax1)
    ax1.axvline(x=startx, color='b', linestyle='--', linewidth=1)
    ax1.axvline(x=endx, color='b', linestyle='--', linewidth=1)
    ax1.legend(loc='best')

    ax2.plot(exposure_times, ResidualsList, 'ro', linewidth=1, label=' LE = $\\pm$ %5.3f %%' % (LinearityError))
    ax2.plot([startx, endx], [0, 0], 'b-', linewidth=1)
    ax2.set_ylim(-3 * LinearityError, +3 * LinearityError)
    ax2.set_ylabel('Residuals [%]')
    ax2.set_xlabel('Exposure [s]')
    ax2.axvline(x=startx, color='b', linestyle='--', linewidth=1)
    ax2.axvline(x=endx, color='b', linestyle='--', linewidth=1)
    ax2.legend(loc='best')
    plt.tight_layout()
    # figure.savefig('Linearity_FFR_12bit_temp_-25.pdf', bbox_inches='tight')
    plt.show()


def set_best_fit_ranges(xdata, ydata, startx, endx):
    best_fit_xdata = []
    best_fit_ydata = []

    for x, y in zip(xdata, ydata):
        if startx < x < endx:
            best_fit_xdata.append(x)
            best_fit_ydata.append(y)

    return best_fit_xdata, best_fit_ydata


def best_fit(xdata, ydata):
    def func(x, a, b):
        return a * x + b

    Gradient = optimization.curve_fit(func, xdata, ydata)[0][0]
    Offset = optimization.curve_fit(func, xdata, ydata)[0][1]

    print('gradient [{}] offset [{}]'.format(Gradient, Offset))
    return Gradient, Offset


class CreateResiduals():
    def __init__(self, X_values, Y_values, Offset, Gradient, max_val, range_factor):
        self.x = X_values
        self.y = Y_values
        self.offset = Offset
        self.gradient = Gradient
        self.max = max_val
        self.range = range_factor
        self.residuals = []
        self.residual_counts = []

        self.create()

    def create(self):
        print('creating residuals')
        for i in range(0, len(self.y)):
            calculated_level = self.offset + (self.gradient * self.x[i])

            Residuals = (self.y[i] - calculated_level) / (
                    self.range * self.max) * 100  # Equation 35 from EMVA1288-V3.1

            residualscts = (self.y[i] - calculated_level)

            self.residuals.append(Residuals)
            self.residual_counts.append(residualscts)

        print('residuals [{}]'.format(self.residuals))
        print('residual counts [{}]'.format(self.residual_counts))


def find_linearity_error(ResidualsList):
    LinearityError = (max(ResidualsList) - min(ResidualsList)) / 2

    return LinearityError


figure = 2
plot_images()
startx = saturation_grey_value * 0.05
endx = saturation_grey_value * 0.95

CorrectedCtsList_5_95, ExposureTimeList_5_95 = set_best_fit_ranges(corrected_counts, exposure_times, startx, endx)
Linearitygradient, LinearityOffset = best_fit(ExposureTimeList_5_95, CorrectedCtsList_5_95)

range_factor = 0.9

ResidualsList = CreateResiduals(exposure_times, corrected_counts, LinearityOffset, Linearitygradient,
                                saturation_grey_value, range_factor).residuals
ResidualsList_5_95 = CreateResiduals(ExposureTimeList_5_95, CorrectedCtsList_5_95, LinearityOffset, Linearitygradient,
                                     saturation_grey_value, range_factor).residuals

LinearityError = find_linearity_error(ResidualsList_5_95)

plot_linearity(exposure_times, ExposureTimeList_5_95, Linearitygradient, LinearityOffset, CorrectedCtsList_5_95,
               ResidualsList_5_95, LinearityError, ResidualsList, corrected_counts, figure)

print('LinearityError (LE) = +/- {} %'.format(LinearityError))

print('Finished Finding PTC\n')

# Linearity Analysis with Weighted Least Squares starts here

Ssat = saturation_grey_value
Smin = 0.05 * Ssat
Smax = 0.95 * Ssat
index = corrected_counts.index(Ssat)
exp_sat = exposure_times[index]


# exp_min = 0.05 * exp_sat
# exp_max = 0.95 * exp_sat


def perform_linearity_analysis():
    y = np.array(corrected_counts)
    x = np.array(exposure_times)

    weights = (1 / y ** 2)

    p_3 = np.polyfit(x[(y >= Smin) & (y <= Smax)], y[(y >= Smin) & (y <= Smax)], 1,
                     w=weights[(y >= Smin) & (y <= Smax)])
    fit_3 = np.poly1d(p_3)
    residuals_3 = (y - fit_3(x)) / fit_3(x) * 100
    NL_3 = np.mean(np.abs(residuals_3[(y >= Smin) & (y <= Smax)]))
    print('Linearity error with weights = +/- {} % '.format(NL_3))

    fig, (ax3, ax6) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    exp_min, exp_max = plot_main_figure(ax3, fit_3)
    plot_linearity_3_residuals(ax6, residuals_3, NL_3, exp_min, exp_max)

    fig.tight_layout()
    plt.show()
    fig.savefig('Linearity_FFR_12bit_temp_-25_W.pdf', bbox_inches='tight')

    return fit_3, residuals_3, NL_3


def plot_main_figure(ax, fit_3):
    corrected_counts = np.array(data["corrected_counts"])
    exposure_times = np.array(data["exposure_times"])
    exp_min = np.min(exposure_times[(corrected_counts >= Smin) & (corrected_counts <= Smax)])
    exp_max = np.max(exposure_times[(corrected_counts >= Smin) & (corrected_counts <= Smax)])
    ax.plot(exposure_times, corrected_counts, 'ro', label='FFR mode')
    ax.plot(exposure_times[(corrected_counts >= Smin) & (corrected_counts <= Smax)],
            fit_3(exposure_times[(corrected_counts >= Smin) & (corrected_counts <= Smax)]), 'b-')
    # label='%5.3f $x$ + %5.3f' % (fit_3[1], fit_3[0]))
    ax.set_ylabel('Signal (ADU)')
    ax.axvline(x=exp_min, color='b', linestyle='--', linewidth=1)
    ax.axvline(x=exp_max, color='b', linestyle='--', linewidth=1)
    ax.legend(loc='best')
    # ax.set_yticks([10000, 20000, 30000, 40000, 50000, 60000])

    return exp_min, exp_max


def plot_linearity_3_residuals(ax, residuals_3, NL_3, exp_min, exp_max):
    ax.plot(exposure_times, residuals_3, 'ro', label=' LE = %5.3f %%' % (NL_3))
    ax.plot([exp_min, exp_max], [0, 0], 'b-', linewidth=1)
    ax.set_xlabel('Exposure time (s)')
    ax.set_ylabel('Residuals (%)')
    ax.set_ylim(-5 * NL_3, +5 * NL_3)
    ax.axvline(x=exp_min, color='b', linestyle='--', linewidth=1)
    ax.axvline(x=exp_max, color='b', linestyle='--', linewidth=1)
    # ax.legend(loc='best')


fit_3, residuals_3, NL_3 = perform_linearity_analysis()
