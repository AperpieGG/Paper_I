import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json
from plot_images import plot_images

plot_images()

# Load data from JSON file
with open('dc_vs_temp.json') as json_file:
    data = json.load(json_file)

# Extract temperature and dark current data
temperature = np.array(data["temperature"])
dc = np.array(data["slopes"])

# Convert temperatures from Celsius to Kelvin
temperature_kelvin = temperature + 273.15

# Calculate 1000 / T in Kelvin^-1
inverse_T = 1000 / temperature_kelvin


# Define the dark current model (Arrhenius-like)
def dark_current_model(T, I_d0, E_a):
    k = 8.617e-5  # Boltzmann constant in eV/K
    return I_d0 * np.exp(-E_a / (k * T))


# Estimate initial guesses based on the range of the data
I_d0_guess = np.max(dc)  # The maximum dark current value in your data
E_a_guess = 0.08  # An average value for activation energy in semiconductors

# Split the data into two segments
temp_segment_1 = temperature_kelvin[temperature >= 0]
dc_segment_1 = dc[temperature >= 0]

temp_segment_2 = temperature_kelvin[temperature < 0]
dc_segment_2 = dc[temperature < 0]

# Fit the model to each segment with better initial guesses
popt1, _ = curve_fit(dark_current_model, temp_segment_1, dc_segment_1, p0=(I_d0_guess, E_a_guess))
popt2, _ = curve_fit(dark_current_model, temp_segment_2, dc_segment_2, p0=(I_d0_guess, E_a_guess))

# Extract the fitted parameters
I_d0_1, E_a_1 = popt1
I_d0_2, E_a_2 = popt2

print(f"Segment 1 - Fitted I_d0: {I_d0_1:.3e} e^-/pixel/sec, Fitted E_a: {E_a_1:.3f} eV")
print(f"Segment 2 - Fitted I_d0: {I_d0_2:.3e} e^-/pixel/sec, Fitted E_a: {E_a_2:.3f} eV")

# Generate fitted values for each segment
fit_segment_1 = dark_current_model(temp_segment_1, *popt1)
fit_segment_2 = dark_current_model(temp_segment_2, *popt2)

# Plot the data and the fitted models
plt.figure()

# Plot the original data
plt.scatter(inverse_T, dc, label='Data', color='black')

# Plot the fitted models
plt.plot(1000 / temp_segment_1, fit_segment_1, label='Fitted model (T >= 0°C)', color='blue')
plt.plot(1000 / temp_segment_2, fit_segment_2, label='Fitted model (T < 0°C)', color='red')

plt.yscale('log')
plt.xlabel(r'1000/T(K$^{-1}$)')
plt.ylabel('Dark Current (e^-/pixel/sec)')
plt.legend()
plt.tight_layout()
plt.show()
