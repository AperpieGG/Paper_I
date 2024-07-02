import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve
import json
from plot_images import plot_images

plot_images()

path = '/Users/u5500483/Documents/GitHub/Paper_I/Results/'
# Load data from JSON file
with open(path + 'DC_vs_T_FFR_Marana.json') as json_file:
    data = json.load(json_file)

# Extract temperature and dark current data
temperature = np.array(data["temperature"])
dc = np.array(data["slopes"]) * 0.632

# exponential_model = lambda A, k1, T: A * np.exp(k1 * T) + 2.114 * np.exp(0.099 * T)
# exponential_model = lambda T, A, k1, C: A * np.exp(k1 * T) + C

# Initialize empty lists to store data
temperature_ccd = []
dc_ccd = []

# Open the file
with open(path + 'DC_vs_T_Ikon-L.json') as json_file:
    # Read each line in the file
    for line in json_file:
        try:
            # Load JSON data from each line
            data = json.loads(line)

            # Extract temperature and dc from each JSON object
            temperature_ccd.append(float(data["x"]))  # Convert to float if needed
            dc_ccd.append(float(data["y"]))  # Convert to float if needed
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON line: {line.strip()}")

# Convert lists to numpy arrays if needed
temperature_ccd = np.array(temperature_ccd) #+ 49
dc_ccd = np.array(dc_ccd)


def exp_model(x, A, k1, B, k2, C):
    y = A * np.exp(k1 * x) + B * np.exp(k2 * x) + C
    return y


parameters, covariance = curve_fit(exp_model, temperature, dc)

fit_A = parameters[0]
fit_k1 = parameters[1]
fit_B = parameters[2]
fit_k2 = parameters[3]
fit_C = parameters[4]
print("A:", fit_A, "k1:", fit_k1, "B:", fit_B, "k2:", fit_k2, "C:", fit_C)
fit_total = exp_model(temperature, fit_A, fit_k1, fit_B, fit_k2, fit_C)
fit_1 = fit_A * np.exp(fit_k1 * temperature) + fit_C
fit_2 = fit_B * np.exp(fit_k2 * temperature)

# Plot the data and fitted models
plt.figure(figsize=(6, 4))
# plt.xticks([-55, -45, -35, -25, -15, -5, 5, 15])

plt.plot(temperature, fit_total, 'b-')
plt.plot(temperature, fit_1, 'r--')
plt.plot(temperature, fit_2, 'y--')
plt.scatter(temperature, dc, color='black')
plt.scatter(temperature_ccd, dc_ccd, color='red')
plt.yscale('log')
plt.xlabel('Temperature (\N{DEGREE SIGN}C)')
plt.ylabel('Dark Current (e$^-$/pixel/sec)')
plt.ylim(1e-1, 1e2)
plt.xlim(-70, 20)
# plt.tight_layout()
# plt.savefig('DC_vs_Temp.pdf', bbox_inches='tight')
plt.savefig('DC_vs_Temp.pdf', bbox_inches='tight')
plt.show()

print('The dc values are:', dc)
# print('The dc values from the fits added are:', fit_total)
# print('The dc values from the first fit are:', fit_1)
# print('The dc values from the second fit are:', fit_2)


I_dark = 0.736  # dark current in e/pix/sec
pixel_size_microns = 11  # pixel size in microns
A_pixel = (pixel_size_microns / 1e6) ** 2  # pixel area in m^2, convert microns to meters and calculate area

# Elementary charge in coulombs
e = 1.602e-19  # coulombs

# Calculate dark current density in e/m^2
J_dark_e = I_dark / A_pixel

# Convert to dark current density in pA/m^2
J_dark_pA = J_dark_e * e * 1e12  # Convert from e to pA, 1e12 converts from C to pA

print(f"Dark Current Density: {J_dark_pA:.3f} pA/m^2")
