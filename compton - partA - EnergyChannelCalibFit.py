import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Open file dialog
Tk().withdraw()
file_path = askopenfilename(title="Select your Excel file", filetypes=[("Excel files", "*.xlsx *.xls")])

# Load the Excel file
df = pd.read_excel(file_path)

# Take only required data
df = df.dropna(subset=["real_energy (keV)", "Dreal_energy (keV)",
                       "channel_center (AU)", "Dchannel_center (AU)"])

# Extract variables
E = df["real_energy (keV)"].values
dE = df["Dreal_energy (keV)"].values
ch = df["channel_center (AU)"].values
dch = df["Dchannel_center (AU)"].values

# Define linear model
def linear(x, a, b):
    return a * x + b

# Weighted least squares fit
popt, pcov = curve_fit(linear, ch, E, sigma=dE, absolute_sigma=True)
a, b = popt
da, db = np.sqrt(np.diag(pcov))

# Compute residuals and chi-square
residuals = E - linear(ch, *popt)
chi2_val = np.sum((residuals / dE)**2)
dof = len(E) - 2
chi2_red = chi2_val / dof
p_val = 1 - chi2.cdf(chi2_val, dof)

# Print results
print(f"Fit: E = a*x + b")
print(f"a = {a:.6f} ± {da:.6f}")
print(f"b = {b:.6f} ± {db:.6f}")
print(f"chi2 = {chi2_val:.2f}, chi2_red = {chi2_red:.2f}, p-value = {p_val:.3f}")

# Plot data
plt.errorbar(ch, E, xerr=dch, yerr=dE, fmt='o', ms=1, elinewidth=3, capsize=3)
# Fit line
x_fit = np.linspace(min(ch), max(ch), 500)
y_fit = linear(x_fit, *popt)

# 1σ error band on fit line from covariance matrix(in the end we didnt use it)
# Fit: y = a*x + b, so uncertainty = sqrt((x*σ_a)^2 + σ_b^2 + 2*x*cov_ab)
sigma_a2 = pcov[0, 0]
sigma_b2 = pcov[1, 1]
cov_ab = pcov[0, 1]
fit_err = np.sqrt(sigma_a2 * x_fit**2 + sigma_b2 + 2 * x_fit * cov_ab)

# Plot fit line and 1σ error band
plt.plot(x_fit, y_fit, 'r-', label='Linear fit')
#plt.fill_between(x_fit, y_fit - fit_err, y_fit + fit_err, color='r', alpha=0.7, label='1σ error band')
plt.xlabel("Channel [AU]")
plt.ylabel("Energy [keV]")
plt.legend()
plt.grid(True)
plt.show()

# Plot residuals
plt.errorbar(ch, residuals, yerr=dE, fmt='o', ms=1, elinewidth=3, capsize=3)
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("Channel [AU]")
plt.ylabel("Residuals E - linear_fit(E) [keV]")
plt.grid(True)
plt.show()
