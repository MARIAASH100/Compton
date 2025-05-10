# Re-import necessary libraries after code execution environment reset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import openpyxl

# Step 1: File and Sheet Selection
Tk().withdraw()
file_path = askopenfilename(title="Select Excel File", filetypes=[("Excel Files", "*.xlsx *.xls")])
xlsx = pd.ExcelFile(file_path)
#sheet_name = "Sheet2"
#df = pd.read_excel(file_path, sheet_name=sheet_name)
df = pd.read_excel(file_path)

# Step 2: Load and clean data
#df = df.dropna(subset=["1-cos(Ang [rad])", "D(1-cos(Ang [rad])", "1/E'", "D(1/E')"])
#x = df["1-cos(Ang [rad])"].values
#dx = df["D(1-cos(Ang [rad])"].values
#y = df["1/E'"].values
#dy = df["D(1/E')"].values
##part A=> Something went really wrong with the parameters
#df = df.dropna(subset=["x", "dx", "y", "dy"])
#x = df["x"].values
#dx = df["dx"].values
#y = df["y"].values
#dy = df["dy"].values
##
##finally put it :(put this names to the col in excel)
df = df.dropna(subset=["1-cos(Ang)", "D(1-cos(Ang))", "1/E_calib", "D(1/E_calib)"])
x = df["1-cos(Ang)"].values
dx = df["D(1-cos(Ang))"].values
y = df["1/E_calib"].values
dy = df["D(1/E_calib)"].values

# Step 3: Linear model and fit
def linear(x, a, b):
    return a * x + b

popt, pcov = curve_fit(linear, x, y, sigma=dy, absolute_sigma=True)
a, b = popt
da, db = np.sqrt(np.diag(pcov))
cov_ab = pcov[0, 1]

# Step 4: Residuals and statistics
residuals = y - linear(x, *popt)
chi2_val = np.sum((residuals / dy)**2)
dof = len(x) - 2
chi2_red = chi2_val / dof
p_val = 1 - chi2.cdf(chi2_val, dof)

# Step 5: Theory comparison (N_sigma check)

E_true=661 #keV  E_gamma for Cs-137
DE_true=0.3 #keV
mc2_true=511 #keV (m_e*c^2) in keV
Dmc2_true=0.000011 #keV
#to the parameters
E=1/b
DE=db/b**2
mc2=1/a
Dmc2=da/a**2

N_sigma_a= abs(mc2-mc2_true) / (Dmc2_true**2+Dmc2**2)**(1/2)
N_sigma_b = abs(E-E_true) / (DE**2+DE_true**2)**(1/2)

# Step 6: Plot data and fit
x_fit = np.linspace(min(x), max(x), 500)
y_fit = linear(x_fit, *popt)
fit_err = np.sqrt((x_fit * da)**2 + db**2 + 2 * x_fit * cov_ab)

plt.errorbar(
    x, y, xerr=dx, yerr=dy,
    fmt='o', markersize=3,
    elinewidth=1.5, capsize=3, capthick=1.5
    #label='Data'
)

plt.plot(x_fit, y_fit,'r-', linewidth=2)
plt.xlabel("1 - cos(θ) [rad]")
plt.ylabel("1/E' [1/keV]")
plt.grid(True)
plt.show()

# Step 7: Residual plot
plt.errorbar(x, residuals, yerr=dy,fmt='o',
    capsize=3,          # length of caps at the ends(point size)
    elinewidth=1.5,     # thickness of error bar lines
    capthick=1.5,       # thickness of caps
    #ecolor='black',     # color of error bars
    #label='Residuals'
)

plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("1 - cos(θ) [rad]")
plt.ylabel(r"$\left(\frac{1}{E_i'}\right) - f\left(\frac{1}{E_i'}\right) [\frac{1}{keV}]$", fontsize=12)
plt.grid(True)
plt.show()

# Print results
print(f"Fit: E = a*x + b")
print(f"a = {a:.6f} ± {da:.6f}")
print(f"b = {b:.6f} ± {db:.6f}")
print(f"chi2 = {chi2_val:.2f}, chi2_red = {chi2_red:.2f}, p-value = {p_val:.3f}")
print(f"N_sigma(mc^2): {N_sigma_a:.6f} ")
print(f"N_sigma(E_incident): {N_sigma_b:.6f} ")
print(f"fit mc^2: {mc2:.6f} ± {Dmc2:.6f} ")
print(f"fit E_incident:{E:.6f} ± {DE:.6f} ")


