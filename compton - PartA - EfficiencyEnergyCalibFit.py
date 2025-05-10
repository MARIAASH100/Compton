import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2
import easygui

# === Load Excel file ===
file_path = easygui.fileopenbox(msg="Select Excel file with efficiency data", filetypes=["*.xlsx", "*.xls"])
df = pd.read_excel(file_path)

# === Read required columns ===I should change it to name extraction(later...)
E = df.iloc[:, 0].values              # Energy in keV
DE = df.iloc[:, 1].values             # Efficiency (N_meas/N_exp)
dE = df.iloc[:, 2].values             # Energy error
dDE = df.iloc[:, 3].values            # Efficiency error

# === Define model ===
def eff_model(E, a0, a1, a2):
    term = np.exp(-a2 * E**a1)
    return a0 * term * (1 - term)

# === Fit ===
p0 = [0.6, -0.8, 0.001]
popt, pcov = curve_fit(eff_model, E, DE, sigma=dDE, absolute_sigma=True, p0=p0)
perr = np.sqrt(np.diag(pcov))

# === Residuals and Fit Stats ===
DE_fit = eff_model(E, *popt)
residuals = DE - DE_fit
chi2_val = np.sum((residuals / dDE)**2)
dof = len(E) - len(popt)
chi2_red = chi2_val / dof
p_val = 1 - chi2.cdf(chi2_val, dof)

# === Plot ===
E_plot = np.linspace(min(E) - 20, max(E) + 100, 500)
DE_fit_plot = eff_model(E_plot, *popt)

plt.errorbar(
    E, DE,
    xerr=dE, yerr=dDE,
    fmt='o',              # circle markers
    markersize=4,         # marker size
    #markerfacecolor='blue',
    #markeredgecolor='black',
    #ecolor='darkblue',    # error bar color
    elinewidth=1.5,       # error bar thickness
    capsize=4,            # width of caps
    capthick=1.5,         # thickness of cap lines
    #alpha=0.9             # transparency
)

plt.plot(E_plot, DE_fit_plot, 'r-')
plt.xlabel("Energy [keV]")
plt.ylabel("Efficiency [a.u.]")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Residuals Plot ===
plt.errorbar(
    E, residuals,
    yerr=dDE,
    fmt='o',                    # circle markers
    markersize=4,               # moderate point size
    #markerfacecolor='blue',    # fill color of markers
    #markeredgecolor='black',   # border color of markers
    #ecolor='blue',             # color of error bars
    elinewidth=1.5,            # thickness of error bars
    capsize=4,                 # width of the caps
    capthick=1.5,              # thickness of cap lines
    #alpha=0.9,                 # transparency
)

plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("Energy [keV]")
plt.ylabel(r"$\varepsilon_i - f(E_i) [a.u.]$", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# === Output ===
print("\nFit Results:")
print(f"a0 = {popt[0]:.5f} ± {perr[0]:.5f}")
print(f"a1 = {popt[1]:.5f} ± {perr[1]:.5f}")
print(f"a2 = {popt[2]:.5f} ± {perr[2]:.5f}")
print(f"\nChi² = {chi2_val:.2f}")
print(f"Reduced Chi² (Chi²/dof) = {chi2_red:.2f}")
print(f"p-value = {p_val:.4f}")

