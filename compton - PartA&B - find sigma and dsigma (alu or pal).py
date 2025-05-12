import os
import numpy as np
import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
from tkinter.simpledialog import askstring

# Initialize Tk window
root = Tk()
root.withdraw()

# Ask the person who use the script
target = askstring("Target Selection", "Enter target type (alu for Aluminum, pal for plastic):")

if target:
    target = target.strip().lower()
    print(f"Target selected: {target}")
else:
    print("No target selected. Exiting Bye Bye.")
    exit()

if target == 'alu':
    Z = 13
    A = 26.98  # g/mol
    rho_m = 2.70  # g/cm³
    V_eff, dV_eff = 0.053407, 0.003141592654
elif target == 'pal':
    Z = 3.85
    A = (12.01 * 4.78 + 1.008 * 5.28) / (4.78 + 5.28)  #g/mol
    rho_m = 1.032  # g/cm³
    #V_eff, dV_eff = 56.81, 0.003141592654  # replace with actual values if known (i tried my best to measure this black cylinder)
    V_eff, dV_eff = 0.0314, 0.003141592654
else:
    raise ValueError("Invalid target type.Put 'alu' or 'si'.")

# === Constants ===
N_A = 6.022e23  # Avogadro's number
rho_e = Z * N_A * rho_m / A  # electron density in electrons/cm³

# Calibration constants
L, dL = 67.0, 0.1  # cm
S, dS = 78.53981634, 0.03141592654  # cm²
R, dR = 15.0, 0.02886751346  # cm
tau, dtau = 43.35298598, 1.298425537  # yr
t, dt = 44, 0.7071067812  # yr
I_gamma, dI_gamma = 0.851, 0.002  # fractional
#A_0 = 100  # activity (assumed units)
A_0 = 3.7e9 #si units =>chane for code (it was used for epsilon SI units)

# Efficiency model parameters
a0, da0 = 2.29411, 0.36815
a1, da1 = -0.83009, 0.25337
a2, da2 = 20.19732, 36.78218

# === Efficiency function and error ===
def efficiency(E):
    inner = E ** a1
    exp1 = np.exp(-a2 * inner)
    return a0 * exp1 * (1 - exp1)

def defficiency(E, dE):
    inner = E**a1
    exp_term = np.exp(-a2 * inner)
    term = (1 - exp_term)
    lnE = np.log(E)

    d_eff_a0 = exp_term * term
    d_eff_a1 = a0 * exp_term * (-a2) * term * lnE * inner + a0 * exp_term * (a2**2) * (inner**2) * lnE
    d_eff_a2 = a0 * exp_term * (-inner) * term + a0 * exp_term * a2 * (inner**2)
    d_eff_E = a0 * exp_term * (-a2) * term * a1 * E**(a1 - 1) + a0 * exp_term * (a2**2) * a1 * E**(2 * a1 - 1)

    d_eff_sq = (
        (d_eff_a0 * da0)**2 +
        (d_eff_a1 * da1)**2 +
        (d_eff_a2 * da2)**2 +
        (d_eff_E * dE)**2
    )
    return np.sqrt(d_eff_sq)

# === dsigma/domega and error ===
def dsigma_domega(N, eff, LT):
    pre = (4 * np.pi * R**2) / (eff * S)
    flux = A_0 * np.exp(-t / tau) * I_gamma * LT * ((1 / (4 * np.pi * L**2)) * rho_e * V_eff)
    return N / (pre * flux)

def dsigma_error(N, dN, eff, deff, LT, dLT):
    pre = (4 * np.pi * R**2) / (eff * S)
    flux = A_0 * np.exp(-t / tau) * I_gamma * LT * ((1 / (4 * np.pi * L**2)) * rho_e * V_eff)
    dsigma = N / (pre * flux)

    # Partial derivatives
    d_sigma_N = 1 / (pre * flux)
    d_sigma_eff = -N / (pre * eff * flux)
    d_sigma_LT = -N / (pre * flux**2) * (
        A_0 * np.exp(-t / tau) * I_gamma * ((1 / (4 * np.pi * L**2)) * rho_e * V_eff)
    )

    d_sigma = np.sqrt(
        (d_sigma_N * dN)**2 +
        (d_sigma_eff * deff)**2 +
        (d_sigma_LT * dLT)**2
    )

    return dsigma, d_sigma

# === File input ===

file_path = askopenfilename(title="Select Excel File with Experimental Data", filetypes=[("Excel Files", "*.xlsx *.xls")])
df = pd.read_excel(file_path)

# === Required columns ===
E = df["E_calibrated [keV]"] #it is keV
dE = df["ΔE_calibrated [keV]"] #it is in keV
N = df["N_meas"]
dN = df["ΔN_meas"]
LT = df["LIVE_TIME (s)"]
dLT = df["DLIVE_TIME"]

# === Computation ===
df["efficiency"] = efficiency(E)
df["Δefficiency"] = defficiency(E, dE)

sigma_list = []
dsigma_list = []

for i in range(len(df)):
    sig, dsig = dsigma_error(N[i], dN[i], df["efficiency"][i], df["Δefficiency"][i], LT[i], dLT[i])
    sigma_list.append(sig)
    dsigma_list.append(dsig)

df["dsigma/dOmega [cm^2/sr]"] = sigma_list
df["Δdsigma/dOmega"] = dsigma_list

# Save to Excel
save_path = asksaveasfilename(
    title="Save Output Excel File",
    defaultextension=".xlsx",
    filetypes=[("Excel files", "*.xlsx")]
)

if save_path:
    df.to_excel(save_path, index=False)
    print(f" Results saved to:\n{save_path}")
else:
    print(" Saving cancelled.")
