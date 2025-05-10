import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2
from tkinter.filedialog import askopenfilename

# === Load Excel File ===
file_path = askopenfilename(title="Select Excel File", filetypes=[("Excel files", "*.xlsx *.xls")])
df = pd.read_excel(file_path)

# === Extract relevant columns ===
theta_exp = df['Ang [rad]'].to_numpy()
dsigma_exp = df['dsigma/dOmega [cm^2/sr]'].to_numpy()
dsigma_err = df['Δdsigma/dOmega'].to_numpy()

# === Remove any negative values to prevent polar plot issues ==={it was like serious problem}
dsigma_exp = np.clip(dsigma_exp, 0, None)
dsigma_err = np.clip(dsigma_err, 0, None)

# === Define Klein-Nishina model ===
def klein_nishina_model(theta, a0, a1):
    cos_theta = np.cos(theta)
    numerator = a1**2 * (1 - cos_theta)**2
    denominator = (1 + cos_theta**2) * (1 + a1 * (1 - cos_theta))
    prefactor = 1 / (1 + a1 * (1 - cos_theta))**2
    return a0 * (1 + cos_theta**2) * prefactor * (1 + numerator / denominator)

# === Define Thomson model ===
def thomson_model(theta, a0):
    return a0 * (1 + np.cos(theta)**2)

# === Fit Klein-Nishina ===
initial_guess_kn = [1e-25, 1.0]
popt_kn, pcov_kn = curve_fit(klein_nishina_model, theta_exp, dsigma_exp, sigma=dsigma_err, absolute_sigma=True)
a0_kn, a1_kn = popt_kn
a0_kn_err, a1_kn_err = np.sqrt(np.diag(pcov_kn))

residuals_kn = dsigma_exp - klein_nishina_model(theta_exp, *popt_kn)
chi2_kn = np.sum((residuals_kn / dsigma_err)**2)
dof_kn = len(dsigma_exp) - len(popt_kn)
chi2_red_kn = chi2_kn / dof_kn
p_val_kn = 1 - chi2.cdf(chi2_kn, dof_kn)

# === Fit Thomson model ===
popt_th, pcov_th = curve_fit(thomson_model, theta_exp, dsigma_exp, sigma=dsigma_err, absolute_sigma=True)
a0_th = popt_th[0]
a0_th_err = np.sqrt(np.diag(pcov_th))[0]

residuals_th = dsigma_exp - thomson_model(theta_exp, a0_th)
chi2_th = np.sum((residuals_th / dsigma_err)**2)
dof_th = len(dsigma_exp) - 1
chi2_red_th = chi2_th / dof_th
p_val_th = 1 - chi2.cdf(chi2_th, dof_th)

# === Generate theoretical curves ===
theta_model = np.linspace(-np.pi, np.pi, 360)
kn_vals = np.clip(klein_nishina_model(theta_model, *popt_kn), 0, None)
th_vals = np.clip(thomson_model(theta_model, a0_th), 0, None)

# === Normalize for plotting ===
#norm_factor = np.max(dsigma_exp) => give scale from 0 to 1
norm_factor = 1e-26 #typical cross section scale in this kind of interactions cm^2

dsigma_exp_norm = dsigma_exp / norm_factor
dsigma_err_norm = dsigma_err / norm_factor
kn_vals_norm = kn_vals / norm_factor
th_vals_norm = th_vals / norm_factor

# === Plot ===
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)
ax.errorbar(theta_exp, dsigma_exp_norm, yerr=dsigma_err_norm, fmt='o', label='Experimental data')
ax.plot(theta_model, kn_vals_norm, label='Klein-Nishina fit', color='orange')
ax.plot(theta_model, th_vals_norm, label='Thomson fit', color='skyblue')
ax.legend(loc='upper right')
plt.tight_layout()
plt.show()

# === Print Results ===
mc2=511 #keV
r0_kn=(2*a0_kn)**(1/2)
dr0_kn=2**(1/2)*a0_kn**(-1/2)*a0_kn_err/2
E_kn=mc2*a1_kn
dE_kn=mc2*a1_kn_err
print("\n--- Klein–Nishina Fit ---")
print(f"a0 = {a0_kn:.2e} ± {a0_kn_err:.2e}")
print(f"a1 = {a1_kn:.3f} ± {a1_kn_err:.3f}")
print(f"Chi² = {chi2_kn:.3f}")
print(f"Reduced Chi² = {chi2_red_kn:.3f}")
print(f"p-value = {p_val_kn:.4f}")
print(f"ro_kn = {r0_kn:.2e} ± {dr0_kn:.2e}")
print(f"E_incident = {E_kn:.3f} ± {dE_kn:.3f}")

print("\n--- Thomson Fit ---")
r0_t=(2*a0_th)**(1/2)
dr0_t=2**(1/2)*a0_th**(-1/2)*a0_th_err/2
print(f"a0 = {a0_th:.2e} ± {a0_th_err:.2e}")
print(f"Chi² = {chi2_th:.3f}")
print(f"Reduced Chi² = {chi2_red_th:.3f}")
print(f"p-value = {p_val_th:.4f}")
print(f"ro_th = {r0_t:.2e} ± {dr0_t:.2e}")


