import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2
from matplotlib.widgets import RectangleSelector
import easygui
from matplotlib.ticker import FuncFormatter #rad convert to deg

# === Load Excel file ===
file_path = easygui.fileopenbox(msg="Select your AngleCalibrationResults.xlsx", filetypes=["*.xlsx"])

if not file_path:
    print("No file selected=> Exiting.")
    exit()

# Read the Excel file
data = pd.read_excel(file_path)

# === Prepare the data ===(based on the previous code excel output)
angle_deg = data['Angle (deg)'].values
amplitude = data['Amplitude a0'].values
amplitude_err = data['Amplitude Error'].values
live_time = data['Live Time (s)'].values

# Normalize amplitude by live time (we did different times so we have no choice...)
norm_amplitude = amplitude / live_time
norm_amplitude_err = norm_amplitude * (amplitude_err / amplitude)

# Convert angles to radians
angle_rad = np.deg2rad(angle_deg)

# === Interactive region selection ===
selected_region = []

def onselect(eclick, erelease):
    global selected_region
    x1, x2 = eclick.xdata, erelease.xdata
    selected_region = [min(x1, x2), max(x1, x2)]
    plt.close()

fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(angle_rad, norm_amplitude, yerr=norm_amplitude_err, fmt='o', capsize=3)
ax.set_xlabel('Angle (rad)')
ax.set_ylabel('Normalized Amplitude (a.u.)')
ax.set_title('Select Region to Fit (Drag mouse and release)')
ax.grid()
selector = RectangleSelector(ax, onselect, useblit=True, button=[1], minspanx=0.01, minspany=0.01, interactive=True)
plt.show()

# === Crop the data ===
if not selected_region:
    print("No region selected. Exiting Bye Bye.")
    exit()

x_min, x_max = selected_region
mask = (angle_rad >= x_min) & (angle_rad <= x_max)

angle_rad_fit = angle_rad[mask]
norm_amplitude_fit = norm_amplitude[mask]
norm_amplitude_err_fit = norm_amplitude_err[mask]

# === Define models ===
def gaussian(x, a, x0, sigma,c):
    return a * np.exp(-((x - x0)**2) / (2 * sigma**2))+c

def parabola(x, a, x0, c):
    return a * (x - x0)**2 + c

def cosine_squared(x, a, x0, c):
    return a * np.cos((x - x0)/2)**2 + c

models = {
    'Gaussian': gaussian,
    'Parabola': parabola,
    'Cosine': cosine_squared
}

# === Fitting ===
fit_results = {}

for model_name, model_func in models.items():
    if model_name == 'Gaussian':
        p0 = [np.max(norm_amplitude_fit), 0, 0.2]
    elif model_name == 'Parabola':
        p0 = [-100, 0, np.mean(norm_amplitude_fit)]
    elif model_name == 'Cosine':
        p0 = [np.max(norm_amplitude_fit), 0, np.min(norm_amplitude_fit)]

    try:
        popt, pcov = curve_fit(
            model_func, angle_rad_fit, norm_amplitude_fit,
            sigma=norm_amplitude_err_fit, absolute_sigma=True, p0=p0
        )
    except Exception as e:
        print(f"Fitting failed for {model_name}: {e}")
        continue

    y_fit = model_func(angle_rad_fit, *popt)
    residuals = norm_amplitude_fit - y_fit
    chi2_val = np.sum((residuals / norm_amplitude_err_fit) ** 2)
    dof = len(angle_rad_fit) - len(popt)
    chi2_reduced = chi2_val / dof
    p_val = 1 - chi2.cdf(chi2_val, dof)

    fit_results[model_name] = {
        'popt': popt,
        'pcov': pcov,
        'chi2_reduced': chi2_reduced,
        'p_value': p_val,
        'y_fit': y_fit,
        'residuals': residuals
    }

# === Plot fits (smooth) FOR RADIANS ===
fig, ax = plt.subplots(figsize=(12, 7))
ax.errorbar(angle_rad_fit, norm_amplitude_fit, yerr=norm_amplitude_err_fit, fmt='o', capsize=3, label='Data')

colors = ['red', 'green', 'blue']
x_fine = np.linspace(np.min(angle_rad_fit), np.max(angle_rad_fit), 500)  # fine grid

for idx, (model_name, result) in enumerate(fit_results.items()):
    model_func = models[model_name]
    popt = result['popt']
    y_fine = model_func(x_fine, *popt)
    ax.plot(x_fine, y_fine, label=f'{model_name} Fit', color=colors[idx])

ax.set_xlabel('Angle [rad]')
ax.set_ylabel('Normalized Amplitude [a.u.]')
ax.legend()
ax.grid()
plt.show()


# === Residuals FOR RADIANS ===
fig, axes = plt.subplots(len(fit_results), 1, figsize=(12, 5 * len(fit_results)))

if len(fit_results) == 1:
    axes = [axes]

for idx, (model_name, result) in enumerate(fit_results.items()):
    axes[idx].errorbar(angle_rad_fit, result['residuals'], yerr=norm_amplitude_err_fit, fmt='o', capsize=3)
    axes[idx].axhline(0, color='red', linestyle='--')
    axes[idx].set_xlabel('Angle (rad)')
    axes[idx].set_ylabel('Residuals')
    axes[idx].set_title(f'Residuals: {model_name}')
    axes[idx].grid()

plt.tight_layout()
plt.show()

# === Print fit ID ===
print("\n=== Fit Comparison Summary ===")
for model_name, result in fit_results.items():
    print(f"{model_name}: χ²_red = {result['chi2_reduced']:.3f}, p-value = {result['p_value']:.4f}")

# === Full detailed fit parameters and errors ===
print("\n=== Full Fit Parameters Summary ===")

fit_summary = []

for model_name, result in fit_results.items():
    popt = result['popt']
    pcov = result['pcov']
    param_errors = np.sqrt(np.diag(pcov))

    if model_name == 'Gaussian':
        amplitude, center, sigma = popt
        amplitude_err, center_err, sigma_err = param_errors
        print(f"\nModel: {model_name}")
        print(f"  Amplitude = {amplitude:.4g} ± {amplitude_err:.2g}")
        print(f"  Center (rad) = {center:.4g} ± {center_err:.2g}")
        print(f"  Width σ (rad) = {sigma:.4g} ± {sigma_err:.2g}")

    elif model_name == 'Parabola':
        a, center, c = popt
        a_err, center_err, c_err = param_errors
        print(f"\nModel: {model_name}")
        print(f"  Curvature a = {a:.4g} ± {a_err:.2g}")
        print(f"  Center (rad) = {center:.4g} ± {center_err:.2g}")
        print(f"  Baseline c = {c:.4g} ± {c_err:.2g}")

    elif model_name == 'Cosine':
        amplitude, center, c = popt
        amplitude_err, center_err, c_err = param_errors
        print(f"\nModel: {model_name}")
        print(f"  Amplitude = {amplitude:.4g} ± {amplitude_err:.2g}")
        print(f"  Center (rad) = {center:.4g} ± {center_err:.2g}")
        print(f"  Baseline c = {c:.4g} ± {c_err:.2g}")
#=== We decided convert to Deg (everyone and what they like, we love Deg)===
deg = 180 / np.pi  # conversion factor

for model_name, result in fit_results.items():
    popt = result['popt']
    pcov = result['pcov']
    param_errors = np.sqrt(np.diag(pcov))

    if model_name == 'Gaussian':
        amplitude, center, sigma = popt
        amplitude_err, center_err, sigma_err = param_errors
        center_deg = center * deg
        center_err_deg = center_err * deg
        sigma_deg = sigma * deg
        sigma_err_deg = sigma_err * deg
        print(f"\nModel: {model_name}")
        print(f"  Amplitude = {amplitude:.4g} ± {amplitude_err:.2g}")
        print(f"  Center = {center_deg:.4g}° ± {center_err_deg:.2g}°")
        print(f"  Width σ = {sigma_deg:.4g}° ± {sigma_err_deg:.2g}°")

    elif model_name == 'Parabola':
        a, center, c = popt
        a_err, center_err, c_err = param_errors
        center_deg = center * deg
        center_err_deg = center_err * deg
        print(f"\nModel: {model_name}")
        print(f"  Curvature a = {a:.4g} ± {a_err:.2g}")
        print(f"  Center = {center_deg:.4g}° ± {center_err_deg:.2g}°")
        print(f"  Baseline c = {c:.4g} ± {c_err:.2g}")

    elif model_name == 'Cosine':
        amplitude, center, c = popt
        amplitude_err, center_err, c_err = param_errors
        center_deg = center * deg
        center_err_deg = center_err * deg
        print(f"\nModel: {model_name}")
        print(f"  Amplitude = {amplitude:.4g} ± {amplitude_err:.2g}")
        print(f"  Center = {center_deg:.4g}° ± {center_err_deg:.2g}°")
        print(f"  Baseline c = {c:.4g} ± {c_err:.2g}")

#===Plots:rad convert to deg===
fig, ax = plt.subplots(figsize=(12, 7))

# Convert x-axis for plotting only
angle_deg_fit = np.rad2deg(angle_rad_fit)
x_fine_deg = np.rad2deg(x_fine)

# Plot in degrees
ax.errorbar(angle_deg_fit, norm_amplitude_fit, yerr=norm_amplitude_err_fit, fmt='o', ms=1,  elinewidth=3, capsize=3, label='Data')


colors = ['red', 'green', 'blue']
for idx, (model_name, result) in enumerate(fit_results.items()):
    model_func = models[model_name]
    popt = result['popt']
    y_fine = model_func(np.deg2rad(x_fine_deg), *popt)  # convert back for model input
    ax.plot(x_fine_deg, y_fine, label=f'{model_name} Fit', color=colors[idx])

ax.set_xlabel('Angle [deg]')
ax.set_ylabel('Normalized Counts [a.u.]')
#ax.set_title('Fits Comparison (Degrees)')
ax.legend()
ax.grid()
plt.show()

# === Plot only Gaussian Fit(after we saw who have best Statistics) ===
if 'Gaussian' in fit_results:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.errorbar(angle_deg_fit, norm_amplitude_fit, yerr=norm_amplitude_err_fit, fmt='o', ms=1,  elinewidth=3, capsize=3)

    gaussian_popt = fit_results['Gaussian']['popt']
    y_gauss = gaussian(np.deg2rad(x_fine_deg), *gaussian_popt)
    ax.plot(x_fine_deg, y_gauss, color='red', linewidth=1.5)
    ax.set_xlabel('Angle [deg]')
    ax.set_ylabel('Normalized Amplitude [a.u.]')
    ax.grid()
    plt.show()
else:
    print("Gaussian fit not available Bassa Sababa.")

# === Residual Plot Gaussian Fit===
residuals = norm_amplitude_fit - gaussian(np.deg2rad(angle_deg_fit), *gaussian_popt)

fig, ax_res = plt.subplots(figsize=(12, 4))
ax_res.errorbar(angle_deg_fit, residuals, yerr=norm_amplitude_err_fit, fmt='o',
                    markersize=3, elinewidth=1.5, capsize=3)
ax_res.axhline(0, color='gray', linestyle='--', linewidth=1)
ax_res.set_xlabel('Angle [deg]')
ax_res.set_ylabel(r'$N_i - f(\theta_i)$ [a.u.]')
ax_res.grid(True)

plt.tight_layout()
plt.show()
