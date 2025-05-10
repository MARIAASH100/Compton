import easygui
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.widgets import RectangleSelector
from scipy.stats import chi2
from scipy.signal import savgol_filter
import pandas as pd
import os
import re  # to extract angle


# === Functions for fitting ===
def linear(x, m, b):
    return m * x + b


def gaussian(x, a0, a1, a2):
    return a0 * np.exp(-((x - a1) ** 2) / (a2 ** 2))


# === Select folder ===
folder_path = easygui.diropenbox(title="Select the folder containing .mca files")

if not folder_path:
    print("No folder selected")
    exit()

# === List all .mca files ===
mca_files = [f for f in os.listdir(folder_path) if f.endswith('.mca')]
mca_files.sort()

# === Initialize list to store results ===
results = []

# === Loop over files ===
for filename in mca_files:
    filepath = os.path.join(folder_path, filename)

    # === Read the MCA file ===
    with open(filepath, 'r', encoding='latin1') as file:
        lines = file.readlines()

    real_time = None
    live_time = None
    counts = []
    reading_counts = False

    for line in lines:
        line = line.strip()
        if line.startswith('REAL_TIME'):
            real_time = float(line.split('-')[1].strip())
        elif line.startswith('LIVE_TIME'):
            live_time = float(line.split('-')[1].strip())
        elif line == '<<DATA>>':
            reading_counts = True
        elif reading_counts:
            if line.isdigit():
                counts.append(int(line))

    channels = np.arange(0, len(counts))
    counts = np.array(counts)

    # === Apply Savitzky-Golay smoothing ===
    window_length = 11  # must be odd
    polyorder = 3
    counts_smooth = savgol_filter(counts, window_length=window_length, polyorder=polyorder)

    # === Plot and select region ===
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(channels, counts, label="Original Spectrum", color='gray', alpha=0.4)
    ax.plot(channels, counts_smooth, label="Smoothed Spectrum", color='blue')
    ax.set_xlabel('Channel')
    ax.set_ylabel('Counts')
    ax.set_title(f'Select region for: {filename}')
    ax.legend()
    plt.grid()

    selected_region = []


    def onselect(eclick, erelease):
        global selected_region
        selected_region = [eclick.xdata, erelease.xdata]
        plt.close()


    toggle_selector = RectangleSelector(ax, onselect, useblit=True, button=[1], minspanx=5, minspany=5,
                                        interactive=True)
    plt.show()

    if not selected_region:
        print(f"No region selected for {filename}.I will skip.")
        continue

    x_min, x_max = sorted(selected_region)
    mask = (channels >= x_min) & (channels <= x_max)
    x_selected = channels[mask]
    y_selected_smooth = counts_smooth[mask]
    y_selected_raw = counts[mask]

    # === Fit background first (linear) ===
    n_side = max(5, len(x_selected) // 10)
    x_background = np.concatenate([x_selected[:n_side], x_selected[-n_side:]])
    y_background = np.concatenate([y_selected_smooth[:n_side], y_selected_smooth[-n_side:]])

    popt_bg, pcov_bg = curve_fit(linear, x_background, y_background)
    background_fit = linear(x_selected, *popt_bg)

    # === Subtract background ===
    y_clean = y_selected_smooth - background_fit
    y_clean = np.clip(y_clean, a_min=0, a_max=None)

    # === Fit Gaussian to background-subtracted peak ===
    a0_guess = np.max(y_clean)
    a1_guess = x_selected[np.argmax(y_clean)]
    a2_guess = (x_max - x_min) / 6

    popt_gauss, pcov_gauss = curve_fit(gaussian, x_selected, y_clean, p0=[a0_guess, a1_guess, a2_guess])
    a0, a1, a2 = popt_gauss
    a0_err, a1_err, a2_err = np.sqrt(np.diag(pcov_gauss))

    # === Calculate FWHM ===
    fwhm = 2 * np.sqrt(2 * np.log(2)) * a2

    # === Calculate chi-squared and p-probability ===
    y_fit = gaussian(x_selected, *popt_gauss)
    residuals = y_clean - y_fit
    errors = np.sqrt(np.maximum(y_selected_raw, 1))  # errors from RAW counts! as it is...
    chi_squared = np.sum((residuals / errors) ** 2)
    dof = len(x_selected) - len(popt_gauss)
    chi_squared_dof = chi_squared / dof
    p_value = 1 - chi2.cdf(chi_squared, dof)

    # === Extract angle from filename ===(super convenient so name it ang(#))
    angle_match = re.search(r'ang\((-?\d+)\)', filename)
    if angle_match:
        angle = int(angle_match.group(1))
    else:
        angle = None  # if something went wrong

    # === Store result ===
    results.append({
        "File": filename,
        "Angle (deg)": angle,
        "Real Time (s)": real_time,
        "Live Time (s)": live_time,
        "Amplitude a0": a0,
        "Amplitude Error": a0_err,
        "Peak Center a1 (channel)": a1,
        "Center Error": a1_err,
        "Width a2 (sigma)": a2,
        "Width Error": a2_err,
        "FWHM (channel units)": fwhm,
        "Chi-squared/dof": chi_squared_dof,
        "P-value": p_value
    })

# === Save all results to Excel ===
df_results = pd.DataFrame(results)
save_path = easygui.filesavebox(default="AngleCalibrationResults.xlsx", title="Save Results As")
if save_path:
    if not save_path.endswith('.xlsx'):
        save_path += '.xlsx'
    df_results.to_excel(save_path, index=False)
    print(f"\n Results saved successfully to {save_path}!")
else:
    print("No save path selected. Results not saved.")
