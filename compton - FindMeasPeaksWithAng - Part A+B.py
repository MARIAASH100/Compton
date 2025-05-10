import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2
import easygui
from matplotlib.widgets import RectangleSelector
import re
import tkinter as tk
from tkinter import filedialog

# === Models ===
def gaussian(x, a0, a1, a2):
    return a0 * np.exp(-((x - a1) ** 2) / (2 * a2 ** 2))

def gauss_linear(x, a0, a1, a2, a3, a4):
    return gaussian(x, a0, a1, a2) + a3 * x + a4

def gauss_exp(x, a0, a1, a2, a3, a4):
    return gaussian(x, a0, a1, a2) + a3 * np.exp(-a4 * x)

def fit_stats(x, y, model, popt, yerr):
    y_fit = model(x, *popt)
    residuals = y - y_fit
    chi2_val = np.sum((residuals / yerr) ** 2)
    dof = len(y) - len(popt)
    pval = 1 - chi2.cdf(chi2_val, dof)
    return chi2_val, chi2_val / dof, pval

def estimate_center_error(fwhm, N):
    return fwhm / (2.35 * np.sqrt(N)) if N > 0 else np.nan

def extract_angle(filename):
    match = re.search(r'ang\((-?\d+)\)', filename)
    return int(match.group(1)) if match else None

# === Load .mca ===
def load_mca_file(filepath):
    with open(filepath, 'r', encoding='latin1') as f:
        lines = f.readlines()
    counts = []
    real_time = live_time = None
    for line in lines:
        if line.startswith('REAL_TIME'):
            real_time = float(line.split('-')[1].strip())
        elif line.startswith('LIVE_TIME'):
            live_time = float(line.split('-')[1].strip())
        elif line.strip().isdigit():
            counts.append(int(line.strip()))
    return np.array(counts), real_time, live_time

def subtract_background(signal, t_signal, background, t_background):
    scale = t_signal / t_background
    corrected = signal - scale * background
    return np.clip(corrected, a_min=0, a_max=None)

# === Interactive Peak Selector ===
class InteractivePeakSelector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.selected_regions = []
        self.fig, self.ax = plt.subplots()
        self.selector = RectangleSelector(
            self.ax, self.on_select, useblit=True,
            button=[1], minspanx=5, minspany=5,
            spancoords='data', interactive=True)
        self.ax.plot(x, y, label="BG-subtracted")
        self.ax.set_title("Drag to select peak region. Press Enter when done :)")
        self.ax.set_xlabel("Channel")
        self.ax.set_ylabel("Counts")
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        plt.legend()
        plt.show()

    def on_select(self, eclick, erelease):
        self.selected_regions.append((int(eclick.xdata), int(erelease.xdata)))
        self.ax.axvspan(eclick.xdata, erelease.xdata, color='red', alpha=0.3)

    def on_key(self, event):
        if event.key == "enter":
            plt.close(self.fig)

    def get_regions(self):
        return self.selected_regions

# === Peak Analysis ===
def analyze_peaks(x, y, regions, angle, real_time, live_time):
    results = []
    for i, (x0, x1) in enumerate(regions):
        mask = (x >= x0) & (x <= x1)
        x_seg = x[mask]
        y_seg = y[mask]
        yerr = np.sqrt(np.maximum(y_seg, 1))

        for model_func, label in [(gauss_linear, 'Gauss+Linear'), (gauss_exp, 'Gauss+Exp')]:
            try:
                if label == 'Gauss+Linear':
                    p0 = [np.max(y_seg), x_seg[np.argmax(y_seg)], 10, 0, np.min(y_seg)]
                else:
                    p0 = [np.max(y_seg), x_seg[np.argmax(y_seg)], 10, np.max(y_seg), 0.001]

                popt, pcov = curve_fit(model_func, x_seg, y_seg, p0=p0, sigma=yerr, absolute_sigma=True)
                perr = np.sqrt(np.diag(pcov))

                fwhm = 2.355 * popt[2]
                fwhm_err = 2.355 * perr[2]
                N = np.sum(gaussian(x_seg, *popt[:3]))
                err_center = estimate_center_error(fwhm, N)
                ##
                # Calibration constants
                calib_a = 0.246334
                calib_da = 0.000190
                calib_b = -71.128255
                calib_db = 0.569967

                # Calibrated energy and its error
                E_prime = calib_a * popt[1] + calib_b
                DE_prime = np.sqrt((popt[1] * calib_da) ** 2 + (calib_a * perr[1]) ** 2 + calib_db ** 2)
                ##
                chi2_val, red_chi2, p_val = fit_stats(x_seg, y_seg, model_func, popt, yerr)
                N_meas = popt[0] * np.sqrt(2 * np.pi) * popt[2]
                DN_meas = N_meas * np.sqrt((perr[0] / popt[0]) ** 2 + (perr[2] / popt[2]) ** 2)

                results.append([
                    angle, i + 1, label, x0, x1,
                    *popt, *perr, fwhm, fwhm_err, N, err_center,
                    N_meas, DN_meas,
                    chi2_val, red_chi2, p_val,
                    E_prime, DE_prime,
                    real_time, live_time
                ])
            except Exception as e:
                print(f"{label} fit failed: {e}")
    return results

# === Main Execution ===
def main():
    bg_file = easygui.fileopenbox("Select background .mca file", filetypes=["*.mca"]) #here we use the linearity of BG (will be explained)
    if not bg_file:
        return
    bg_counts, _, bg_live = load_mca_file(bg_file)

    tk.Tk().withdraw()
    folder = filedialog.askdirectory(title="Select folder of angle MCA files")
    if not folder:
        return

    all_results = []
    for fname in sorted(os.listdir(folder)):
        if fname.endswith('.mca'):
            full_path = os.path.join(folder, fname)
            angle = extract_angle(fname)
            if angle is None:
                continue

            counts, real_time, live_time = load_mca_file(full_path)
            corrected = subtract_background(counts, live_time, bg_counts, bg_live)

            x = np.arange(len(corrected))
            selector = InteractivePeakSelector(x, corrected)
            regions = selector.get_regions()
            res = analyze_peaks(x, corrected, regions, angle, real_time, live_time)
            all_results.extend(res)

    columns = [
        "angle", "Peak #", "Fit Type", "x_start", "x_end",
        "a0", "a1", "a2", "a3", "a4",
        "a0_err", "a1_err", "a2_err", "a3_err", "a4_err",
        "FWHM", "FWHM Error", "Counts (N)", "σ_center (FWHM/2.35√N)",
        "N_meas (a0*sqrt(2pi)*a2)", "ΔN_meas", "Chi2", "Chi2/dof", "p-value",
        "E_calibrated", "ΔE_calibrated",
        "REAL_TIME (s)", "LIVE_TIME (s)"
    ]

    df = pd.DataFrame(all_results, columns=columns)
    output_path = easygui.filesavebox(default="ComptonAngle_Fits.xlsx", filetypes=["*.xlsx"])
    if output_path:
        if not output_path.endswith(".xlsx"):
            output_path += ".xlsx"
        df.to_excel(output_path, index=False)
        print(f" Results saved to {output_path}")

    # Save full results
    df.to_excel(output_path, index=False)
    print(f" Results saved to {output_path}")

    # Extract best fit (minimum |Chi2/dof - 1| for each angle & peak) => as the p-values always bad
    best_fit_df = df.loc[
        df.groupby(["angle", "Peak #"])["Chi2/dof"].apply(lambda g: (g - 1).abs().idxmin())
    ].reset_index(drop=True)

    # Save best fit to another file
    base, ext = os.path.splitext(output_path)
    best_fit_path = base + "_best_fit" + ext
    best_fit_df.to_excel(best_fit_path, index=False)
    print(f"✅ Best fits saved to {best_fit_path}")


main()


