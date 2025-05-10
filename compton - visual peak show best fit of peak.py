import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2
import easygui
from matplotlib.widgets import RectangleSelector

# === Models ===
def gaussian(x, a0, a1, a2):
    return a0 * np.exp(-((x - a1)**2) / (2 * a2**2))

def gauss_linear(x, a0, a1, a2, a3, a4):
    return gaussian(x, a0, a1, a2) + a3 * x + a4

def gauss_exp(x, a0, a1, a2, a3, a4):
    return gaussian(x, a0, a1, a2) + a3 * np.exp(-a4 * x)

# === Helper Functions ===
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

def fit_stats(x, y, model, popt, yerr):
    y_fit = model(x, *popt)
    residuals = y - y_fit
    chi2_val = np.sum((residuals / yerr)**2)
    dof = len(y) - len(popt)
    pval = 1 - chi2.cdf(chi2_val, dof)
    return chi2_val, chi2_val / dof, pval

def estimate_center_error(fwhm, N):
    return fwhm / (2.35 * np.sqrt(N)) if N > 0 else np.nan

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
        self.ax.plot(x, y, label="Raw Spectrum")
        self.ax.set_title("Drag to select peak region. Press Enter when done.")
        self.ax.set_xlabel("Channel")
        self.ax.set_ylabel("Counts")
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        plt.legend()
        plt.show()

    def on_select(self, eclick, erelease):
        self.selected_regions.append((eclick.xdata, erelease.xdata))
        self.ax.axvspan(eclick.xdata, erelease.xdata, color='red', alpha=0.3)

    def on_key(self, event):
        if event.key == "enter":
            plt.close(self.fig)

    def get_regions(self):
        return self.selected_regions

# === Main peak analysis ===
def analyze_peaks(x, y, regions, filename):
    results = []
    fits_to_plot = []  # collect best fits for final plot
    material = os.path.basename(filename).split('.')[0]

    for i, (x0, x1) in enumerate(regions):
        mask = (x >= x0) & (x <= x1)
        x_seg = x[mask]
        y_seg = y[mask]
        yerr = np.sqrt(np.maximum(y_seg, 1))
        x_start = int(x_seg[0])
        x_end = int(x_seg[-1])

        best_fit = None

        # Gaussian + Linear
        try:
            p0 = [np.max(y_seg), x_seg[np.argmax(y_seg)], 10, 0, np.min(y_seg)]
            popt1, pcov1 = curve_fit(gauss_linear, x_seg, y_seg, p0=p0, sigma=yerr, absolute_sigma=True)
            perr1 = np.sqrt(np.diag(pcov1))
            chi2_1, redchi2_1, pval1 = fit_stats(x_seg, y_seg, gauss_linear, popt1, yerr)
            best_fit = ('Gauss+Linear', gauss_linear, popt1, x_seg, redchi2_1, perr1, chi2_1, pval1)
        except Exception as e:
            print(f"Gauss+Linear fit failed: {e}")

        # Gaussian + Exponential
        try:
            p0 = [np.max(y_seg), x_seg[np.argmax(y_seg)], 10, np.max(y_seg), 0.001]
            popt2, pcov2 = curve_fit(gauss_exp, x_seg, y_seg, p0=p0, sigma=yerr, absolute_sigma=True)
            perr2 = np.sqrt(np.diag(pcov2))
            chi2_2, redchi2_2, pval2 = fit_stats(x_seg, y_seg, gauss_exp, popt2, yerr)
            if best_fit is None or redchi2_2 < best_fit[4]:
                best_fit = ('Gauss+Exp', gauss_exp, popt2, x_seg, redchi2_2, perr2, chi2_2, pval2)
        except Exception as e:
            print(f"Gauss+Exp fit failed: {e}")

        if best_fit:
            label, model, popt, x_seg, redchi2, perr, chi2_val, pval = best_fit
            print(f"\nBest fit for peak {i+1} ({label}):")
            for j, (param, err) in enumerate(zip(popt, perr)):
                print(f"  Param {j}: {param:.3f} ± {err:.3f}")
            print(f"  Reduced Chi²: {redchi2:.3f}")
            print(f"  p-value: {pval:.4f}")

            plt.plot(x_seg, y[x_seg], 'b.', label='Data')
            plt.plot(x_seg, model(x_seg, *popt), 'orange', label=f'{label}')
            plt.plot(x_seg, gaussian(x_seg, *popt[:3]), 'red', label='Gaussian only')
            plt.legend()
            plt.title(f"{material} Peak {i+1} Best Fit")
            plt.xlabel("Channel")
            plt.ylabel("Counts")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            fits_to_plot.append((x_seg, model, popt))

    # === Overlay best fits on full spectrum ===
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b.', markersize=1, label='Data Points')
    for i, (x_seg, model, popt) in enumerate(fits_to_plot):
        label_bg = 'Gauss with BG' if i == 0 else None
        label_g = 'Gauss no BG' if i == 0 else None
        plt.plot(x_seg, model(x_seg, *popt), 'red', label=label_bg)
        plt.plot(x_seg, gaussian(x_seg, *popt[:3]), 'green', label=label_g)
    plt.xlabel("Channel [a.u]")
    plt.ylabel("Counts [a.u]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return results

# === Full Workflow ===
def main():
    sample_files = easygui.fileopenbox(msg="Select sample .mca files", filetypes=["*.mca"], multiple=True)
    if not sample_files:
        return

    for filepath in sample_files:
        counts, real_time, live_time = load_mca_file(filepath)
        x = np.arange(len(counts))

        plt.figure(figsize=(10, 6))
        plt.plot(x, counts, label="Raw Spectrum", color="blue")
        plt.xlabel("Channel")
        plt.ylabel("Counts")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        selector = InteractivePeakSelector(x, counts)
        regions = selector.get_regions()
        print(f"Regions for {os.path.basename(filepath)}: {regions}")

        analyze_peaks(x, counts, regions, filepath)

if __name__ == "__main__":
    main()
