import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
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

# ========
def load_mca_file(filepath):
    with open(filepath, 'r', encoding='latin1') as f:
        lines = f.readlines()
    counts = []
    real_time = live_time = None #If the file contains those lines, you extract their values, otherwise their values is none
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
        self.ax.plot(x, y, label="Background-Subtracted Spectrum")
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

# ===Main peak analysis===
def analyze_peaks(x, y, regions, filename):
    results = []
    material = os.path.basename(filename).split('.')[0]

    for i, (x0, x1) in enumerate(regions):
        mask = (x >= x0) & (x <= x1)
        x_seg = x[mask]
        y_seg = y[mask]
        yerr = np.sqrt(np.maximum(y_seg, 1))
        x_start = int(x_seg[0])
        x_end = int(x_seg[-1])

        # Fit 1: Gaussian + Linear
        try:
            p0 = [np.max(y_seg), x_seg[np.argmax(y_seg)], 10, 0, np.min(y_seg)]
            popt1, pcov1 = curve_fit(gauss_linear, x_seg, y_seg, p0=p0, sigma=yerr, absolute_sigma=True)
            perr1 = np.sqrt(np.diag(pcov1))
            fwhm1 = 2.355 * popt1[2]
            err_center1 = estimate_center_error(fwhm1, np.sum(gaussian(x_seg, *popt1[:3])))
            chi2_1, redchi2_1, pval1 = fit_stats(x_seg, y_seg, gauss_linear, popt1, yerr)

            # Gaussian-only parameters
            a0, a1, a2 = popt1[:3]
            a0_err, _, a2_err = perr1[:3]
            N_meas1 = a0 * np.sqrt(2 * np.pi) * a2
            dN_meas1 = np.sqrt((a0_err * np.sqrt(2 * np.pi) * a2) ** 2 + (a2_err * np.sqrt(2 * np.pi) * a0) ** 2)

            # Plot Gaussian-only component
            plt.figure()
            plt.plot(x_seg, y_seg, label='Data', color='green')
            plt.plot(x_seg, gaussian(x_seg, *popt1[:3]), color='orange', label='Gaussian (no linear BG)')
            plt.title(f'{material} Peak {i + 1} - Gauss+Linear Subtracted')
            plt.xlabel("Channel")
            plt.ylabel("Counts")
            plt.legend()
            plt.tight_layout()
            plt.show()

            results.append([
                material, i + 1, 'Gauss+Linear',
                x_start, x_end,
                *popt1, *perr1,
                fwhm1, 2.355 * perr1[2],
                np.sum(gaussian(x_seg, *popt1[:3])), err_center1,
                N_meas1, dN_meas1, chi2_1, redchi2_1, pval1
            ])
            # Visualize Gauss+Linear full fit and isolated Gaussian
            plt.figure()
            plt.plot(x_seg, y_seg, 'b.', markersize=2, label='Data')
            plt.plot(x_seg, gauss_linear(x_seg, *popt1), 'orange', label='Gaussian + Linear')
            plt.plot(x_seg, gaussian(x_seg, *popt1[:3]), 'red', label='Gaussian only')
            plt.title(f"{material} Peak {i + 1} (Gauss+Linear)")
            plt.xlabel("Channel")
            plt.ylabel("Counts")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            # ===FULL SPECTRUM PLOT (Gauss+Linear)===
            plt.figure()
            plt.plot(x, y, 'b.', markersize=1, label='Full Spectrum')
            plt.plot(x_seg, gauss_linear(x_seg, *popt1), 'orange', label='Gaussian + Linear')
            plt.plot(x_seg, gaussian(x_seg, *popt1[:3]), 'red', label='Gaussian only')
            plt.title(f"{material} Peak {i + 1} (Gauss+Linear)")
            plt.xlabel("Channel")
            plt.ylabel("Counts")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()


        except Exception as e:
            print(f"Gaussian+Linear fit failed: {e}")

        # Fit 2: Gaussian + Exponential
        try:
            p0 = [np.max(y_seg), x_seg[np.argmax(y_seg)], 10, np.max(y_seg), 0.001]
            popt2, pcov2 = curve_fit(gauss_exp, x_seg, y_seg, p0=p0, sigma=yerr, absolute_sigma=True)
            perr2 = np.sqrt(np.diag(pcov2))
            fwhm2 = 2.355 * popt2[2]
            err_center2 = estimate_center_error(fwhm2, np.sum(gaussian(x_seg, *popt2[:3])))
            chi2_2, redchi2_2, pval2 = fit_stats(x_seg, y_seg, gauss_exp, popt2, yerr)

            # Gaussian-only
            a0, a1, a2 = popt2[:3]
            a0_err, _, a2_err = perr2[:3]
            N_meas2 = a0 * np.sqrt(2 * np.pi) * a2
            dN_meas2 = np.sqrt((a0_err * np.sqrt(2 * np.pi) * a2) ** 2 + (a2_err * np.sqrt(2 * np.pi) * a0) ** 2)

            # Plot Gaussian-only
            plt.figure()
            plt.plot(x_seg, y_seg, label='Data', color='green')
            plt.plot(x_seg, gaussian(x_seg, *popt2[:3]), color='orange', label='Gaussian (no exp BG)')
            plt.title(f'{material} Peak {i + 1} - Gauss+Exp Subtracted')
            plt.xlabel("Channel")
            plt.ylabel("Counts")
            plt.legend()
            plt.tight_layout()
            plt.show()

            results.append([
                material, i + 1, 'Gauss+Exp',
                x_start, x_end,
                *popt2, *perr2,
                fwhm2, 2.355 * perr2[2],
                np.sum(gaussian(x_seg, *popt2[:3])), err_center2,
                N_meas2, dN_meas2,chi2_1,redchi2_2,pval2
            ])
            # Visualize Gauss+Exp full fit and isolated Gaussian
            plt.figure()
            plt.plot(x_seg, y_seg, 'b.', markersize=2, label='Data')
            plt.plot(x_seg, gauss_exp(x_seg, *popt2), 'orange', label='Gaussian + Exponential')
            plt.plot(x_seg, gaussian(x_seg, *popt2[:3]), 'red', label='Gaussian only')
            plt.title(f"{material} Peak {i + 1} (Gauss+Exp)")
            plt.xlabel("Channel")
            plt.ylabel("Counts")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            # === FULL-SPECTRUM PLOT (Gauss+Exp) ===
            plt.figure()
            plt.plot(x, y, 'b.', markersize=1, label='Full Spectrum')
            plt.plot(x_seg, gauss_exp(x_seg, *popt2), 'orange', label='Gaussian + Exponential')
            plt.plot(x_seg, gaussian(x_seg, *popt2[:3]), 'red', label='Gaussian only')
            plt.title(f"{material} Peak {i + 1} (Gauss+Exp)")
            plt.xlabel("Channel")
            plt.ylabel("Counts")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()


        except Exception as e:
            print(f"Gaussian+Exp fit failed: {e}")

    return results


# === Full Workflow ===
def main():
    bg_file = easygui.fileopenbox(msg="Select background .mca file", filetypes=["*.mca"])
    if not bg_file:
        return
    ##bg_counts, _, bg_live = load_mca_file(bg_file)
    bg_counts, bg_real, bg_live = load_mca_file(bg_file)

    sample_files = easygui.fileopenbox(msg="Select sample .mca files", filetypes=["*.mca"], multiple=True)
    if not sample_files:
        return

    all_results = []

    for filepath in sample_files:
        ##sample_counts, _, sample_live = load_mca_file(filepath)
        sample_counts, sample_real, sample_live = load_mca_file(filepath)
        print("live time:",sample_live,"real time:",sample_real)
        corrected_counts = subtract_background(sample_counts, sample_live, bg_counts, bg_live)
        x = np.arange(len(corrected_counts))
        # Plot original, background, and corrected spectra(later addition for vis)
        plt.figure(figsize=(10, 6))
        plt.plot(x, sample_counts, label="Sample", color="blue", alpha=0.6)
        plt.plot(x, (sample_live / bg_live) * bg_counts, label="Scaled Background", color="gray", alpha=0.6)
        plt.plot(x, corrected_counts, label="Corrected (Sample - Background)", color="green", alpha=0.8)
        plt.title(f"Spectrum Before and After Background Subtraction\n{os.path.basename(filepath)}")
        plt.xlabel("Channel")
        plt.ylabel("Counts")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


        selector = InteractivePeakSelector(x, corrected_counts)
        regions = selector.get_regions()
        print(f"Regions for {os.path.basename(filepath)}: {regions}")

        results = analyze_peaks(x, corrected_counts, regions, filepath)
        all_results.extend(results)
    columns = [
        "Material", "Peak #", "Fit Type", "x_start", "x_end",
        "a0", "a1", "a2", "a3", "a4",
        "a0_err", "a1_err", "a2_err", "a3_err", "a4_err",
        "FWHM", "FWHM Error", "Counts (N)", "σ_center (FWHM/2.35√N)",
        "N_meas (a0*sqrt(2pi)*a2)", "ΔN_meas",  # <-- new
        "Chi2", "Chi2/dof", "p-value"
    ]

    df = pd.DataFrame(all_results, columns=columns)
    print(df.to_string(index=False))

    save_path = easygui.filesavebox(default="EnergyCalibFits_BGsubtracted.xlsx", filetypes=["*.xlsx"])
    if save_path:
        if not save_path.endswith(".xlsx"):
            save_path += ".xlsx"
        df.to_excel(save_path, index=False)
        print(f" Saved results to {save_path}")

if __name__ == "__main__":
    main()