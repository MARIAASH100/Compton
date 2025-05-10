import numpy as np
import matplotlib.pyplot as plt
import os

# === Functions For Hard Times ===
def load_mca_file_with_time(filepath):
    counts = []
    live_time = None
    with open(filepath, 'r', encoding='latin1') as f:
        for line in f:
            line = line.strip()
            if line.startswith("LIVE_TIME"):
                live_time = float(line.split('-')[1].strip())
            elif line.isdigit():
                counts.append(int(line))
    return np.array(counts), live_time

def subtract_background(signal, t_signal, background, t_background):
    scale = t_signal / t_background
    corrected = signal - scale * background
    return np.clip(corrected, a_min=0, a_max=None)

# === File Paths ===
base_folder = r"C:\Users\anna1\PycharmProjects\LAB C - projects\ComptonScattering\background and energy calib\for together"

file_dict = {
    "Am-241": "Am241.mca",
    "Ba-133": "ba133.mca",
    "Cs-137": "Cs137Close.mca",
    "Na-22": "Na22.mca"
}
background_file = "no S - background.mca" #we took a long meas of it (will be explained)

# === Load Background ===
bg_path = os.path.join(base_folder, background_file)
background_counts, bg_live_time = load_mca_file_with_time(bg_path)

# === Plotting ===
plt.figure(figsize=(12, 6))
colors = ['red', 'blue', 'green', 'purple']

for i, (label, fname) in enumerate(file_dict.items()):
    sample_path = os.path.join(base_folder, fname)
    sample_counts, sample_live_time = load_mca_file_with_time(sample_path)

    # Subtract background
    corrected = subtract_background(sample_counts, sample_live_time, background_counts, bg_live_time)

    plt.plot(np.arange(len(corrected)), corrected, label=label, color=colors[i % len(colors)], alpha=0.85)

plt.title("Background-Subtracted Gamma-Ray Spectra ALL")
plt.xlabel("Channel [a.u.]")
plt.ylabel("Counts [a.u.]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




