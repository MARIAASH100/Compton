import pandas as pd
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Constants
pi = np.pi

# Open file
Tk().withdraw()
file_path = askopenfilename(title="Select Excel file", filetypes=[("Excel files", "*.xlsx *.xls")])

# Read file
#df = pd.read_excel(file_path)
df = pd.read_excel(file_path, sheet_name="Sheet2")

# Convert units (μCi to Bq)
df["A0_Bq"] = df["A_0(microCi)"] * 3.7e4 #yeah we did it previously so we must do it here to

# Extract columns
A0 = df["A0_Bq"]
##A0 = df["A_0(microCi)"]
tau = df["tau"]
Dtau = df["Dtau"]
t = df["t(yr)"]
Dt = df["Dt(yr)"]
I_gamma = df["I_gamma(%)"] / 100 #when you put in excel the I_gamma: if it is 85% so put 85
DI_gamma = df["DI_gamma(%)"] / 100
T = df["Live_Time(sec)"]
DT = df["DLive_Time(sec)"]
S = df["S(cm^2)"]
DS = df["DS(cm^2)"]
R2 = df["R^2 (cm^2)"]
DR2 = df["DR^2 (cm^2)"]

# Calculate N_exp
exp_term = np.exp(-t / tau)
N_exp = A0 * exp_term * I_gamma * T * S / (4 * pi * R2)

# Partial derivatives
dN_dtau = A0 * exp_term * I_gamma * T * S / (4 * pi * R2) * (t / tau**2)
dN_dI = A0 * exp_term * T * S / (4 * pi * R2)
dN_dt = -A0 * exp_term * I_gamma * T * S / (4 * pi * R2) * (1 / tau)
dN_dT = A0 * exp_term * I_gamma * S / (4 * pi * R2)
dN_dS = A0 * exp_term * I_gamma * T / (4 * pi * R2)
dN_dR2 = -A0 * exp_term * I_gamma * T * S / (4 * pi * R2**2)

# Total uncertainty
DN_exp = np.sqrt(
    (dN_dtau * Dtau)**2 +
    (dN_dI * DI_gamma)**2 +
    (dN_dt * Dt)**2 +
    (dN_dT * DT)**2 +
    (dN_dS * DS)**2 +
    (dN_dR2 * DR2)**2
)

# Add to DataFrame
df["N_exp"] = N_exp
df["DN_exp"] = DN_exp

# Save to Excel
output_path = file_path.replace(".xlsx", "_Nexp_Output.xlsx")
df.to_excel(output_path, index=False)

print(f"Done! Results saved to:\n{output_path}")
