
# Compton-Scattering
A Compton scattering experiment using a Cs-137 source to measure angular-dependent gamma scattering and validate the Klein–Nishina cross section against classical and quantum predictions.
# The system's user manual for part B and A  .docx file are in the final stages of preparation and will be uploaded on 15/05
# Computational methods with statistics with equations( will be uplad till 16/05 )
1)For the extrapolation of the photopeak in the angle calibration, we chose either linear or exponential extrapolation, as mentioned in an article that discusses which background model is more suitable for different gamma ray energies("Ng, Edwin. "The Kinematics and Electron Cross Sections of Compton Scattering."")
2) How do you know if your resolution is bad? If the Gaussian is under 5σ and it’s hard to clearly see the peak, then the resolution probably isn’t good enough. First, figure out how long you’d need to measure to get a 5σ peak. If that’s not doable, try using some filtering to help.
Suggested solution for poor resolution: To smooth noise in the gamma spectra, we will use the Savitzky–Golay filter: 
* https://crp2a.github.io/gamma/reference/smooth.html
* Good for low statistics of gamma : “Analysis of Gamma Spectrum with low statistics”S. Balakrishnan1 , P. K. Joshi2 https://sympnp.org/proceedings/68/G28.pdf => reduce noise and enhance peak clarity in gamma spectra and we are happier
3) The MCA system used ORTEC 4001A modules with very accurate Live Time, so any error from Live Time was much smaller than the usual measurement and fitting errors, and could be ignored (we included it).
4) According to Krane, page 224, we may take the statistical uncertainty of the peak centroid (channel position) as:
$$
\sigma_{{centroid}} = \frac{{FWHM}}{2.35 \sqrt{N}}
$$
This formula gives the statistical uncertaintyin the position of the peak centroid, when: 
FWHM is the full width at half maximum of the peak.
N is the total number of counts under the peak.
see: https://www.ortec-online.com/-/media/ametekortec/third-edition-experiments/10-compton-scattering.pdf?la=en&revision=dc0de3b8-aacc-498f-adf5-610df8b47915&hash=41EE9FF9FB8303EC409C0023BE06A4C2
5) We too ideas from "The Mirion Nuclear Science Experiments Manual" , good for compton scattering experiments because it provides detailed procedures, theoretical background, and necessary equipment specifications to accurately measure and analyze gamma ray interactions, see: https://assets-mirion.mirion.com/prod-20220822/cms4_mirion/files/images/brochure-images/ops-1476_nuclear_science_experiments_manual.pdf
6) For Part A, we tried both background subtraction methods: scaled background subtraction and local background fitting using exponential and linear models. We found that the scaling method gave a little better results. In the scaling method, we take a background measurement without the radioactive source to see how the spectrum behaves across all channels. Since this background has counts across the full range, we scale it by the ratio of live time between the background and the sample measurements. This approach is commonly used in video and image processing, where background subtraction relies on pixel-based scaling. We applied a similar idea here, assuming that the background behaves linearly and can be corrected by scaling in the same way.

# How to measure (updated till 15)
# Problems occurred during the measurement & How to aviod them (updated till 16)
1) For angle calibration : We measured the angles at different times (which we recommend not to do). There's no real need for that just set the live time to 2 minutes for each measurement. And if you do use different times, we suggest normalizing the results by the live time. Without normalizing by live time, the results might look incorrect just because the measurements were taken for different amounts of time not because anything actually changed. Normalizing makes sure the angles are compared fairly. see: https://link.springer.com/article/10.1007/s10967-024-09729-7
2) dead time and pile-up can cause missed counts and make nuclear measurements less accurate (not perfectly random like Poisson), and shows ways to fix or reduce those errors(for the first part maybe good - we didn't do that), Uncertainty of nuclear counting: https://iopscience.iop.org/article/10.1088/0026-1394/52/3/S3/pdf . Maybe checking is the dead time can be excluded by making sure the count rate is low enough that its effect is negligible: https://phyusdb.wordpress.com/wp-content/uploads/2013/03/radiationdetectionandmeasurementbyknoll.pdf
3) (good for part B) The electron’s kinetic energy depends on the energy of the scattered gamma. for example, it gets higher when the gamma is scattered backward. So in the future, we want to find the connection between voltage and electron energy, to help us tell apart true gamma scattering events. https://indico.cern.ch/event/426015/contributions/1047617/attachments/906081/1278753/Compton_experiment_2005.pdf
# Angle Calibration: 
## compton-angleCalib3.py
Loads multiple .mca files from a selected folder(you choose), and the user interactively select a peak region in each spectrum, fits and subtracts a linear background(Since we use a high energy gamma source (Cs-137), the background is expected to vary slowly with energy. Therefore, we assume it can be well approximated by a linear function within the selected region), fits a Gaussian to the peak, extracts peak parameters (amplitude, center, width), and saves all results (including statistical fit quality and angle from filename) into an Excel file.
Input: A folder containing .mca files with gamma spectra, where filenames include the angle and names as ang(30).mca for example.
Output: An Excel file summarizing the Gaussian fit results for each spectrum, including angle, peak parameters, errors, FWHM, and chi-squared statistics.
## CF - AngCalibFit.py
Loads fitted peak data from an Excel file (from the previous spectrum analysis), and the user interactively select an angular region, fits multiple models (Gaussian, Parabola, Cosine) to the "normalized Compton scattering amplitude vs angle data", compares fit quality, and visualizes the results and residuals in both radians and degrees.
Input: An Excel file you got in the previous script containing fitted peak parameters and angles from MCA spectra.
Output: Interactive plots of fitted curves and residuals, along with printed best-fit parameters and goodness-of-fit statistics for each model.
Note: The best fitting model (Gaussian) was plotted separately after it was identified as the most suitable fit based on the statistical comparison. :)
# Energy Channel Calib: 
## "compton - visual peak show best fit of peak.py" or "compton - energychannel - step1(excel&no BG&all parameters&N_meas).py"

first sctipt:  Loads .mca gamma spectra files, allows the user to interactively select peak regions, fits each selected peak using both "Gaussian + linear background" and "Gaussian + exponential background" models, compares the fits based on statistical criteria (χ² and p-value), and visualizes both the individual and overlaid fits for each peak.
Input: One or more .mca files(Am,Cs,Ba,Na) containing gamma ray spectra with real/live.
Output: Interactive plots showing raw spectra, best fit models per peak (including Gaussian-only and full background models), and printed fit parameters with uncertainties and goodness of its statistics.

Second script: The script performs background subtraction on gamma-ray spectra using a selected background .mca file (obtained by turning off all radioactive sources and measuring for an extended time to capture the environmental background only). It then allows the user to interactively select peak regions in multiple sample spectra, fits each peak with both Gaussian+linear and Gaussian+exponential models, extracts peak parameters and statistical metrics, and saves the complete analysis results to an Excel file.
Input: One background .mca file and one or more sample .mca files(Am,Cs,Ba,Na) containing gamma ray spectra with real/live time metadata.
Output: Interactive plots of raw, background, and corrected spectra with fitted peaks, plus an Excel file (EnergyCalibFits_BGsubtracted.xlsx) summarizing peak positions, widths, counts, uncertainties, and fit quality statistics for each peak.
Note: This script is particularly suitable for elements like Americium (Am), whose decay rate is relatively slow, making the environmental background more significant and harder to distinguish.
## compton - CountsVsChannel all elements toget
This script loads gamma ray spectra from several radioactive sources, subtracts a long measured background spectrum scaled by live time, and plots all the corrected spectra on a single graph for visual comparison.
Input: .mca files for each isotope ( Am-241, Cs-137...) and a background .mca file with live time data.
Output: A single overlaid plot showing the background-subtracted spectra of all sources.
Note: This method is especially useful when resolution is poor or conditions are challenging (“life is hard and yes... GOD play dice and we’re left fitting Gaussians to figure out what just happened”) — so by plotting all isotopes together, we can visually assess whether the channel energy calibration is consistent and identify anomalies like the characteristic of another peak of Am-241 near 300 keV.
## compton - partA - EnergyChannelCalibFit.py
This script loads calibration data from an Excel file, performs a weighted linear fit between channel number and known gamma ray energies, computes uncertainties and statistical metrics (χ², χ² red, p-value, Residuals etc...), and visualizes the calibration fit and residuals.
Input: An Excel file containing columns for real energies and their uncertainties (from LiveChart), and channel centers and their uncertainties (extracted from the previous peak fitting script).
Output: Printed best fit parameters with uncertainties and statistical quality, a plot of the calibration line with data points, and a residuals plot showing the fit accuracy.
# Efficiency Energy Calib:
## compton - PartA - calculate N_exp with error.py
This script calculates the expected gamma ray counts $N_{\text{exp}}$ theoretically detected by a detector from a radioactive source at the current time, based on the decay law and geometric/detection parameters.
* The computation is based on the exponential decay formula, taking into account decay time, live time, emission intensity, and detector geometry.
* The uncertainties in all input parameters are propagated analytically to estimate the total uncertainty Δ$N_{\text{exp}}$
* The values used in the calculation are: Source-specific constants (half-life, gamma intensity) obtained from LiveChart, Detector geometry and measurement time (area, distance, live time) -> All values are entered into an Excel file, which serves as the input for this script.
* Bq units was used as they directly represents disintegrations per second, which matches the units expected in the formula for predicting the number of counts recorded over time by the MCA (Multi Channel Analyzer).
Input: Excel file containing theoretical values: Initial activity $A_0$ in $\mu$Ci ,Lifetime $\tau$,  time since calibration t, gamma ray fruction $I_{\gamma}$, Detector live time T, Detector surface area S, Distance squared $R^2$ , And the associated uncertainties for each of these parameters.
Output: A new Excel file with added columns $N_{\text{exp}}$ expected number of detected gamma counts and Δ$N_{\text{exp}}$ total propagated uncertainty.
## compton - PartA - EfficiencyEnergyCalibFit.py
This script fits an analytical efficiency model to gamma ray detector efficiency data (measured as $\varepsilon = \frac{N_{\text{meas}}}{N_{\text{exp}}}$) as a function of energy, evaluates the fit quality (χ², p-value), and visualizes both the fit and residuals.
Input: An Excel file with four columns: energy values (keV), efficiency values, and their respective uncertainties.
Output: Printed fit parameters with uncertainties and statistical fit quality (χ², reduced χ², p-value), and two plots: the efficiency fit vs. energy and the residuals.
# Part A and B 
## "compton - FindMeasPeaksWithAng - Part A+B(Not include BG).py" or "compton - FindMeasPeaksWithAng - Part A+B.py"

=First Script=: Processes a folder of .mca gamma ray spectra files labeled by scattering angle, allows the user to interactively select peaks in each spectrum, fits each peak with both Gaussian+linear and Gaussian+exponential background models, calculates fit parameters and detector calibrated energies, and exports all results along with the best fit per peak to Excel.
Input: A folder containing .mca files named with angle as ang(30).mca 
Output: Two Excel files=> one with full fit results (ComptonAngle_Fits.xlsx) and one with the best fit per peak (ComptonAngle_Fits_best_fit.xlsx), plus visual plots of each fit during execution.

Second script: This script loads a background .mca file and a folder of angle-labeled .mca spectra files, performs background subtraction, allows interactive peak selection, fits each selected peak with both Gaussian+linear and Gaussian+exponential models, applies an energy calibration, and saves full fit results and the best fits to Excel.
Input: One background .mca file (measured without the source), a folder containing sample .mca files named with angles like ang(30).mca, each file must contain REAL_TIME and LIVE_TIME.
Otput: ComptonAngle_Fits.xlsx: full table of all peak fits (parameters, uncertainties, calibrated energies, fit statistics) and ComptonAngle_Fits_best_fit.xlsx=> a filtered table with the best fit for each angle/peak based on reduced chi-squared

Note: We use the first script in Part B for all the angles, as it introduces less error from external environmental contamination.
## compton - PartA&B - find sigma and dsigma (alu or pal).py
This script calculates the differential cross section dσ/dΩ and its uncertainty for Compton scattering using experimental peak data, detector efficiency, and physical constants specific to the selected target material (Aluminum or Silicon).
Input: elected target material (alu or si) + an Excel file containing calibrated energy, counts, live time, and their uncertainties.
Output: A new Excel file with added columns for detector efficiency, its uncertainty + dσ/dΩ and its uncertainty.
## compton - partA&B - linear fit for energy shift.p
This script performs a weighted linear fit to Compton scattering data by plotting 1/E' vs 1-cos extracts the incident photon energy and electron rest mass from the fit, compares them to theoretical values using Nσ and visualizes both the fit and residuals.
Input: An Excel file containing calibrated energies and angles with uncertainties (generated by the previous script).
Output: Printed fit parameters and physical constants and two plots: the linear fit and the residuals.
## compton - PartA&B - Plar rep try 7.py
This script fits Compton scattering differential cross section data (from the previous script’s Excel output) using both the Klein–Nishina and Thomson models, compares the quality of fits (χ², p-value), visualizes the results on a polar plot, and extracts physical constants such as the classical electron radius r0 and incident photon energy.
Input: An Excel file containing columns for scattering angle (in radians), differential cross section dσ/dΩ and its uncertainty — generated by the previous script.
Output: A polar plot comparing experimental data to both theoretical models, and printed fit parameters, statistics, and extracted physical quantities.
