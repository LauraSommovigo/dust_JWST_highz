#my functions
from scipy.integrate import quad
from matplotlib import cm
col_f = cm.get_cmap('viridis')
from librerie import *
fold_in = '../'
from scipy.optimize import fsolve
from scipy.optimize import curve_fit
from astropy.stats import bootstrap
from scipy.stats import expon
from astropy.cosmology import Planck15 as cosmo
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np
import cmasher as cmr
import emcee
from scipy.optimize import minimize
import corner
import h5py
import numpy.polynomial.polynomial as poly
from matplotlib import gridspec
import matplotlib.ticker as ticker
import pickle
from scipy.interpolate import interp1d
import pandas as pd
import seaborn as sns
import os


###################useful constants##################
alphab=2.6e-13
G = 6.67e-8#cgs
bi = -2.5 #UV spectral slope for MW
yr=3.154e+7#in s
#p0=numpy.array([10**4.0,10**8.0])
Msol=1.99e33#g
Lsol=3.9e33#erg/s
mp=1.66e-24#g, proton mass
kb = 1.38064852e-16#cgs
h = 6.63e-27 #ergs
c = 3e10 #cm/s
pc=3.09e18#cm
kpc=1e3*pc
To = 2.725#TCMB
Zsole=0.0142
D=1.0/163.0#dust to gas ratio MW
cost_fd=1.0
kv=3.482*1e4#cm^2/g, MW dust Draine+03



# Function definitions for attenuation curve parameters
def logAv_logSigma_SFR_y(logSigma_SFR_y):
    return -0.18902186 + 0.29927106 * logSigma_SFR_y

def logAv_logSigma_SFR(logSigma_SFR):
    return 0.0725817 + 0.40366813 * logSigma_SFR

def logc4_logAv(logAV):
    return -0.58556778 * logAV - 1.42466711

def logc1_logAv(logAV):
    return -0.37203682 * logAV + 0.74979339

def c3_logc1(logc1):
    return 1.21202635 * logc1 - 1.33081963

# Function to compute attenuation curve
def Li_08_fit_noratio(lam_micron, c1, c2, c3, c4):
    return (
        c1 / ((lam_micron / 0.08) ** c2 + (lam_micron / 0.08) ** -c2 + c3)
        + (233. * (1. - c1 / (6.88 ** c2 + 0.145 ** c2 + c3) - c4 / 4.6))
        / ((lam_micron / 0.046) ** 2. + (lam_micron / 0.046) ** -2. + 90.)
        + c4 / ((lam_micron / 0.2175) ** 2. + (lam_micron / 0.2175) ** -2. - 1.95)
    )

# Function to find the nearest wavelength index
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# Load TNG and Observational Data
tng_file = "//Users/lsommovigo/Desktop/Scripts/txt_files/LTU_att_curve/TNG_Att_SDSSbands.txt"
obs_file = "//Users/lsommovigo/Desktop/Scripts/txt_files/LTU_att_curve/OBS_Salim18_Att_SDSSbands.txt"

tng_columns = ["id", "los", "sfr100", "age", "log_Mstar", "Z", "Sigma_SFR_y", "Sigma_SFR", "A1500", "A2175", "Av", "Au", "Ag", "Ar", "Ai", "Az"]
obs_columns = ["id", "sfr100", "sSFR_100", "age", "log_Mstar", "Z", "Sigma_SFR", "A1500", "A2175","Av", "Au", "Ag", "Ar", "Ai", "Az"]

tng_data = pd.read_csv(tng_file, comment='#', delimiter=',', header=None, names=tng_columns)
obs_data = pd.read_csv(obs_file, comment='#', delimiter=',', header=None, names=obs_columns)

# Convert to numeric, coerce errors to NaN
tng_data["Sigma_SFR"] = pd.to_numeric(tng_data["Sigma_SFR"], errors="coerce")
tng_data["Sigma_SFR_y"] = pd.to_numeric(tng_data["Sigma_SFR_y"], errors="coerce")
# Drop rows where either Sigma_SFR or Sigma_SFR_y is NaN
tng_data = tng_data.dropna(subset=["Sigma_SFR", "Sigma_SFR_y"])
# Compute log(Sigma_SFR) only for valid values
tng_data["logSigma_SFR"] = np.log10(tng_data["Sigma_SFR"])
tng_data["logSigma_SFR_y"] = np.log10(tng_data["Sigma_SFR_y"])


# Convert Sigma_SFR to numeric, coercing errors to NaN
obs_data["Sigma_SFR"] = pd.to_numeric(obs_data["Sigma_SFR"], errors="coerce")
# Drop rows where Sigma_SFR is NaN before applying log10
obs_data = obs_data.dropna(subset=["Sigma_SFR"])
# Compute log(Sigma_SFR) only for valid values
obs_data["logSigma_SFR"] = np.log10(obs_data["Sigma_SFR"])

# Compute attenuation curve parameters for TNG galaxies
tng_data["logAv_scaling"] = logAv_logSigma_SFR_y(tng_data["logSigma_SFR_y"])
tng_data["Av_scaling"] = 10**tng_data["logAv_scaling"]  # Convert log(A_V) to A_V
tng_data["logc1_scaling"] = logc1_logAv(tng_data["logAv_scaling"])
tng_data["logc4_scaling"] = logc4_logAv(tng_data["logAv_scaling"])
tng_data["c1"] = 10**tng_data["logc1_scaling"]
tng_data["c4"] = 10**tng_data["logc4_scaling"]
tng_data["c3"] = c3_logc1(tng_data["logc1_scaling"])


# Compute attenuation curve parameters for observational sample
obs_data["logAv_scaling"] = logAv_logSigma_SFR_y(obs_data["logSigma_SFR"])
obs_data["Av_scaling"] = 10**obs_data["logAv_scaling"]  # Convert log(A_V) to A_V
obs_data["logc1_scaling"] = logc1_logAv(obs_data["logAv_scaling"])
obs_data["logc4_scaling"] = logc4_logAv(obs_data["logAv_scaling"])
obs_data["c1"] = 10**obs_data["logc1_scaling"]
obs_data["c4"] = 10**obs_data["logc4_scaling"]
obs_data["c3"] = c3_logc1(obs_data["logc1_scaling"])

# Define wavelength array (microns)
lam_micron = np.logspace(-1, 0, 1000)  # Wavelength range from 0.1 to 1 microns

# Compute attenuation curves for each galaxy in TNG sample
attenuation_curves = np.array([
    Li_08_fit_noratio(lam_micron, row["c1"], 1.87, row["c3"], row["c4"])
    for _, row in tng_data.iterrows()
])

# Compute attenuation curves for each galaxy in the observational sample
obs_attenuation_curves = np.array([
    Li_08_fit_noratio(lam_micron, row["c1"], 1.89, row["c3"], row["c4"])
    for _, row in obs_data.iterrows()
])

# Define SDSS band wavelengths in microns
band_wavelengths = {
    "Av": 0.551, "A1500":0.1500, "A2175":0.2175, "Au": 0.3543, "Ag": 0.4770,
    "Ar": 0.6231, "Ai": 0.7625, "Az": 0.9
}


# Remove "Av" from band_wavelengths dictionary
bands_to_plot = {key: val for key, val in band_wavelengths.items() if key != "Av"}

# Initialize dictionary to store computed attenuation values
tng_AC_scaling = {}

# Initialize dictionary to store computed attenuation values for observations
obs_AC_scaling = {}

# Compute attenuation values at SDSS bands (excluding "Av")
for band, wavelength in bands_to_plot.items():
    idx = find_nearest(lam_micron, wavelength)
    tng_AC_scaling[band] = attenuation_curves[:, idx]
    obs_AC_scaling[band] = obs_attenuation_curves[:, idx]
    #print(f"Band: {band}, Index: {idx}, First 10 values:")
    #print(obs_AC_scaling[band][:10])

colors_los = cmr.take_cmap_colors("cmr.fusion", 51, cmap_range=(0.1, 1.),return_fmt='hex')

band_colors = {"Av": 'none', "A1500": 'Greys_r', "A2175": 'Greys_r',"Au": 'Blues_r', "Ag": 'Greens_r',
    "Ar": 'Reds_r', "Ai": 'Purples_r', "Az":'copper'}
band_lab = {"Av": "V", "A1500": "$1500 \dot{A}$", "A2175": "$2175 \dot{A}$", "Au": "U", "Ag": "G",
    "Ar": "R", "Ai": "I", "Az":"Z"}

## directory to save the plots
save_directory = "/Users/lsommovigo/Desktop"


# Now, plot comparisons
for band in bands_to_plot.keys():
    #if band not in tng_AC_scaling or np.all(np.isnan(tng_AC_scaling[band])):
    #    continue  # Skip this band if no valid data

    # Compute the scaled attenuation values
    scaled_values = tng_data["Av_scaling"] * tng_AC_scaling[band]

    # Compute the scaled attenuation values for observations
    scaled_values_obs = obs_data["Av_scaling"] * obs_AC_scaling[band]
    
    # Remove NaNs before plotting
    mask = (~np.isnan(scaled_values)) & (~np.isnan(tng_data[band])) & (tng_data[band] > 1e-2) & (tng_data["Z"] > 0) & (tng_data["Av"] > 0)
    x_data, y_data, los_values, logmstar_data, sfr_data,zgas_data, sSFR_data, age_data, av_data  = tng_data[band][mask], scaled_values[mask], tng_data["los"][mask], tng_data["log_Mstar"][mask], tng_data["sfr100"][mask], tng_data["Z"][mask], 10**tng_data["log_Mstar"][mask]/tng_data["sfr100"][mask], tng_data["age"][mask], tng_data["Av"][mask]

    # Remove NaNs before plotting (Observations)
    mask_obs = (~np.isnan(scaled_values_obs)) & (~np.isnan(obs_data[band])) & (obs_data[band] > 1e-2) & (obs_data["Z"] > 0) & (obs_data["Av"] > 0)
    x_data_obs, y_data_obs = obs_data[band], scaled_values_obs

    # Get the darkest color from the colormap
    cmap = plt.get_cmap(band_colors[band])  # Load colormap
    darkest_color = cmap(0)  # Near the end of the colormap range (darkest)

    
    # Create figure
    plt.figure(figsize=(6, 6))

    # Scatter plot of individual data points
    #plt.scatter(x_data, y_data, alpha=0.1, s=10,edgecolor=darkest_color,linewidth=0.04,marker='o',color='none',zorder=-10000)
    
    # Scatter plot for Observational data
    #plt.scatter(x_data_obs, y_data_obs, alpha=0.8, s=100, edgecolor="red", linewidth=1.4, marker='s', color='none', label="Observations", zorder=9000)
    #print(x_data_obs/y_data)
    
    # Contour density plot (only if enough valid data points exist)
    if len(x_data) > 10:  # Avoid KDE errors with very few points
        sns.kdeplot(x=x_data, y=y_data, levels=np.linspace(0.3,0.7,5), cmap=band_colors[band], fill=True, linewidth=2,alpha=.7,clip=(1e-4,4.), thresh=0, gridsize=200,zorder=1000, label='TNG, '+str(band_lab[band]))
        sns.kdeplot(x=x_data_obs, y=y_data_obs, levels=[0.3,0.5,0.7], color='black', fill=False, linewidth=5,alpha=.5,clip=(1e-4,4.), thresh=0, gridsize=200,zorder=2000,ls=':', label='Obs., '+str(band_lab[band]))

    # 1:1 reference line
    plt.plot([min(x_data), max(x_data)],
             [min(x_data), max(x_data)], ls='--',lw=2.5,alpha=0.2,zorder=1000,color='black')

    
    # Labels & Title
    plt.xlabel("$A_{\lambda}$ (RT)",fontsize=20)
    plt.ylabel("$A_{\lambda}$ (scaling rel.)", fontsize=20)
    #plt.title(f"Comparison for {band}")
    plt.legend(fontsize=15,frameon=False)

    plt.ylim(0.,np.percentile(x_data,92))
    plt.xlim(0.,np.percentile(x_data,92))
    
    # Save the plot with the band name in the specified directory
    save_path = os.path.join(save_directory, f"Obs_AC_{band}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    # Show plot
    plt.tight_layout()
    #plt.show()
    plt.close()

    '''

    #log sSFR
    qtity=np.log10(sSFR_data)#zgas_data)
    
    plt.scatter(qtity,(y_data-x_data)/y_data,alpha=0.1, s=10,edgecolor='grey',linewidth=0.05,marker='o',color='none',zorder=-10000)
    sns.kdeplot(x=qtity, y=(y_data-x_data)/y_data, levels=[0.16,0.3,0.5,0.7,0.84], cmap=band_colors[band], fill=True, linewidth=2,alpha=.7,thresh=0, gridsize=200,zorder=1000)
    plt.ylabel('$(A_{\lambda,RT}-A_{\lambda,SR})/A_{\lambda,SR}$')
    plt.xlabel('$ \log (sSFR/yr^{-1})$')#'$\log(Z/Z_{\odot})$')#'SFR [$M_{\odot}$/yr]')
    plt.axhline(0,ls='--',lw=2.5,alpha=0.2,zorder=1000,color='black')
    #plt.xscale('log')
    #plt.yscale('log')
    plt.ylim(-4.,4.)
    plt.tight_layout()
    plt.show()
    
    
    #log age
    qtity=np.log10(age_data)#zgas_data)
    
    plt.scatter(qtity,(y_data-x_data)/y_data,alpha=0.1, s=10,edgecolor='grey',linewidth=0.05,marker='o',color='none',zorder=-10000)
    sns.kdeplot(x=qtity, y=(y_data-x_data)/y_data, levels=[0.16,0.3,0.5,0.7,0.84], cmap=band_colors[band], fill=True, linewidth=2,alpha=.7,thresh=0, gridsize=200,zorder=1000)
    plt.ylabel('$(A_{\lambda,RT}-A_{\lambda,SR})/A_{\lambda,SR}$')
    plt.xlabel('$ \log (Age/Gyr)$')#'$\log(Z/Z_{\odot})$')#'SFR [$M_{\odot}$/yr]')
    plt.axhline(0,ls='--',lw=2.5,alpha=0.2,zorder=1000,color='black')
    #plt.xscale('log')
    #plt.yscale('log')
    plt.ylim(-4.,4.)
    plt.xlim(0.3,1.1)
    plt.tight_layout()
    plt.show()
    
    
    #log SFR
    qtity=np.log10(sfr_data)#zgas_data)
    
    plt.scatter(qtity,(y_data-x_data)/y_data,alpha=0.1, s=10,edgecolor='grey',linewidth=0.05,marker='o',color='none',zorder=-10000)
    sns.kdeplot(x=qtity, y=(y_data-x_data)/y_data, levels=[0.16,0.3,0.5,0.7,0.84], cmap=band_colors[band], fill=True, linewidth=2,alpha=.7,thresh=0, gridsize=200,zorder=1000)
    plt.ylabel('$(A_{\lambda,RT}-A_{\lambda,SR})/A_{\lambda,SR}$')
    plt.xlabel('$ \log (SFR/M_{\odot}yr^{-1})$')#'$\log(Z/Z_{\odot})$')#'SFR [$M_{\odot}$/yr]')
    plt.axhline(0,ls='--',lw=2.5,alpha=0.2,zorder=1000,color='black')
    #plt.xscale('log')
    #plt.yscale('log')
    plt.ylim(-4.,4.)
    plt.tight_layout()
    plt.show()

    
    #log Z
    qtity=np.log10(zgas_data)#zgas_data)
    
    plt.scatter(qtity,(y_data-x_data)/y_data,alpha=0.1, s=10,edgecolor='grey',linewidth=0.05,marker='o',color='none',zorder=-10000)
    sns.kdeplot(x=qtity, y=(y_data-x_data)/y_data, levels=[0.16,0.3,0.5,0.7,0.84], cmap=band_colors[band], fill=True, linewidth=2,alpha=.7,thresh=0, gridsize=200,zorder=1000)
    plt.ylabel('$(A_{\lambda,RT}-A_{\lambda,SR})/A_{\lambda,SR}$')
    plt.xlabel('$ \log (Z/Z_{\odot})$')#'$\log(Z/Z_{\odot})$')#'SFR [$M_{\odot}$/yr]')
    plt.axhline(0,ls='--',lw=2.5,alpha=0.2,zorder=1000,color='black')
    #plt.xscale('log')
    #plt.yscale('log')
    plt.ylim(-4.,4.)
    plt.tight_layout()
    plt.show()
    
    
    #log Mstar
    qtity=logmstar_data#zgas_data)
    
    plt.scatter(qtity,(y_data-x_data)/y_data,alpha=0.1, s=10,edgecolor='grey',linewidth=0.05,marker='o',color='none',zorder=-10000)
    sns.kdeplot(x=qtity, y=(y_data-x_data)/y_data, levels=[0.16,0.3,0.5,0.7,0.84], cmap=band_colors[band], fill=True, linewidth=2,alpha=.7,thresh=0, gridsize=200,zorder=1000)
    plt.ylabel('$(A_{\lambda,RT}-A_{\lambda,SR})/A_{\lambda,SR}$')
    plt.xlabel('$ \log (M_{\star}/M_{\odot})$')#'$\log(Z/Z_{\odot})$')#'SFR [$M_{\odot}$/yr]')
    plt.axhline(0,ls='--',lw=2.5,alpha=0.2,zorder=1000,color='black')
    #plt.xscale('log')
    #plt.yscale('log')
    plt.ylim(-4.,4.)
    plt.tight_layout()
    plt.show()


    #Av
    qtity=av_data#zgas_data)
    
    plt.scatter(qtity,(y_data-x_data)/y_data,alpha=0.1, s=10,edgecolor='grey',linewidth=0.05,marker='o',color='none',zorder=-10000)
    sns.kdeplot(x=qtity, y=(y_data-x_data)/y_data, levels=[0.16,0.3,0.5,0.7,0.84], cmap=band_colors[band], fill=True, linewidth=2,alpha=.7,thresh=0, gridsize=200,zorder=1000, clip=((0,3),(-4,4)))
    plt.ylabel('$(A_{\lambda,RT}-A_{\lambda,SR})/A_{\lambda,SR}$')
    plt.xlabel('$ A_{V}$ [mag]')#'$\log(Z/Z_{\odot})$')#'SFR [$M_{\odot}$/yr]')
    plt.axhline(0,ls='--',lw=2.5,alpha=0.2,zorder=1000,color='black')
    #plt.xscale('log')
    #plt.yscale('log')
    plt.ylim(-1.5,1.5)
    plt.xlim(-0.01,1.5)
    
    # Save the plot with the band name in the specified directory
    save_path = os.path.join(save_directory, f"err_Av_{band}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    # Show plot
    plt.tight_layout()
    #plt.show()
    plt.close()


    #inclination
    qtity=los_values#zgas_data)
    
    plt.scatter(qtity,(y_data-x_data)/y_data,alpha=0.1, s=10,edgecolor='grey',linewidth=0.05,marker='o',color='none',zorder=-10000)
    sns.kdeplot(x=qtity, y=(y_data-x_data)/y_data, levels=[0.16,0.3,0.5,0.7,0.84], cmap=band_colors[band], fill=True, linewidth=2,alpha=.7,thresh=0, gridsize=200,zorder=1000)
    plt.ylabel('$(A_{\lambda,RT}-A_{\lambda,SR})/A_{\lambda,SR}$')
    plt.xlabel('los')#'$\log(Z/Z_{\odot})$')#'SFR [$M_{\odot}$/yr]')
    plt.axhline(0,ls='--',lw=2.5,alpha=0.2,zorder=1000,color='black')
    #plt.xscale('log')
    #plt.yscale('log')
    plt.ylim(-1.5,1.5)
    plt.tight_layout()
    plt.show()
    '''
