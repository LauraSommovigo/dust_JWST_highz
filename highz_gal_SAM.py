import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import odeint, quad
from astropy.cosmology import Planck18 as cosmo, z_at_value
import astropy.units as u
import matplotlib.cm as cm
from numpy import random
from hmf import MassFunction
import matplotlib.colors as colors
import pandas as pd
from matplotlib import gridspec
from scipy.interpolate import interp1d


# --- YOUR SFH here ---
sfh_times = np.array([0, 10, 20, 30, 40, 50])  # in Myr, 0 = latest/current

# ======== CONSTANTS ===========
kb     = 1.38064852e-16      # Boltzmann constant [erg/K]
c      = 3e10                # Speed of light [cm/s]
pc_to_cm     = 3.09e18             # Parsec [cm]
Mpc_to_cm    = 1e6 * pc_to_cm            # Megaparsec [cm]
kpc_to_cm    = 1e3 * pc_to_cm            # Kiloparsec [cm]
mp     = 1.67262192e-24      # Proton mass [g]
alphab = 2.6e-13             # Case B recombination coefficient [cgs]
G      = 6.6743e-8           # Gravitational constant [cgs]
Lsun   = 3.9e33              # Solar luminosity [erg/s]
Msun   = 1.98840987e33       # Solar mass [g]
yr     = 3.154e+7            # Year [s]
Myr    = 1e6 * yr            # Megayear [s]
mu_gas     = 1.22                # Mean molecular weight (typical ISM)
To     = 2.725               # CMB temperature at z=0 [K]
h      = 6.63e-27            # Planck constant [erg*s]
Zsun   = 0.0142              # Solar metallicity
D_MW    = 1. / 162.           # Milky Way dust-to-gas ratio
hlittle = cosmo.h            # Dimensionless Hubble parameter



# == Dust Model Parameters (computed in dust_JWST_z10_GSD.py; in principle other GSD can be tried) ==
kUV_hir = 18559.54915539  # 1500 Angstrom opacity [cm^2/g] for Hirashita+22 stellar dust
kUV_hir_abs = 9679.87080737  # 1500 Angstrom absorption opacity [cm^2/g] for Hirashita+22 stellar dust
kv_hir = 21158.51486479  # V-band opacity [cm^2/g] for Hirashita+22 stellar dust
kIR_hir = 8.32842305  # 158 micron opacity [cm^2/g] for Hirashita+22 stellar dust

kUV_drn = 68476.44934565  # 1500 Angstrom opacity [cm^2/g] for MW-like dust (Draine 2003)
kUV_drn_abs = 33961.82896294  # 1500 Angstrom absorption opacity [cm^2/g] for MW-like dust (Draine 2003)
kv_drn= 20593.65113121  # V-band opacity [cm^2/g] for MW-like dust (Draine 2003)
kIR_drn = 12.85127581  # 158 micron opacity [cm^2/g] for MW-like dust (Draine 2003)



### Fixing random seed for spin and sigma gas distribution
# Remove unused numpy random import for clarity
np.random.seed(42)


# =============================== #
# ========== FUNCTIONS =========== #
# =============================== #

# --- Halo to Stellar Mass Conversion
def halo_to_stellar_mass(Mh, fb, epsilon):
    """
    Return stellar mass from halo mass.
    """
    return epsilon * fb * Mh

def halo_to_stellar_mass_pl(Mh, fb, epsilon, alpha):
    """
    Return stellar mass from halo mass using a power-law scaling.
    """
    return epsilon * fb * Mh * (Mh/1e10)**alpha

# --- Halo Mass Function Fitting: Yung+23 (Appendix A)
def sigma_funct(Mh):
    """RMS density fluctuation for halo mass Mh [Msun]."""
    y = 1e12 / Mh
    return 26.80004233 * y**0.40695158 / (1. + 6.18130098 * y**0.23076433 + 4.64104008 * y**0.36760939)

def chi(chi_0, chi_1, chi_2, z):
    """Generic quadratic redshift (z) dependence"""
    return chi_0 + chi_1 * z + chi_2 * z**2.

def g_GUREFT(z):
    """Growth suppression function used by Yung+23 in Gureft."""
    a  = cosmo.scale_factor(z)
    om = cosmo.Om(z)
    ol = cosmo.Ode(z)
    return (0.4 * om * a)/(np.power(om,0.571428571)-ol+(1+om*0.5)*(1+ol*0.014285714))


def g_RP2016(z):
    a  = cosmo.scale_factor(z)
    om = cosmo.Om(z)
    ol = cosmo.Ode(z)
    num = 2.5 * om * a
    den = (om - ol) + (1.0 + 0.5*om) / (1.0 + ol/70.0)
    return num / den


def grad_funct(f, x, dx=1e-6):
    """
    Numerically compute the gradient of function f at x using central finite differences.

    Parameters
    ----------
    f : callable
        Function to differentiate.
    x : float
        Point at which to evaluate the derivative.
    dx : float, optional
        Step size for finite difference (default: 1e-6).

    Returns
    -------
    float
        Numerical derivative of f at x.
    """
    return (f(x + dx) - f(x - dx)) / (2 * dx)

def D(z):
    """Linear growth factor normalized to z=0"""
    return g_GUREFT(z) / g_GUREFT(0.0)

def f_sigma(Mh, z):
    """Fitting function for the mass function shape at various redshift z"""
    sigma = sigma_funct(Mh) * D(z)
    A_GFT = chi(0.13765772, -0.01003821, 0.00102964, z)
    a_GFT = chi(1.06641384, 0.02475576, -0.00283342, z)
    b_GFT = chi(4.86693806, 0.09212356, -0.01426283, z)
    c_GFT = chi(1.19837952, -0.00142967, -0.00033074, z)
    return A_GFT * ((sigma / b_GFT)**-a_GFT + 1.) * np.exp(-c_GFT / sigma**2.)

def dn_dlogMh_GUREFT(log10Mh, z):
    """
    Halo mass function dn/dlog10Mh using Yung+23 fit.
    Input: log10Mh (array), redshift (z)
    Output: number density [Mpc^-3 dex^-1]
    """
    rho_m = cosmo.Om(0) * cosmo.critical_density(0).value / cosmo.h**2
    Mh = 10**log10Mh
    # Chain rule for converting between sigma(M) and Mh
    d_log_sigma = np.gradient(-np.log(sigma_funct(Mh)), Mh)
    d_log_Mh = np.gradient(np.log(Mh), Mh)
    return (np.log(10) * f_sigma(Mh, z) * rho_m / (Mh * Msun) *
            np.abs(d_log_sigma / d_log_Mh) * Mpc_to_cm**3)

def dn_dMh_GUREFT(Mh, z):
    """Halo mass function per unit mass at redshift z"""
    rho_m = cosmo.Om(0) * cosmo.critical_density(0).value / cosmo.h**2
    d_log_sigma = np.gradient(-np.log(sigma_funct(Mh)), Mh)
    d_log_Mh = np.gradient(np.log(Mh), Mh)
    return (f_sigma(Mh, z) * rho_m / Mh**2 *
            np.abs(d_log_sigma / d_log_Mh) * Mpc_to_cm**3 / Msun)

# --- Star Formation Rate (SFR) / Mass Accretion
def dMhdt_GUREFT(Mh, z):
    """dMh/dt for halo growth (Yung+23), at redshift z"""
    return 10**logbeta_funct(z) * (1e-12 * Mh * E(z))**alpha_funct(z)  # Msun/yr


def alpha_funct(z):
    a = 1. / (1. + z)
    return chi(0.858, 1.554, -1.176, a)

def logbeta_funct(z):
    a = 1. / (1. + z)
    return chi(2.578, -0.989, -1.545, a)

def E(z):
    """E(z) = H(z)/H0"""
    return np.sqrt(cosmo.Om0 * (1 + z)**3 + cosmo.Ode0)

### Accretion rate from Sommovigo+22
def dMhdt_num_sims(Mh,redshift):
    dMdt=69.3*f(redshift)*E(redshift)*Mh/1e12#Msun/yr
    return dMdt#Msun/yr
def f(z):
    return -0.24 + 0.75 * (1.+z)


def Build_SFH_funct(Mh, z, tstep, eps):
    """
    Build SFH given accretion recipe, integrating until stellar mass reached.
    Returns: SFR_t, logMst (stellar mass), time_SFH [all arrays]
    """
    i = 0
    logMst = [0.0]
    SFH_z = []
    reds_arr = []
    # Mstar at the given z
    fb = cosmo.Ob(z) / cosmo.Om(z)
    Mstar = eps * fb * Mh
    while logMst[-1] < np.log10(Mstar):
        reds = z_at_value(cosmo.age, (cosmo.age(z).value - 1e-3 * tstep * i) * u.Gyr, method='bounded').value
        reds_arr.append(reds)
        sfr_val = eps * fb * dMhdt_GUREFT(Mh, reds)
        SFH_z.append(sfr_val)
        logMst.append(np.log10(tstep * 1e6 * np.sum(SFH_z)))
        i += 1
    time_SFH = np.linspace(0, i * tstep, i)
    SFR_t = np.flip(np.array(SFH_z))
    logMst = np.array(logMst[1:])
    return SFR_t, logMst, time_SFH


# --- Star Formation History (SFH) Construction
def Build_SFH_funct_pl(Mh, z, tstep, eps, alpha):
    """
    Build SFH given accretion recipe, integrating until stellar mass reached.
    Returns: SFR_t, logMst (stellar mass), time_SFH [all arrays]
    """
    i = 0
    logMst = [0.0]
    SFH_z = []
    reds_arr = []
    # Mstar at the given z
    fb = cosmo.Ob(z) / cosmo.Om(z)
    Mstar = eps * fb * Mh * (Mh/1e10)**alpha
    while logMst[-1] < np.log10(Mstar):
        reds = z_at_value(cosmo.age, (cosmo.age(z).value - 1e-3 * tstep * i) * u.Gyr, method='bounded').value
        reds_arr.append(reds)
        sfr_val = eps * fb * dMhdt_GUREFT(Mh, reds) * (Mh/1e10)**alpha
        SFH_z.append(sfr_val)
        logMst.append(np.log10(tstep * 1e6 * np.sum(SFH_z)))
        i += 1
    time_SFH = np.linspace(0, i * tstep, i)
    SFR_t = np.flip(np.array(SFH_z))
    logMst = np.array(logMst[1:])
    return SFR_t, logMst, time_SFH


def compute_L1500_steps_KS(age, tstep, SFH, time_yr_L1500, L1500_SB99):
    """
    Compute the 1500 Å monochromatic luminosity using the KS+98 scaling:
    L_1500 [erg/s/Hz] = 7.14e27 * SFR [Msun/yr]
    This is a direct conversion—no convolution with SB99 here!

    Parameters:
    - SFH: array-like, star formation rate [Msun/yr] at each time step

    Returns:
    - L1500: np.ndarray, UV luminosity at 1500 Å in [erg/s/Hz]
    """
    L1500 = 7.14e27 * SFH  # [erg/s/Hz], L_nu
    return np.array(L1500)

def L1500_to_MUV_conv(L_1500_nu):
    """
    Convert L_1500 [erg/s/Hz] (L_nu) to absolute AB magnitude at 1500 Å.

    Parameters:
    - L_1500_nu: float or array, luminosity in [erg/s/Hz]

    Returns:
    - MUV: float or array, absolute AB magnitude at 1500 Å
    """
    return -2.5 * np.log10(L_1500_nu) + 51.63

def L1500_lambda_to_Lnu(L1500_ang):
    """
    Convert L_1500 [erg/s/Å] (L_lambda) to L_nu [erg/s/Hz] at 1500 Å.
    """
    lambda_ang = 1500.0  # wavelength in Angstrom
    return L1500_ang * lambda_ang**2* 1e-8 /c

def compute_L1500_steps(age, tstep, SFH, time_yr_L1500, L1500_SB99):
    """
    Compute the monochromatic luminosity at 1500 Å (L_1500) [erg/s/Hz] at each time step,
    by convolving the SFH with the SB99 single stellar population (SSP) output.

    For each time step, sum the contribution from all previous bursts, accounting for their age.
    SB99 file gives log10(L_1500) [erg/s/Å] for an *instantaneous burst* of 1e6 Msun.

    Returns:
    - L1500_arr: np.ndarray, total L1500 [erg/s/Hz] at each age step
    """
    L1500_arr = []
    for ind in range(len(age)):
        L1500_step = 0
        for t in range(ind + 1):
            age_delay = 1e6 * tstep * (ind - t)  # time since burst, in years
            # interpolate to get L_1500 for this delay (per 1e6 Msun burst)
            L1500_ssp = 10**np.interp(age_delay, time_yr_L1500, L1500_SB99)
            mass_formed = SFH[t] * tstep * 1e6   # Msun formed in this burst (tstep in Myr!)
            L1500_step += L1500_ssp * (mass_formed / 1e6)  # rescale to burst mass
        L1500_arr.append(L1500_step)
    return L1500_lambda_to_Lnu(np.array(L1500_arr))


    
def compute_dotNion_steps(age, tstep, SFH, time_yr_Nion, log_dotNion):
    """
    Computes the instantaneous ionizing photon production rate (dot_Nion) at each age step.

    Parameters:
    -----------
    age : array-like
        Age grid (in Myr).
    tstep : float
        Time step size (in Myr).
    SFH : array-like
        Star formation rate history (Msun/yr) at each age step.
    time_yr_Nion : array-like
        SSP times (in yr) corresponding to the tabulated log_dotNion.
    log_dotNion : array-like
        log10 of the ionizing photon rate [photons/s] from stellar population models.

    Returns:
    --------
    dotNion_arr : np.ndarray
        Array of instantaneous ionizing photon rate [photons/s] at each age step.
    """

    dotNion_arr = []
    for ind in range(len(age)):
        dotNion_step = 0.0
        for t in range(ind+1):
            # Interpolate the SSP ionizing rate for the elapsed time since formation (in yr)
            dt_yr = 1e6 * (tstep * (ind - t))
            dotNion_SSP = 10**np.interp(dt_yr, time_yr_Nion, log_dotNion)
            dotNion_step += dotNion_SSP * SFH[t] * tstep
        dotNion_arr.append(dotNion_step)
    return np.array(dotNion_arr)



#----------------------------------------------------------
# --- DUST build up functions: from halo/dust properties
#----------------------------------------------------------

def r_vir(z, Mh):
    """
    Virial radius [kpc] for a given halo mass and redshift z.
    From Barkana & Loeb (2001).
    """
    deltac = 18 * np.pi**2 + 82 * (cosmo.Om(z) - 1.) - 39. * (cosmo.Om(z) - 1.)**2
    return (0.784 / cosmo.h * (Mh / (1e8 / cosmo.h))**(1. / 3.) *
            (cosmo.Om0 * deltac / (cosmo.Om(z) * 18 * np.pi**2))**(-1. / 3.) / ((1 + z) / 10.))

def rd_kpc(z, Mh, spin):
    """
    Disk scale length in kpc, for given halo mass and spin parameter.
    """
    return 4.5 * spin * r_vir(z, Mh)  # Disk scale length formula, 4.5 *

def compute_Mdust_steps(age, tstep, SFH, time_yr, logSNr_yr, yd):
    """
    Computes cumulative number of SNe and Mdust at each step.

    Parameters:
    - age: array of time steps (in Myr)
    - tstep: time step (in Myr)
    - SFH: star formation history (SFR at each step) [Msun/yr]
    - time_yr: time array for SNR in years
    - logSNr_yr: log10(SNR) array [per yr] for SN rate
    - yd: dust yield per SN [Msun]

    Returns:
    - N_SN_arr: cumulative SNe at each age step
    - Md_arr: Mdust at each age step [Msun]
    """
    N_SN = 0
    N_SN_arr = []
    for ind in range(len(age)):
        t = 0
        N_SN_step = 0
        while t <= ind:
            # Interpolate SN rate as a function of time delay (in yr)
            N_SN_interp = 10**np.interp(1e6 * (tstep * (ind-t)), time_yr, logSNr_yr)
            N_SN_step += N_SN_interp * 1e6 * tstep * SFH[t] * tstep
            t += 1
        N_SN += N_SN_step
        N_SN_arr.append(N_SN)
    N_SN_arr = np.array(N_SN_arr)
    Md_arr = N_SN_arr * yd
    return N_SN_arr, Md_arr


def tau_pred(klam, Md, Mh, spin, z):
    """
    Optical depth tau_V or tau_lambda, for MW-like geometry (sphere).
    """
    fmu = 4. / 3.   #sphere, mixed
    #fmu=0.841 #slab
    return klam * Md / (fmu * np.pi * rd_kpc(z, Mh, spin)**2) * Msun / kpc_to_cm**2

# --- UV Attenuation: T_1500
# Slab geometry
def int_W_l(t, x):
    chi = 2.0
    return (1.0 - t)**(chi - 1.0) * np.cosh(x * t)

def W_l(x):
    """
    Vectorized W_l(x) using scipy.integrate.quad for each scalar x.
    Works for scalars or numpy arrays.
    """
    chi = 2.0
    x_arr = np.atleast_1d(x).astype(float)

    W_vals = np.empty_like(x_arr, dtype=float)
    for i, xi in enumerate(x_arr):
        # quad expects args as a *tuple*
        W_vals[i] = chi * quad(int_W_l, 0.0, 1.0, args=(xi,))[0]

    # return scalar if input was scalar
    if np.isscalar(x):
        return W_vals[0]
    return W_vals

def T_slab(tau_lambda, mu, omega_lambda):
    """
    Slab geometry with scattering (Henyey-Greenstein).
    tau_lambda   : array-like, optical depth at each wavelength
    mu           : float, cos(inclination)
    omega_lambda : array-like, albedo at each wavelength
    """
    tau_lambda   = np.asarray(tau_lambda, dtype=float)
    omega_lambda = np.asarray(omega_lambda, dtype=float)

    x = (1.0 - omega_lambda) * tau_lambda / (2.0 * mu)
    W = W_l(x)                     # now safely vectorized
    Tl = (1.0 / mu) * np.exp(-x) * W
    return Tl

# Sphere geometry, point source, Code1973
def T_1500_sphere(tau_1500):
    """Transmission for spherical geometry, with central source. This accounts for scattering"""
    om_1500 = 0.3807#albedo
    g_1500 = 0.6633#scattering asimmetry factor
    eta = np.sqrt((1. - om_1500) / (1. - om_1500 * g_1500))
    psi = np.sqrt((1. - om_1500) * (1. - om_1500 * g_1500))
    Tl = 2. / ((1. + eta) * np.exp(psi * tau_1500) + (1. - eta) * np.exp(-psi * tau_1500))
    return Tl


# Sphere geometry, mixed, Otenbrock1989
def T_1500_sphere_im(tau_1500):
    """Transmission for spherical geometry, with stars and dust mixed. This accounts for geometry only."""
    Tl = 3./(4.*tau_1500) * (1. - 1./(2*tau_1500**2) + (1./tau_1500 + 1./(2*tau_1500**2)) * np.exp(-2.*tau_1500))
    #Tl=np.exp(-tau_1500)
    return Tl

# --- Functional form for the Attenuation Curve (not used at this stage) ---
def Li_08(lam_micron, c1, c2, c3, c4, model):
    """
    Computes an attenuation curve A(λ)/A_V following the Li+08 parameterization,
    with optional override for specific galaxy templates (Calzetti, SMC, MW, LMC).

    Parameters
    ----------
    lam_micron : float or ndarray
        Wavelength(s) in microns.
    c1, c2, c3, c4 : float
        Free parameters controlling the curve shape (ignored if model is specified).
    model : str
        If set to one of 'Calzetti', 'SMC', 'MW', 'LMC', uses fixed parameter values.

    Returns
    -------
    A_lam_v : float or ndarray
        Attenuation curve values A(λ)/A_V at the input wavelength(s).
    """

    # Override parameters for named models
    if model == 'Calzetti':
        c1, c2, c3, c4 = 44.9, 7.56, 61.2, 0.
    elif model == 'SMC':
        c1, c2, c3, c4 = 38.7, 3.83, 6.34, 0.
    elif model == 'MW':
        c1, c2, c3, c4 = 14.4, 6.52, 2.04, 0.0519
    elif model == 'LMC':
        c1, c2, c3, c4 = 4.47, 2.39, -0.988, 0.0221
    elif model == 'None':
        pass  # Use supplied c1–c4

    # Li+08 three-term parameterization, normalized to A_V
    A_lam_v = (
        c1 / ((lam_micron / 0.08)**c2 + (lam_micron / 0.08)**-c2 + c3) +
        (233. * (1 - c1 / (6.88**c2 + 0.145**c2 + c3) - c4 / 4.6)) /
        ((lam_micron / 0.046)**2. + (lam_micron / 0.046)**-2. + 90.) +
        c4 / ((lam_micron / 0.2175)**2. + (lam_micron / 0.2175)**-2. - 1.95)
    )

    return A_lam_v




#----- Parameters for dust grain size dristribution from Weingartner & Draine (2001) for Milky Way-type dust
#----- Different values of Rv and bC (carbon abundance in very small grains)
WD01_MW_PARAMS = {

    3.1: {
        "A": [
            {'bC':0.0, 'alpha_g':-2.25, 'beta_g':-0.0648, 'at_g':0.00745, 'ac_g':0.606, 'Cg':9.94e-11,
                           'alpha_s':-1.48, 'beta_s':-9.34,    'at_s':0.172, 'Cs':1.02e-12},
            {'bC':1.0, 'alpha_g':-2.17, 'beta_g':-0.0382, 'at_g':0.00373, 'ac_g':0.586, 'Cg':3.79e-10,
                           'alpha_s':-1.46, 'beta_s':-10.3,   'at_s':0.174, 'Cs':1.09e-12},
            {'bC':2.0, 'alpha_g':-2.04, 'beta_g':-0.111,  'at_g':0.00828, 'ac_g':0.543, 'Cg':5.57e-11,
                           'alpha_s':-1.43, 'beta_s':-11.7,   'at_s':0.173, 'Cs':1.27e-12},
            {'bC':3.0, 'alpha_g':-1.91, 'beta_g':-0.125,  'at_g':0.00837, 'ac_g':0.499, 'Cg':4.15e-11,
                           'alpha_s':-1.41, 'beta_s':-11.5,   'at_s':0.171, 'Cs':1.33e-12},
            {'bC':4.0, 'alpha_g':-1.84, 'beta_g':-0.132,  'at_g':0.00898, 'ac_g':0.489, 'Cg':2.90e-11,
                           'alpha_s':-2.10, 'beta_s':-0.14,   'at_s':0.169, 'Cs':1.26e-13},
            {'bC':5.0, 'alpha_g':-1.72, 'beta_g':-0.322,  'at_g':0.0254,  'ac_g':0.438, 'Cg':3.32e-12,
                           'alpha_s':-2.10, 'beta_s':-0.0407, 'at_s':0.166, 'Cs':1.27e-13},
            {'bC':6.0, 'alpha_g':-1.54, 'beta_g':-0.165,  'at_g':0.0107,  'ac_g':0.428, 'Cg':9.99e-12,
                           'alpha_s':-2.21, 'beta_s':0.300,   'at_s':0.164, 'Cs':1.00e-13},
        ]
    },

    4.0: {
        "A": [
            {'bC':0.0, 'alpha_g':-2.26, 'beta_g':-0.199, 'at_g':0.0241, 'ac_g':0.861, 'Cg':5.47e-12,
                           'alpha_s':-2.03, 'beta_s':0.668, 'at_s':0.189, 'Cs':5.20e-14},
            {'bC':1.0, 'alpha_g':-2.16, 'beta_g':-0.0862,'at_g':0.00867,'ac_g':0.803,'Cg':4.58e-11,
                           'alpha_s':-2.05, 'beta_s':0.832, 'at_s':0.188, 'Cs':4.81e-14},
            {'bC':2.0, 'alpha_g':-2.01, 'beta_g':-0.0973,'at_g':0.00811,'ac_g':0.696,'Cg':3.96e-11,
                           'alpha_s':-2.06, 'beta_s':0.995, 'at_s':0.185, 'Cs':4.70e-14},
            {'bC':3.0, 'alpha_g':-1.83, 'beta_g':-0.175, 'at_g':0.0117, 'ac_g':0.604,'Cg':1.42e-11,
                           'alpha_s':-2.08, 'beta_s':1.29,  'at_s':0.184, 'Cs':4.26e-14},
            {'bC':4.0, 'alpha_g':-1.64, 'beta_g':-0.247, 'at_g':0.0152, 'ac_g':0.536,'Cg':5.83e-12,
                           'alpha_s':-2.09, 'beta_s':1.58,  'at_s':0.183, 'Cs':3.94e-14},
        ],
        "B": [
            {'bC':0.0, 'alpha_g':-2.62,'beta_g':-0.0144,'at_g':0.0187,'ac_g':5.74,'Cg':6.40e-12,
                           'alpha_s':-2.01,'beta_s':0.894,'at_s':0.198,'Cs':4.95e-14},
            {'bC':1.0, 'alpha_g':-2.52,'beta_g':-0.0541,'at_g':0.0366,'ac_g':6.65,'Cg':1.08e-12,
                           'alpha_s':-2.11,'beta_s':1.58, 'at_s':0.197,'Cs':3.69e-14},
            {'bC':2.0, 'alpha_g':-2.36,'beta_g':-0.0957,'at_g':0.0305,'ac_g':6.44,'Cg':1.62e-12,
                           'alpha_s':-2.05,'beta_s':1.19, 'at_s':0.197,'Cs':4.37e-14},
            {'bC':3.0, 'alpha_g':-2.09,'beta_g':-0.193, 'at_g':0.0199,'ac_g':4.60,'Cg':4.21e-12,
                           'alpha_s':-2.10,'beta_s':1.64, 'at_s':0.198,'Cs':3.63e-14},
            {'bC':4.0, 'alpha_g':-1.96,'beta_g':-0.813, 'at_g':0.0693,'ac_g':3.48,'Cg':2.95e-12,
                           'alpha_s':-2.11,'beta_s':0.996,'at_s':0.199,'Cs':3.13e-14},
        ]
    },

    5.5: {
        "A": [
            {'bC':0.0,'alpha_g':-2.35,'beta_g':-0.668,'at_g':0.148,'ac_g':1.96,'Cg':4.82e-14,
                         'alpha_s':-1.57,'beta_s':1.10,'at_s':0.198,'Cs':4.24e-14},
            {'bC':1.0,'alpha_g':-2.12,'beta_g':-0.670,'at_g':0.0686,'ac_g':1.35,'Cg':3.65e-13,
                         'alpha_s':-1.57,'beta_s':1.25,'at_s':0.197,'Cs':4.00e-14},
            {'bC':2.0,'alpha_g':-1.94,'beta_g':-0.853,'at_g':0.0786,'ac_g':0.921,'Cg':2.57e-13,
                         'alpha_s':-1.55,'beta_s':1.33,'at_s':0.195,'Cs':4.05e-14},
            {'bC':3.0,'alpha_g':-1.61,'beta_g':-0.722,'at_g':0.0418,'ac_g':0.720,'Cg':7.58e-13,
                         'alpha_s':-1.59,'beta_s':2.12,'at_s':0.193,'Cs':2.61e-14},
        ],
        "B": [
            {'bC':0.0,'alpha_g':-2.80,'beta_g':0.0356,'at_g':0.0203,'ac_g':3.43,'Cg':2.74e-12,
                         'alpha_s':-1.09,'beta_s':-0.370,'at_s':0.218,'Cs':1.17e-13},
            {'bC':1.0,'alpha_g':-2.67,'beta_g':0.0129,'at_g':0.0134,'ac_g':3.44,'Cg':7.27e-12,
                         'alpha_s':-1.14,'beta_s':-0.195,'at_s':0.216,'Cs':1.05e-13},
            {'bC':2.0,'alpha_g':-2.45,'beta_g':-0.00132,'at_g':0.0275,'ac_g':5.14,'Cg':8.79e-13,
                         'alpha_s':-1.08,'beta_s':-0.336,'at_s':0.216,'Cs':1.17e-13},
            {'bC':3.0,'alpha_g':-1.90,'beta_g':-0.0517,'at_g':0.0120,'ac_g':7.28,'Cg':2.86e-12,
                         'alpha_s':-1.13,'beta_s':-0.109,'at_s':0.211,'Cs':1.04e-13},
        ]
    }
}




#----------------------------------------------------------
#--- Turbulent Gas Surface Density Distribution  functions 
#----------------------------------------------------------
def compute_R(Mach):
    """
    Computes the compression ratio R as defined in Equation (15) of Thompson et al. (2016),
    assuming a power-law index alpha = 1/3 (Kolmogorov turbulence).

    Parameters
    ----------
    Mach : float
        Turbulent Mach number

    Returns
    -------
    R : float
        Compression ratio R.
    """
    alpha = 2.5
    numerator = 1 - Mach**(2 * (2 - alpha))
    denominator = 1 - Mach**(2 * (3 - alpha))
    if np.any(denominator == 0):
        raise ValueError("Denominator goes to zero; choose a different Mach.")

    prefactor = 0.5 * (3 - alpha) / (2 - alpha)
    R = prefactor * (numerator / denominator)
    return R


def draw_sigma_distribution(mu_sigma, Mach, nsamples=10000):
    """
    Generates a log-normal distribution of gas surface densities (Sigma),
    with median mu_sigma and variance determined by turbulence via
    Thompson+16 Eq. (15).

    Parameters
    ----------
    mu_sigma : float
        Median of the Sigma distribution [arbitrary units]
    Mach : float
        Turbulent Mach number
    nsamples : int, optional
        Number of samples to draw (default is 10,000)

    Returns
    -------
    np.ndarray
        Array of Sigma values drawn from the log-normal distribution
    """
    R = compute_R(Mach)
    sigma_lnSigma_sq = np.log(1 + (R * Mach**2) / 4)
    sigma_ln = np.sqrt(sigma_lnSigma_sq)
    mu_ln = np.log(mu_sigma)
    #mu_ln = -sigma_lnSigma_sq/2
    
    return np.random.lognormal(mean=mu_ln, sigma=sigma_ln, size=nsamples)


def compute_vrot(z,Mh):
    """
    Compute the rotational velocity in km/s from halo mass and virial radius.

    Parameters:
    -----------
    Mvir : float or array-like
        Virial mass in solar masses (M_sun)
    Rvir : float or array-like
        Virial radius in kiloparsecs (kpc)

    Returns:
    --------
    Vrot : float or array-like
        Rotational velocity in km/s
    """
    G = 4.302e-6  # gravitational constant in kpc * (km/s)^2 / M_sun
    Rvir=r_vir(z, Mh)
    return (G * Mh / Rvir)**0.5



# ======================================== #
# ========== OBS. DATA SECTION =========== #
# ======================================== #

# --- SED Fitting dervied properties for spec-z confirmed galaxies, at z>10
# Path to your file
csv_path = "/Users/lsommovigo/Desktop/Scripts/txt_files/JWST_dust_z10/JWST_z10_galaxies.csv"
# Set header=1 to tell pandas to use the second line (0-based) as header
df = pd.read_csv(csv_path, skipinitialspace=True, na_values='-999')
print("Column names:", df.columns.tolist())
names                = df['name'].values
tau_v_meas           = df['tau_v'].values
err_tau_meas_low     = df['err_tau_low'].values
err_tau_meas_up      = df['err_tau_up'].values
redshift_meas        = df['redshift'].values
re_pc_meas           = df['re_pc'].values
err_re_pc_low        = df['err_re_low'].values
err_re_pc_up         = df['err_re_up'].values
SFR_meas             = df['SFR'].values
err_SFR_low_meas     = df['err_SFR_low'].values
err_SFR_up_meas      = df['err_SFR_up'].values
age_meas             = df['age'].values
MUV_meas             = df['MUV'].values
log_Mstar_meas       = df['log_Mstar'].values
err_logMstar_low_meas  = df['err_logMstar_low'].values
err_logMstar_high_meas = df['err_logMstar_high'].values
IR_Flux_meas         = df['IR_Flux'].values
uplims_IRFlux        = df['uplim_IRFlux'].values
lambda_obs           = df['lambda_obs'].values
betaUV_meas=df['betaUV_obs'].values



###--- SED Fitting derived properties for REBELS Galaxies (z=7)
### REFs: Sommovigo+22b, Stefanon in prep.
REBELS_index=np.array([5, 8, 12, 14, 18, 19, 25, 27, 29, 32, 38, 39, 40])
MUV_REB=np.array([-21.57, -21.82, -22.47, -22.66, -22.37, -21.6, -21.67, -21.93, -22.24, -21.65, -21.87, -22.71, -21.84])
##-- Non-par SFH (Topping+22)
## Attenuation V Band
Av_REB_npSFH=1.086*np.array([0.18,0.009, 0.158,0.006, 0.069, 0.01, 0.052, 0.019, 0.085, 0.152, 0.027, 0.064, 0.097])
errm_Av_REB_npSFH=1.086*np.array([0.043, 0.002, 0.09, 0.002, 0.009, 0.002, 0.005, 0.003, 0.032, 0.092, 0.003, 0.019, 0.018])
errp_Av_REB_npSFH=1.86*np.array([0.264, 0.046, 0.22, 0.021, 0.136, 0.05, 0.197, 0.097, 0.127, 0.196, 0.121, 0.105, 0.164])
#logMd_REB_AV_npSFH=np.log10(Md_from_Av(Av_REB_npSFH,1.,0.841))

## Log Stellar Masses
logMstar_REB_npSFH=np.array([10.09, 9.56, 9.94, 9.22, 9.82, 9.38, 10.27, 10.16, 10.04, 9.78, 10.37, 9.24, 9.82])
errm_logMstar_REB_npSFH=np.array([0.5, 0.46, 0.42, 0.41, 0.45, 0.51, 0.23, 0.19, 0.18, 0.35, 0.37, 0.32, 0.5])
errp_logMstar_REB_npSFH=np.array([0.32, 0.34, 0.32, 0.38, 0.34, 0.39, 0.1, 0.12, 0.13, 0.24, 0.15, 0.37, 0.3])

## Log Dust Masses
logMd_REB_npSFH=np.array([7.178553842550373, 7.222072062056873, 7.302253265513234, 7.016965485868306, 7.308617989378732, 7.232033622294462, 7.555296005147596, 7.138938465954253, 7.107044369891009, 7.216636655590892, 7.462844752786296, 7.197760252551811, 7.0777444193759385])
errm_logMd_REB_npSFH=np.array([0.32205552379404256, 0.2856291197206957, 0.31652104427750505, 0.27006874273331416, 0.3283003848819428, 0.31201464818425517, 0.21897227509967898, 0.3249105784703952, 0.32298631791288734, 0.3274635773089871, 0.313705663664531, 0.30308570535902657, 0.3221830378196415])
errp_logMd_REB_npSFH=np.array([0.3682902509353072, 0.3503539376977258, 0.3709013733916553, 0.33160205946943133, 0.36848755472335704, 0.33690851009108247, 0.33518855225741007, 0.3729431006573245, 0.3726309477680845, 0.36764758316417634, 0.3671520150887915, 0.31535345886411204, 0.37662779362592946])

## Dust Temperatures
Td_REB_npSFH=np.array([41.19473069695478, 47.72080464438527, 44.143263869734, 48.13426947613128, 38.62655724127967, 43.5887546733149, 44., 40.9330560250899, 41.728466054254284, 39.1468047356576, 45.63654800581859, 44.85008744958079, 43.11543190136595])
errm_Td_REB_npSFH=np.array([9.748473968176071, 12.165882049530957, 10.894199726070454, 11.857514090831785, 7.58339172499814, 9.921377104589517, 8., 9.286868051046618, 10.063639512970191, 9.005512560297468, 11.658986983741556, 10.034765553277339, 10.374304972033379])
errp_Td_REB_npSFH=np.array([15.62809297953136, 17.651906340201407, 17.343763170150424, 16.43677034660398, 12.446707897590422, 16.66582942001719, 13., 14.852010152296295, 16.051344203653287, 14.598531683460166, 17.993236351298435, 16.846211588842557, 16.785079519062414])

## NB: Age is from Mauro's Fits, not consistent 100%, cause above non par SFH, hereas Mauro uses CSFH
Age_REB=10**np.array([7.57, 7.82, 6.9, 7.13, 8.03, 7.63, 8.69, 8.44, 8.32, 7.92, 8.09, 6.42, 8.14])
Age_REB=1e-6*Age_REB
errm_Age_REB=10**np.array([6.66, 7.1, 6.52, 6.49, 7.29, 6.96, 8.45, 8.02, 8.14, 7.66, 6.8, 6.28, 6.97])
errm_Age_REB=Age_REB-1e-6*errm_Age_REB
errp_Age_REB=10**np.array([8.61, 8.47, 7.63, 7.77, 8.52, 8.29, 8.81, 8.69, 8.51, 8.16, 8.73, 6.61, 8.67])
errp_Age_REB=-Age_REB + 1e-6*errp_Age_REB

reds_REB=np.array([6.49632623080003, 6.74949312222826, 7.34591349960637, 7.08424192680435, 7.67499921115377, 7.37010028069009, 7.30651374725317, 7.08975543485707, 6.68474280897578, 6.72902239593933, 6.57701183413199, 6.84488626164266, 7.36495347204348])
#computing Im
betaUV_REB=np.array([-1.29, -2.17, -1.99, -2.21, -1.34, -2.33, -1.85, -1.79, -1.61, -1.50, -2.18, -1.96, -1.44])
#print 'betaUV med and dev reb', betaUV_REB.mean(), numpy.std(betaUV_REB)
IR_Flux_meas= np.array([67.233017562467, 101.443285027657, 86.7784069441866, 59.9864001077979, 52.8727962380168, 71.1547654942816, 259.549851951265, 50.5904947746512, 56.0849016628018, 60.3849166209817, 162.997677399763, 79.7361487750138, 48.2836056474205])#microJy
err_IR_Flux_meas= np.array([13., 20., 24., 15., 10., 20., 22., 10., 13., 17., 23, 16., 13.])#microJy


def Plot_LF_Data(z, ax):
    """
    Plot observed UV luminosity function (LF) data from various literature sources for a given redshift.

    Parameters
    ----------
    z : float or int
        Redshift at which to plot the LF data.
    ax : matplotlib.axes.Axes
        Matplotlib axis object on which to plot the LF data.

    Returns
    -------
    None
        The function adds data points and error bars to the provided axis but does not return any value.
    """
    ##--- Bouwens+21, LF pre JWST, z<10 
    if z < 10:
        # Load digitized observational data
        obs_data = np.genfromtxt(
            "/Users/lsommovigo/Desktop/Scripts/txt_files/JWST_dust_z10/Bouwens21_z2-9.txt",
            names=True, dtype=None, encoding=None
        )

        mask = obs_data['Redshift'] == z

        # Helper to safely convert to float and replace non-numerics with NaN
        def to_float(arr):
            out = []
            for x in arr:
                try:
                    out.append(float(x))
                except:
                    out.append(np.nan)
            return np.array(out, dtype=float)

        M_UV = to_float(obs_data['MUV'][mask])
        Phi  = to_float(obs_data['Phi'][mask])
        Err_Phi_low = to_float(obs_data['Err_Phi_low'][mask])
        Err_Phi_up  = to_float(obs_data['Err_Phi_up'][mask])

        # Identify upper limits: missing or huge upper error
        is_upper_limit = np.isnan(Err_Phi_up) | (Err_Phi_up > Phi)
        is_detection = ~is_upper_limit

        # --- Detections: asymmetric errors ---
        Err_Phi_low_det = Err_Phi_low[is_detection].copy()
        Err_Phi_up_det  = Err_Phi_up[is_detection].copy()

        # Cap absurd errors
        max_err = 0.99 * Phi[is_detection]
        Err_Phi_low_det = np.minimum(Err_Phi_low_det, max_err)
        Err_Phi_up_det  = np.minimum(Err_Phi_up_det, max_err)

        ax.errorbar(
            M_UV[is_detection],
            Phi[is_detection],
            yerr=[Err_Phi_low_det, Err_Phi_up_det],
            fmt='s', markersize=10, capsize=4, elinewidth=0.8,
            color='grey', alpha=0.7, markeredgecolor='black',
            linestyle='none',
            label='Obs, Bouwens+21, $z=$'+str(int(z))
        )

        # --- Upper limits ---
        ax.errorbar(
            M_UV[is_upper_limit],
            Phi[is_upper_limit],
            yerr=0.2 * Phi[is_upper_limit],
            uplims=True,
            fmt='s', markersize=10, capsize=4, elinewidth=0.8,
            color='grey', markeredgecolor='black',
            linestyle='none', alpha=0.7
        )

    
    if z==10:
        ###--- Harikane+23: spec-z, Tab. 7 in https://doi.org/10.3847/1538-4357/ad0b7e
        MUV_H23_z9_det = [-21.03, -20.03, -19.03, -18.03]
        phi_H23_z9_det = [4.00e-5, 4.08e-5, 2.24e-4, 1.12e-3]
        phi_H23_z9_uerr = [9.42e-5 , 9.60e-5 , 1.87e-4, 1.03e-3]
        phi_H23_z9_lerr = [3.85e-5, 3.92e-5, 1.46e-4, 0.90e-3]
        # Upper limits (first two bins)
        MUV_H23_z9_ulim = [-23.03, -22.03]
        phi_H23_z9_ulim = [6.95e-5, 7.67e-5]
        # Plot detections with asymmetric errors
        ax.errorbar(MUV_H23_z9_det, phi_H23_z9_det,
                     yerr=[phi_H23_z9_lerr, phi_H23_z9_uerr], ls='None',ms=12.,marker='h',capsize=5.,alpha=0.5,color='teal', label='spec, Harikane+23, $z=9$', mew=1.5,mec='black',elinewidth=0.8)
        # Plot upper limits as downward arrows
        ax.errorbar(MUV_H23_z9_ulim, phi_H23_z9_ulim,
                     yerr=0.4 * np.array(phi_H23_z9_ulim),  # length of arrow
                     uplims=True, ls='None',ms=12.,marker='h',capsize=5.,alpha=0.5,color='teal', mew=1.5,mec='black',elinewidth=0.8)
                     
        ###--- Donnan+24, Tab. 2 in  https://arxiv.org/pdf/2403.03171
        MUV_D23=np.array([-20.75,-20.25,-19.75,-19.25,-18.55,-18.05,-17.55])
        n_D23=np.array([4e-6,27e-6,92e-6,177e-6,321e-6,686e-6,1278e-6])
        lerr_D23=np.array([4e-6,10e-6,20e-6,45e-6,111e-6,223e-6,432e-6])
        uerr_D23=np.array([10e-6,13e-6,25e-6,53e-6,127e-6,245e-6,486e-6])
        ax.errorbar(MUV_D23,n_D23,yerr=[lerr_D23,uerr_D23],ls='None',ms=8.,marker='o',capsize=5.,alpha=0.4,color='black', label='phot, Donnan+24, $z=10$',mec='black',mew=1.5,elinewidth=0.8)

        ###--- Oesh+18, pre JWST
        MUV_O18=np.loadtxt('/Users/lsommovigo/Desktop/Scripts/txt_files/JWST_dust_z10/Oesch_z10.txt',usecols=0)
        n_O18=1e-4*np.loadtxt('/Users/lsommovigo/Desktop/Scripts/txt_files/JWST_dust_z10/Oesch_z10.txt',usecols=1)
        lerr_O18=1e-4*np.loadtxt('/Users/lsommovigo/Desktop/Scripts/txt_files/JWST_dust_z10/Oesch_z10.txt',usecols=3)
        uerr_O18=1e-4*np.loadtxt('/Users/lsommovigo/Desktop/Scripts/txt_files/JWST_dust_z10/Oesch_z10.txt',usecols=2)
        ax.errorbar(MUV_O18,n_O18,yerr=[lerr_O18,uerr_O18],ls='None',ms=8.,marker='s',capsize=5.,alpha=0.5,color='grey', label='phot, Oesch+18, $z\sim 10$',mec='black',mew=1.5,elinewidth=0.8)

        # Whitler+25, F115W dropouts (z_med = 9.8)
        MUV_Whitler25 = np.array([-21.4, -20.4, -19.4, -18.4, -17.4])
        phi_Whitler25 = 1e-5 * np.array([0.40, 3.6, 17, 78, 330])
        phi_err_low = 1e-5 * np.array([0.26, 1.1, 3, 9, 40])
        phi_err_up  = 1e-5 * np.array([0.45, 1.2, 3, 10, 50])
        ax.errorbar(MUV_Whitler25, phi_Whitler25,
                     yerr=[phi_err_low, phi_err_up],
                     fmt='*', color='dimgrey', mec='black', label='Whitler+25, $z_{\\rm med}=9.8$', alpha=0.6,ms=10,mew=1.5,capsize=5,elinewidth=0.8)


    if z==12:
        ###--- McLeod+23 (z ~ 11), including Donnan+23a
        MUV_McLeod23_z11 = [-22.57, -21.80, -20.80, -20.05, -19.55, -18.85, -18.23]
        phi_McLeod23_z11 = [0.012, 0.129, 1.254, 3.974, 9.863, 23.490, 63.080]
        err_McLeod23_z11 = [0.012, 0.128, 0.428, 1.340, 4.197, 9.190, 28.650]
        ax.errorbar(MUV_McLeod23_z11, [v*1e-5 for v in phi_McLeod23_z11],
                     yerr=[e*1e-5 for e in err_McLeod23_z11],label='McLeod+23, $z=9.5-12.5$',
                     ls='None',ms=8.,marker='d',capsize=5.,alpha=0.5,color='grey', mec='black',mew=1.5,elinewidth=0.8)
                     
        ###--- Donnan+24, z=11.5-12.5
        MUV_D24 = [-21.25, -20.75, -20.25, -19.75, -19.25, -18.75, -18.25]
        phi_D24 = [3e-6, 4e-6, 16e-6, 34e-6, 43e-6, 80e-6, 217e-6]
        phi_D24_uerr = [4e-6, 5e-6, 9e-6, 23e-6, 35e-6, 51e-6, 153e-6]
        phi_D24_lerr = [2e-6, 3e-6, 6e-6, 15e-6, 22e-6, 36e-6, 104e-6]
        ax.errorbar(MUV_D24, phi_D24,
                     yerr=[phi_D24_lerr, phi_D24_uerr],
                     ls='None', ms=10., marker='o', capsize=4,
                     alpha=0.4, color='black',
                     label='Donnan+24, $z=11.5-12.5$', mew=1.5, mec='black', elinewidth=0.8)

        ###--- CEERS, Finkelstein+23
        MUV_CRS = [-20.5, -20.0, -19.5, -19.0, -18.5]
        phi_CRS = [1.8e-5, 5.4e-5, 7.6e-5, 17.6e-5, 26.3e-5]
        phi_CRS_uerr = [1.2e-5, 2.7e-5, 3.9e-5, 10.3e-5, 18.2e-5]
        phi_CRS_lerr = [0.9e-5, 2.1e-5, 3.0e-5, 7.9e-5, 13.3e-5]
        ax.errorbar(MUV_CRS, phi_CRS,
                     yerr=[phi_CRS_lerr, phi_CRS_uerr],
                     ls='None', ms=10., marker='x', capsize=4,
                     alpha=0.6, color='grey',
                     label=r'CEERS, $z=9.7-13$',
                     mew=1.5, mec='black', elinewidth=0.8)

        ###--- z = 11 LF Casey+23
        MUV_Casey = [-22.0, -21.2]
        phi_Casey = [1.0e-6, 1.4e-6]
        phi_Casey_uerr = [0.3e-6, 0.5e-6]
        phi_Casey_lerr = [0.4e-6, 0.5e-6]
        ax.errorbar(MUV_Casey, phi_Casey,
                     yerr=[phi_Casey_lerr, phi_Casey_uerr],
                     ls='None', ms=10., marker='D', capsize=4,
                     alpha=0.6, color='silver',
                     label=r'Casey+23, $z=9.5-12.5$',
                     mew=1.5, mec='black', elinewidth=0.8)

        ###--- Leung+23: z = 11 from Table in 2306.06244
        MUV_Leung23 = [-19.35, -18.65, -17.95, -17.25]
        phi_Leung23 = [18.5e-5, 27.7e-5, 59.1e-5, 269.0e-5]  # Convert to Mpc⁻³ mag⁻¹
        phi_Leung23_uerr = [11.9e-5, 18.3e-5, 41.9e-5, 166.0e-5]
        phi_Leung23_lerr = [8.3e-5, 13.0e-5, 29.3e-5, 124.0e-5]
        ax.errorbar(MUV_Leung23, phi_Leung23,
                     yerr=[phi_Leung23_lerr, phi_Leung23_uerr],
                     ls='None', marker='s', ms=10., capsize=4., alpha=0.5,
                     color='grey', label='Leung+23, $z=11$',
                     mew=1.2, mec='black', elinewidth=0.8)

        ###--- z = 12.8 (F150W dropouts), Whitler+25
        MUV_z12p8 = [-20.5, -19.5, -18.5, -17.5]
        phi_z12p8 = [0.84e-5, 3.4e-5, 29e-5, 71e-5]
        phi_z12p8_uerr = [0.63e-5, 1.5e-5, 6e-5, 28e-5]
        phi_z12p8_lerr = [0.43e-5, 1.3e-5, 6e-5, 23e-5]
        ax.errorbar(MUV_z12p8, phi_z12p8,
                     yerr=[phi_z12p8_lerr, phi_z12p8_uerr],
                     ls='None', marker='*', ms=10., capsize=4., alpha=0.6,
                     color='dimgrey', label='Whitler+25, $z=12.8$',
                     mew=1.2, mec='black', elinewidth=0.8)

        ###--- 11.5 < z < 13.5, Robertson+24
        MUV_11p5_13p5 = [-18.5, -18.0, -17.6]
        phi_11p5_13p5 = [1.22e-4, 3.20e-4, 1.54e-4]
        phi_11p5_13p5_err = [0.94e-4, 2.46e-4, 1.18e-4]

        ax.errorbar(MUV_11p5_13p5, phi_11p5_13p5,
                     yerr=phi_11p5_13p5_err,
                     ls='None', marker='+', ms=9., capsize=5., alpha=0.6,
                     color='grey', label='Robertson+24, $11.5 < z < 13.5$', mew=1.2, mec='black')


    if z==14:
        ###---Casey+23
        ax.errorbar(-21, 0.81e-6,
                     yerr=[[0.42e-6], [0.42e-6]],
                     ls='None', ms=10., marker='D', capsize=4,
                     alpha=0.4, color='silver',
                     label=r'Casey+23, $z=13-15$',
                     mew=1.5, mec='black', elinewidth=0.8)

        ###--- CEERS, Finkelstein+23? (2311.04279)
        MUV_2311_zgt13_det = [-20.0, -19.5]
        phi_2311_zgt13_det = [2.6e-5, 7.3e-5]
        phi_2311_zgt13_uerr = [3.3e-5, 6.9e-5]
        phi_2311_zgt13_lerr = [1.8e-5, 4.4e-5]
        # Upper limit (first point)
        MUV_2311_zgt13_ulim = [-20.5]
        phi_2311_zgt13_ulim = [1.8e-5]
        # Plot detections
        ax.errorbar(MUV_2311_zgt13_det, phi_2311_zgt13_det,
                     yerr=[phi_2311_zgt13_lerr, phi_2311_zgt13_uerr],
                     ls='None', ms=10., marker='x', capsize=4,
                     alpha=0.6, color='grey',
                     label=r'CEERS, $z>13$',
                     mew=1.5, mec='black', elinewidth=0.8)
        # Plot upper limit
        ax.errorbar(MUV_2311_zgt13_ulim, phi_2311_zgt13_ulim,
                     yerr=0.4 * np.array(phi_2311_zgt13_ulim),  # arrow length
                     uplims=True, ls='None', ms=10., marker='D', capsize=4,
                     alpha=0.6, color='silver',
                     mew=1.5, mec='black', elinewidth=0.8)
        
        ###--- Donnan+24, z = 14.5
        ax.errorbar([-20.25], [3e-6],
                     yerr=[[2e-6], [6e-6]],
                     ls='None', marker='o', ms=9, capsize=5., alpha=0.4,
                     color='black', label='Donnan+24, $z=14.5$',
                     mew=1.2, mec='black', elinewidth=0.8)

        ###--- McLeod+23: 12.5 < z < 14.5 from Table 4 of 2304.14469
        MUV_McLeod24 = [-19.45, -18.95]
        phi_McLeod24 = [2.469e-6, 6.199e-6]  # Convert from 1.0E-06 units
        phi_McLeod24_err = [1.659e-6, 3.974e-6]

        ax.errorbar(MUV_McLeod24, phi_McLeod24, yerr=phi_McLeod24_err, ls='None',
                     marker='d', ms=9., capsize=4., alpha=0.6, color='grey',
                     label='McLeod+23, $12.5 < z < 14.5$', mew=1.2, mec='black', elinewidth=0.8)

        ###--- z ≥ 14 (F150W dropouts, z_median = 14.3), Whitler+25
        MUV_z14p3 = [-20.2, -18.9]
        phi_z14p3 = [3.2e-5, 12e-5]
        phi_z14p3_uerr = [2.1e-5, 7e-5]
        phi_z14p3_lerr = [1.5e-5, 5e-5]

        ax.errorbar(MUV_z14p3, phi_z14p3,
                     yerr=[phi_z14p3_lerr, phi_z14p3_uerr],
                     ls='None', marker='*', ms=10., capsize=4., alpha=0.6,
                     color='dimgrey', label='Whitler+25, $z=14.3$',
                     mew=1.2, mec='black', elinewidth=0.8)
        
        ###--- Robertson+24, 13.5 < z < 15
        MUV_13p5_15 = [-20.8, -18.4, -18.1]
        phi_13p5_15 = [0.371e-4, 2.56e-4, 0.783e-4]
        phi_13p5_15_err = [0.357e-4, 2.46e-4, 0.754e-4]

        ax.errorbar(MUV_13p5_15, phi_13p5_15,
                     yerr=phi_13p5_15_err,
                     ls='None', marker='+', ms=9., capsize=5., alpha=0.6,
                     color='grey', label='Robertson+24, $13.5 < z < 15$', mew=1.2, mec='black')
    print('Data ADDED!\n\n')
    return


