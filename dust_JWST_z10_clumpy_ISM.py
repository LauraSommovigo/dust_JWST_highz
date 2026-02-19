# ============================================================
# dust_JWST_z10_clumpy_ISM.py
# ============================================================
# Studies how ISM turbulence affects dust attenuation.
# A uniform dust shell (as in the other scripts) is replaced by
# a log-normal distribution of dust surface densities Sigma_d,
# parameterised by the turbulent Mach number (Fischera & Dopita 2004).
# Higher Mach -> wider log-normal -> some sightlines are optically
# thin even if the average is optically thick -> galaxy appears
# brighter in UV but with a broader MUV distribution.
#
# Main outputs:
#   1) Two-panel figure: Sigma_d and MUV PDFs at different Mach numbers
#   2) Far-IR SED comparison for MW vs stellar dust at Mach=10,100
# ============================================================

from scipy.integrate import quad
from matplotlib import cm
col_f = cm.get_cmap('gray')
from librerie import *
from highz_gal_SAM import *
from general import name_and_save, increase_ticklabels, set_labels, do_minorticks, do_log, equal_axes, \
    set_colorbar_labels, set_ticklabels
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.gridspec as gridspec
from scipy.stats import norm, gaussian_kde
import matplotlib.pyplot as plt
import numpy as np

# ======== DUST CONSTANTS ===========
# Loaded from highz_gal_SAM.py, originally computed in GSD.py
bd=2.03              # dust emissivity index beta_d
kUV=kUV_drn          # UV extinction opacity [cm^2/g] (MW dust)
kUV_abs=kUV_drn_abs  # UV absorption opacity (extinction * (1 - albedo))
kV=kv_drn            # V-band extinction opacity [cm^2/g]

# ============================================================
#  SECTION 1: Grain size distribution (Draine+03, MW silicates)
# ============================================================
# These functions are only used for plotting the GSD shape;
# the actual kappa values come from GSD.py tables.
# ============================================================

def dn_da(a):
    """
    Grain size distribution from Draine+03 for MW-like silicates.
    """
    cs = 1.02e-12
    ats = 0.172e-4  # cm
    alphas = -1.48
    bs = -9.34
    acs = 0.1e-4    # cm

    if a <= ats:
        return (cs / ats**alphas) * a**(alphas - 1) / (1 - bs * a / ats)
    else:
        return (cs / ats**alphas) * a**(alphas - 1) / (1 - bs * a / ats) * \
               np.exp(-((a - ats) / acs)**3)

def massagrain_cgs(a):
    """
    Mass of a spherical dust grain of radius a (in cm).
    Assumes silicate density of 3.5 g/cm^3.
    """
    d = 3.5  # g/cm^3
    return (4.0 / 3.0) * np.pi * a**3 * d

def dn_da_massg(a):
    """
    Mass-weighted grain size distribution (dn/da × mass of grain).
    """
    return dn_da(a) * massagrain_cgs(a)

def na(a):
    """
    Returns normalized number-weighted grain distribution in bins of a.
    """
    weights = np.zeros(len(a) - 1)
    for s in range(1, len(a)):
        weights[s - 1] = quad(dn_da, a[s - 1], a[s])[0]
    return weights / np.sum(weights)

def na3(a):
    """
    Returns normalized mass-weighted grain distribution in bins of a.
    """
    weights = np.zeros(len(a) - 1)
    for s in range(1, len(a)):
        weights[s - 1] = quad(dn_da_massg, a[s - 1], a[s])[0]
    return weights / np.sum(weights)

##############################################


# ======== INPUT DATA ===========
# Load SB99 instantaneous burst tables
logSNr_yr = np.loadtxt(os.path.join(SCRIPTS_DIR, 'txt_files/SB99/snr_inst_Z001.txt'), usecols=1)
time_yr = np.loadtxt(os.path.join(SCRIPTS_DIR, 'txt_files/SB99/snr_inst_Z001.txt'), usecols=0)
L1500_SB99 = np.loadtxt(os.path.join(SCRIPTS_DIR, 'txt_files/SB99/L1500_inst_Z001.txt'), usecols=1)
time_yr_L1500 = np.loadtxt(os.path.join(SCRIPTS_DIR, 'txt_files/SB99/L1500_inst_Z001.txt'), usecols=0)


# ============================================================
#  SECTION 2: Halo model setup
# ============================================================
# Pick a single characteristic halo, build its SFH, and compute
# the final dust mass, L_UV, and tau_UV from the spin distribution.
# ============================================================
redshift = 12.37#internmeidate redshift for IR followed up BMs z=12.37, otherwise z=7
logMh = 10.68#10.68#11.6 (for z=7), good intermediate values for BMs (z=12.37) is 10.68
fb = cosmo.Ob(redshift) / cosmo.Om(redshift)
epsilon = 0.1
yd = 0.1

# Stellar mass from halo mass
Mstar_array = halo_to_stellar_mass(10**logMh, fb, epsilon)
print('\n\n log Mstar/Msun -->', np.log10(Mstar_array))

# Build SFH
tstep = 1  # [Myr]
len_sp_dis = 1000
spin_param_distr = np.random.lognormal(mean=np.log(10**-1.5677), sigma=0.5390, size=len_sp_dis)
SFH, logMst_build, age = Build_SFH_funct(10**logMh, redshift, tstep, epsilon)

# Compute dust mass over time using SN rate
N_SN_arr, Md_arr = compute_Mdust_steps(age, tstep, SFH, time_yr, logSNr_yr, yd)

# Compute 1500A luminosity [erg/s/Hz]
L1500_arr = compute_L1500_steps(age, tstep, SFH, time_yr_L1500, L1500_SB99)
Lintr = L1500_arr[-1]  # only last step (final galaxy age) # erg/s/Hz
MUV_intr = L1500_to_MUV_conv(Lintr)
L1500_lambda = Lintr * c / (1500.**2 * 1e-8)  # erg/s/Å
print('MUV (intrinsic) -->', MUV_intr)


# ============================================================
#  SECTION 3: Turbulent Sigma_d distribution
# ============================================================
# For a uniform shell: Sigma_d = Mdust / (pi * rd^2), one value
# per halo/spin combination.
# Turbulence creates a LOG-NORMAL spread in Sigma_d along
# different lines of sight (Fischera & Dopita 2004):
#   sigma_ln^2 = ln(1 + R * Mach^2 / 4)
# where R = R(Mach, alpha) from Thompson+16 (in highz_gal_SAM.py).
# Higher Mach -> wider distribution -> more scatter in tau and MUV.
# ============================================================

# Fixed total dust mass [Msun]
M_dust = Md_arr[-1]
print('log Mdust/Msun -->', np.log10(M_dust))
print('log Mstar/Msun -->', np.log10(Mstar_array))
print('median logMstar(Msun) [REBELS] -->', np.median(logMstar_REB_npSFH))
print('rdisk (kpc) -->', rd_kpc(redshift, 10**logMh, np.median(spin_param_distr)))
print('Median log(Sigmad (Msun/kpc^2)) [unif] -->', np.log10(M_dust / (np.pi * (rd_kpc(redshift, 10**logMh, np.median(spin_param_distr)))**2)))
# Median sigma_d for uniform distribution [g/cm^2]
tauUV_arr = tau_pred(kUV, M_dust, 10**logMh, spin_param_distr, redshift)
Sigmad_arr = tauUV_arr / kUV  # g/cm^2            

# Log-normal draw for sigma_d, width set by Mach number
#Mach=3000  # turbulent Mach number
Sigmad_distr_10 = draw_sigma_distribution(mu_sigma=np.median(Sigmad_arr), Mach=10)
Sigmad_distr_100 = draw_sigma_distribution(mu_sigma=np.median(Sigmad_arr), Mach=100)



# Build the y-grid (dex) covering your histogram range
R_10 = compute_R(10)
sigma_lnSigma_sq_10 = np.log(1 + (R_10 * 10**2) / 4)
sigma_ln_10 = np.sqrt(sigma_lnSigma_sq_10)
mu_ln_10 = np.log(np.median(Sigmad_arr))
y_vals = np.linspace(-10, -2, 1000)                  # dex
x_vals = 10**y_vals

# Lognormal PDF in x (same as your function)
def lognormal_pdf_x(x, mu, sigma):
    return np.exp(-(np.log(x) - mu)**2 / (2*sigma**2)) / (x * sigma * np.sqrt(2*np.pi))

# parameters you already computed:
# mu_ln_10 = np.log(np.median(Sigmad_arr))
# sigma_ln_10 = np.sqrt(np.log(1 + (R_10 * 10**2) / 4))

# Transform to the PDF in y = log10 x (units: 1/dex)
pdf_y_10 = np.log(10) * x_vals * lognormal_pdf_x(x_vals, mu_ln_10, sigma_ln_10)
# Overlay the correctly transformed PDF (no arbitrary /max)
#plt.plot(y_vals, pdf_y_10, label='PDF Mach=10 (correct in log10-space)', lw=2)




# ============================================================
#  SECTION 4: MUV distributions — spin scatter vs turbulence
# ============================================================
# Two sources of scatter in attenuated MUV:
#   a) Galaxy-to-galaxy: different spin -> different rd -> different Sigma_d
#   b) Line-of-sight (LOS): turbulence within a single galaxy
# We combine both using K_SPINS seed quantiles from the spin-driven
# Sigma_d array, and for each seed draw N_LOS sightlines from the
# lognormal turbulent distribution.
# ============================================================
K_SPINS   = 13      # odd, so we include the exact median seed (u=0.5)
W_BLEND   = 0.6     # blend between median-seed transmission and mean over seeds
N_LOS     = 600     # LOS draws per Mach (distributed across seeds)
albedo    = 0.3807
Mach_array = np.array([5, 10, 20, 30, 40, 50, 100])

# exact-median spin seed and symmetric quantiles (data-driven via Sigmad_arr)
u_left  = (np.arange(1, (K_SPINS//2)+1) - 0.5) / K_SPINS
u_mid   = np.array([0.5])
u_right = 1.0 - u_left[::-1]
u_seeds = np.concatenate([u_left, u_mid, u_right])  # length = K_SPINS
mid_idx = K_SPINS // 2


def sigma_ln_from_Mach(Mach):
    R = compute_R(Mach)  # (Thompson+16; alpha=2.5 in your code)
    return np.sqrt(np.log(1 + (R * Mach**2) / 4.0))


# --- Spin-driven (no turbulent scatter): use uniform shell with spin spread ---
T_1500_uniform = T_sphere_mixed(tauUV_arr)   # one value per halo/spin
L1500_uniform  = Lintr * T_1500_uniform
MUV_spin = -2.5 * np.log10(L1500_uniform) + 51.60

# --- Helper to get MUV draws for a given Mach using the clumpy LOS logic ---
def get_MUV_clumpy(Mach):
    sigma_ln = sigma_ln_from_Mach(Mach)
    # seeds from empirical Σ_d distribution (same as before)
    mu_sigmas = np.quantile(Sigmad_arr, u_seeds)   # shape (K_SPINS,)

    # LOS quantiles
    u_los = (np.arange(1, N_LOS+1) - 0.5) / N_LOS
    z_los = norm.ppf(u_los)[None, :]               # (1, N_LOS)

    # Σ_d LOS, then τ, then T, then LUV
    Sigmad_LOS = np.exp(np.log(mu_sigmas)[:, None] + sigma_ln * z_los)  # (K_SPINS, N_LOS)
    tauUV_LOS  = kUV * Sigmad_LOS
    T_uv_LOS   = T_sphere_mixed(tauUV_LOS)
    LUV_LOS    = Lintr * T_uv_LOS
    LUV_draws  = LUV_LOS.ravel()

    return -2.5 * np.log10(LUV_draws) + 51.60     # MUV_att draws

# ------------------------------------
# Build extra Mach MUV distributions
# ------------------------------------
MUV_clumpy_10   = get_MUV_clumpy(10)
MUV_clumpy_30   = get_MUV_clumpy(30)
MUV_clumpy_50  = get_MUV_clumpy(50)
MUV_clumpy_100  = get_MUV_clumpy(100)

# Mach values for the clumpy case
mach_vals = np.array([10, 30, 50, 100])

# Build colormap for Mach curves (shared by left & right panels)
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=10):
    new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval), cmap(np.linspace(minval, maxval, n)))
    return new_cmap
cmap = truncate_colormap(plt.cm.coolwarm_r, 0., 1.)
mach_norm = (np.log10(mach_vals) - np.log10(mach_vals.min())) / \
            (np.log10(mach_vals.max()) - np.log10(mach_vals.min()))
mach_colors = [cmap(v) for v in mach_norm]   # [color(M=10), color(30), color(50), color(100)]

# ============================================================
# FIGURE 1: Two-panel — Sigma_d PDFs (left) + MUV PDFs (right)
# Left: galaxy-to-galaxy scatter (uniform) vs LOS scatter (M=10,100)
# Right: resulting MUV distributions at multiple Mach numbers
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ---- LEFT PANEL: Σ_d PDFs in log10-space (KDEs) ----

# conversion factor: g/cm^2 -> Msun/kpc^2
gcm2_to_Msun_kpc2 = (kpc_to_cm**2) / Msun

# convert Σ_d samples to Msun/kpc^2 and take log10
logSig_spin = np.log10(Sigmad_arr * gcm2_to_Msun_kpc2)
logSig_10   = np.log10(Sigmad_distr_10 * gcm2_to_Msun_kpc2)
logSig_100  = np.log10(Sigmad_distr_100 * gcm2_to_Msun_kpc2)

logs_list = [logSig_spin, logSig_10, logSig_100]

# common x-grid in log10(Σ_d / Msun kpc^-2)
x_Sig_min = min(l.min() for l in logs_list)
x_Sig_max = max(l.max() for l in logs_list)
x_Sig = np.linspace(x_Sig_min, x_Sig_max, 400)

# colors & labels (left panel); use Mach colormap for 10 and 100
colors_Sigma = [
    'dimgrey',        # uniform / spin-driven
    mach_colors[0],   # M=10
    mach_colors[-1]   # M=100
]
labels_Sigma = [
    'Galaxy-to-Galaxy scatter\n (Size driven)',
    'Single galaxy, LOS scatter\n (Turbulence-driven, $\\mathcal{M}=10$)',
    '(Turbulence-driven, $\\mathcal{M}=100$)'
]

for i, (logs, col, lab) in enumerate(zip(logs_list, colors_Sigma, labels_Sigma)):
    kde = gaussian_kde(logs)     # KDE in log-space
    pdf = kde(x_Sig)

    if i == 0:
        # Uniform: galaxy-to-galaxy scatter → thick solid line, no fill
        axes[0].plot(x_Sig, pdf, color=col, lw=2.8, label=lab)
        p16, p84 = np.percentile(logs, [16, 84])
        axes[0].axvline(p16, color=col, linestyle='-', lw=2.0, alpha=0.7)
        axes[0].axvline(p84, color=col, linestyle='-', lw=2.0, alpha=0.7)
    else:
        # Turbulence: LOS-to-LOS scatter → shaded + dashed 16–84% lines
        axes[0].fill_between(x_Sig, 0*x_Sig, pdf, color=col, alpha=0.1, label=lab, zorder=-1000+i)
        axes[0].plot(x_Sig, pdf, color=col, lw=2.8, alpha=0.2)
        p16, p84 = np.percentile(logs, [16, 84])
        axes[0].axvline(p16, color=col, linestyle='--', lw=1.8, alpha=0.7)
        axes[0].axvline(p84, color=col, linestyle='--', lw=1.8, alpha=0.7)

axes[0].set_xlabel(r'$\log_{10}\!\left(\Sigma_{d}/M_{\odot}\,\mathrm{kpc}^{-2}\right)$')
axes[0].set_ylabel(r'PDF')
axes[0].set_xlim(x_Sig_min, x_Sig_max)
axes[0].set_ylim(0, 1.5)
axes[0].legend(frameon=False, fontsize=14, loc='upper left')

# ---- RIGHT PANEL: MUV_att PDFs (KDEs), same Mach colors ----

# List including new Mach numbers
MUV_list = [
    MUV_spin,          # uniform, galaxy-to-galaxy
    MUV_clumpy_10,     # LOS-to-LOS
    MUV_clumpy_30,
    MUV_clumpy_50,
    MUV_clumpy_100
]

# Colors & labels for right panel:
#   - first: uniform
#   - then: 10, 30, 50, 100 with colormap colors
colors_MUV = ['dimgrey'] + mach_colors
labels_MUV = [
    'Galaxy-to-Galaxy scatter',
    r'Single galaxy, LOS scatter'+'\n'+r'$\mathcal{M}=10$',
    r'$\mathcal{M}=30$',
    r'$\mathcal{M}=50$',
    r'$\mathcal{M}=100$'
]

# Build x-grid for MUV PDFs
all_MUV = np.concatenate(MUV_list)
x_MUV_min, x_MUV_max = all_MUV.min(), all_MUV.max()
x_MUV = np.linspace(x_MUV_min, x_MUV_max, 400)

# add Muv itrinsic line
axes[1].axvline(MUV_intr, color='black', linestyle=':', lw=2.0, label=r'Intrinsic')

for i, (arr, col, lab) in enumerate(zip(MUV_list, colors_MUV, labels_MUV)):
    kde = gaussian_kde(arr)
    pdf = kde(x_MUV)

    if i == 0:
        # Uniform: galaxy-to-galaxy scatter (solid, no fill)
        axes[1].plot(x_MUV, pdf, color=col, lw=2.8, label=lab)
        p16, p84 = np.percentile(arr, [16, 84])
        axes[1].axvline(p16, color=col, linestyle='-', lw=2.0, alpha=0.7)
        axes[1].axvline(p84, color=col, linestyle='-', lw=2.0, alpha=0.7)
    else:
        # Turbulence: LOS-to-LOS scatter → *shaded only* + dashed percentiles
        axes[1].fill_between(x_MUV, 0*x_MUV, pdf, color=col, alpha=0.1, label=lab)
        axes[1].plot(x_MUV, pdf, color=col, lw=2.8, alpha=0.2)
        p16, p84 = np.percentile(arr, [16, 84])
        axes[1].axvline(p16, color=col, linestyle='--', lw=1.8, alpha=0.7)
        axes[1].axvline(p84, color=col, linestyle='--', lw=1.8, alpha=0.7)


axes[1].set_xlabel(r'$M_{\rm UV}$')
axes[1].set_ylabel(r'PDF')
axes[1].set_ylim(0, 0.46)
axes[1].set_xlim(-14.5, MUV_intr-0.5)   # keeps brighter magnitudes on the left
axes[1].legend(frameon=False, fontsize=14, loc='upper left')

plt.tight_layout()
plt.show()





# ============================================================
#  SECTION 5: Far-IR SED comparison
# ============================================================
# For each dust model (MW WD01, stellar H19) and each Mach number,
# compute the absorbed UV luminosity -> L_IR, then invert the
# single-T greybody to get T_dust, and plot the observed-frame SED.
# Key physics: L_IR = L_UV,intr * f_abs, where f_abs is the
# seed-averaged absorbed fraction from the turbulent Sigma_d PDF.
# ============================================================

def seedavg_LIR_from(kUV_here_abs, Mach_here, Sigmad_arr, Lintr, K_SPINS=13, K_U=24):
    """
    Mirrors the LF logic:
      - pick K_SPINS empirical quantiles of Σ_d as 'seeds' (μ_sigma)
      - around each seed, integrate absorption over a lognormal with width set by Mach
      - average the absorbed fractions across seeds

    Returns
    -------
    LIR : float
        Bolometric IR luminosity [erg/s], using the same mapping
        L_IR ≈ L_1500,int * f_abs * (c / λ_1500).
    """
    # exact-median seed and symmetric quantiles
    u_left  = (np.arange(1, (K_SPINS//2) + 1) - 0.5) / K_SPINS
    u_mid   = np.array([0.5])
    u_right = 1.0 - u_left[::-1]
    u_seeds = np.concatenate([u_left, u_mid, u_right])

    mu_sigmas = np.quantile(Sigmad_arr, u_seeds)  # (K_SPINS,)
    sig_ln    = sigma_ln_from_Mach(Mach_here)

    # Gauss–Legendre nodes in Normal-quantile space
    xu, wu  = np.polynomial.legendre.leggauss(K_U)
    u_nodes = np.clip(0.5 * (xu + 1.0), 1e-12, 1 - 1e-12)
    w_nodes = 0.5 * wu
    z_nodes = norm.ppf(u_nodes)

    # nodes in Σ_d for each seed, then absorption
    x_nodes = np.exp(np.log(mu_sigmas)[:, None] + sig_ln * z_nodes[None, :])  # (K_SPINS, K_U)
    tau_abs_nodes = kUV_here_abs * x_nodes
    T_abs_nodes   = T_sphere_mixed(tau_abs_nodes)
    A_nodes       = 1.0 - T_abs_nodes   # absorbed fraction at 1500 Å

    # average over quadrature nodes per seed, then over seeds
    A_mean_seed   = np.sum(w_nodes[None, :] * A_nodes, axis=1)  # (K_SPINS,)
    f_abs_seedavg = A_mean_seed.mean()

    # Convert absorbed L_1500 to IR as in your earlier code:
    # L_IR ≈ L_1500,int * f_abs * (nu_1500)
    LIR = Lintr * f_abs_seedavg * (3e10 / (1500e-8))  # erg/s
    return LIR


def freq_convert(lamda_micron):
    """Converts wavelength [micron] to frequency [Hz]."""
    return c * 1e4 / lamda_micron


def T_CMB_corr(Td, redshift):
    """
    Corrects dust temperature Td for the CMB thermal bath at high z.
    At z~7 the CMB is ~22 K, which sets a floor on T_dust and reduces
    the contrast against the CMB background (da Cunha+13).
    """
    return (Td**(4+bd) + (To**(4+bd)) * ((1.0 + redshift)**(4+bd) - 1.0))**(1.0 / (4.0 + bd))

if "compute_Td_single" not in globals():
    def compute_Td_single(logMdust, betad, logLdust, kabs_158):
        """
        Compute single-T greybody temperature Td [K] from:
          - logMdust [log10 Msun]
          - betad    [emissivity index]
          - logLdust [log10 Lsun]  (bolometric IR luminosity)
        using kappa_nu anchored at 158 um.

        NB: the exponents kb^(4+beta) / h^(3+beta) come from the
        modified Planck integral: L = Md * kappa_0 * (8*pi/c^2) *
        (nu_0^-beta) * (kb/h)^(4+beta) * Gamma(4+beta) * zeta(4+beta) * Td^(4+beta).
        The asymmetry (4+beta vs 3+beta) arises because one power of h
        is absorbed into the 2h/c^2 prefactor of B_nu.
        """
        Teta = ((np.pi * 8.0 / c**2) *
                (kabs_158 / freq_convert(158.)**betad) *
                (kb**(4.0 + betad) / h**(3.0 + betad)) *
                scipy.special.zeta(4.0 + betad) *
                scipy.special.gamma(4.0 + betad))
        logTd = (logLdust - logMdust - np.log10(Teta) - np.log10(Msun / Lsun)) / (4.0 + betad)
        return 10**logTd

if "Fnu_funct" not in globals():
    def Fnu_funct(lamda_cm, Td, betad, logMd, redshift, kabs_158):
        """
        Observed-frame F_nu [μJy] from a single-T greybody.

        lamda_cm : REST-frame wavelength [cm]
        Td       : dust temperature [K]
        betad    : emissivity index β
        logMd    : log10(M_dust / Msun)
        redshift : source redshift
        kabs_158 : κ_ν at 158 μm [cm^2/g] (see also GSD.py for the consistency check)
        """
        # correct Tdust for CMB heating
        Td = T_CMB_corr(Td, redshift)
        
        # luminosity distance in cm (luminosity_distance is in Mpc)
        DL = cosmo.luminosity_distance(redshift).value * Mpc_to_cm  # [cm]

        # conversion to μJy
        cost = 1e29 * lamda_cm**2 / c / (4.0 * np.pi * DL**2)  # μJy

        # CMB correction
        TCMB = To * (1.0 + redshift)
        CMB  = 1.0 / (np.exp(h * c / (kb * TCMB * lamda_cm)) - 1.0)

        # --- κ_λ scaling anchored at 158 μm ---
        lam_158_cm = 158.0 * 1e-4              # 158 µm in cm
        # κ_λ = κ_158 * (λ / 158 µm)^(-β)
        kappa_lambda = kabs_158 * (lamda_cm / lam_158_cm)**(-betad)

        # Planck λ-form: B_λ ∝ λ^-5 / (exp(hc/λkT) - 1)
        F_nu = ((1.0 + redshift) * cost *
                (h * c**2 * np.pi * 8.0) *
                kappa_lambda * lamda_cm**(-5.0) *
                10.0**logMd * Msun *
                (1.0 / (np.exp(h * c / (kb * Td * lamda_cm)) - 1.0) - CMB))

        return F_nu



def Td_from_LIR_Md(LIR_erg_s, Mdust_Msun, beta_d, kabs_158):
    """Wrapper: LIR in erg/s → Td [K] via single-T greybody inversion."""
    logLdust = np.log10(LIR_erg_s / Lsun)
    logMdust = np.log10(Mdust_Msun)
    return compute_Td_single(logMdust, beta_d, logLdust, kabs_158)


# ---- Build and plot the 4 SEDs ----
# --- Dust models used in the SED comparison ---
dust_models = [
    {
        "name":  "MW (WD01)",
        "color": "crimson",
        "kUV_abs": kUV_drn_abs,   # κ_1500,abs for MW dust (extinction × (1-ω))
        "kIR_158": kIR_drn,       # κ_158 for MW dust
    },
    {
        "name":  "Stellar dust (H19)",
        "color": "teal",
        "kUV_abs": kUV_hir_abs,   # κ_1500,abs for stellar dust
        "kIR_158": kIR_hir,       # κ_158 for stellar dust
    },
]


Mach_list = [10, 100]
linestyle_map = {10: "-", 100: "--"}

# wavelength grids (observed) and rest-frame conversion
lambda_obs  = np.logspace(2, 4, 2000)         # micron (obs)
lambda_rest = lambda_obs / (1.0 + redshift)   # micron (rest)
lam_cm_rest = lambda_rest * 1e-4              # cm (rest)

plt.figure(figsize=(7.2, 5.2))

for model in dust_models:
    label   = model["name"]
    color   = model["color"]
    kUV_abs_model  = model["kUV_abs"]   # κ_1500,abs for this model
    kappa_158_model = model["kIR_158"]  # κ_158 for this model

    for Mnum in [10, 100]:
        # L_IR from clumpy ISM with the *correct* UV opacity
        LIR_case = seedavg_LIR_from(kUV_abs_model, Mnum, Sigmad_arr, Lintr,
                                    K_SPINS=13, K_U=24)
        print ('log (LIR/Lsun) for dust model', label, 'Mach =', Mnum, '=', np.log10(LIR_case / Lsun))
        
        # Td from L_IR, Mdust, and κ_158 of this model
        Td_case  = Td_from_LIR_Md(LIR_case, M_dust, bd, kappa_158_model)
        print('\n', label, 'Mach =', Mnum, 'Td =', Td_case)

        # SED in μJy using the same κ_158
        Fnu_case = Fnu_funct(lam_cm_rest, Td_case, bd,
                             np.log10(M_dust), redshift, kappa_158_model)

        plt.plot(np.log10(lambda_obs), np.log10(Fnu_case),
                 color=color, ls=linestyle_map[Mnum], lw=2.2,
                 label=f"{label}, $\\mathcal{{M}}$={Mnum} ($T_d$={Td_case:.1f} K)",
                 alpha=0.7)


# --- REBELS data points overplotted ---
## colormap (same as tau_v plot for consistency)
custom_colormap_base = cm.coolwarm
custom_colormap = truncate_colormap(custom_colormap_base, 0.1, 1.)

# REBELS data
if redshift == 7:
    for jj in range(len(REBELS_index)):
        plt.errorbar(np.log10(158*reds_REB[jj]), np.log10(IR_Flux_REB[jj]), yerr=[[np.log10(IR_Flux_REB[jj] + err_IR_Flux_REB[jj]) - np.log10(IR_Flux_REB[jj])], [-np.log10(IR_Flux_REB[jj] - err_IR_Flux_REB[jj]) + np.log10(IR_Flux_REB[jj])]], ms=10.,marker='s',capsize=2.5,mec='black',elinewidth=0.5,alpha=0.8, color=custom_colormap((logMstar_REB_npSFH[jj] -6.2)/4.2), mew=0.3)

# Blue monsters upper limits
if redshift>7: 
    for pp in range(len(redshift_meas)):
        if IR_Flux_meas[pp]>0:
            print('\n z=', redshift_meas[pp])
            print('lambda_rf/micron ->', lambda_obs_meas[pp])
            print('log (lambda_obs/micron)->', np.log10(lambda_obs_meas[pp]*(1.+redshift_meas[pp])))
            plt.errorbar(np.log10(lambda_obs_meas[pp]*(1.+redshift_meas[pp])), np.log10(IR_Flux_meas[pp]), uplims=bool(uplims_IRFlux[pp]), yerr=0.01*IR_Flux_meas[pp], ms=10.,marker='h',capsize=2.5,mec='black',elinewidth=0.5,alpha=0.8, color=custom_colormap((log_Mstar_meas[pp]- 6.19904499)/(10.19904499- 6.19904499)) ,mew=0.3)
            plt.text(np.log10(lambda_obs_meas[pp] * (1. + redshift_meas[pp])) - 0.04, np.log10(IR_Flux_meas[pp])+0.04, str(names[pp]), fontsize=10, color=custom_colormap((log_Mstar_meas[pp] - 6.19904499) / (10.19904499 - 6.19904499)))

cmap = plt.cm.magma      # choose your colormap here
colors = cmap(np.linspace(0.2, 0.8, 4))  # 4 nice evenly spaced colors

# y-range for the fills
yr = np.linspace(-5, 4.3, 100)
# Band 6
plt.fill_betweenx(yr, np.log10(1.1e3), np.log10(1.4e3),
                  color=colors[0], alpha=0.1, zorder=-100)
# Band 7
plt.fill_betweenx(yr, np.log10(0.8e3), np.log10(1.1e3),
                  color=colors[1], alpha=0.1, zorder=-100)
# Band 8
plt.fill_betweenx(yr, np.log10(0.6e3), np.log10(0.8e3),
                  color=colors[2], alpha=0.1, zorder=-100)
# Band 9
plt.fill_betweenx(yr, np.log10(0.4e3), np.log10(0.5e3),
                  color=colors[3], alpha=0.1, zorder=-100)

if redshift==7:
    plt.ylim(0.77, 2.45)
    plt.text(np.log10(1.19e3), 1, '6', fontsize=16, color=colors[0], alpha=0.5)
    plt.text(np.log10(0.9e3), 1, '7', fontsize=16, color=colors[1], alpha=0.5)
    plt.text(np.log10(0.65e3), 1, '8', fontsize=16, color=colors[2], alpha=0.5)
    plt.text(np.log10(0.45e3), 1, '9', fontsize=16, color=colors[3], alpha=0.5)
    plt.text(2.7, 0.85, 'ALMA Bands', fontsize=18, alpha=0.3)
else:
    plt.ylim(-0.7,1.8)
    plt.text(np.log10(1.19e3), -0.5, '6', fontsize=16, color=colors[0], alpha=0.5)
    plt.text(np.log10(0.9e3), -0.5, '7', fontsize=16, color=colors[1], alpha=0.5)
    plt.text(np.log10(0.65e3), -0.5, '8', fontsize=16, color=colors[2], alpha=0.5)
    plt.text(np.log10(0.45e3), -0.5, '9', fontsize=16, color=colors[3], alpha=0.5)
    plt.text(2.7, -0.65, 'ALMA Bands', fontsize=18, alpha=0.3)

plt.xlim(2.,3.35)
plt.xlabel(r'$\log\,(\lambda_{\rm obs}/\mu{\rm m})$')
plt.ylabel(r'$\log\,(F_{\nu}/{\rm \mu Jy})$')
plt.legend(frameon=False, fontsize=11, ncol=1, loc='upper left')
plt.tight_layout()
plt.show()


