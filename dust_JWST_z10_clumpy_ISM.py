
from scipy.integrate import quad
from matplotlib import cm
col_f = cm.get_cmap('gray')
from librerie import *
fold_in = '../'
from highz_gal_SAM import *
from general import name_and_save, increase_ticklabels, set_labels, do_minorticks, do_log, equal_axes, \
    set_colorbar_labels, set_ticklabels
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.gridspec as gridspec


# ======== CONSTANTS ===========        # Msun in grams
mu = 2.0                 # mean molecular weight (not used anymore)
bd=2.03
kUV=kUV_drn #cm^2/g
kUV_abs=kUV_drn_abs

##############################################
##########   DUST DISTRIBUTIONS    ###########
##############################################

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

def massagrain_cgs(a):
    """
    Mass of a spherical dust grain of radius a (in cm).
    Assumes silicate density of 3.5 g/cm^3.
    """
    d = 3.5  # g/cm^3
    return (4.0 / 3.0) * np.pi * a**3 * d
##############################################


# ======== INPUT DATA ===========
# Load SB99 instantaneous burst tables
logSNr_yr = np.loadtxt('/Users/lsommovigo/Desktop/Scripts/txt_files/snr_inst_Z001.txt', usecols=1)
time_yr = np.loadtxt('/Users/lsommovigo/Desktop/Scripts/txt_files/snr_inst_Z001.txt', usecols=0)
L1500_SB99 = np.loadtxt('/Users/lsommovigo/Desktop/Scripts/txt_files/L1500_inst_Z001.txt', usecols=1)
time_yr_L1500 = np.loadtxt('/Users/lsommovigo/Desktop/Scripts/txt_files/L1500_inst_Z001.txt', usecols=0)


# ======== HALO MODEL PARAMETERS ===========
redshift = 7.0
logMh = 11.#0.9
fb = cosmo.Ob(redshift) / cosmo.Om(redshift)
epsilon = 0.1
yd = 0.1

# Stellar mass from halo mass
Mstar_array = halo_to_stellar_mass(10**logMh, fb, epsilon)
print('log Mstar/Msun -->', np.log10(Mstar_array))

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


# ======== BUILD DUST SURFACE DENSITY DISTRIBUTION ===========
### === Setting up shell model with log-normal surface density fluctuations ===

# Fixed total dust mass [Msun]
M_dust = Md_arr[-1]
print('log Mdust/Msun -->', np.log10(M_dust))
print('log Mstar/Msun -->', np.log10(Mstar_array))

# Median sigma_d for uniform distribution [g/cm^2]
tauUV_arr = tau_pred(kUV, M_dust, 10**logMh, spin_param_distr, redshift)
Sigmad_arr = tauUV_arr / kUV  # g/cm^2            

# Log-normal draw for sigma_d, width set by Mach number
#Mach=3000  # turbulent Mach number
Sigmad_distr_10 = draw_sigma_distribution(mu_sigma=np.median(Sigmad_arr), Mach=10)
Sigmad_distr_300 = draw_sigma_distribution(mu_sigma=np.median(Sigmad_arr), Mach=300)



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

          
#--- Plot Sigma_dust distribution
plt.figure(figsize=(9, 6))
plt.hist(np.log10(Sigmad_distr_10), bins=30, alpha=0.6, histtype='step', color='mistyrose',edgecolor='firebrick',label='Turbulence-driven (Mach=10)',ls='--',lw=1.5,density=True)
plt.hist(np.log10(Sigmad_distr_300), bins=30, alpha=0.3, histtype='step', color='midnightblue',edgecolor='midnightblue',label='Turbulence-driven(Mach=300)',ls='--',lw=1., density=True)
plt.hist(np.log10(Sigmad_arr), bins=30, alpha=0.2,lw=2.5, histtype='stepfilled', color='grey',edgecolor='dimgrey',label='Spin-driven', density=True)

plt.axvline(np.log10(np.percentile(Sigmad_distr_10,16)), color='firebrick', linestyle='--',lw=2.,alpha=0.6)
plt.axvline(np.log10(np.percentile(Sigmad_distr_10,84)), color='firebrick', linestyle='--',lw=2.,alpha=0.6)

plt.axvline(np.log10(np.percentile(Sigmad_distr_300,16)), color='midnightblue', linestyle='--',alpha=0.4)
plt.axvline(np.log10(np.percentile(Sigmad_distr_300,84)), color='midnightblue', linestyle='--',alpha=0.4)

plt.axvline(np.log10(np.percentile(Sigmad_arr,16)), color='dimgrey',lw=2.5,alpha=0.4)
plt.axvline(np.log10(np.percentile(Sigmad_arr,84)), color='dimgrey',lw=2.5,alpha=0.4)
plt.xlabel(r'$\log_{10}(\Sigma_{d}/M_{\odot}\,\mathrm{kpc}^{-2})$')
plt.ylabel('PDF')
plt.legend(frameon=False, fontsize=14,loc='upper left')
plt.xlim(-7,-1)
plt.tight_layout()
plt.show()


'''
#--- Plot width of log-normal dust surface density distribution vs Mach number
Mach_arr = np.linspace(5, 1000, 1000) 
R = compute_R(Mach_arr)
sigma_lnSigma_sq = np.log(1 + (R * Mach_arr**2) / 4)
plt.plot(Mach_arr, np.sqrt(sigma_lnSigma_sq),color='grey',alpha=0.5)
plt.xscale('log')
plt.xlabel('Mach number')
plt.ylabel(r'$\sigma_{\ln \Sigma_d}$')
#plt.title('Width of log-normal dust surface density distribution')
plt.grid(True, which='both', ls=':', alpha=0.3)
plt.tight_layout()
plt.show()
'''



# --- knobs to mirror the LF logic ---
from scipy.stats import norm

K_SPINS   = 13      # odd, so we include the exact median seed (u=0.5)
W_BLEND   = 0.6     # blend between median-seed transmission and mean over seeds
N_LOS     = 600     # LOS draws per Mach (distributed across seeds)
albedo    = 0.3807
Mach_array = np.array([5, 10, 20, 30, 40, 50, 100, 200, 300])

# exact-median spin seed and symmetric quantiles (data-driven via Sigmad_arr)
u_left  = (np.arange(1, (K_SPINS//2)+1) - 0.5) / K_SPINS
u_mid   = np.array([0.5])
u_right = 1.0 - u_left[::-1]
u_seeds = np.concatenate([u_left, u_mid, u_right])  # length = K_SPINS
mid_idx = K_SPINS // 2

def sigma_ln_from_Mach(Mach):
    R = compute_R(Mach)  # (Thompson+16; alpha=2.5 in your code)
    return np.sqrt(np.log(1 + (R * Mach**2) / 4.0))

# containers for the text summary you print
LUV_med_vals, LUV_84_vals, LIR_seedavg_vals, LIR_singlemu_vals = [], [], [], []

plt.figure(figsize=(9, 6))

for i, Mach in enumerate(Mach_array):
    # lognormal width for the clumpy scatter (same for all LOS at this Mach)
    sigma_ln = sigma_ln_from_Mach(Mach)

    # -------- seeds from the *empirical* Σ_d distribution (mirrors population code) --------
    # we take K_SPINS empirical quantiles of Sigmad_arr as the seed means μ_sigma
    mu_sigmas = np.quantile(Sigmad_arr, u_seeds)   # shape (K_SPINS,)

    # -------- UV: build LOS distribution by scattering around each seed and stacking --------
    # use common quantiles for LOS to reduce noise, like in the LF code
    u_los = (np.arange(1, N_LOS+1) - 0.5) / N_LOS
    z_los = norm.ppf(u_los)[None, :]                       # (1, N_LOS)
    # draw Σ_d samples per seed (broadcast), then flatten
    Sigmad_LOS = np.exp(np.log(mu_sigmas)[:, None] + sigma_ln * z_los)  # (K_SPINS, N_LOS)
    tauUV_LOS  = kUV * Sigmad_LOS
    T_uv_LOS   = T_1500_sphere_im(tauUV_LOS)
    LUV_LOS    = Lintr * T_uv_LOS
    LUV_draws  = LUV_LOS.ravel()

    # violin on log10 but draw in linear axis:
    parts = plt.violinplot(np.log10(LUV_draws),
                           positions=[Mach], widths=0.3*Mach,
                           showmeans=False, showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        y = pc.get_paths()[0].vertices[:, 1]
        pc.get_paths()[0].vertices[:, 1] = 10**y   # back to linear y
        pc.set_facecolor('royalblue'); pc.set_edgecolor('darkblue'); pc.set_alpha(0.15)
        pc.set_zorder(1)

    # reference stats from the LOS distribution (these are what the violin shows)
    LUV_med   = np.median(LUV_draws)
    LUV_p84   = np.percentile(LUV_draws, 84)
    LUV_med_vals.append(LUV_med); LUV_84_vals.append(LUV_p84)

    # -------- UV: overlay the *uniform blended* point built from the SEEDS (no scatter) --------
    # transmission at each seed (no clumpy perturbation) and blend like in LF:
    TK_seeds   = T_1500_sphere_im(kUV * mu_sigmas)         # shape (K_SPINS,)
    T_med_seed = TK_seeds[mid_idx]                         # exact median seed
    T_mean     = TK_seeds.mean()
    T_eff      = (1.0 - W_BLEND) * T_med_seed + W_BLEND * T_mean
    LUV_eff    = Lintr * T_eff

    # Plot the blended "uniform" point on top of the violin
    plt.scatter(Mach, LUV_eff, s=70, marker='o',
                facecolor='royalblue', edgecolor='black', linewidths=0.6, alpha=0.9,
                label=r'$L_{\rm UV}$ (seed-blend)' if i==0 else None)

    # Optionally also show the LOS-median and 84th as open markers (comment out if cluttered)
    plt.scatter(Mach, LUV_med, s=55, marker='o',
                facecolor='none', edgecolor='royalblue', linewidths=1.2, alpha=0.8,
                label=r'$L_{\rm UV}$ (LOS median)' if i==0 else None)
    # plt.scatter(Mach, LUV_p84, s=50, marker='o',
    #             facecolor='none', edgecolor='navy', linewidths=1.0, alpha=0.7,
    #             label=r'$L_{\rm UV,84}$' if i==0 else None)

    # -------- IR: integrate against the continuous PDF around each SEED, average across seeds --------
    # Gauss–Legendre in Normal quantile space (fast + mirrors population code)
    K_U = 24
    xu, wu  = np.polynomial.legendre.leggauss(K_U)
    u_nodes = np.clip(0.5*(xu+1.0), 1e-12, 1-1e-12)
    w_nodes = 0.5*wu
    z_nodes = norm.ppf(u_nodes)

    # nodes in Σ_d for each seed
    x_nodes = np.exp(np.log(mu_sigmas)[:, None] + sigma_ln * z_nodes[None, :])      # (K_SPINS, K_U)
    tau_abs_nodes = kUV * (1.0 - albedo) * x_nodes
    T_abs_nodes   = T_1500_sphere_im(tau_abs_nodes)
    A_nodes       = 1.0 - T_abs_nodes
    A_mean_seed   = np.sum(w_nodes[None, :] * A_nodes, axis=1)                       # (K_SPINS,)
    f_abs_seedavg = A_mean_seed.mean()
    LIR_seedavg   = Lintr * f_abs_seedavg
    LIR_seedavg_vals.append(LIR_seedavg)

    # For comparison: your original single-μ IR using μ = median(Sigmad_arr)
    mu_ln_lin = np.log(np.median(Sigmad_arr))
    x = np.logspace(np.log10(np.exp(mu_ln_lin - 6*sigma_ln)),
                    np.log10(np.exp(mu_ln_lin + 6*sigma_ln)), 2000)
    p_x = lognormal_pdf_x(x, mu_ln_lin, sigma_ln)
    tau_abs = kUV * x * (1.0 - albedo)
    T_abs   = T_1500_sphere_im(tau_abs)
    LIR_singlemu = Lintr * np.trapz((1.0 - T_abs) * p_x, x)
    LIR_singlemu_vals.append(LIR_singlemu)

    # Plot IR points:
    # seed-averaged (solid diamond) and single-μ (hollow diamond) for comparison
    plt.scatter(Mach, LIR_seedavg, facecolor='coral', edgecolor='darkred', alpha=0.95,
                label=r'$L_{\rm IR}$ (seed-avg PDF)' if i==0 else None, marker='d', s=80, zorder=4)
    plt.scatter(Mach, LIR_singlemu, facecolor='none', edgecolor='darkred', alpha=0.85,
                label=r'$L_{\rm IR}$ (single-$\mu$ PDF)' if i==0 else None, marker='d', s=70, zorder=4)

# --- Uniform case (unchanged) ---
T_1500_uniform = T_1500_sphere_im(tauUV_arr)
L1500_uniform  = Lintr * T_1500_uniform
L1500_abs_uniform = Lintr * (1 - T_1500_uniform) * (1 - albedo)
LIR_uniform    = L1500_abs_uniform
L1500_uniform_med = np.median(L1500_uniform)
LIR_uniform_med   = np.median(LIR_uniform)

print(f'Uniform case --> Luv={L1500_uniform_med:.2e}, Lir={LIR_uniform_med:.2e}, '
      f'Lir/(Luv+Lir)={LIR_uniform_med/(LIR_uniform_med + L1500_uniform_med):.2f}')
print('Clumpy case (Mach=10) --> Luv_med=', LUV_med_vals[1], ', Lir_seedavg=', LIR_seedavg_vals[1],
      ', Lir/(Luv+Lir)=', LIR_seedavg_vals[1]/(LUV_med_vals[1]+LIR_seedavg_vals[1]))
print('Clumpy case (Mach=300) --> Luv_med=', LUV_med_vals[-1], ', Lir_seedavg=', LIR_seedavg_vals[-1],
      ', Lir/(Luv+Lir)=', LIR_seedavg_vals[-1]/(LUV_med_vals[-1]+LIR_seedavg_vals[-1]))

# --- finalize plot ---
plt.axhline(Lintr, color='black', label='$L_{\\rm intr}$', lw=2, alpha=0.5, zorder=2)
plt.axhline(L1500_uniform_med, color='royalblue', linestyle='--',
            label='$L_{\\rm UV}$ (uniform)', lw=2, alpha=0.5, zorder=2)
plt.axhline(LIR_uniform_med, color='coral', linestyle='--',
            label='$L_{\\rm IR}$ (uniform)', lw=2, alpha=0.5, zorder=2)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Mach number $\\mathcal{M}$')
plt.ylabel('Luminosity [erg/s/Å]')
plt.ylim(1e25, 5e29)
plt.legend(frameon=False, fontsize=13, ncol=2)
plt.tight_layout()
plt.show()




# ============================
# IR SEDs for (kUV_drn, kUV_hir) × (Mach=10, 300)
# ============================

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
    T_abs_nodes   = T_1500_sphere_im(tau_abs_nodes)
    A_nodes       = 1.0 - T_abs_nodes   # absorbed fraction at 1500 Å

    # average over quadrature nodes per seed, then over seeds
    A_mean_seed   = np.sum(w_nodes[None, :] * A_nodes, axis=1)  # (K_SPINS,)
    f_abs_seedavg = A_mean_seed.mean()

    # Convert absorbed L_1500 to IR as in your earlier code:
    # L_IR ≈ L_1500,int * f_abs * (c / λ_1500)
    LIR = Lintr * f_abs_seedavg * (3e10 / (1500e-8))  # erg/s
    return LIR


def freq_convert(lamda_micron):
    """Converts wavelength [micron] to frequency [Hz]."""
    return c * 1e4 / lamda_micron


def T_CMB_corr(Td, redshift):
    """
    Corrects dust temperature Td for the effect of the CMB acting as a thermal bath at a given redshift (matters at high-z).
    """
    return (Td**(4+bd) + (To**(4+bd)) * ((1.0 + redshift)**(4+bd) - 1.0))**(1.0 / (4.0 + bd))

if "compute_Td_single" not in globals():
    def compute_Td_single(logMdust, betad, logLdust, kabs_158):
        """
        Compute single-T greybody temperature Td [K] from:
          - logMdust [log10 Msun]
          - betad    [emissivity index]
          - logLdust [log10 Lsun]  (bolometric IR luminosity)
        using κ_ν anchored at 158 μm.
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
        DL = cosmo.luminosity_distance(redshift).value * kpc_to_cm  # [cm]

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
cases = [
    ("Draine MW",            kUV_drn,  kIR_drn,  "crimson"),
    ("Hirashita large-grain", kUV_hir, kIR_hir, "teal"),
]

Mach_list = [10, 300]
linestyle_map = {10: "-", 300: "--"}

# wavelength grids (observed) and rest-frame conversion
lambda_obs  = np.logspace(2, 4, 2000)         # micron (obs)
lambda_rest = lambda_obs / (1.0 + redshift)   # micron (rest)
lam_cm_rest = lambda_rest * 1e-4              # cm (rest)

plt.figure(figsize=(7.2, 5.2))

for label_k, kappa_UV, kappa_IR_158, color in cases:
    for Mnum in Mach_list:
        # L_IR from clumpy ISM (uses κ_UV)
        LIR_case = seedavg_LIR_from(kUV_abs, Mnum, Sigmad_arr, Lintr,
                                    albedo=albedo, K_SPINS=13, K_U=24)

        # Td from L_IR, Mdust, and κ_158
        Td_case  = Td_from_LIR_Md(LIR_case, M_dust, bd, kappa_IR_158)
        print('\n', label_k, 'Mach =', Mnum, 'Td =', Td_case)

        # SED in μJy
        Fnu_case = Fnu_funct(lam_cm_rest, Td_case, bd,
                             np.log10(M_dust), redshift, kappa_IR_158)

        plt.plot(np.log10(lambda_obs), np.log10(Fnu_case),
                 color=color, ls=linestyle_map[Mnum], lw=2.2,
                 label=f"{label_k}, Mach={Mnum} (T_d={Td_case:.1f} K)")


# --- ALMA band shading etc. (your original code) ---
plt.fill_betweenx(np.linspace(-5, 4.3, 100),
                  np.log10(1.1e3), np.log10(1.4e3),
                  color='gainsboro', alpha=0.2, zorder=-100)
plt.text(np.log10(1.19e3), 1.2, '6', fontsize=12, alpha=0.65)

plt.fill_betweenx(np.linspace(-5, 4.3, 100),
                  np.log10(0.8e3), np.log10(1.1e3),
                  color='silver', alpha=0.2, zorder=-100)
plt.text(np.log10(0.9e3), 1.2, '7', fontsize=12, alpha=0.65)

plt.fill_betweenx(np.linspace(-5, 4.3, 100),
                  np.log10(0.6e3), np.log10(0.8e3),
                  color='grey', alpha=0.2, zorder=-100)
plt.text(np.log10(0.65e3), 1.2, '8', fontsize=12, alpha=0.65)

plt.fill_betweenx(np.linspace(-5, 4.3, 100),
                  np.log10(0.4e3), np.log10(0.5e3),
                  color='dimgrey', alpha=0.3, zorder=-100)
plt.text(np.log10(0.47e3), 1.2, '9', fontsize=12, alpha=0.65)

plt.ylim(-3.8, 1.7)
plt.xlabel(r'$\log\,(\lambda_{\rm obs}/\mu{\rm m})$')
plt.ylabel(r'$\log\,(F_{\nu}/{\rm \mu Jy})$')
plt.legend(frameon=False, fontsize=11, ncol=1, loc='lower right')
plt.tight_layout()
plt.show()

