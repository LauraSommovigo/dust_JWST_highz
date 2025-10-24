
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



##############################################
##########   GENERAL FUNCTIONS     ###########
##############################################

def T_CMB_corr(Td, redshift):
    """
    Corrects dust temperature Td for the effect of the CMB at a given redshift.
    """
    return (Td**(4+bd) + (To**(4+bd)) * ((1.0 + redshift)**(4+bd) - 1.0))**(1.0 / (4.0 + bd))

def Plank_funct(y, bd):
    """
    Planck function integrand for computing the total emitted power.
    """
    return (y**(3+bd)) / (np.exp(y) - 1)

def Planck_integral(bd):
    """
    Integral of the Planck function needed for computing grain temperatures.
    """
    return quad(Plank_funct, 0.0, 35.0, args=(bd))[0]

##############################################
##########    INPUT ARRAYS         ###########
##############################################

# Dust absorption wavelengths [Å], normalized to 1600 Å
londa = np.array([
    4217., 3981., 3758., 3548., 3350., 3162., 2985., 2818., 2661., 2512.,
    2371., 2239., 2113., 1995., 1884., 1778., 1679., 1585., 1496., 1413.,
    1334., 1259., 1189., 1122., 1059., 1000., 944., 900.
]) / 1600.0  # adimensional

# Grain size bins [cm], logarithmic in microns
a = np.array([
    1.000e-03, 1.5849e-03, 2.5119e-03, 3.9811e-03, 6.3096e-03, 1.0000e-02,
    1.5849e-02, 2.5119e-02, 3.9811e-02, 6.3096e-02, 1.0000e-01, 1.5849e-01,
    2.5119e-01, 3.9811e-01, 6.3096e-01, 1.0000e+00
]) * 1e-4  # convert microns to cm

# Q_abs: wavelength and grain-size-dependent absorption efficiency
# Shape should be [len(wavelengths) * (len(a) - 1)], flattened
Qabs = np.loadtxt(
    '/Users/lsommovigo/Desktop/Scripts/txt_files/Clouds_Fedb/Q_abs_Draine_silicates.txt',
    unpack=True, delimiter=','
)

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

# ======== CONSTANTS ===========        # Msun in grams
kpc_to_cm = 3.086e21     # 1 kpc in cm
mu = 2.0                 # mean molecular weight (not used anymore)
bd=2.03
kUV=kUV_drn #cm^2/g
#kUV=kUV_hir #cm^2/g

# ======== INPUT DATA ===========
# Load SB99 instantaneous burst tables
logSNr_yr = np.loadtxt('/Users/lsommovigo/Desktop/Scripts/txt_files/snr_inst_Z001.txt', usecols=1)
time_yr = np.loadtxt('/Users/lsommovigo/Desktop/Scripts/txt_files/snr_inst_Z001.txt', usecols=0)
L1500_SB99 = np.loadtxt('/Users/lsommovigo/Desktop/Scripts/txt_files/L1500_inst_Z001.txt', usecols=1)
time_yr_L1500 = np.loadtxt('/Users/lsommovigo/Desktop/Scripts/txt_files/L1500_inst_Z001.txt', usecols=0)


# ======== HALO MODEL PARAMETERS ===========
redshift = 10.0
logMh = 10.9
fb = cosmo.Ob(redshift) / cosmo.Om(redshift)
epsilon = 0.1
yd = 0.1

# Stellar mass from halo mass
Mstar_array = halo_to_stellar_mass(10**logMh, fb, epsilon)

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



# ====== PLOT 1: Attenuation curve and Grain size distribution comparison ======
# to do: i want to compare the grain size distribution used here with the one from Hirashita+19, and make it an inset plot in the one with the attenuation curves

## Constants for the dust model (Draine+03 or Hirashita+19)
print('\n\n kUV/kV (tabulated input Draine):', kUV / kv)
print("kUV from Hirashita:", kUV_hir)
print("kUV from Draine model:", kUV_drn)
print("kUV/kV (Hirashita):", kUV_hir / kv)
print("kUV/kV (interp. Draine):", kUV_drn / kv)
# --- Plot in lambda [Angstrom] ---
fig, att_axis = plt.subplots(figsize=(5, 5))
att_axis.plot(lambda_ang_hir, Alambda_over_Av_hir, lw=2, alpha=0.8, label='Hirashita+19, (0.1–0.3) Gyr', color='teal')
att_axis.plot(lambda_ang_drn, drn_interp(lambda_ang_drn), lw=2.5, alpha=0.5, color='crimson', label='Draine+03, MW')
att_axis.axvline(lambda_UV, color='gray', linestyle='--', lw=1)
att_axis.scatter(lambda_UV, kUV_drn / kv, label='$k_{1500}/k_V$', color='crimson', marker='D', s=60, edgecolor='black',alpha=0.4)
att_axis.scatter(lambda_UV, kUV_hir / kv, color='teal', marker='D', s=60, edgecolor='black',alpha=0.7)
# Set custom ticks
att_axis.set_xticks([1500, 2175, 3543., 4770., 6231., 7625., 9134.])
att_axis.set_xscale('log')
att_axis.set_xlabel('$\lambda\ \mathrm{[\dot{A}]}$', fontsize=20)
att_axis.set_ylabel('$\\tau_{\lambda} / \\tau_V$', fontsize=20)
#plt.title('Attenuation Curves in Wavelength Space', fontsize=14)
#att_axis.set_xscale('log')
att_axis.text(1550,0,'$\lambda=1500\ \mathrm{[\dot{A}]}$',fontsize=16,color='gray')
att_axis.set_xlim(1e3,2e4)
att_axis.set_ylim(-0.2,5)
att_axis.grid(True, alpha=0.2, lw=0.5)
plt.legend(frameon=False, fontsize=14, loc='upper left')
plt.tight_layout()

#--- Inset: Grain size distribution comparison ---
inset_ax = att_axis.inset_axes([0.55, 0.5, 0.4, 0.4])
# Load Hirashita data
hir_a, hir_a4n = np.loadtxt('/Users/lsommovigo/Desktop/Scripts/txt_files/JWST_dust_z10/Hirashita_a4n_dense_01Gyr.txt', unpack=True, delimiter=',')
hir_a_cm = hir_a * 1e-4  # convert micron to cm

# Draine+03 grain size distribution (a^4 * n(a))
a_cm=np.linspace(1e-8, 1e-4, 10000)  # cm
a_mids = 0.5 * (a_cm[:-1] + a_cm[1:])
draine_a4n = a_mids**4 * na(a_cm) / np.diff(a_cm) # to get n(a) in per cm units
# Plot both distributions
inset_ax.plot(hir_a, hir_a4n / np.max(hir_a4n), label='Hirashita+19', color='teal', alpha=0.8)
inset_ax.plot(1e4*a_mids, draine_a4n / np.max(draine_a4n), label='Draine+03 - MW', color='crimson', alpha=0.5)
inset_ax.plot(1e4*a_mids, a_mids**0.5/np.max(a_mids**0.5), lw=0.8, ls=':', color='crimson', alpha=0.5, label='MRN')
inset_ax.set_xscale('log')
inset_ax.set_yscale('log')
increase_ticklabels(inset_ax, 16)
inset_ax.set_xlabel('Grain size $a$ [$\mu$m]',fontsize=16)
inset_ax.set_ylabel('$a^4 n(a)$ (normalized)',fontsize=16)
inset_ax.set_ylim(1e-3, 2)
inset_ax.set_xlim(1e-3,1)

plt.grid(True, which='both', ls=':', alpha=0.1)
#plt.savefig('grain_size_dist_comparison.png', dpi=300)
plt.show()






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
plt.hist(np.log10(Sigmad_distr_10), bins=30, alpha=0.6, histtype='step', color='mistyrose',edgecolor='firebrick',label='Turbulence-driven distribution (Mach=10)',ls='--',lw=1.5,density=True)
plt.hist(np.log10(Sigmad_distr_300), bins=30, alpha=0.3, histtype='step', color='midnightblue',edgecolor='midnightblue',label='Turbulence-driven distribution (Mach=300)',ls='--',lw=1., density=True)
plt.hist(np.log10(Sigmad_arr), bins=30, alpha=0.2,lw=2.5, histtype='stepfilled', color='grey',edgecolor='dimgrey',label='Spin-driven distribution', density=True)

plt.axvline(np.log10(np.percentile(Sigmad_distr_10,16)), color='firebrick', linestyle='--',lw=2.,alpha=0.6)
plt.axvline(np.log10(np.percentile(Sigmad_distr_10,84)), color='firebrick', linestyle='--',lw=2.,alpha=0.6)

plt.axvline(np.log10(np.percentile(Sigmad_distr_300,16)), color='midnightblue', linestyle='--',alpha=0.4)
plt.axvline(np.log10(np.percentile(Sigmad_distr_300,84)), color='midnightblue', linestyle='--',alpha=0.4)

plt.axvline(np.log10(np.percentile(Sigmad_arr,16)), color='dimgrey',lw=2.5,alpha=0.4)
plt.axvline(np.log10(np.percentile(Sigmad_arr,84)), color='dimgrey',lw=2.5,alpha=0.4)
plt.xlabel(r'$\log_{10}(\Sigma_{d}/M_{\odot}\,\mathrm{kpc}^{-2})$')
plt.ylabel('PDF')
plt.legend(frameon=False, fontsize=14,loc='upper left')
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

# --- Helpers ---
# Function to compute sigma_ln from Mach
def sigma_ln_from_Mach(Mach):
    R = compute_R(Mach)
    return np.sqrt(np.log(1 + (R * Mach**2) / 4.0))

albedo = 0.3807
Mach_array = np.array([5,10,20,30,40,50,100,200,300])

LUV_mean_vals, LIR_vals = [], []

#--- Loop over Mach numbers ---
plt.figure(figsize=(7, 6))
for i, Mach in enumerate(Mach_array):
    # --- parameters of Σ_d lognormal (in linear space) ---
    mu_ln   = np.log(np.median(Sigmad_arr))
    sigma_ln = sigma_ln_from_Mach(Mach)

    # === UV: keep the Monte-Carlo distribution (NO PDF integration) ===
    Sigmad_distr = draw_sigma_distribution(mu_sigma=np.median(Sigmad_arr), Mach=Mach)  # linear Σ_d
    tauUV_distr  = kUV * Sigmad_distr
    T_uv_draws   = T_1500_sphere_im(tauUV_distr)                 # per-LOS transmission
    L1500_distr  = Lintr * T_uv_draws                             # per-LOS UV luminosities
    L1500_mean   = np.median(L1500_distr)                           # optional sample mean (not a PDF integral)
    LUV_mean_vals.append(L1500_mean)

    # plot the distribution (spread) and the sample mean for UV
    plt.scatter(Mach*np.ones(len(L1500_distr)), L1500_distr, facecolor='royalblue', edgecolor='darkblue', alpha=0.25, s=8)
    plt.scatter(Mach, L1500_mean, facecolor='royalblue', edgecolor='darkblue', alpha=0.5, label='$L_{\\rm UV}$ (diff. LOS)' if i==0 else None)

    # === IR: integrate against the continuous PDF ===
    # build an x-grid that captures essentially all probability
    x_min = np.exp(mu_ln - 6*sigma_ln)
    x_max = np.exp(mu_ln + 6*sigma_ln)
    x = np.logspace(np.log10(x_min), np.log10(x_max), 2000)
    p_x = lognormal_pdf_x(x, mu_ln, sigma_ln)                     # ~normalized over [x_min, x_max]

    tau_abs = kUV * x * (1.0 - albedo)                            # absorption optical depth
    T_abs   = T_1500_sphere_im(tau_abs)

    LIR_total = Lintr * np.trapz((1.0 - T_abs) * p_x, x)          # E[absorbed UV]
    LIR_vals.append(LIR_total)

    plt.scatter(Mach, LIR_total, facecolor='coral', edgecolor='darkred', alpha=0.9, label='$L_{\\rm IR}$ (PDF-integrated)' if i==0 else None)

# -- Uniform case (unchanged logic) --
T_1500_uniform = T_1500_sphere_im(tauUV_arr)
L1500_uniform = Lintr * T_1500_uniform
L1500_absorbed_uniform = Lintr * (1 - T_1500_uniform) * (1 - albedo)
LIR_uniform = L1500_absorbed_uniform
L1500_uniform_mean = np.median(L1500_uniform)
LIR_uniform_mean   = np.median(LIR_uniform)
print(f'Uniform case --> Luv={L1500_uniform_mean:.2e}, Lir={LIR_uniform_mean:.2e}, '
      f'Lir/Luv={LIR_uniform_mean/L1500_uniform_mean:.2f}')


# -- Finalize plot --
plt.axhline(Lintr, color='black', label='$L_{\\rm intr}$',lw=2,alpha=0.5)
plt.axhline(np.median(L1500_uniform), color='royalblue', linestyle='--', label='$L_{\\rm UV}$ (uniform)',lw=2,alpha=0.5)
plt.axhline(np.median(LIR_uniform), color='coral', linestyle='--', label='$L_{\\rm IR}$ (uniform)',lw=2,alpha=0.5)
plt.xscale('log'); plt.yscale('log')
plt.xlabel('Mach number $\mathcal{M}$ ')
plt.ylabel('Luminosity [erg/s/Å]')
plt.legend(frameon=False, fontsize=14)
plt.tight_layout()
plt.show()
