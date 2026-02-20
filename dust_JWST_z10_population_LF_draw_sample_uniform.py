# ============================================================
# dust_JWST_z10_population_LF_draw_sample_uniform.py
# ============================================================
# Computes the UV and IR luminosity functions (LFs) for a
# population of halos at a given redshift, assuming a UNIFORM
# dust shell (no turbulent clumpiness).
#
# Method:
#   For each halo mass Mh on a grid:
#     1) Build SFH -> get final L1500 and Md (at yd=1, then scale)
#     2) Draw K_SPINS stratified spin parameters -> tau_UV -> T(tau)
#     3) Blend median and mean T to get T_eff (knob W_BLEND)
#     4) MUV_att = -2.5 log10(T_eff * L1500) + 51.63
#   The UV LF is then phi(MUV) = (dn/dlogMh) / |dMUV/dlogMh|
#   (Jacobian transformation of the halo mass function).
#   The IR LF uses L_IR = (1-T_abs) * L1500 * nu_1500.
#
# Outputs:
#   1) UV luminosity function for multiple (epsilon, yd) pairs
#   2) IR luminosity function (same parameter grid)
# ============================================================

from highz_gal_SAM import *
from general import name_and_save, increase_ticklabels, set_labels, do_minorticks, do_log, equal_axes, \
    set_colorbar_labels, set_ticklabels
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.gridspec as gridspec
from scipy.stats import norm  # for inverse CDF of normal (lognormal quantiles)
from scipy.signal import savgol_filter  # for smoothing LF
from scipy.ndimage import median_filter
from matplotlib.lines import Line2D

# ============================================================
#  HELPER FUNCTIONS
# ============================================================
def redistribute_phi(MUV_array, phi_array):
    """
    Fix non-monotonic MUV(Mh) curves caused by dust attenuation
    making massive halos fainter than expected.
    When MUV turns around (starts getting fainter at higher Mh),
    redistribute the phi from the non-monotonic tail back into
    the nearest earlier bin with similar MUV.
    """
    MUV_array = np.array(MUV_array)
    phi_array = np.array(phi_array)

    # Detect where MUV stops getting brighter and starts getting fainter
    diffs = np.diff(MUV_array)
    break_mask = diffs >= 0.05  # look for first increase in MUV (fainter galaxy)
    break_idx = np.argmax(break_mask) + 1 if np.any(break_mask) else len(MUV_array)

    if break_idx < len(MUV_array):
        print(f"Monotonicity breaks at index {break_idx} (MUV = {MUV_array[break_idx]:.3f})")
    else:
        print("MUV is monotonic — no redistribution needed.")

    phi_corr = np.copy(phi_array)

    if break_idx < len(MUV_array):
        for j in range(break_idx, len(MUV_array)):
            muv_j = MUV_array[j]

            prev = MUV_array[:break_idx]
            # Look for the closest earlier point
            diffs = prev - muv_j
            diffs[diffs > 0] = np.inf  # only consider brighter (more negative) previous values

            if np.all(np.isinf(diffs)):
                idx_match = break_idx - 1  # fallback: last monotonic bin
            else:
                idx_match = np.argmin(np.abs(diffs))

            phi_corr[idx_match] += phi_array[j]
            #print(f"→ Reassigning Phi[{j}] (MUV={muv_j:.2f}, Phi={phi_array[j]:.2e}) "
            #      f"→ Phi[{idx_match}] (MUV={MUV_array[idx_match]:.2f})")

        # Trim non-monotonic tail
        MUV_array = MUV_array[:break_idx]
        phi_corr = phi_corr[:break_idx]

    #print("MUV output:", MUV_array)
    #print("phi output:", phi_corr)
    #print("--- END redistribute_phi ---\n")

    return MUV_array, phi_corr


# --- slope-based masking: drop bins where |dMUV/dlogMh| is too small (spike source) ---
SLOPE_FLOOR = 5e-3       # try 5e-3; if any spike stays try 7e-3

def mask_by_slope(x_curve, phi_curve, slope_floor=SLOPE_FLOOR, hard_cap=None):
    """Remove points where the numerical derivative is too small (→ φ spikes).
       Optionally cap absurd φ with a hard ceiling."""
    dx = np.gradient(x_curve, logMh_array)
    mask = np.isfinite(phi_curve) & (np.abs(dx) >= slope_floor)
    if hard_cap is not None:
        med = np.nanmedian(phi_curve[mask]) if np.any(mask) else np.nanmedian(phi_curve)
        mask &= (phi_curve <= hard_cap * med)
    return x_curve[mask], phi_curve[mask]



# ---- colormap ----
costum_colormap = cm.inferno
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=10):
    new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval), cmap(np.linspace(minval, maxval, n)))
    return new_cmap
costum_colormap = truncate_colormap(costum_colormap, 0., 0.7)






# ============================================================
#  MODEL PARAMETERS
# ============================================================
redshift=7.           # target redshift (>7)
fb=cosmo.Ob(redshift)/cosmo.Om(redshift)
lumDistpc = cosmo.luminosity_distance(redshift).value * 1e6  # luminosity distance [pc]
logMh_array=np.linspace(8,13,23)  # halo mass grid [log10 Msun]; same range as Yung+23

# ---- Quick HMF comparison plot ----
plt.plot(logMh_array, dn_dlogMh_GUREFT(logMh_array,redshift),alpha=0.5,zorder=-10, label='Yung+23, GUREFT, z='+str(redshift))
    
# HMF package HMF (convert to physical units)
mf = MassFunction(z=redshift, Mmin=3, Mmax=12, dlog10m=0.05, hmf_model="ST")
m_phys = mf.m * hlittle            # Msun (physical)
dndlogm_phys = mf.dndlog10m / hlittle**3   # [Mpc^-3 Msun^-1] (physical)
plt.plot(np.log10(m_phys), dndlogm_phys, linestyle='--', color='grey',label=f"z={redshift} (hmf package)")

#--- Ulli simulation
def schecter_log10(L_log10, phi_star, L_star_log10, alpha):
    L = 10**L_log10
    L_star = 10**L_star_log10
    return np.log10(phi_star) + alpha * np.log10(L/L_star) - np.log10(np.e) * L / L_star
plt.plot(logMh_array, 10**schecter_log10(logMh_array, 1.75088404e-03,  1.04160168e+01, -1.47813953e+00), linestyle=':',label='Ulli')
#---

plt.yscale('log')
plt.ylabel('$\mathrm{dn/d}\log\mathrm{M_h\ [Mpc^{-3}]}$')
plt.xlabel('$\log (M_h/M_{\odot})$')
#plt.ylim(1e-7,5e3)
plt.xlim(7,12)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()






# ---- SFH and SB99 settings ----
tstep=1               # SFH timestep [Myr]; must resolve SN rate
logSNr_yr=np.loadtxt(os.path.join(SCRIPTS_DIR, 'txt_files/SB99', 'snr_inst_Z001.txt'), usecols=1)
time_yr=np.loadtxt(os.path.join(SCRIPTS_DIR, 'txt_files/SB99', 'snr_inst_Z001.txt'), usecols=0)
L1500_SB99=np.loadtxt(os.path.join(SCRIPTS_DIR, 'txt_files/SB99', 'L1500_inst_Z001.txt'), usecols=1)
time_yr_L1500=np.loadtxt(os.path.join(SCRIPTS_DIR, 'txt_files/SB99', 'L1500_inst_Z001.txt'), usecols=0)

# ---- Dust opacity model ----
# Choose _hir (stellar/Hirashita+19) or _drn (MW/WD01)
kUV=kUV_drn
kUV_abs=kUV_drn_abs
kv=kv_drn




# ============================================================
#  PARAMETER GRID: star-formation efficiency and dust yield
# ============================================================
arr_e=np.array([0.05, 0.10, 0.20])   # SF efficiency
arr_yd=np.array([0.02,0.3])           # dust yield per SN [Msun/SN]
colors=np.array([costum_colormap(0), costum_colormap(0.5), costum_colormap(1.)])

# if i only run one
#colors=np.array([costum_colormap(0.5), costum_colormap(0.5), costum_colormap(0.5)])


# Spin distribution (fixed) reused everywhere
len_sp_dis = 1000
spin_param_distr = np.random.lognormal(mean=np.log(10**-1.5677), sigma=0.5390, size=len_sp_dis)


# ============================================================
#  UV LUMINOSITY FUNCTION (uniform dust shell)
# ============================================================
# Strategy: instead of drawing 1000 random spins per halo, use
# K_SPINS deterministic quantiles of the lognormal spin distribution
# and blend median/mean transmission (W_BLEND knob).
# This gives smooth, reproducible results with far fewer evaluations.
# ============================================================

K_SPINS = 7        # stratified spin quantiles per halo mass (5-9 works well)
W_BLEND = 0.6      # blend weight: 0 = pure median T, 1 = pure mean T

# ---- shared setup ----
dndlogM = dn_dlogMh_GUREFT(logMh_array, redshift)  # HMF [Mpc^-3 dex^-1]
mu_ln, sig_ln = np.log(10**-1.5677), 0.5390         # lognormal spin params (Yung+23)

# deterministic stratified quantiles of the spin distribution
u = (np.arange(1, K_SPINS+1) - 0.5) / K_SPINS
z = norm.ppf(u)                          # standard-normal quantiles
spin_quant = np.exp(mu_ln + sig_ln*z)    # lognormal quantiles for spin, shape (K_SPINS,)

fig, ax = plt.subplots(figsize=(9,7))

for e, epsilon in enumerate(arr_e):
    # cache intrinsic + dust kernel once per (epsilon, Mh)
    Mh_grid    = 10**logMh_array
    L1500_grid = np.empty_like(logMh_array, float)
    Md1_grid   = np.empty_like(logMh_array, float)   # final Md at yd=1

    for j, Mh in enumerate(Mh_grid):
        SFH, logMst_build, age = Build_SFH_funct(Mh, redshift, tstep, epsilon)
        L1500_grid[j] = compute_L1500_steps(age, tstep, SFH, time_yr_L1500, L1500_SB99)[-1]
        Md1_grid[j]   = compute_Mdust_steps(age, tstep, SFH, time_yr, logSNr_yr, yd=1.0)[1][-1]

    MUV_intr = L1500_to_MUV_conv(L1500_grid)

    for yd in arr_yd:
        ls = ':' if yd == arr_yd[1] else '-.' #if yd == arr_yd[0] else ('--' if yd == arr_yd[1] else '-'))

        # scale dust kernel to this yd
        Md_grid = yd * Md1_grid

        # compute T for each stratified spin; blend mean/median => T_eff
        # loop over Mh (cheap: K_SPINS is small)
        T_eff = np.empty_like(Mh_grid, float)
        for j, Mh in enumerate(Mh_grid):
            # tau for K spins at this Mh
            tauK = tau_pred(kUV, Md_grid[j], Mh, spin_quant, redshift)   # expects vector spins
            TK   = T_sphere_mixed(tauK)                                   # shape (K_SPINS,)

            T_med  = np.median(TK)
            T_mean = np.mean(TK)
            T_eff[j] = (1.0 - W_BLEND)*T_med + W_BLEND*T_mean

        # attenuated mags using the blended transmission
        MUV_att = L1500_to_MUV_conv(T_eff * L1500_grid)

        # ---- Jacobian transformation: HMF -> UV LF ----
        # phi(MUV) = (dn/dlogMh) / |dMUV/dlogMh|
        # This is a change-of-variable from halo mass to UV magnitude.
        dMUV_dlogM_att  = np.gradient(MUV_att,  logMh_array)
        dMUV_dlogM_intr = np.gradient(MUV_intr, logMh_array)

        # avoid zeros in derivative
        eps_der = 1e-6
        dMUV_dlogM_att  = np.where(np.abs(dMUV_dlogM_att)  < eps_der, np.sign(dMUV_dlogM_att)*eps_der,  dMUV_dlogM_att)
        dMUV_dlogM_intr = np.where(np.abs(dMUV_dlogM_intr) < eps_der, np.sign(dMUV_dlogM_intr)*eps_der, dMUV_dlogM_intr)

        phi_att  = dndlogM / np.abs(dMUV_dlogM_att)
        phi_intr = dndlogM / np.abs(dMUV_dlogM_intr)

        # ---- monotonicity check + redistribution (both curves) ----
        def monotonic(x):
            return np.all(np.diff(x) < 0) or np.all(np.diff(x) > 0)

        if not monotonic(MUV_att):
            MUV_att,  phi_att  = redistribute_phi(MUV_att,  phi_att)
        if not monotonic(MUV_intr):
            MUV_intr, phi_intr = redistribute_phi(MUV_intr, phi_intr)

        # numpy safety
        phi_att  = np.where(np.isfinite(phi_att)  & (phi_att > 0),  phi_att,  np.nan)
        phi_intr = np.where(np.isfinite(phi_intr) & (phi_intr > 0), phi_intr, np.nan)
        
        ax.plot(MUV_att,  phi_att,  lw=3.0, ls=ls, color=colors[e], alpha=0.8)#,label=fr"ATT: $\epsilon$={epsilon}, $y_d$={yd}", alpha=0.9)#(K={K_SPINS}, w={W_BLEND})
            
    if redshift>=10:
        ax.plot(MUV_intr, phi_intr, lw=3., color=colors[e], alpha=0.4,zorder=-10000)#label=fr"Intrinsic, $\epsilon$={epsilon}"
        # in line text for clarity, specifying SF efficiency
        idx = int(0.6 * len(MUV_intr))
        x_eps  = MUV_intr[idx-1]
        y_eps  = 1.3*phi_intr[idx-1]
        ax.text(x_eps, y_eps,fr'$\epsilon_\star={epsilon*100:.0f}\%$',fontsize=17, color=colors[e],
                rotation=-50, rotation_mode='anchor', ha='left', va='bottom', alpha=0.5)


# ---- Big free-floating labels + short horizontal line segments ----
# y_d = 1e-3  (dash-dot)
x0, y0 = 0.78, 0.91#0.74, 0.81   # text anchor
line_length = 0.10    # fraction of axes width
ax.plot([x0 - line_length, x0 - 0.01], [y0, y0], transform=ax.transAxes, color='black', lw=3, ls='-.')
ax.text(x0, y0, r'$ y_d = 0.02\,\mathrm{M_\odot}$',transform=ax.transAxes,fontsize=16, color='black',ha='left', va='center')
# y_d = 0.3 (dashed)
x1, y1 =0.78, 0.85#0.74, 0.75
ax.plot([x1 - line_length, x1 - 0.01], [y1, y1],transform=ax.transAxes,color='black', lw=3, ls=':')
ax.text(x1, y1,r'$y_d = 0.3\,\mathrm{M_\odot}$',transform=ax.transAxes,fontsize=16, color='black', ha='left', va='center')
# specify redhsift
#ax.text(x1, 0.9*y1, fr'$z={redshift:.0f}$',transform=ax.transAxes,fontsize=20, color='black', ha='left', va='center')


# ---- ε_star colored horizontal labels (for z!=10)----
if redshift!=10: 
    x_eps0 = 0.7          # left margin (axes fraction)
    y_eps0 = 0.78          # top starting height
    dy     = 0.06          # vertical spacing
    line_length = 0.08     # horizontal line length (axes fraction)

    for e, epsilon in enumerate(arr_e):
        y_here = y_eps0 - e * dy

        # short solid line in same color as curves
        ax.plot([x_eps0, x_eps0 + line_length],[y_here, y_here],transform=ax.transAxes,color=colors[e],lw=4,ls='-')

        # text label
        ax.text(x_eps0 + line_length + 0.01,y_here,fr'$\epsilon_\star={epsilon*100:.0f}\%$',transform=ax.transAxes,fontsize=17,color=colors[e],ha='left',va='center')



# Data & axes
Plot_LF_Data(redshift, ax=ax)
ax.set_yscale('log')
ax.set_ylabel(r'$\phi(M_{UV})\ [\mathrm{Mpc}^{-3}\,\mathrm{mag}^{-1}]$')
ax.set_xlabel(r'$M_{UV}$')
if redshift==7:
    #plot vertican lines marking REBELS range
    ax.axvspan(-23, -21.,
                color='lightskyblue', alpha=0.1,
                edgecolor='none',zorder=-1000)
    ax.text(-21.6,0.8e-7,'REBELS', color='lightskyblue',alpha=0.35)
    ax.set_xlim(-19,-24.1)
    ax.set_ylim(0.5e-7, 1e-2)
else:
    ax.set_xlim(-18,-23.5)   # more negative MUV = brighter; bright end on RIGHT
    ax.set_ylim(0.5e-7, 1e-2)

ax.legend(fontsize=12, ncols=2, loc='upper right')

plt.subplots_adjust(left=0.12, bottom=0.087, right=0.983, top=0.958, wspace=0.2, hspace=0.2)

plt.show()






# ============================================================
#  IR LUMINOSITY FUNCTION (uniform dust shell)
# ============================================================
# L_IR = f_abs * L_1500 * nu_1500, where f_abs = 1 - T(tau_abs).
# Uses the ABSORPTION opacity (kUV_abs) not the extinction opacity,
# because scattered photons are not absorbed and don't heat dust.
# ============================================================
dndlogM_IR = dndlogM
Mh_grid = 10**logMh_array
nu_1500 = 3e10 / (1500e-8)  # Hz

plt.figure(figsize=(9,7))

for e, epsilon in enumerate(arr_e):

    print('\n')
    print('epsilon = ', epsilon)
    
    # cache per-Mh: L1500 and dust kernel at yd=1 to scale later
    L1500_grid = np.empty_like(logMh_array, float)
    Md1_grid   = np.empty_like(logMh_array, float)
    for j, Mh in enumerate(Mh_grid):
        SFH, logMst_build, age = Build_SFH_funct(Mh, redshift, tstep, epsilon)
        L1500_grid[j] = compute_L1500_steps(age, tstep, SFH, time_yr_L1500, L1500_SB99)[-1]
        Md1_grid[j]   = compute_Mdust_steps(age, tstep, SFH, time_yr, logSNr_yr, yd=1.0)[1][-1]

    for yd in arr_yd:
        
        #ls='-.'
        ls = ':' if yd == arr_yd[1] else '-.' #if yd == arr_yd[0] else ('--' if yd == arr_yd[1] else '-'))

        # scale dust kernel to this yd
        Md_grid = yd * Md1_grid

        # Effective absorbed fraction via stratified spins + blend
        A_eff = np.empty_like(Mh_grid, float)
        for j, Mh in enumerate(Mh_grid):
            # tau for K spins: need both extinction (for T) and absorption (for f_abs)
            tauK     = tau_pred(kUV,     Md_grid[j], Mh, spin_quant, redshift)
            tauK_abs = tau_pred(kUV_abs, Md_grid[j], Mh, spin_quant, redshift)

            TK   = T_sphere_mixed(tauK)        # transmission given spin
            AK   = 1.0 - T_sphere_mixed(tauK_abs)                    # absorbed fraction given spin
            A_eff[j] = (1.0 - W_BLEND) * np.median(AK) + W_BLEND * np.mean(AK)

        # IR luminosity from absorbed UV: L_IR = f_abs * L_UV * nu_UV
        LIR = A_eff * L1500_grid * nu_1500
        logLIR = np.log10(LIR / Lsun)

        # Jacobian: phi(LIR) = (dn/dlogMh) / |d logLIR / d logMh|
        dlogLIR_dlogM = np.gradient(logLIR, logMh_array)
        eps_der = 1e-6
        dlogLIR_dlogM = np.where(np.abs(dlogLIR_dlogM) < eps_der,
                                 np.sign(dlogLIR_dlogM)*eps_der, dlogLIR_dlogM)
        phi_IR = dndlogM_IR / np.abs(dlogLIR_dlogM)

        # monotonicity check + redistribution in logLIR space
        mono = np.all(np.diff(logLIR) > 0) or np.all(np.diff(logLIR) < 0)
        if not mono:
            logLIR, phi_IR = redistribute_phi(logLIR, phi_IR)

        plt.plot(logLIR, phi_IR, lw=3., ls=ls, color=colors[e])#,label=fr"$\epsilon$={epsilon}, $y_d$={yd}", alpha=0.9)

# Barrufet+23 (REBELS) data at z=7 
if redshift == 7:
    # REBELS z=7 points, table 1 Barrufet+23
    logLIR_dat = np.array([11.45, 11.75, 12.05])
    log_phi = np.array([-4.3, -4.6, -5.5])
    #max err btween detection and non detection, cause hihgly uncertain
    log_phi_uerr = np.array([0.2, 0.3, 0.4])
    log_phi_lerr = np.array([0.2, 0.3, 0.5])
    phi_data = 10**log_phi
    phi_uerr = 10**(log_phi + log_phi_uerr) - phi_data
    phi_lerr = phi_data - 10**(log_phi - log_phi_lerr)
    plt.errorbar(logLIR_dat, phi_data, yerr=[phi_lerr, phi_uerr], ls='none',
                   marker='s', ms=10, capsize=5, alpha=0.7, color='darkred',
                   label='REBELS\n Barrufet+23, $z=7$', mew=1.5, mec='black',
                   elinewidth=0.8)

plt.yscale('log')
plt.ylabel(r'$\phi(L_{\rm IR})\ [\mathrm{Mpc^{-3}\,dex^{-1}}]$')
plt.xlabel(r'$\log (L_{\rm IR}/L_{\odot})$')
plt.ylim(1.e-7, 1e-3)
plt.xlim(10.4, 13.4)

# specify redhsift
plt.text(12.9, 3e-4, fr'$z={redshift:.0f}$', fontsize=20, color='black')

plt.legend(fontsize=12)
plt.subplots_adjust(left=0.12, bottom=0.087, right=0.983, top=0.958, wspace=0.2, hspace=0.2)
plt.show()




