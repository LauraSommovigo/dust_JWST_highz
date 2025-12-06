from highz_gal_SAM import *
from general import name_and_save, increase_ticklabels, set_labels, do_minorticks, do_log, equal_axes, \
    set_colorbar_labels, set_ticklabels
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.gridspec as gridspec
from scipy.stats import norm  # for inverse CDF of normal (lognormal quantiles)
from scipy.signal import savgol_filter# for smoothing LF
from scipy.ndimage import median_filter
from matplotlib.lines import Line2D
from scipy import special

##---- extra functions needed:
def redistribute_phi(MUV_array, phi_array):
    MUV_array = np.array(MUV_array)
    phi_array = np.array(phi_array)

    #print("\n--- ENTERING redistribute_phi ---")
    #print("MUV input:", MUV_array)
    #print("phi input:", phi_array)

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



##---- colormap
costum_colormap = cm.inferno
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=10):
    new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval), cmap(np.linspace(minval, maxval, n)))
    return new_cmap
costum_colormap = truncate_colormap(costum_colormap, 0., 0.7)






### Choosing redshift, Mh range, dust model
redshift=14.#7.#10.
fb=cosmo.Ob(redshift)/cosmo.Om(redshift)
lumDistpc = cosmo.luminosity_distance(redshift)*(1e6)#pc
lumDistpc = lumDistpc.value
logMh_array=np.linspace(8,13,70)##NB: important to use the same mass range used for the fitting in Yung+23

## Plot HMF
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






### Fixing tstep for SFH, NB: deve essere breve abbastanza affinche' SN rate interpolato bene, 1-2 Myr vanno bene
tstep=1#in [Myr] units
metall=0.001/Zsun
### Loading Sn rate from SB99 for instantaneous SFR, and Metallicity=0.004 (1/5 Zsun), Salpeter IMF 1-100 Msun
logSNr_yr=np.loadtxt('/Users/lsommovigo/Desktop/Scripts/txt_files/snr_inst_Z001.txt',usecols=1)
time_yr=np.loadtxt('/Users/lsommovigo/Desktop/Scripts/txt_files/snr_inst_Z001.txt',usecols=0)
## logL1500, same assumptions, [erg/s/angstrom] units (need to multiply by angstrom)
L1500_SB99=np.loadtxt('/Users/lsommovigo/Desktop/Scripts/txt_files/L1500_inst_Z001.txt',usecols=1)
time_yr_L1500=np.loadtxt('/Users/lsommovigo/Desktop/Scripts/txt_files/L1500_inst_Z001.txt',usecols=0)

###Fixing the dust model
kUV=kUV_drn #kUV_drn  # choose dust opacity model here
kUV_abs=kUV_drn_abs
kv=kv_drn




#------------------------------------------------------------------------------------------
############# Loop in epsilon and yd: compute fract. of optically thick galaxies in the V-band
#------------------------------------------------------------------------------------------
arr_e=np.array([0.05, 0.10, 0.20])#[0.2])#0.05, 0.1, 0.6])
arr_yd=np.array([0.02,0.3])#, 0.3])
colors=np.array([costum_colormap(0), costum_colormap(0.5), costum_colormap(1.)])
# if i only run one
#colors=np.array([costum_colormap(0.5), costum_colormap(0.5), costum_colormap(0.5)])




'''
# ============================================
# Clumpy ISM LFs: UV (LOS percentiles) + IR (PDF-integrated)
# Two-panel (UV, IR) only when redshift == 7; otherwise UV-only
# Uses odd K_SPINS with an exact-median spin seed (u=0.5) and blends:
#   T_eff = (1-W_BLEND)*T_med_seed + W_BLEND*mean(TK)
# Montecarlo sampling nd shaded
# ============================================

# ---- knobs ----
K_SPINS  = 21     # odd; includes exact median spin seed (u=0.5). Try 13–21 for stability.
W_BLEND  = 0.6      # 0=median (seed), 1=mean (over seeds)
N_LOS    = 300      # LOS draws per seed for UV clumpy sampling
Mach     = 30      # clumpy ISM width
albedo   = 0.3807   # UV albedo
K_U      = 24       # Gauss–Legendre nodes for IR integral
SLOPE_FLOOR = 1e-3  # for spike masking (|dMUV/dlogMh| too small)

# ---- shared stuff ----
Mh_grid   = 10**logMh_array
dndlogM   = dn_dlogMh_GUREFT(logMh_array, redshift)  # [Mpc^-3 dex^-1]
mu_ln, sig_ln = np.log(10**-1.5677), 0.5390

# spin quantile seeds — include exact median at u=0.5
u_left  = (np.arange(1, (K_SPINS//2)+1) - 0.5) / K_SPINS
u_mid   = np.array([0.5])
u_right = 1.0 - u_left[::-1]
u = np.concatenate([u_left, u_mid, u_right])          # length = K_SPINS
z = norm.ppf(u)
spin_quant = np.exp(mu_ln + sig_ln*z)                 # (K_SPINS,)
mid_idx = K_SPINS // 2                                # index of exact-median seed

# clumpy width from Mach (for Σ_d)
def sigma_ln_from_Mach(Mach):
    R = compute_R(Mach)  # Thompson+16 R(M) with alpha=2.5
    return np.sqrt(np.log(1.0 + (R * Mach**2) / 4.0))
sigma_ln = sigma_ln_from_Mach(Mach)

# Gauss–Legendre nodes for IR integral (in u-space)
xu, wu  = np.polynomial.legendre.leggauss(K_U)
u_nodes = np.clip(0.5*(xu+1.0), 1e-12, 1-1e-12)
w_nodes = 0.5*wu
z_nodes = norm.ppf(u_nodes)

# helpers
def _nz(x):
    return np.where(np.abs(x) < 1e-6, np.sign(x)*1e-6, x)

def _lf_from_curve_sg(x_of_mh, window= 9, poly=2):
    # choose an odd window: 7/9/11 depending on len(logMh_array)
    win = max(7, min(window, len(logMh_array) - (1 - len(logMh_array)%2)))
    dx_dlogMh = savgol_filter(x_of_mh, win, poly, deriv=1,
                              delta=np.mean(np.diff(logMh_array)), mode='interp')
    dx_dlogMh = _nz(dx_dlogMh)
    return dndlogM / np.abs(dx_dlogMh)

def _ensure_mono(xc, phi):
    mono = np.all(np.diff(xc) < 0) or np.all(np.diff(xc) > 0)
    return (xc, phi) if mono else redistribute_phi(xc, phi)

def mask_by_slope(x_curve, phi_curve, slope_floor=SLOPE_FLOOR, hard_cap=10.0):
    """
    Remove/clip points where |d x / d logMh| is too small (spiky Jacobian).
    Keeps arrays aligned by masking both x and phi at the same indices.
    """
    dx = np.abs(_nz(np.gradient(x_curve, logMh_array)))
    bad = (dx < slope_floor) | ~np.isfinite(dx) | ~np.isfinite(phi_curve) | (phi_curve <= 0)
    # optional hard cap on phi (log spikes)
    phi_clip = np.minimum(phi_curve, hard_cap)
    return x_curve[~bad], phi_clip[~bad]

# ============================================
# Figure setup: IR panel only if z=7
# ============================================
if redshift == 7:
    gs = gridspec.GridSpec(1, 2, width_ratios=[1,1])
    ax_uv = plt.subplot(gs[0])
    ax_ir = plt.subplot(gs[1])
    show_IR = True
else:
    fig, ax_uv = plt.subplots(figsize=(6.8,6.4))
    ax_ir = None
    show_IR = False

_uniform_uv_plotted = False

for e, epsilon in enumerate(arr_e):
    color = colors[e]
    print('\n'); print('epsilon = ', epsilon)

    # ---------- intrinsic L1500 and Md kernel per Mh (ε-dependent) ----------
    L1500_grid = np.empty_like(logMh_array, float)
    Md1_grid   = np.empty_like(logMh_array, float)   # Md at yd=1
    for j, Mh in enumerate(Mh_grid):
        SFH, logMst_build, age = Build_SFH_funct(Mh, redshift, tstep, epsilon)
        L1500_grid[j] = compute_L1500_steps(age, tstep, SFH, time_yr_L1500, L1500_SB99)[-1]
        Md1_grid[j]   = compute_Mdust_steps(age, tstep, SFH, time_yr, logSNr_yr, yd=1.0)[1][-1]

    # Intrinsic UV LF (no dust):
    MUV_intr = L1500_to_MUV_conv(L1500_grid)
    phi_intr = _lf_from_curve_sg(MUV_intr)
    MUV_intr, phi_intr = _ensure_mono(MUV_intr, phi_intr)

    for yd in arr_yd:
        ls = ':' if yd == arr_yd[1] else '-.' #if yd == arr_yd[0] else ('--' if yd == arr_yd[1] else '-'))

        # scale dust kernel to this yd
        Md_grid = yd * Md1_grid

        USE_PURE_MEDIAN = True   # set True to match estimators

        T_eff = np.empty_like(Mh_grid, float)
        for j, Mh in enumerate(Mh_grid):
            tauK = tau_pred(kUV, Md_grid[j], Mh, spin_quant, redshift)  # τ for K seeds
            TK   = T_1500_sphere_im(tauK)
            if USE_PURE_MEDIAN:
                T_eff[j] = TK[mid_idx]                      # exact median seed only
            else:
                T_med_seed = TK[mid_idx]
                T_mean     = TK.mean()
                T_eff[j]   = (1.0 - W_BLEND)*T_med_seed + W_BLEND*T_mean

        MUV_uniform = L1500_to_MUV_conv(T_eff * L1500_grid)
        phi_uniform = _lf_from_curve_sg(MUV_uniform)
        MUV_uniform, phi_uniform = _ensure_mono(MUV_uniform, phi_uniform)

        # Plot uniform baseline once (commented out by default to avoid clutter)
        # if not _uniform_uv_plotted:
        #     ax_uv.plot(MUV_uniform, phi_uniform, lw=1.3, ls='--', color='dimgray',
        #                alpha=0.8, label='Uniform (median-seed blend)')
        #     _uniform_uv_plotted = True

        # ---------- CLUMPY UV: LOS sampling around the SAME seeds (Mach scatter) ----------
        # common LOS quantiles reduce noise across Mh
        u_LOS  = (np.arange(1, N_LOS+1) - 0.5) / N_LOS
        z_LOS  = norm.ppf(u_LOS)[None, :]   # (1, N_LOS) -> broadcast to (K_SPINS, N_LOS)

        MUV_p30 = np.empty_like(logMh_array, float)
        MUV_p50 = np.empty_like(logMh_array, float)
        MUV_p70 = np.empty_like(logMh_array, float)

        # IR (isotropic) integrated over clumpy PDF
        LIR_med = np.empty_like(logMh_array, float)

        for j, Mh in enumerate(Mh_grid):
            L1500 = L1500_grid[j]
            Md    = Md_grid[j]

            # Σ_d means from size/geometry seeds
            tauK = tau_pred(kUV, Md, Mh, spin_quant, redshift)
            mu_sigmas = tauK / kUV  # (K_SPINS,)

            # --- UV LOS draws: K_SPINS seeds × N_LOS per seed, using common z_LOS ---
            Sigmad_LOS = np.exp(np.log(mu_sigmas)[:, None] + sigma_ln * z_LOS)   # (K_SPINS, N_LOS)
            #print('tauV LOS sample -->', np.min(kUV * Sigmad_LOS), '--', np.max(kUV * Sigmad_LOS))
            T_uv       = T_1500_sphere_im(kUV * Sigmad_LOS)
            LUV        = L1500 * T_uv
            MUV_LOS    = L1500_to_MUV_conv(LUV.ravel())

            MUV_p30[j] = np.percentile(MUV_LOS, 16)
            MUV_p50[j] = np.percentile(MUV_LOS, 50)
            MUV_p70[j] = np.percentile(MUV_LOS, 84)

            # --- IR isotropic: integral over clumpy PDF around SAME seeds, then average seeds ---
            x_nodes = np.exp(np.log(mu_sigmas)[:, None] + sigma_ln*z_nodes[None, :])     # (K_SPINS, K_U)
            T_abs_nodes = T_1500_sphere_im(kUV_abs * x_nodes)
            A_nodes     = 1.0 - T_abs_nodes
            A_mean_seed = np.sum(w_nodes[None, :] * A_nodes, axis=1)  # (K_SPINS,)
            f_abs       = np.mean(A_mean_seed)
            LIR_med[j]  = (L1500 * f_abs * (3e10/(1500e-8))) / Lsun

        # --- light smoothing BEFORE Jacobian ---
        win = max(5, min(11, (len(logMh_array)//5)*2 + 1))  # odd window 7/9/11
        MUV_p50 = savgol_filter(MUV_p50, win, 2, mode='interp')

        # Only smooth p30/p70 when band is shown 
        show_band = (
            (np.isclose(yd, arr_yd[-1])  and np.isclose(epsilon, arr_e[1])) #or
            #(np.isclose(yd, 0.1) and np.isclose(epsilon, 0.5))
        )
        if show_band:
            MUV_p30 = savgol_filter(MUV_p30, win, 2, mode='interp')
            MUV_p70 = savgol_filter(MUV_p70, win, 2, mode='interp')

        # --- LFs from percentile curves (UV) ---
        phi_30 = _lf_from_curve_sg(MUV_p30)
        phi_50 = _lf_from_curve_sg(MUV_p50)
        phi_70 = _lf_from_curve_sg(MUV_p70)

        # avoid spikes due to near-zero slope in the discretized mapping
        try:
            MUV_p50, phi_50 = mask_by_slope(MUV_p50, phi_50, slope_floor=SLOPE_FLOOR, hard_cap=10.0)
            MUV_p30, phi_30 = mask_by_slope(MUV_p30, phi_30, slope_floor=SLOPE_FLOOR, hard_cap=10.0)
            MUV_p70, phi_70 = mask_by_slope(MUV_p70, phi_70, slope_floor=SLOPE_FLOOR, hard_cap=10.0)
        except Exception:
            # if mask_by_slope not defined in the environment, skip masking
            pass

        # enforce monotonicity after masking
        MUV_p30, phi_30 = _ensure_mono(MUV_p30, phi_30)
        MUV_p50, phi_50 = _ensure_mono(MUV_p50, phi_50)
        MUV_p70, phi_70 = _ensure_mono(MUV_p70, phi_70)

        # --- IR LF ---
        logLIR_med = np.log10(LIR_med)
        phi_IR     = _lf_from_curve_sg(logLIR_med)
        logLIR_med, phi_IR = _ensure_mono(logLIR_med, phi_IR)

        # --- UV plotting ---
        ax_uv.plot(MUV_p50, phi_50, lw=3., ls=ls, color=color)#,label=fr"ATT (Mach {Mach}): $\epsilon$={epsilon}, $y_d$={yd}", alpha=0.95)

        if show_band:
            ax_uv.plot(MUV_p30, phi_30, lw=1.5, ls=ls, color=color, alpha=0.35)
            ax_uv.plot(MUV_p70, phi_70, lw=1.5, ls=ls, color=color, alpha=0.35)
            verts_uv = np.concatenate([
                np.column_stack([MUV_p30,       phi_30]),
                np.column_stack([MUV_p70[::-1], phi_70[::-1]])
            ])
            ax_uv.add_patch(Polygon(verts_uv, closed=True, facecolor=color, alpha=0.18, edgecolor='none'))

        # --- IR plotting (only if z=7) ---
        if show_IR:
            ax_ir.plot(logLIR_med, phi_IR, lw=3., ls=ls, color=color, alpha=0.95)#,label=fr"$\epsilon$={epsilon}, $y_d$={yd}")

    # Intrinsic UV LF (no dust), once per epsilon
    ax_uv.plot(MUV_intr, phi_intr, lw=4., color=color, alpha=0.50)#,label=fr"Intrinsic, $\epsilon$={epsilon}")

    # in line text for clarity, specifying SF efficiency
    idx = int(0.63 * len(MUV_intr))
    x_eps  = MUV_intr[idx]
    y_eps  = 1.1*phi_intr[idx]
    ax_uv.text(x_eps, y_eps,fr'$\epsilon_\star={epsilon*100:.0f}\%$',fontsize=20, color=colors[e],
            rotation=-30, rotation_mode='anchor', ha='left', va='bottom', alpha=0.6)


# ---- Big free-floating labels + short horizontal line segments ----
# y_d = 1e-3  (dash-dot)
x0, y0 = 0.74, 0.915   # text anchor
line_length = 0.10    # fraction of axes width
ax_uv.plot([x0 - line_length, x0 - 0.01], [y0, y0], transform=ax_uv.transAxes, color='black', lw=3, ls='-.')
ax_uv.text(x0, y0, r'$y_d = 10^{-3}\,\mathrm{M_\odot}$',transform=ax_uv.transAxes,fontsize=18, color='black',ha='left', va='center')
# y_d = 0.3 (dashed)
x1, y1 = 0.74, 0.865
ax_uv.plot([x1 - line_length, x1 - 0.01], [y1, y1],transform=ax_uv.transAxes,color='black', lw=3, ls=':')
ax_uv.text(x1, y1,r'$y_d = 0.3\,\mathrm{M_\odot}$',transform=ax_uv.transAxes,fontsize=18, color='black', ha='left', va='center')




# --- cosmetics ---
Plot_LF_Data(redshift, ax=ax_uv)
ax_uv.set_yscale('log')
ax_uv.set_ylabel(r'$\phi(M_{UV})\ [\mathrm{Mpc}^{-3}\,\mathrm{mag}^{-1}]$')
ax_uv.set_xlabel(r'$M_{UV}$')
ax_uv.set_xlim(-17.5,-23.); 
if redshift != 7:
    ax_uv.set_ylim(0.5e-7, 1e-1)
else:
    ax_uv.set_ylim(0.5e-6, 1e-1)

if redshift==7:
    ax_uv.set_xlim(-18.4,-24.1)
    ax_uv.set_ylim(0.5e-7, 1e-1)
else:
    ax_uv.set_xlim(-17.5,-23.)   # more negative MUV = brighter; bright end on RIGHT
    ax_uv.set_ylim(0.5e-7, 1e-1)

ax_uv.legend(fontsize=12, ncols=2, loc='upper right')

if show_IR:
    # REBELS z=7 points
    logLIR = np.array([11.45, 11.75, 12.05])
    log_phi = np.array([-4.4, -4.6, -5.1])
    log_phi_uerr = np.array([0.2, 0.3, 0.2])
    log_phi_lerr = np.array([0.2, 0.3, 0.5])
    phi = 10**log_phi
    phi_uerr = 10**(log_phi + log_phi_uerr) - phi
    phi_lerr = phi - 10**(log_phi - log_phi_lerr)
    ax_ir.errorbar(logLIR, phi, yerr=[phi_lerr, phi_uerr], ls='none',
                   marker='s', ms=10, capsize=5, alpha=0.7, color='darkred',
                   label='Barrufet+23, $z=7$', mew=1.5, mec='black', elinewidth=0.8)

    ax_ir.set_yscale('log')
    ax_ir.set_ylabel(r'$\phi(\log_{10} L_{\mathrm{IR}})\ [\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}]$')
    ax_ir.set_xlabel(r'$\log_{10}(L_{\mathrm{IR}}/L_{\odot})$')
    ax_ir.set_ylim(1.e-6, 1e-3)
    ax_ir.set_xlim(10.4, 13.4)
    ax_ir.legend(fontsize=12, ncols=2, loc='upper right')
    plt.subplots_adjust(left=0.07, bottom=0.087, right=0.983, top=0.958, wspace=0.169, hspace=0.2)
else:
    plt.subplots_adjust(left=0.12, bottom=0.087, right=0.96, top=0.95)

plt.show()
'''




# ============================================
# Clumpy ISM LFs: UV + IR with Σ_d PDF and T_1500_sphere_im
# Two-panel (UV, IR) only when redshift == 7; otherwise UV-only
# NB: Lachlan Method
# ============================================

# ---- knobs ----
K_SPINS  = 21      # used only to estimate geometry-induced Σ_d scatter
Mach     = 30      # clumpy ISM width (lognormal σ for Σ_d from turbulence)
albedo   = 0.3807
K_U      = 24      # Gauss–Legendre nodes for Σ_d integral (IR)

# ---- shared stuff ----
Mh_grid   = 10**logMh_array
dndlogM   = dn_dlogMh_GUREFT(logMh_array, redshift)  # [Mpc^-3 dex^-1]

# spin distribution (for geometry scatter)
mu_ln_spin, sig_ln_spin = np.log(10**-1.5677), 0.5390
u_left  = (np.arange(1, (K_SPINS//2)+1) - 0.5) / K_SPINS
u_mid   = np.array([0.5])
u_right = 1.0 - u_left[::-1]
u = np.concatenate([u_left, u_mid, u_right])          # length = K_SPINS
z_spin = norm.ppf(u)
spin_quant = np.exp(mu_ln_spin + sig_ln_spin*z_spin)  # (K_SPINS,)
mid_idx = K_SPINS // 2

# clumpy width from Mach (for Σ_d lognormal)
def sigma_ln_from_Mach(Mach):
    R = compute_R(Mach)  # Thompson+16 R(M) with alpha=2.5
    return np.sqrt(np.log(1.0 + (R * Mach**2) / 4.0))
sigma_turb = sigma_ln_from_Mach(Mach)

# Gauss–Legendre nodes for Σ_d integral (in z-space)
xu, wu  = np.polynomial.legendre.leggauss(K_U)
u_nodes = np.clip(0.5*(xu+1.0), 1e-12, 1-1e-12)
w_nodes = 0.5*wu
z_nodes = norm.ppf(u_nodes)

# --- invert T_1500_sphere_im: precompute τ(T) lookup once ---
tau_max_lookup = 20.0
N_tau          = 4000
tau_grid = np.linspace(0.0, tau_max_lookup, N_tau)
T_grid  = T_1500_sphere_im(tau_grid)          # T(τ) for UV

# make T increasing for interpolation τ(T)
T_rev   = T_grid[::-1]
tau_rev = tau_grid[::-1]

def tau_of_T(T_target):
    """
    Invert T_1500_sphere_im(τ) ≈ T_target by interpolation.
    T_target in (0, 1]; vectorized.
    """
    T_target = np.asarray(T_target)
    # clip into the range of precomputed T
    T_clipped = np.clip(T_target, T_rev[0], T_rev[-1])
    return np.interp(T_clipped, T_rev, tau_rev)

# helpers
def _nz(x):
    return np.where(np.abs(x) < 1e-6, np.sign(x)*1e-6, x)

def _lf_from_curve_sg(x_of_mh, window=9, poly=2):
    # choose an odd window: 7/9/11 depending on len(logMh_array)
    win = max(7, min(window, len(logMh_array) - (1 - len(logMh_array)%2)))
    dx_dlogMh = savgol_filter(x_of_mh, win, poly, deriv=1,
                              delta=np.mean(np.diff(logMh_array)), mode='interp')
    dx_dlogMh = _nz(dx_dlogMh)
    return dndlogM / np.abs(dx_dlogMh)

def _ensure_mono(xc, phi):
    mono = np.all(np.diff(xc) < 0) or np.all(np.diff(xc) > 0)
    return (xc, phi) if mono else redistribute_phi(xc, phi)


# ============================================
# Figure setup: IR panel only if z=7
# ============================================
if redshift == 7:
    gs = gridspec.GridSpec(1, 2, width_ratios=[1,1])
    ax_uv = plt.subplot(gs[0])
    ax_ir = plt.subplot(gs[1])
    show_IR = True
else:
    fig, ax_uv = plt.subplots(figsize=(6.8,6.4))
    ax_ir = None
    show_IR = False

for e, epsilon in enumerate(arr_e):
    color = colors[e]
    print('\n'); print('epsilon = ', epsilon)

    # ---------- intrinsic L1500 and Md kernel per Mh (ε-dependent) ----------
    L1500_grid = np.empty_like(logMh_array, float)
    Md1_grid   = np.empty_like(logMh_array, float)   # Md at yd=1
    for j, Mh in enumerate(Mh_grid):
        SFH, logMst_build, age = Build_SFH_funct(Mh, redshift, tstep, epsilon)
        L1500_grid[j] = compute_L1500_steps(
            age, tstep, SFH, time_yr_L1500, L1500_SB99
        )[-1]
        Md1_grid[j]   = compute_Mdust_steps(
            age, tstep, SFH, time_yr, logSNr_yr, yd=1.0
        )[1][-1]

    # Intrinsic UV LF (no dust):
    MUV_intr_per_halo = L1500_to_MUV_conv(L1500_grid)   # per halo
    MUV_intr          = MUV_intr_per_halo.copy()
    phi_intr          = _lf_from_curve_sg(MUV_intr)
    MUV_intr, phi_intr = _ensure_mono(MUV_intr, phi_intr)

    # magnitude grid for analytic UV LF
    Mbright   = MUV_intr.min() - 1.0   # slightly brighter
    Mfaint    = MUV_intr.max() + 1.0   # slightly fainter
    MUV_grid  = np.linspace(Mbright, Mfaint, 250)
    nMh       = len(Mh_grid)
    nM        = len(MUV_grid)

    for yd in arr_yd:
        ls = ':' if yd == arr_yd[1] else '-.' 

        # scale dust kernel to this yd
        Md_grid = yd * Md1_grid

        # ---------- geometry-induced Σ_d from spins (per halo) ----------
        tauK_grid   = np.empty((nMh, K_SPINS))
        Sigma0      = np.empty(nMh)
        sigma_geom  = np.empty(nMh)
        for j, Mh in enumerate(Mh_grid):
            tauK = tau_pred(kUV, Md_grid[j], Mh, spin_quant, redshift)  # (K_SPINS,)
            tauK_grid[j, :] = tauK
            SigmaK = tauK / kUV

            # lognormal fit to Σ_d distribution from geometry
            lnSigmaK = np.log(SigmaK)
            Sigma0[j]     = np.exp(np.mean(lnSigmaK))      # geometric mean
            sigma_geom[j] = np.std(lnSigmaK)               # ln-space scatter

        # effective Σ_d scatter: geometry + turbulence (independent)
        sigma_eff = np.sqrt(sigma_geom**2 + sigma_turb**2)

        # ---------- uniform-screen UV LF (no clumpiness; median spin) ----------
        tau_med = tauK_grid[:, mid_idx]       # median over spins
        T_uniform = T_1500_sphere_im(tau_med)
        MUV_uniform = L1500_to_MUV_conv(T_uniform * L1500_grid)
        phi_uniform = _lf_from_curve_sg(MUV_uniform)
        MUV_uniform, phi_uniform = _ensure_mono(MUV_uniform, phi_uniform)

        # show uniform curve in faint dashed for reference
        #ax_uv.plot(MUV_uniform, phi_uniform, lw=0.5, ls=ls,
        #           color=color, alpha=0.4)

        # ---------- CLUMPY UV: use lognormal Σ_d + T_1500_sphere_im ----------
        # f_all[j, i] = fraction of galaxies at Mh_j brighter than MUV_grid[i]
        f_all  = np.zeros((nMh, nM))
        dfdM   = np.zeros_like(f_all)

        for j in range(nMh):
            M_int_j    = MUV_intr_per_halo[j]
            Sigma0_j   = Sigma0[j]
            sigma_j    = sigma_eff[j]

            if not np.isfinite(Sigma0_j) or Sigma0_j <= 0.0 or sigma_j <= 0.0:
                continue

            # required transmission for each M: T_star = Lobs/Lint
            T_star = 10**(-0.4 * (MUV_grid - M_int_j))   # can be >1 (unphysical)
            # physically, T_star > 1 means "need brightening" → f = 0
            valid_T = (T_star > 0.0) & (T_star <= 1.0)

            tau_max = np.zeros_like(MUV_grid)
            if np.any(valid_T):
                tau_max[valid_T] = tau_of_T(T_star[valid_T])

            Sigma_max = tau_max / kUV   # Σ_d,max(M, Mh_j)

            # lognormal CDF: f = P(Σ_d < Σ_max)
            f_Mj = np.zeros_like(MUV_grid)
            positive = Sigma_max > 0.0
            if np.any(positive):
                z_ln = (np.log(Sigma_max[positive]) - np.log(Sigma0_j)) / (
                    np.sqrt(2.0) * sigma_j
                )
                f_Mj[positive] = 0.5 * (1.0 + special.erf(z_ln))

            f_all[j, :] = f_Mj

            # derivative df/dM for this halo (central finite difference)
            df = np.zeros_like(MUV_grid)
            # central points
            df[1:-1] = (f_Mj[2:] - f_Mj[:-2]) / (MUV_grid[2:] - MUV_grid[:-2])
            # edges
            df[0]  = (f_Mj[1] - f_Mj[0]) / (MUV_grid[1] - MUV_grid[0])
            df[-1] = (f_Mj[-1] - f_Mj[-2]) / (MUV_grid[-1] - MUV_grid[-2])
            dfdM[j, :] = df

        # integrate over halo mass to get φ(M)
        phi_clumpy = np.zeros_like(MUV_grid)
        for i in range(nM):
            integrand = dndlogM * dfdM[:, i]
            phi_clumpy[i] = np.trapz(integrand, logMh_array)

        # smooth φ(M) in log-space to reduce numerical wiggles
        logphi = np.log10(np.clip(phi_clumpy, 1e-12, None))
        win_M  = max(11, min(41, (len(MUV_grid)//3)*2 + 1))  # odd window
        logphi_s = savgol_filter(logphi, win_M, 3, mode='interp')
        phi_clumpy_s = 10**logphi_s

        # enforce monotonic non-decreasing φ(M) as M increases (fainter → higher φ)
        phi_mon = phi_clumpy_s.copy()
        for i in range(1, len(phi_mon)):
            if phi_mon[i] < phi_mon[i-1]:
                phi_mon[i] = phi_mon[i-1]
        phi_clumpy_s = phi_mon

        # ---------- IR LF: same Σ_d lognormal + T_1500_sphere_im ----------
        LIR_med = np.empty_like(logMh_array, float)
        for j, Mh in enumerate(Mh_grid):
            L1500 = L1500_grid[j]
            mu_lnSigma = np.log(Sigma0[j])
            sigma_j    = sigma_eff[j]

            x_nodes = np.exp(mu_lnSigma + sigma_j * z_nodes)     # Σ_d quadrature nodes
            T_abs_nodes = T_1500_sphere_im(kUV_abs * x_nodes)
            A_nodes     = 1.0 - T_abs_nodes

            f_abs = np.sum(w_nodes * A_nodes)   # average absorbed fraction
            LIR_med[j]  = (L1500 * f_abs * (3e10/(1500e-8))) / Lsun

        logLIR_med = np.log10(LIR_med)
        phi_IR     = _lf_from_curve_sg(logLIR_med)
        logLIR_med, phi_IR = _ensure_mono(logLIR_med, phi_IR)

        # ---------- plotting ----------
        # clumpy UV LF
        ax_uv.plot(MUV_grid, phi_clumpy_s, lw=3., ls=ls, color=color)

        # IR LF (only if z=7)
        if show_IR:
            ax_ir.plot(logLIR_med, phi_IR, lw=3., ls=ls, color=color, alpha=0.95)

    # Intrinsic UV LF (no dust), once per ε
    ax_uv.plot(MUV_intr, phi_intr, lw=4., color=color, alpha=0.50)

    # in-line text for clarity, specifying SF efficiency
    idx = int(0.5 * len(MUV_intr))#0.63 z=7, 0.53 z=14
    x_eps  = MUV_intr[idx]
    y_eps  = 1.3*phi_intr[idx]#1.3 for z=14
    ax_uv.text(x_eps, y_eps, fr'$\epsilon_\star={epsilon*100:.0f}\%$',
               fontsize=20, color=colors[e],
               rotation=-50, rotation_mode='anchor',
               ha='left', va='bottom', alpha=0.6)#-30 for z=7 drn, -50 z14


# ---- Big free-floating labels + short horizontal line segments ----
x0, y0 = 0.74, 0.815   # text anchor, 0.74, 0.915 for z=7, for z=14 0.74, 0.815
line_length = 0.10
ax_uv.plot([x0 - line_length, x0 - 0.01], [y0, y0],
           transform=ax_uv.transAxes, color='black', lw=3, ls='-.')
ax_uv.text(x0, y0, r'$y_d = 0.02\,\mathrm{M_\odot}$',
           transform=ax_uv.transAxes, fontsize=18, color='black',
           ha='left', va='center')
x1, y1 = 0.74, 0.755 ## text anchor, 0.74, 0.865 for z=7, for z=14 0.74, 0.755
ax_uv.plot([x1 - line_length, x1 - 0.01], [y1, y1],
           transform=ax_uv.transAxes, color='black', lw=3, ls=':')
ax_uv.text(x1, y1, r'$y_d = 0.3\,\mathrm{M_\odot}$',
           transform=ax_uv.transAxes, fontsize=18, color='black',
           ha='left', va='center')

# --- cosmetics ---
Plot_LF_Data(redshift, ax=ax_uv)
ax_uv.set_yscale('log')
ax_uv.set_ylabel(r'$\phi(M_{UV})\ [\mathrm{Mpc}^{-3}\,\mathrm{mag}^{-1}]$')
ax_uv.set_xlabel(r'$M_{UV}$')

if redshift == 7:
    ax_uv.set_xlim(-19., -24.1)
    ax_uv.set_ylim(0.5e-7, 1e-2)

    #plot vertican lines marking REBELS range
    ax_uv.axvspan(-23, -21., color='lightskyblue', alpha=0.1, edgecolor='none',zorder=-1000)
    ax_uv.text(-21.6,0.8e-7,'REBELS', color='lightskyblue',alpha=0.35)

else:
    ax_uv.set_xlim(-18, -23.5)
    ax_uv.set_ylim(0.5e-7, 1e-2)

ax_uv.legend(fontsize=12, ncols=2, loc='upper right')

if show_IR:
    # REBELS z=7 points, table 1 Barrufet+23
    logLIR_dat = np.array([11.45, 11.75, 12.05])
    log_phi = np.array([-4.3, -4.6, -5.5])
    #max err btween detection and non detection, cause hihgly uncertain
    log_phi_uerr = np.array([0.2, 0.3, 0.4])
    log_phi_lerr = np.array([0.2, 0.3, 0.5])
    phi_data = 10**log_phi
    phi_uerr = 10**(log_phi + log_phi_uerr) - phi_data
    phi_lerr = phi_data - 10**(log_phi - log_phi_lerr)
    ax_ir.errorbar(logLIR_dat, phi_data, yerr=[phi_lerr, phi_uerr], ls='none',
                   marker='s', ms=10, capsize=5, alpha=0.7, color='darkred',
                   label='REBELS\n Barrufet+23, $z=7$', mew=1.5, mec='black',
                   elinewidth=0.8)

    ax_ir.set_yscale('log')
    ax_ir.set_ylabel(r'$\phi(\log_{10} L_{\mathrm{IR}})\ [\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}]$')
    ax_ir.set_xlabel(r'$\log_{10}(L_{\mathrm{IR}}/L_{\odot})$')
    ax_ir.set_ylim(1.e-7, 1e-3)
    ax_ir.set_xlim(10.4, 13.4)
    ax_ir.legend(fontsize=12, ncols=2, loc='upper right')
    plt.subplots_adjust(left=0.07, bottom=0.087, right=0.983, top=0.958,
                        wspace=0.169, hspace=0.2)
else:
    plt.subplots_adjust(left=0.12, bottom=0.087, right=0.96, top=0.95)

plt.show()
