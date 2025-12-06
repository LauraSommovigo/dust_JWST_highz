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
redshift=10.#7.#10.
fb=cosmo.Ob(redshift)/cosmo.Om(redshift)
lumDistpc = cosmo.luminosity_distance(redshift)*(1e6)#pc
lumDistpc = lumDistpc.value
logMh_array=np.linspace(8,13,20)##NB: important to use the same mass range used for the fitting in Yung+23

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

# ---------------- FAST: cache + compute f_obsc(Mh) with tau_V > 0.1 ----------------
# Assumes: logMh_array, arr_e, arr_yd, fb, redshift, tstep, time_yr, time_yr_L1500, 
#          L1500_SB99, logSNr_yr, halo_fredsto_stellar_mass, Build_SFH_funct, 
#          compute_Mdust_steps, tau_pred, kv, kUV_drn, kUV_hir are already defined.

# Spin distribution (fixed) reused everywhere
len_sp_dis = 1000
spin_param_distr = np.random.lognormal(mean=np.log(10**-1.5677), sigma=0.5390, size=len_sp_dis)
''' Loop over epsilon and yd to compute fraction of obscured galaxies (tau_V > 0.1) 
for epsilon in arr_e:
    print('\n')
    print('epsilon = ', epsilon)

    # ---------- cache per-(epsilon, Mh): SFH/age + dust kernel Md_fin at yd=1 ----------
    Mh_grid = 10**logMh_array
    cached_age = [None]*len(logMh_array)
    cached_SFH = [None]*len(logMh_array)
    Md1_fin    = np.empty(len(logMh_array), dtype=float)   # final Md for yd=1 (kernel)
    for j, Mh in enumerate(Mh_grid):
        SFH, logMst_build, age = Build_SFH_funct(Mh, redshift, tstep, epsilon)
        cached_SFH[j] = SFH
        cached_age[j] = age
        # run dust once with yd=1.0, then scale later
        _, Md_arr_1 = compute_Mdust_steps(age, tstep, SFH, time_yr, logSNr_yr, yd=1.0)
        Md1_fin[j] = Md_arr_1[-1]

    # stellar mass only depends on epsilon (not on yd) -> compute once
    Mstar_array = halo_to_stellar_mass(Mh_grid, fb, epsilon)
    logMstar = np.log10(Mstar_array)

    for yd in arr_yd:
        print('yd = ', yd)

        # scale dust kernel to the requested yield
        Md_fin = yd * Md1_fin   # final Md at this yd for each mass bin

        # per-Mh obscured fraction (tau_V > 0.1) using the spin ensemble
        # NOTE: we DO NOT accumulate across masses; this is bin-by-bin.
        f_obsc = np.empty(len(logMh_array), dtype=float)

        for j, Mh in enumerate(Mh_grid):
            tauV = tau_pred(kv, Md_fin[j], Mh, spin_param_distr, redshift)  # array over spins
            f_obsc[j] = np.mean(tauV > 0.1)

        # ------- SAVE -------
        outpath_base = '/Users/lsommovigo/Desktop/Scripts/txt_files/JWST_dust_z10/Low_Obscured_fract'
        suffix = f"_1e3yd{int(1e3*yd)}_100eps{int(100*epsilon)}_z{int(redshift)}"
        if kv == kUV_hir:
            outfile = outpath_base + suffix + '_HiroDust.txt'
        else:
            outfile = outpath_base + suffix + '.txt'

        header = (
            f"yd={yd}, e_star={epsilon}\n"
            "log(Mhalo/Msun)      log(Mstar/Msun)      n(tau_V>1e-1)/n per unit Mpc^-3"
        )
        np.savetxt(outfile, np.c_[logMh_array, logMstar, f_obsc], header=header)
        print("Saved:", outfile)


#### Plotting fraction of obscured galaxies as a function of epsilon_star and yd
fig = plt.figure()
ax1 = fig.add_subplot(111)
for e in range(len(arr_e)):
    epsilon=arr_e[e]
    
    for y in range(len(arr_yd)):
        yd=arr_yd[y]
        
        if yd==0.1:
            ls=':'
        
        if yd==0.001:
            ls='-.'
        
        if yd==0.01:
            ls='--'
        
        ## Loading data
        # --- Read the correct obscured-fraction file depending on dust opacity model ---
        base = "/Users/lsommovigo/Desktop/Scripts/txt_files/JWST_dust_z10/Low_Obscured_fract"
        suffix = f"_1e3yd{int(1e3*yd)}_100eps{int(100*epsilon)}_z{int(redshift)}"

        if kUV == kUV_hir:
            filename = base + suffix + "_HiroDust.txt"
        elif kUV == kUV_drn:
            filename = base + suffix + ".txt"
        else:
            raise ValueError("Unknown kUV model: must match kUV_hir or kUV_drn")

        # --- Load columns ---
        logMh_list, logMst_list, fobsc_list = np.loadtxt(filename, usecols=(0, 1, 2), unpack=True)


        ## Plot obscured (tau_V>0.1) fraction as a function of Mstar
        plt.plot(logMst_list, fobsc_list, label='$y_d/\mathrm{M_{\odot}}$ = '+str(yd), color=colors[e], ls=ls, lw=2., alpha=0.6)
        #plt.scatter(logMst_list, fobsc_list, edgecolors=costum_colormap(epsilon), alpha=0.2, s=35, color='none')

        #### Fraction of obscured (tau_V>0.1) galaxies per unit volume (multiplying previous number for HMF and integrating over Mh range)
        #print( 'num of obscured galaxies per unit volume ->', np.trapz(dn_dlogMh_GUREFT(logMh_list,redshift)*fobsc_list,logMh_list), ' -- for eps=', epsilon, ', yd=', yd)
        #print( 'num of gal. per unit volume ->', (cosmo.comoving_volume(redshift+0.01)-cosmo.comoving_volume(redshift-0.01))*np.trapz(dn_dlogMh_GUREFT(logMh_list,redshift),logMh_list))
        print('ratio of obscured (tau_V>0.1) galaxies (%) -->', 100*np.trapz(dn_dlogMh_GUREFT(logMh_list,redshift)*fobsc_list,logMh_list) / np.trapz(dn_dlogMh_GUREFT(logMh_list,redshift),logMh_list))
        print()
        print()
        
        if e==0:
            plt.legend(fontsize=12, loc='lower left')

ax1.set_yscale('log')
ax1.set_xlabel('$\log (M_{\star}/M_{\odot})$', fontsize=18)
ax1.set_ylabel(' $n(\\tau_V>0.1) / n_{tot}$', fontsize=18)
##
ax1.plot(np.linspace(4, 12,10),0.5*np.ones(10),color='grey', alpha=0.2,lw=4.)
ax1.text(10.4,0.49,'50\%', color='grey', fontsize=18,alpha=0.9,fontweight='bold')
ax1.set_xlim(7,10.9)
ax1.set_ylim(0.9e-2,1.05)
#ax1.text(5.9, 0.0033, '$\epsilon_{\star}=$0.1', color=colors[1], fontsize=20, fontweight='bold')
ax1.text(7.1, 0.023, '$\epsilon_{\star}=$0.05, ', color=colors[0], fontsize=20, fontweight='bold')
ax1.text(7.67, 0.023, '   0.1,', color=colors[1], fontsize=20, fontweight='bold')
ax1.text(7.9, 0.023, ' 0.5', color=colors[2], fontsize=20, fontweight='bold')

plt.tight_layout()
plt.show()

'''





#-------------------------------------------------------
######### UV LF with SN dust correction (z=redshift)
#-------------------------------------------------------

# ---- knobs for the "in-between" behavior ----
K_SPINS = 7        # small number of stratified spins per mass (5–9 works well)
W_BLEND = 0.6      # 0=median, 1=mean; pick ~0.5–0.7 for "in-between"

# ---- setup shared stuff ----
dndlogM = dn_dlogMh_GUREFT(logMh_array, redshift)  # [Mpc^-3 dex^-1]
mu_ln, sig_ln = np.log(10**-1.5677), 0.5390

# fixed stratified quantiles in (0,1)
u = (np.arange(1, K_SPINS+1) - 0.5) / K_SPINS
z = norm.ppf(u)                          # standard-normal quantiles
spin_quant = np.exp(mu_ln + sig_ln*z)    # lognormal quantiles for lambda, shape (K_SPINS,)

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
            TK   = T_1500_sphere_im(tauK)                                   # shape (K_SPINS,)

            T_med  = np.median(TK)
            T_mean = np.mean(TK)
            T_eff[j] = (1.0 - W_BLEND)*T_med + W_BLEND*T_mean

        # attenuated mags using the blended transmission
        MUV_att = L1500_to_MUV_conv(T_eff * L1500_grid)

        # ---- Jacobian -> LFs ----
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
        
        ax.plot(MUV_att,  phi_att,  lw=3.0, ls=ls, color=colors[e])#,label=fr"ATT: $\epsilon$={epsilon}, $y_d$={yd}", alpha=0.9)#(K={K_SPINS}, w={W_BLEND})
            
    
    ax.plot(MUV_intr, phi_intr, lw=4., color=colors[e], alpha=0.45)#label=fr"Intrinsic, $\epsilon$={epsilon}"
    
    # in line text for clarity, specifying SF efficiency
    idx = int(0.58 * len(MUV_intr))
    x_eps  = MUV_intr[idx]
    y_eps  = 1.3*phi_intr[idx]
    ax.text(x_eps, y_eps,fr'$\epsilon_\star={epsilon*100:.0f}\%$',fontsize=20, color=colors[e],
            rotation=-50, rotation_mode='anchor', ha='left', va='bottom', alpha=0.6)


# ---- Big free-floating labels + short horizontal line segments ----
# y_d = 1e-3  (dash-dot)
x0, y0 = 0.74, 0.815   # text anchor
line_length = 0.10    # fraction of axes width
ax.plot([x0 - line_length, x0 - 0.01], [y0, y0], transform=ax.transAxes, color='black', lw=3, ls='-.')
ax.text(x0, y0, r'$y_d = 0.02\,\mathrm{M_\odot}$',transform=ax.transAxes,fontsize=18, color='black',ha='left', va='center')
# y_d = 0.3 (dashed)
x1, y1 = 0.74, 0.755
ax.plot([x1 - line_length, x1 - 0.01], [y1, y1],transform=ax.transAxes,color='black', lw=3, ls=':')
ax.text(x1, y1,r'$y_d = 0.3\,\mathrm{M_\odot}$',transform=ax.transAxes,fontsize=18, color='black', ha='left', va='center')



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





'''
#-------------------------------------------------------
######### IR LF (z=redshift)
#-------------------------------------------------------

# knobs for the in-between estimator (define only if missing)
if 'K_SPINS' not in locals(): K_SPINS = 7    # 5–9 works nicely
if 'W_BLEND' not in locals(): W_BLEND = 0.6  # 0=median, 1=mean
if 'albedo'  not in locals(): albedo  = 0.3807

# spin quantiles (deterministic stratified sampling) — define only if missing
if 'spin_quant' not in locals():
    from scipy.stats import norm
    if 'mu_ln' not in locals() or 'sig_ln' not in locals():
        mu_ln, sig_ln = np.log(10**-1.5677), 0.5390
    u = (np.arange(1, K_SPINS+1) - 0.5) / K_SPINS
    z = norm.ppf(u)
    spin_quant = np.exp(mu_ln + sig_ln * z)  # shape (K_SPINS,)

# HMF (use existing if available)
dndlogM_IR = dndlogM if 'dndlogM' in locals() else dn_dlogMh_GUREFT(logMh_array, redshift)
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

        # effective absorbed fraction via stratified spins + mean/median blend
        A_eff = np.empty_like(Mh_grid, float)
        for j, Mh in enumerate(Mh_grid):
            Md_j = float(Md_grid[j])        # scalar
            Mh_j = float(Mh)                # scalar
            
            # tau_ext (scatt + abs) for K spins at this Mh
            tauK = np.array(
                [tau_pred(kUV, Md_j, Mh_j, float(lam), redshift)
                 for lam in spin_quant],
                dtype=float
            )  # shape (K_SPINS,)
            
            # tau_abs (abs only) for K spins at this Mh
            tauK_abs = np.array(
                [tau_pred(kUV_abs, Md_j, Mh_j, float(lam), redshift)
                 for lam in spin_quant],
                dtype=float
            )  # shape (K_SPINS,)

            TK   = T_1500_sphere_im(tauK)        # transmission given spin
            AK   = 1.0 - T_1500_sphere_im(tauK_abs)                    # absorbed fraction given spin
            A_eff[j] = (1.0 - W_BLEND) * np.median(AK) + W_BLEND * np.mean(AK)

        # IR luminosity from absorbed UV 
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
plt.legend(fontsize=12)
plt.subplots_adjust(left=0.12, bottom=0.087, right=0.983, top=0.958, wspace=0.2, hspace=0.2)
plt.show()




'''