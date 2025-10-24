from highz_gal_SAM import *
from general import name_and_save, increase_ticklabels, set_labels, do_minorticks, do_log, equal_axes, \
    set_colorbar_labels, set_ticklabels
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.gridspec as gridspec

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




##---- colormap
costum_colormap = cm.inferno
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=10):
    new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval), cmap(np.linspace(minval, maxval, n)))
    return new_cmap
costum_colormap = truncate_colormap(costum_colormap, 0., 0.7)






### Choosing redshift, Mh range, dust model
redshift=7.#10.
fb=cosmo.Ob(redshift)/cosmo.Om(redshift)
lumDistpc = cosmo.luminosity_distance(redshift)*(1e6)#pc
lumDistpc = lumDistpc.value
logMh_array=np.linspace(6,12,30)##NB: important to use the same mass rnge used for the fitting in Yung+23

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
kUV=kUV_drn


#####################################################
############# Loop in epsilon and yd: compute fract. of optically thick galaxies in the V-band
#####################################################
arr_e=np.array([0.05,0.1,0.5])
arr_yd=np.array([0.001,0.01, 0.1])
colors=np.array([costum_colormap(0), costum_colormap(0.5), costum_colormap(1.)])
# if i only run one
#colors=np.array([costum_colormap(0.5), costum_colormap(0.5), costum_colormap(0.5)])
'''
for epsilon in arr_e:
    print('\n')
    print('epsilon = ', epsilon)
    
    for yd in arr_yd:
        print('yd = ', yd)
        
        ### Computing Stellar properties at the given z for the chosen M_halo
        Mstar_array=halo_to_stellar_mass(10**logMh_array, fb, epsilon)# Stellar masses corresponding to Mh_array at z=10, in [Msun] units
        print(fb)
        print('\n Stellar masses log(Mstar/Msun) ->', np.log10(Mstar_array), '\n log(M_halo/Msun) ->', logMh_array)
        print('\n')
        
        
        ### Distr. of spin. param from GUREFT, Yung+23
        len_sp_dis=1000
        spin_param_distr=random.lognormal(mean=np.log(10**-1.5677),sigma=0.5390,size=len_sp_dis)
        
        ### SFH, Mstar and Md bild up over time for my Halo masses
        j=0
        fract_obs_Mh=[]
        tauv_fin_array=[]
        for j in range(len(logMh_array)):
            col=costum_colormap(float(j)/len(logMh_array))
            #print(j)
            SFH, logMst_build, age=Build_SFH_funct(10**logMh_array[j],redshift,tstep,epsilon)
            #print('At z='+str(redshift)+': log(Mh/Msun)=', round(logMh_array[j],3), ', log(Mstar/Msun)=', round(logMst_build[-1],3), ', SFR/Msun/yr=', round(SFH[-1],2))
            
            ### Md build upolo: assuming no dust is ejected
            N_SN_arr, Md_arr = compute_Mdust_steps(age, tstep, SFH, time_yr, logSNr_yr, yd)
            
            ##Computing tau_v
            tauV=tau_pred(kv,Md_arr[-1],10**logMh_array[j],spin_param_distr,redshift)
            
            ### computing final (i.e. at z=10) tauV in the given Mh_bin
            tauv_fin_array=np.append(tauv_fin_array,tauV)

            #computing number of obscured galaxies in the given Mh_bin
            fract_obs_Mh=np.append(fract_obs_Mh, sum(i > 0.1 for i in tauv_fin_array)/len(tauv_fin_array))
            #print('In this Mh bin, fract of obscured = ', sum(i > 1 for i in tauv_fin_array)/len(tauv_fin_array))
            

        ### Saving the fraction of obscured galaxies in different Mh bins, for the chosen epsilon_star, yd
        np.savetxt('/Users/lsommovigo/Desktop/Scripts/txt_files/JWST_dust_z10/Low_Obscured_fract'+'_1e3yd'+str(int(1e3*yd))+'_100eps'+str(int(100*epsilon))+'_z'+str(int(redshift))+'_HiroDust.txt',np.c_[logMh_array, np.log10(Mstar_array), fract_obs_Mh], header='yd='+str(yd)+', e_star='+str(epsilon)+ '\n log(Mhalo/Msun)      log(Mstar/Msun)      n(tau_v>1e-1)/n per unit Mpc^-3')

'''





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
        
        logMh_list = np.loadtxt('/Users/lsommovigo/Desktop/Scripts/txt_files/JWST_dust_z10/Low_Obscured_fract'+'_1e3yd'+str(int(1e3*yd))+'_100eps'+str(int(100*epsilon))+'_z'+str(int(redshift))+'.txt', usecols=0)
        
        logMst_list = np.loadtxt('/Users/lsommovigo/Desktop/Scripts/txt_files/JWST_dust_z10/Low_Obscured_fract'+'_1e3yd'+str(int(1e3*yd))+'_100eps'+str(int(100*epsilon))+'_z'+str(int(redshift))+'.txt', usecols=1)
        
        fobsc_list = np.loadtxt('/Users/lsommovigo/Desktop/Scripts/txt_files/JWST_dust_z10/Low_Obscured_fract'+'_1e3yd'+str(int(1e3*yd))+'_100eps'+str(int(100*epsilon))+'_z'+str(int(redshift))+'.txt', usecols=2)
        
        ## Plot obscured fraction as a function of Mstar
        plt.plot(logMst_list, fobsc_list, label='$y_d/\mathrm{M_{\odot}}$ = '+str(yd), color=colors[e], ls=ls, lw=2., alpha=0.6)
        #plt.scatter(logMst_list, fobsc_list, edgecolors=costum_colormap(epsilon), alpha=0.2, s=35, color='none')
        
        #### Fraction of obscured galaxies per unit volume (multiplying previous number for HMF and integrating over Mh range)
        print( 'num of obscured galaxies per unit volume ->', np.trapz(dn_dlogMh_GUREFT(logMh_list,redshift)*fobsc_list,logMh_list), ' -- for eps=', epsilon, ', yd=', yd)
        print( 'num of gal. per unit volume ->', (cosmo.comoving_volume(redshift+0.01)-cosmo.comoving_volume(redshift-0.01))*np.trapz(dn_dlogMh_GUREFT(logMh_list,redshift),logMh_list))
        print('ratio (%) -->', 100*np.trapz(dn_dlogMh_GUREFT(logMh_list,redshift)*fobsc_list,logMh_list) / np.trapz(dn_dlogMh_GUREFT(logMh_list,redshift),logMh_list))
        print()
        print()
        
        if e==0:
            plt.legend(fontsize=14, loc='lower left')

ax1.set_yscale('log')
ax1.set_xlabel('$\log (M_{\star}/M_{\odot})$', fontsize=18)
ax1.set_ylabel(' $n(\\tau_V>0.1) / n_{tot}$', fontsize=18)
##
ax1.plot(np.linspace(4, 12,10),0.5*np.ones(10),color='grey', alpha=0.2,lw=4.)
ax1.text(10.4,0.49,'50\%', color='grey', fontsize=18,alpha=0.9,fontweight='bold')
ax1.set_xlim(7,10.9)
ax1.set_ylim(0.9e-3,1.05)
#ax1.text(5.9, 0.0033, '$\epsilon_{\star}=$0.1', color=colors[1], fontsize=20, fontweight='bold')
ax1.text(7.1, 0.0033, '$\epsilon_{\star}=$0.05, ', color=colors[0], fontsize=20, fontweight='bold')
ax1.text(7.67, 0.0033, '   0.1,', color=colors[1], fontsize=20, fontweight='bold')
ax1.text(7.9, 0.0033, ' 0.5', color=colors[2], fontsize=20, fontweight='bold')

plt.tight_layout()
plt.show()




################################################
######### UV LF with SN dust correction (z=10)
################################################
### Computing luminosity and UV mag of galaxies as a function of time

fig, ax = plt.subplots(figsize=(9,7))

for e in range(len(arr_e)):
    epsilon = arr_e[e]
    print('epsilon=', epsilon)
    for yd in arr_yd:
        print('yd=', yd)

        if yd == 0.1:
            ls = ':'
        elif yd == 0.001:
            ls = '-.'
        elif yd == 0.01:
            ls = '--'

        Mstar_array = halo_to_stellar_mass(10**logMh_array, fb, epsilon)
        spin_param_distr = random.lognormal(mean=np.log(10**-1.5677), sigma=0.5390, size=1000)

        # Prepare arrays to store per-halo results
        MUV_fin_array = []
        MUVatt_fin_array = []
        MUVatt_fin_array_u = []
        MUVatt_fin_array_l = []

        # Loop over halo mass bins (original non-clumpy approach)
        for j in range(len(logMh_array)):
            SFH, logMst_build, age = Build_SFH_funct(10**logMh_array[j], redshift, tstep, epsilon)
            L1500_arr = compute_L1500_steps(age, tstep, SFH, time_yr_L1500, L1500_SB99)
            N_SN_arr, Md_arr = compute_Mdust_steps(age, tstep, SFH, time_yr, logSNr_yr, yd)

            # tauUV distribution from spin parameter draws
            tauUV = tau_pred(kUV, Md_arr[-1], 10**logMh_array[j], spin_param_distr, redshift)

            # Use median and 16/84 percentiles of tauUV to get central and +/- uncertainties
            tau_med = np.percentile(tauUV, 50)
            tau_l = np.percentile(tauUV, 16)
            tau_u = np.percentile(tauUV, 84)

            # Transmission using the spherical geometry routine
            T_med = T_1500_sphere(tau_med)
            T_l = T_1500_sphere(tau_l)
            T_u = T_1500_sphere(tau_u)

            # Attenuated UV luminosities and convert to MUV
            LUV_att_med = T_med * L1500_arr[-1]
            LUV_att_l = T_l * L1500_arr[-1]
            LUV_att_u = T_u * L1500_arr[-1]

            MUV_noatt = L1500_to_MUV_conv(L1500_arr[-1])
            MUV_att_med = L1500_to_MUV_conv(LUV_att_med)
            MUV_att_l = L1500_to_MUV_conv(LUV_att_l)
            MUV_att_u = L1500_to_MUV_conv(LUV_att_u)

            MUV_fin_array.append(MUV_noatt)
            MUVatt_fin_array.append(MUV_att_med)
            MUVatt_fin_array_u.append(MUV_att_u)
            MUVatt_fin_array_l.append(MUV_att_l)

        # Convert to arrays
        MUV_fin_array = np.array(MUV_fin_array)
        MUVatt_fin_array = np.array(MUVatt_fin_array)
        MUVatt_fin_array_u = np.array(MUVatt_fin_array_u)
        MUVatt_fin_array_l = np.array(MUVatt_fin_array_l)

        # Jacobian / LF conversion
        dndlogM = dn_dlogMh_GUREFT(logMh_array, redshift)

        dMUVdlogM_att = np.gradient(MUVatt_fin_array, logMh_array)
        dMUVdlogM_att_u = np.gradient(MUVatt_fin_array_u, logMh_array)
        dMUVdlogM_att_l = np.gradient(MUVatt_fin_array_l, logMh_array)

        phi_MUV_att = dndlogM / np.abs(dMUVdlogM_att)
        phi_MUV_att_u = dndlogM / np.abs(dMUVdlogM_att_u)
        phi_MUV_att_l = dndlogM / np.abs(dMUVdlogM_att_l)

        # Check monotonicity and fix if necessary (keeps original behaviour)
        is_monotonic = np.all(np.diff(MUVatt_fin_array) < 0) or np.all(np.diff(MUVatt_fin_array) > 0)
        if not is_monotonic:
            MUVatt_fin_array, phi_MUV_att = redistribute_phi(MUVatt_fin_array, phi_MUV_att)
            MUVatt_fin_array_l, phi_MUV_att_l = redistribute_phi(MUVatt_fin_array_l, phi_MUV_att_l)
            MUVatt_fin_array_u, phi_MUV_att_u = redistribute_phi(MUVatt_fin_array_u, phi_MUV_att_u)

        # Plot attenuated LF
        ax.plot(MUVatt_fin_array, phi_MUV_att, lw=2., label=f"ATT: $\\epsilon$={epsilon}, $y_d$={yd}", ls=ls, color=colors[e], alpha=0.8)
        #ax.plot(MUVatt_fin_array_l, phi_MUV_att_l, lw=2, ls=ls, color=colors[e], alpha=0.08)
        #ax.plot(MUVatt_fin_array_u, phi_MUV_att_u, lw=2, ls=ls, color=colors[e], alpha=0.08)

        # Intrinsic LF (always monotonic)
        dMUVdlogM_noatt = np.gradient(MUV_fin_array, logMh_array)
        phi_MUV_noatt = dndlogM / np.abs(dMUVdlogM_noatt)
        ax.plot(MUV_fin_array, phi_MUV_noatt, lw=1.5, label=f"Intrinsic, $\\epsilon$={epsilon}", color=colors[e], alpha=0.6)

# Add measurements / data
Plot_LF_Data(redshift, ax=ax)

ax.set_yscale('log')
ax.set_ylabel(r'$\phi(M_{UV})\ [\mathrm{Mpc}^{-3}\,\mathrm{mag}^{-1}]$')
ax.set_xlabel(r'$M_{UV}$')
ax.set_xlim(-18, -24)
ax.set_ylim(2e-8, 1.5)
ax.legend(fontsize=11, ncols=2, loc='upper right')
plt.tight_layout()
plt.show()




################################################
######### IR LF with SN dust correction (z=7)
################################################

plt.figure(figsize=(9,7))

for e in range(len(arr_e)):
    epsilon = arr_e[e]
    print('epsilon=', epsilon)
    
    for yd in arr_yd:
        print('yd=',yd)
        
        if yd == 0.1:
            ls = ':'
        elif yd == 0.001:
            ls = '-.'
        elif yd == 0.01:
            ls = '--'

        Mstar_array = halo_to_stellar_mass(10**logMh_array, fb, epsilon)
        spin_param_distr = random.lognormal(mean=np.log(10**-1.5677), sigma=0.5390, size=1000)

        LIR_array_abs = []
        LIR_array_abs_l = []
        LIR_array_abs_u = []

        for j in range(len(logMh_array)):
            SFH, logMst_build, age = Build_SFH_funct(10**logMh_array[j], redshift, tstep, epsilon)
            L1500_arr = compute_L1500_steps(age, tstep, SFH, time_yr_L1500, L1500_SB99)
            N_SN_arr, Md_arr = compute_Mdust_steps(age, tstep, SFH, time_yr, logSNr_yr, yd)

            # here we account for albedo, to make sure only absorbed UV light is re-emitted in the IR
            tauUV = tau_pred(kUV * (1 - 0.3807), Md_arr[-1], 10**logMh_array[j], spin_param_distr, redshift)

            nu_1500 = 3e10 / (1500e-8)  # Hz
            LIR_abs = (1 - np.exp(-tauUV)) * L1500_arr[-1] * nu_1500

            LIR_array_abs.append(np.median(LIR_abs))
            LIR_array_abs_l.append(np.percentile(LIR_abs, 16))
            LIR_array_abs_u.append(np.percentile(LIR_abs, 84))

        LIR_array_abs = np.array(LIR_array_abs) / Lsun
        LIR_array_abs_l = np.array(LIR_array_abs_l) / Lsun
        LIR_array_abs_u = np.array(LIR_array_abs_u) / Lsun

        dndlogM = dn_dlogMh_GUREFT(logMh_array, redshift)
        dLIR = np.gradient(np.log10(LIR_array_abs), logMh_array)
        dLIR_l = np.gradient(np.log10(LIR_array_abs_l), logMh_array)
        dLIR_u = np.gradient(np.log10(LIR_array_abs_u), logMh_array)

        phi_IR = dndlogM / np.abs(dLIR)
        phi_IR_l = dndlogM / np.abs(dLIR_l)
        phi_IR_u = dndlogM / np.abs(dLIR_u)

        plt.plot(np.log10(LIR_array_abs), phi_IR, lw=2.5,
                 label=f"$\\epsilon$={epsilon}, $y_d$={yd}",
                 ls=ls, color=colors[e])
        
        if epsilon==0.1:
            verts = np.concatenate([
                np.column_stack([np.log10(LIR_array_abs_l), phi_IR_l]),
                np.column_stack([np.log10(LIR_array_abs_u)[::-1], phi_IR_u[::-1]])
            ])
            poly = Polygon(verts, closed=True, facecolor=colors[e], alpha=0.08, edgecolor=None)
            plt.gca().add_patch(poly)

if redshift==7:
    ### Barrufet+23 (REBELS)
    logLIR = np.array([11.45, 11.75, 12.05])
    log_phi = np.array([-4.4, -4.6, -5.1])
    log_phi_uerr = np.array([0.2, 0.3, 0.2])
    log_phi_lerr = np.array([0.2, 0.3, 0.5])
    
    phi = 10**log_phi
    phi_uerr = 10**(log_phi + log_phi_uerr) - phi
    phi_lerr = phi - 10**(log_phi - log_phi_lerr)

    plt.errorbar(logLIR, phi,
                 yerr=[phi_lerr, phi_uerr],
                 ls='none', marker='s', ms=10, capsize=5,
                 alpha=0.7, color='darkred',
                 label='Barrufet+23, $z=7$',
                 mew=1.5, mec='black', elinewidth=0.8)

plt.yscale('log')
plt.ylabel(r'$\phi(L_{IR})\ [\mathrm{Mpc^{-3} dex^{-1}}]$')
plt.xlabel(r'$\log (L_{IR}/L_{\odot})$')
plt.ylim(1e-6, 1e-3)
plt.xlim(10.5, 13.5)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()
        




###-------------------------------------------------------------------------------------
###-------- We now change the gas, thus dust surface density
###-------- and see what percentile needs to contribute to the UV and IR
###-------------------------------------------------------------------------------------

# --- helpers for clumpy-ISM math (place once above the main loop) ---
def lognormal_pdf_x(x, mu_ln, sigma_ln):
    return np.exp(-(np.log(x) - mu_ln)**2 / (2.0 * sigma_ln**2)) / (x * sigma_ln * np.sqrt(2.0*np.pi))

def sigma_ln_from_Mach(Mach):
    R = compute_R(Mach)  # your Thompson+16 R(M) function (alpha=2.5)
    return np.sqrt(np.log(1.0 + (R * Mach**2) / 4.0))

#--- fixed parameters ---
albedo = 0.3807
Mach = 300                 # <-- set your fiducial Mach here
N_samples = 1000 


#fig = plt.figure(figsize=(14, 7))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
ax_uv = plt.subplot(gs[0])
ax_ir = plt.subplot(gs[1])

for e in range(len(arr_e)):
    epsilon = arr_e[e]
    print('epsilon=', epsilon)
    for yd in arr_yd:
        print('yd=', yd)
        if yd == 0.1:
            ls = ':'
        elif yd == 0.001:
            ls = '-.'
        elif yd == 0.01:
            ls = '--'

        Mstar_array = halo_to_stellar_mass(10**logMh_array, fb, epsilon)
        spin_param_distr = random.lognormal(mean=np.log(10**-1.5677), sigma=0.5390, size=1000)

        # ============ UV LUMINOSITY FUNCTION ============ #
        MUV_fin_array = []
        MUVatt_fin_array = []
        MUVatt_fin_array_u = []
        MUVatt_fin_array_l = []
        
        #with no change in sigma_d between uv and ir
        MUVatt_baseline_array = []
        LIR_abs_baseline_array = []
        phi_MUV_baseline = []
        phi_IR_baseline = []

        LIR_array_abs, LIR_array_abs_l, LIR_array_abs_u = [], [], []
        
        
        for j in range(len(logMh_array)):
            SFH, logMst_build, age = Build_SFH_funct(10**logMh_array[j], redshift, tstep, epsilon)
            L1500_arr = compute_L1500_steps(age, tstep, SFH, time_yr_L1500, L1500_SB99)
            N_SN_arr, Md_arr = compute_Mdust_steps(age, tstep, SFH, time_yr, logSNr_yr, yd)

            # size/spin-driven variation of tau (array over many sizes)
            tauUV = tau_pred(kUV, Md_arr[-1], 10**logMh_array[j], spin_param_distr, redshift)

            # mean surface density for the uniform-ISM benchmark
            Mean_Sigmad = np.percentile(tauUV, 50) / kUV

            # ------------------------------------------------------------------
            # CLUMPY ISM treatment
            #   - UV: Monte Carlo LOS draws   (keep distribution)
            #   - IR: continuous-PDF integral (single value)
            # ------------------------------------------------------------------

            # (i) UV — Monte Carlo LOS for each size realization
            mu_array = tauUV / kUV                          # Σ_d "means" from size variation
            sigma_ln = sigma_ln_from_Mach(Mach)             # same width for all sightlines at given Mach

            # draw Σ_d samples for each element of mu_array and flatten
            Sigmad_draws = []
            for mu_sigma in mu_array:                       # loop over sizes/spins
                Sigmad_draws.append(
                    draw_sigma_distribution(mu_sigma=mu_sigma, Mach=Mach, nsamples=N_samples)
                )
            Sigmad_draws = np.asarray(Sigmad_draws).reshape(-1)   # (N_tau * N_samples)

            # UV transmissions and luminosities per LOS
            T_uv_draws = T_1500_sphere(kUV * Sigmad_draws)        # geometry-consistent transmission
            LUV_att_arr = L1500_arr[-1] * T_uv_draws               # per-LOS UV
            MUV_att_dist = L1500_to_MUV_conv(np.atleast_1d(LUV_att_arr))

            # store median and 16–84% for UV (for your error ribbon / error bars)
            MUV_att = np.median(MUV_att_dist)
            MUVatt_u = np.percentile(MUV_att_dist, 84)
            MUVatt_l = np.percentile(MUV_att_dist, 16)

            # also store intrinsic MUV (no attenuation)
            MUV_noatt = L1500_to_MUV_conv(L1500_arr[-1])

            MUV_fin_array.append(MUV_noatt)
            MUVatt_fin_array.append(MUV_att)
            MUVatt_fin_array_u.append(MUVatt_u)
            MUVatt_fin_array_l.append(MUVatt_l)

            # (ii) IR — single integrated value per halo
            # For each size realization with mean mu_sigma, integrate over the continuous lognormal Σ_d PDF
            # and then average across the size distribution.
            nu_1500 = 3e10 / (1500e-8)                       # Hz
            absorbed_means = []
            for mu_sigma in mu_array:
                mu_ln = np.log(mu_sigma)
                # grid covering ~all probability mass
                x_min = np.exp(mu_ln - 6.0*sigma_ln)
                x_max = np.exp(mu_ln + 6.0*sigma_ln)
                x = np.logspace(np.log10(x_min), np.log10(x_max), 1200)

                p_x = lognormal_pdf_x(x, mu_ln, sigma_ln)
                tau_abs = kUV * x * (1.0 - albedo)
                T_abs = T_1500_sphere(tau_abs)

                # E[absorbed fraction] for this size realization
                absorbed_means.append(np.trapz((1.0 - T_abs) * p_x, x))

            # average over size distribution (mixture in mu)
            f_abs = np.mean(absorbed_means)
            LIR_abs_val = L1500_arr[-1] * f_abs * nu_1500    # single number per halo

            LIR_array_abs.append(LIR_abs_val / Lsun)
            # keep lower/upper equal to the central value (single integrated value as requested)
            LIR_array_abs_l.append(LIR_abs_val / Lsun)
            LIR_array_abs_u.append(LIR_abs_val / Lsun)

            # ------------------------------------------------------------------
            # UNIFORM ISM benchmark (unchanged)
            # ------------------------------------------------------------------
            tauUV_mean = kUV * Mean_Sigmad
            T1500_sf_mean = T_1500_sphere(tauUV_mean)

            LUV_att_baseline = T1500_sf_mean * L1500_arr[-1]
            MUV_att_baseline = L1500_to_MUV_conv(LUV_att_baseline)
            LIR_abs_baseline = (1.0 - T1500_sf_mean) * L1500_arr[-1] * nu_1500

            MUVatt_baseline_array.append(MUV_att_baseline)
            LIR_abs_baseline_array.append(LIR_abs_baseline / Lsun)


        MUV_fin_array = np.array(MUV_fin_array)
        MUVatt_fin_array = np.array(MUVatt_fin_array)
        MUVatt_fin_array_u = np.array(MUVatt_fin_array_u)
        MUVatt_fin_array_l = np.array(MUVatt_fin_array_l)

        LIR_array_abs = np.array(LIR_array_abs) 
        LIR_array_abs_l = np.array(LIR_array_abs_l) 
        LIR_array_abs_u = np.array(LIR_array_abs_u) 
        #Uniform ISM comparison
        MUVatt_baseline_array = np.array(MUVatt_baseline_array)
        LIR_abs_baseline_array = np.array(LIR_abs_baseline_array)


        dlogMh = np.gradient(logMh_array)
        dndlogM = dn_dlogMh_GUREFT(logMh_array, redshift)

        # UV LF
        dMUVdlogM_att = np.gradient(MUVatt_fin_array, logMh_array)
        dMUVdlogM_att_u = np.gradient(MUVatt_fin_array_u, logMh_array)
        dMUVdlogM_att_l = np.gradient(MUVatt_fin_array_l, logMh_array)
        dMUVdlogM_att_base = np.gradient(MUVatt_baseline_array, logMh_array)

        phi_MUV_att = dndlogM / np.abs(dMUVdlogM_att)
        phi_MUV_att_u = dndlogM / np.abs(dMUVdlogM_att_u)
        phi_MUV_att_l = dndlogM / np.abs(dMUVdlogM_att_l)
        #Uniform ISM comparison
        phi_MUV_baseline = dndlogM / np.abs(dMUVdlogM_att_base)

        is_monotonic = np.all(np.diff(MUVatt_fin_array) < 0) or np.all(np.diff(MUVatt_fin_array) > 0)
        is_monotonic_base = np.all(np.diff(MUVatt_baseline_array) < 0) or np.all(np.diff(MUVatt_baseline_array) > 0)


        if not is_monotonic:
            MUVatt_fin_array, phi_MUV_att = redistribute_phi(MUVatt_fin_array, phi_MUV_att)
            MUVatt_fin_array_l, phi_MUV_att_l = redistribute_phi(MUVatt_fin_array_l, phi_MUV_att_l)
            MUVatt_fin_array_u, phi_MUV_att_u = redistribute_phi(MUVatt_fin_array_u, phi_MUV_att_u)
        
        if not is_monotonic_base:
            #Uniform ISM comparison
            MUVatt_baseline_array, phi_MUV_baseline = redistribute_phi(MUVatt_baseline_array, phi_MUV_baseline)

        ax_uv.plot(MUVatt_fin_array, phi_MUV_att, lw=2., label=f"ATT: $\\epsilon$={epsilon}, $y_d$={yd}", ls=ls, color=colors[e], alpha=0.8)
        
        ax_uv.plot(MUVatt_baseline_array, phi_MUV_baseline, lw=1., ls=ls,
           color=colors[e], alpha=0.4, label=f"Uniform ISM, $\\epsilon$={epsilon}, $y_d$={yd}")
           
        ax_uv.plot(MUVatt_fin_array_l, phi_MUV_att_l, lw=2, ls=ls, color=colors[e], alpha=0.08)
        ax_uv.plot(MUVatt_fin_array_u, phi_MUV_att_u, lw=2, ls=ls, color=colors[e], alpha=0.08)
        verts_uv = np.concatenate([np.column_stack([MUVatt_fin_array_l, phi_MUV_att_l]), np.column_stack([MUVatt_fin_array_u[::-1], phi_MUV_att_u[::-1]])])
        poly_uv = Polygon(verts_uv, closed=True, facecolor=colors[e], alpha=0.18, edgecolor=None)
        ax_uv.add_patch(poly_uv)

        dMUVdlogM_noatt = np.gradient(MUV_fin_array, logMh_array)
        phi_MUV_noatt = dndlogM / np.abs(dMUVdlogM_noatt)
        if yd==0.001:
            ax_uv.plot(MUV_fin_array, phi_MUV_noatt, lw=1.5, label=f"Intrinsic, $\\epsilon$={epsilon}", color=colors[e], alpha=0.6)

        # IR LF
        dLIR = np.gradient(np.log10(LIR_array_abs), logMh_array)
        dLIR_l = np.gradient(np.log10(LIR_array_abs_l), logMh_array)
        dLIR_u = np.gradient(np.log10(LIR_array_abs_u), logMh_array)
        dLIR_base = np.gradient(np.log10(LIR_abs_baseline_array), logMh_array)

        phi_IR = dndlogM / np.abs(dLIR)
        phi_IR_l = dndlogM / np.abs(dLIR_l)
        phi_IR_u = dndlogM / np.abs(dLIR_u)
        
        phi_IR_baseline = dndlogM / np.abs(dLIR_base)

        ax_ir.plot(np.log10(LIR_array_abs), phi_IR, lw=2.5, label=f"$\\epsilon$={epsilon}, $y_d$={yd}", ls=ls, color=colors[e])
        print('phi_IR (with clumpy ISM)-->', phi_IR)

        ax_ir.plot(np.log10(LIR_abs_baseline_array), phi_IR_baseline, lw=1, ls=ls,
           color=colors[e], alpha=0.4, label=f"Uniform ISM, $\\epsilon$={epsilon}, $y_d$={yd}")

        if True:
            verts_ir = np.concatenate([np.column_stack([np.log10(LIR_array_abs_l), phi_IR_l]), np.column_stack([np.log10(LIR_array_abs_u)[::-1], phi_IR_u[::-1]])])
            poly_ir = Polygon(verts_ir, closed=True, facecolor=colors[e], alpha=0.18, edgecolor=None)
            ax_ir.add_patch(poly_ir)

# Add Data UV panel
Plot_LF_Data(redshift, ax=ax_uv)
ax_uv.set_yscale('log')
ax_uv.set_ylabel(r'$\phi(M_{UV})\ [\mathrm{Mpc}^{-3}\,\mathrm{mag}^{-1}]$')
ax_uv.set_xlabel(r'$M_{UV}$')
ax_uv.set_xlim(-18, -24)
ax_uv.set_ylim(2e-8, 1.5)
ax_uv.legend(fontsize=11, ncols=2, loc='upper right')

# Add Data IR panel (only for z=7)
if redshift == 7:
    logLIR = np.array([11.45, 11.75, 12.05])
    log_phi = np.array([-4.4, -4.6, -5.1])
    log_phi_uerr = np.array([0.2, 0.3, 0.2])
    log_phi_lerr = np.array([0.2, 0.3, 0.5])
    phi = 10**log_phi
    phi_uerr = 10**(log_phi + log_phi_uerr) - phi
    phi_lerr = phi - 10**(log_phi - log_phi_lerr)
    ax_ir.errorbar(logLIR, phi, yerr=[phi_lerr, phi_uerr], ls='none', marker='s', ms=10, capsize=5, alpha=0.7, color='darkred', label='Barrufet+23, $z=7$', mew=1.5, mec='black', elinewidth=0.8)

ax_ir.set_yscale('log')
ax_ir.set_ylabel(r'$\phi(L_{IR})\ [\mathrm{Mpc^{-3} dex^{-1}}]$')
ax_ir.set_xlabel(r'$\log (L_{IR}/L_{\odot})$')
ax_ir.set_ylim(1e-6, 1e-3)
ax_ir.set_xlim(10.5, 13.5)
ax_ir.legend(fontsize=11)

plt.tight_layout()
plt.show()


