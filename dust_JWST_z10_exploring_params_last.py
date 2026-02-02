from statistics import median
from highz_gal_SAM import *
from matplotlib.colors import ListedColormap
from general import name_and_save, increase_ticklabels, set_labels, do_minorticks, do_log, equal_axes, \
    set_colorbar_labels, set_ticklabels
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



###########################
##### MY FUNCTIONS
###########################

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# numerical density cgs, from Sommovigo+20, for  auniform spherical cloud
def num_dens_cgs(p=0.0, sigma=1e-10):
        mu=1.22
        n=p/(mp*mu*sigma**2)#gcm^-3
        return n

## colormap
custom_colormap_base = cm.coolwarm
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=10):
    new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval), cmap(np.linspace(minval, maxval, n)))
    return new_cmap
custom_colormap = truncate_colormap(custom_colormap_base, 0.1, 1.)

### Loading Sn rate from SB99 for instantaneous SFR, and Metallicity=0.001 (1/10 Zsun), Salpeter IMF 1-100 Msun
metall=0.001/Zsun
logSNr_yr=np.loadtxt('/Users/lsommovigo/Desktop/Scripts/txt_files/snr_inst_Z001.txt',usecols=1)
time_yr=np.loadtxt('/Users/lsommovigo/Desktop/Scripts/txt_files/snr_inst_Z001.txt',usecols=0)

## Loading Ionizing photon rate  from SB99 for instantaneous SFR, and Metallicity=0.001 (1/10 Zsun), Salpeter IMF 1-100 Msun
log_dotNion=np.loadtxt('/Users/lsommovigo/Desktop/Scripts/txt_files/Ni_inst_Z001.txt',usecols=1)
time_yr_Nion=np.loadtxt('/Users/lsommovigo/Desktop/Scripts/txt_files/Ni_inst_Z001.txt',usecols=0)

## logL1500, same assumptions as before [erg/s/angstrom] units
L1500_SB99=np.loadtxt('/Users/lsommovigo/Desktop/Scripts/txt_files/L1500_inst_Z001.txt',usecols=1)
time_yr_L1500=np.loadtxt('/Users/lsommovigo/Desktop/Scripts/txt_files/L1500_inst_Z001.txt',usecols=0)

L1500_SB99_cont=np.loadtxt('/Users/lsommovigo/Desktop/Scripts/txt_files/L1500_cont_Z001.txt',usecols=1)
time_yr_L1500_cont=np.loadtxt('/Users/lsommovigo/Desktop/Scripts/txt_files/L1500_cont_Z001.txt',usecols=0)
###########################
###########################



### Choosing redshift, and Mh range
redshift=7#7#10.
Mh_array=np.logspace(8,12,8)##NB: important to use the same mass rnge used for the fitting in Yung+23, 8 in plot version
fb=cosmo.Ob(redshift)/cosmo.Om(redshift)
print('\n\nfb at z=0 -->', cosmo.Ob(0)/cosmo.Om(0), ', vs z=10 -->', fb)
print('rvir at z=10 for Mh=10^10.6/(0.1*fb) -->', r_vir(0 , (10**10.6)/(0.1*fb) ))
 
### Fixing SF efficiency
epsilon=0.1#0.1#1#0.5

###Fixing the dust yiled per each Sn event, based on Bocchio+16 discussion on SN 1987A
yd=1e-1#NB: /sim 2 orders of magnitude below rebels inferred value

###Choosing the attenuation curve (_drn or_hir)
kUV=kUV_drn
kv=kv_drn

###Fixing spin parameter
spin_param=0.027

# Fixing tstep for SFH, NB: deve essere breve abbastanza affinche' SN rate interpolato bene, 1-2 Myr vanno bene
tstep=1#in [Myr] units



### Computing Stellar properties at the given z
Mstar_array=halo_to_stellar_mass(Mh_array, fb, epsilon)# Stellar masses corresponding to Mh_array at z=10, in [Msun] units
Mstar_array_05=halo_to_stellar_mass(Mh_array, fb, 0.5)# Stellar masses corresponding to Mh_array at z=10, in [Msun] units
Mstar_array_01=halo_to_stellar_mass(Mh_array, fb, 0.1)# Stellar masses corresponding to Mh_array at z=10, in [Msun] units
print('\n#########################################')
print('NB: [ ASSUMPTIONS: (epsilon, yd, spin param)='+str([epsilon,yd,spin_param])+' ]')
print('Stellar masses log(Mstar/Msun) ->', np.log10(Mstar_array))
print('\n Halo masses log(M_halo/Msun) ->', np.log10(Mh_array))
print('Yung+23, Mh/(dMh/dt) at z=7, [Myr] -->', 1e-6*Mh_array/dMhdt_GUREFT(Mh_array,7))
print('Yung+23, Mh/(dMh/dt) at z=10, [Myr] -->', 1e-6*Mh_array/dMhdt_GUREFT(Mh_array,10))
print('Age of the Universe z=10, z=7, difference --> ', cosmo.age(10), cosmo.age(7), cosmo.age(7)-cosmo.age(10))
print('timescale of a wind at 200km/s to spazzare la galssia di raggio di cui sopra [Myr] -->', 2.*spin_param*r_vir(redshift,Mh_array)*kpc_to_cm/(200*1e5)/(Myr))
print('#########################################')
print('\n')


#------------------------------------------------------------------------------------------
### Plot instantaeous SFR for given M_halo and M_star at z=10: GUREFT vs. Sommovigo+22
#------------------------------------------------------------------------------------------
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(Mh_array,epsilon*fb*dMhdt_GUREFT(Mh_array,redshift), label='Yung+23, z='+str(redshift),s=50.,marker='o',color=custom_colormap(np.linspace(0,1,len(Mh_array))))
ax1.scatter(Mh_array,epsilon*fb*dMhdt_num_sims(Mh_array,redshift), label='Sommovigo+22 (refs. therein), z='+str(redshift),s=107.,marker='*',color=custom_colormap(np.linspace(0,1,len(Mh_array))))
ax1.plot(Mh_array,epsilon*fb*dMhdt_GUREFT(Mh_array,redshift), color='grey', alpha=0.5,zorder=-10)
ax1.plot(Mh_array,epsilon*fb*dMhdt_num_sims(Mh_array,redshift),color='grey', alpha=0.5,zorder=-10)

ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_xlim(5e7, 1e15)
plt.legend(fontsize=12)
ax1.set_ylabel('$\mathrm{inst.\ SFR}= \epsilon_{\star}\ f_b\ \mathrm{dM_h/dt}$ $\mathrm{[M_{\odot} yr^{-1}}]$')
ax1.set_xlabel('$M_h [M_{\odot}]$')

### Secondary x_axis, M_star
ax2 = ax1.twiny()
ax2.set_xlim(5e7*fb*epsilon,1e15*fb*epsilon)
ax2.set_xscale('log')
ax2.set_xlabel('$M_{\star} [M_{\odot}]$')
plt.tight_layout()
plt.show()



### Building SFH and dust build up from SN for each halo mass
### Distr. of spin. param from GUREFT, Yung+23
spin_param_distr=random.lognormal(mean=-3.6,sigma=0.5677,size=1000)#Yung+23 says mu=-1.5677 (?)
#print('16-84th percentile spin parameter distribution', np.percentile(spin_param_distr,16), np.percentile(spin_param_distr,84))
#plt.hist(spin_param_distr,bins=20)
#plt.show()
#print(np.mean(spin_param_distr))

#------------------------------------------------------------------------------------------
### Plot SFH, Mstar and Md build up over time for my Halo masses
#------------------------------------------------------------------------------------------
fig, (ax_sfr, ax_mstar, ax_md, ax_ratio) = plt.subplots(4, 1, figsize=(8, 17), sharex=True)  # width, height in inches

plt.subplots_adjust(
    left=0.14,
    bottom=0.045,
    right=0.99,
    top=0.993,
    wspace=0.2,
    hspace=0.0
)

j = 0

## transition to opt. thick at: age_ott
age_ott = []
for j in range(len(Mh_array)):
    col = custom_colormap(float(j) / len(Mh_array))
    print('\n Considero alone # ---> ', j)
    
    SFH, logMst_build, age = Build_SFH_funct(Mh_array[j], redshift, tstep, epsilon)
    print('At z='+str(redshift)+': log(Mh/Msun)=', round(np.log10(Mh_array[j]),3),
          ', log(Mstar/Msun)=', round(logMst_build[-1],3),
          ', SFR/Msun/yr=', round(SFH[-1],2),
          'sSFR/Gyr^-1=', round(1e9*SFH[-1]/(10.**logMst_build[-1]),2))
    
    # --- compute derived quantities ---
    L1500_arr = compute_L1500_steps(age, tstep, SFH, time_yr_L1500, L1500_SB99)
    N_SN_arr, Md_arr = compute_Mdust_steps(age, tstep, SFH, time_yr, logSNr_yr, yd)
    dotNion_arr = compute_dotNion_steps(age, tstep, SFH, time_yr_Nion, log_dotNion)
    
    # --- SFR panel: line + sparse star markers every 10 timesteps ---
    ax_sfr.plot(
        age, np.log10(SFH),
        color=col, lw=1.2, alpha=0.85, zorder=2
    )

    # sample every 10 steps, to make less crowded
    step = 10
    ax_sfr.scatter(
        age[::step], np.log10(SFH)[::step],
        marker='*', s=80, color=col,
        edgecolor='k', linewidth=0.1,
        alpha=0.4, zorder=3
    )


    if j == 0:
        ax_sfr.set_ylabel(r'$\log\ \mathrm{SFR}\ [M_\odot\,\mathrm{yr}^{-1}]$')
    
    # --- Mstar panel ---
    ax_mstar.plot(age, logMst_build, ls='--', lw=1.5, dashes=[5, 5], color=col,
                  label=r'$\log (M_\star/M_\odot)$' if j == 0 else None)
    if j == 0:
        ax_mstar.set_ylabel(r'$\log (M_\star/M_\odot)$')
    
    # --- Md panel ---
    logMd = np.log10(Md_arr)
    ax_md.plot(age, logMd, lw=2., alpha=0.8, color=col,
               label=r'$\log (M_{\rm d}/M_\odot)$' if j == 0 else None)
    if j == 0:
        ax_md.set_ylabel(r'$\log (M_{\rm d}/M_\odot)$')
    
    # --- Md/Mstar panel ---
    ratio = logMd - logMst_build
    ax_ratio.plot(age, ratio, lw=2., ls='-.', alpha=0.8, color=col,
                  label=r'$\log (M_{\rm d}/M_\star)$' if j == 0 else None)
    
    # --- tau_V and optically thick transition (unchanged logic) ---
    tauV     = tau_pred(kv, Md_arr, Mh_array[j], spin_param_distr.mean(), redshift)
    tauV_max = tau_pred(kv, Md_arr, Mh_array[j], np.percentile(spin_param_distr,16), redshift)
    tauV_min = tau_pred(kv, Md_arr, Mh_array[j], np.percentile(spin_param_distr,84), redshift)
    
    if tauV[-1] >= 0.9:
        print('\n time after which given halo is optically thick in the V-band -->',
              tauV[find_nearest(tauV,1.)], age[find_nearest(tauV,1.)])
        
        reds_thick = z_at_value(
            cosmo.age,
            (cosmo.age(redshift) -
             ((age[-1] - age[find_nearest(tauV,1.)]) * u.Myr).to(u.Gyr)),
            method='bounded'
        )
        print('corresp to z=', reds_thick)
    else:
        print('Stays optically thin up to z=red')
        age_ott = np.append(age_ott, -999)
    
    # --- Save outputs as before ---
    np.savetxt(
        '/Users/lsommovigo/Desktop/Scripts/txt_files/JWST_dust_z10/Properties_10logMH'
        + str(int(10*np.log10(Mh_array[j]))) + '_1e3yd' + str(int(1e3*yd))
        + '_100eps' + str(int(100*epsilon)) + '_1e3lam' + str(int(1e3*spin_param))
        + '_NoEjection_red' + str(int(redshift)) + '.txt',
        np.c_[age, SFH, logMst_build, logMd, tauV, tauV_min, tauV_max, L1500_arr],
        header='log(Mhalo/Msun)='+str(round(np.log10(Mh_array[j]),3))
               + '\n yd='+str(yd)+', e_star='+str(epsilon)
               + ', spin='+str(spin_param)
               + '\n t[Myr]        SFR[Msun/yr]        log(Mstar/Msun)        '
               + 'log(Md/Msun)        tau_v(MW dust)        '
               + 'tau_v,16th perc.(MW dust)        tau_v,84th perc.(MW dust)        '
               + 'L_1500[erg/s]'
    )

# --- Global formatting / ranges ---
for ax in (ax_sfr, ax_mstar, ax_md, ax_ratio):
    ax.tick_params(axis='both', which='major', labelsize=18, pad=6)

# y-limits so everything fits reasonably; tweak if needed
ax_sfr.set_ylim(-2, 2.5)         # adjust based on your SFR range
ax_mstar.set_ylim(3, 10.7)
ax_md.set_ylim(-0.7, 8.2)
ax_ratio.set_ylim(-4.7, -2)

# shared x-limit
ax_ratio.set_xlim(-1, 1.2 * age[-1])

# bottom panel: labels, REBELS band, z text, 3.5 Myr marker
ax_ratio.set_xlabel(r'$\mathrm{Galaxy\ Age\ [Myr]}$')
ax_ratio.set_ylabel(r'$\log (M_{\rm d}/M_{\star})$')
ax_ratio.text(0.98*age[-1], -4.5, '$z=$' + str(int(redshift)))

# 3.5 Myr vertical line (draw on all panels)
for ax in (ax_mstar, ax_md, ax_ratio):
    ax.axvline(3.5, color='grey', lw=2., alpha=0.2)
ax_ratio.text(0., -4.2, '3.5 Myr',
              rotation=90, fontsize=14, color='grey', alpha=0.5)

# REBELS Md/Mstar region on bottom panel only
ax_ratio.axhspan(
    np.percentile(logMd_REB_npSFH - logMstar_REB_npSFH, 16),
    np.percentile(logMd_REB_npSFH - logMstar_REB_npSFH, 84),
    color='grey', alpha=0.1
)
ax_ratio.text(
    100, np.log10(0.0015),
    r'$< M_{\rm d}/M_{\star}>$ REBELS (z=7)',
    fontsize=16, color='grey', alpha=0.6
)

# --- small legend with quantity labels (use only first-halo labels) ---
#ax_mstar.legend(fontsize=10, loc='upper left')
#ax_md.legend(fontsize=10, loc='upper left')
#ax_ratio.legend(fontsize=10, loc='upper left')

####### colorbar Mh ########
bar = np.zeros((1, len(Mh_array)))
bar[0][:] = np.linspace(np.log10(Mh_array).min(),
                        np.log10(Mh_array).max(),
                        len(Mh_array))

cbaxis = inset_axes(ax_ratio, width="35%", height="6%",
                    loc='center right', borderpad=1.2)
ext = (np.log10(Mh_array).min(), np.log10(Mh_array).max(), 0.0, 0.8)
cbaxis.imshow(
    bar, cmap=custom_colormap, extent=ext, aspect='auto',
    vmin=np.log10(Mh_array).min(), vmax=np.log10(Mh_array).max(), alpha=0.8
)
cbaxis.set_xlabel(r'$\log (M_{h}/M_{\odot})$')
cbaxis.tick_params(axis='both', which='major', pad=4)
cbaxis.set_yticks([])

fig.savefig(
    "/Users/lsommovigo/Desktop/multipanel_SFH_plot.png",
    dpi=200
)

plt.tight_layout()
#plt.show()
plt.close()




#------------------------------------------------------------------------------------------
##### Separate plot for tau_V build up for the different halo masses
#------------------------------------------------------------------------------------------
j=0
fig = plt.figure()
ax1 = fig.add_subplot(111)

for j in range(len(Mh_array)):
    col=custom_colormap(float(j)/len(Mh_array))
    print('\n log(Mh/Msol)=', np.log10(Mh_array[j]))
    print('log(Mstar/Msol)=', np.log10(Mstar_array[j]))
    print('log(Md/Msol)=',np.loadtxt('/Users/lsommovigo/Desktop/Scripts/txt_files/JWST_dust_z10/Properties_10logMH'+str(int(10*np.log10(Mh_array[j])))+'_1e3yd'+str(int(1e3*yd))+'_100eps'+str(int(100*epsilon))+ '_1e3lam'+str(int(1e3*spin_param))+'_NoEjection_red'+str(int(redshift))+'.txt', usecols=3)[-1])
    print( 'rd/kpc -->', rd_kpc(redshift,Mh_array[j],spin_param_distr.mean()), '-',  rd_kpc(redshift,Mh_array[j],spin_param_distr.mean()) - rd_kpc(redshift,Mh_array[j],np.percentile(spin_param_distr,16)), '+', -rd_kpc(redshift,Mh_array[j],spin_param_distr.mean()) + rd_kpc(redshift,Mh_array[j],np.percentile(spin_param_distr,84)))  
    ## Importo info necessarie dalla computazione precedente
    time_arr= np.loadtxt('/Users/lsommovigo/Desktop/Scripts/txt_files/JWST_dust_z10/Properties_10logMH'+str(int(10*np.log10(Mh_array[j])))+'_1e3yd'+str(int(1e3*yd))+'_100eps'+str(int(100*epsilon))+ '_1e3lam'+str(int(1e3*spin_param))+'_NoEjection_red'+str(int(redshift))+'.txt', usecols=0)
    tauV_arr= np.loadtxt('/Users/lsommovigo/Desktop/Scripts/txt_files/JWST_dust_z10/Properties_10logMH'+str(int(10*np.log10(Mh_array[j])))+'_1e3yd'+str(int(1e3*yd))+'_100eps'+str(int(100*epsilon))+ '_1e3lam'+str(int(1e3*spin_param))+'_NoEjection_red'+str(int(redshift))+'.txt', usecols=4)
    tauV_min_arr= np.loadtxt('/Users/lsommovigo/Desktop/Scripts/txt_files/JWST_dust_z10/Properties_10logMH'+str(int(10*np.log10(Mh_array[j])))+'_1e3yd'+str(int(1e3*yd))+'_100eps'+str(int(100*epsilon))+ '_1e3lam'+str(int(1e3*spin_param))+'_NoEjection_red'+str(int(redshift))+'.txt', usecols=5)
    tauV_max_arr= np.loadtxt('/Users/lsommovigo/Desktop/Scripts/txt_files/JWST_dust_z10/Properties_10logMH'+str(int(10*np.log10(Mh_array[j])))+'_1e3yd'+str(int(1e3*yd))+'_100eps'+str(int(100*epsilon))+ '_1e3lam'+str(int(1e3*spin_param))+'_NoEjection_red'+str(int(redshift))+'.txt', usecols=6)
    ##
    lw=1.
    
    ### highlighting fiducial value and lowest mass we consider
    if j==5:
        ax1.fill_between(time_arr,tauV_min_arr,tauV_max_arr,color=col,alpha=0.2)
        ax1.errorbar(time_arr[find_nearest(tauV_arr,1.)],1., xerr=[[-time_arr[find_nearest(tauV_max_arr,1.)]+time_arr[find_nearest(tauV_arr,1.)]],[time_arr[find_nearest(tauV_min_arr,1.)]-time_arr[find_nearest(tauV_arr,1.)]]], capsize=6, lw=4., color=col,alpha=0.65,capthick=2.,ls=':')
        #ax1.text(6.3,0.7,'$r_d$ - 1$\sigma$ var. \n($M_{\star}/M_{\odot}=10^{9.1})$',color=col,fontsize=12)
        lw=2.5
    
    ax1.plot(time_arr,tauV_arr,color=col, linewidth=lw, alpha=0.8)
    #print('\n')

### Evidenzio dove tau_V=0.1, mean measured value
ax1.axhline(0.1,ls=':',color='black',alpha=0.25,zorder=-1000)
ax1.text(2.4,0.11,'$\\tau_{\\rm V}=0.1$',color='black',alpha=0.4,fontsize=12)
ax1.axhline(1.,ls='--',color='black',alpha=0.4,zorder=-1000)
ax1.text(2.4,1.1,'$\\tau_{\\rm V}=1$',color='black',alpha=0.7,fontsize=12)
ax1.plot(3.5*np.ones(10),np.linspace(1e-29,80,10),color='grey',lw=4.,alpha=0.2,zorder=-1000)
ax1.text(3.0,3e-3,'3.5 Myr',rotation=90,fontsize=14,color='grey',alpha=0.4)

##
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylim(2e-3,13)
ax1.set_xlim(2,4*time_arr.max())
ax1.set_ylabel('$\\tau_V$')
ax1.set_xlabel('$\mathrm{Galaxy\ Age [Myr]}$')

####### colorbar Mh or Mstar ########
bar=np.zeros((1,len(Mstar_array)))
bar[0][:]=np.linspace(np.log10(Mstar_array).min(),np.log10(Mstar_array).max(),len(Mstar_array))
##
cbaxis = inset_axes(ax1, width="35%", height="4%", loc='upper right', borderpad=1.2)
ext=(np.log10(Mstar_array).min(),np.log10(Mstar_array).max(),0.0,0.8)
cbaxis.imshow(bar,cmap=custom_colormap,extent=ext,aspect='auto',vmin=np.log10(Mstar_array).min(),vmax=np.log10(Mstar_array).max(),alpha=0.8)
cbaxis.set_xlabel('$\log\ (M_{\star}/M_{\odot})$',fontsize=14)
cbaxis.tick_params(axis='both', which='major', labelsize=14, pad=1)
cbaxis.set_yticks([])
## second colorbar Mstar
#cbaxis2 = cbaxis.twinx()
#cbaxis2.set_xlim(np.log10(5e7*fb*epsilon),np.log10(1e15*fb*epsilon))
#cbaxis2.set_xlabel('$M_{\star} [M_{\odot}]$')

### Plotting "corresponding" Observed galaxies
if redshift>=10:
    for jj in range(len(names)-1):
        if tau_v_meas[jj]>0 and log_Mstar_meas[jj]>0 and age_meas[jj]>0:
            low_xerr=0.6*age_meas
            high_xerr=0.6*age_meas
            if err_tau_meas_low[jj]>=tau_v_meas[jj] or np.isnan(err_tau_meas_low[jj]):
                uplim=True
                err_tau_meas_low[jj] = 0.1*tau_v_meas[jj]
            else:
                uplim=False
            ax1.errorbar(age_meas[jj],tau_v_meas[jj],yerr=[[np.abs(err_tau_meas_up[jj])],[np.abs(err_tau_meas_low[jj])]],xerr=[[low_xerr[jj]],[high_xerr[jj]]], ms=10.,marker='h',capsize=2.5,mec='black',elinewidth=0.5,alpha=0.8, color=custom_colormap((log_Mstar_meas[jj]-np.log10(Mstar_array).min())/(np.log10(Mstar_array).max()-np.log10(Mstar_array).min())), label=str(names[jj]),mew=0.3,uplims=uplim)#,label='JWST sel., spec $z>10$'

if redshift==7:
    for jj in range(len(REBELS_index)):
       ## adding data points for REBELS continuum detected galaxies
       ax1.errorbar(Age_REB[jj],Av_REB_npSFH[jj]/1.086, yerr=[[errm_Av_REB_npSFH[jj]],[errp_Av_REB_npSFH[jj]]], xerr=[[errm_Age_REB[jj]],[errp_Age_REB[jj]]], ms=10.,marker='s',capsize=2.5,mec='black',elinewidth=0.5,alpha=0.8, color=custom_colormap((logMstar_REB_npSFH[jj] -np.log10(Mstar_array).min())/(np.log10(Mstar_array).max()-np.log10(Mstar_array).min())),mew=0.3)
    
#ax1.legend(fontsize=11, loc='upper left',ncol=3)
ax1.text(0.9*age[-1],6.8, '$z=$'+str(int(redshift)), fontsize=18)
plt.tight_layout()
plt.show()



#------------------------------------------------------------------------------------------
### MUV vs. M_Star plot (intrinsic)
#------------------------------------------------------------------------------------------
j=0
LUV_z10=[]
LUV_z10_05=[]
LUV_z10_01=[]
for j in range(len(Mh_array)):
    #L1500_z10= np.loadtxt('/Users/lsommovigo/Desktop/Scripts/txt_files/JWST_dust_z10/Properties_10logMH'+str(int(10*np.log10(Mh_array[j])))+'_1e3yd'+str(int(1e3*yd))+'_100eps'+str(int(100*epsilon))+ '_1e3lam'+str(int(1e3*spin_param))+'_NoEjection.txt', usecols=7)[-1]
    
    LUV_z10=np.append(LUV_z10,np.loadtxt('/Users/lsommovigo/Desktop/Scripts/txt_files/JWST_dust_z10/Properties_10logMH'+str(int(10*np.log10(Mh_array[j])))+'_1e3yd'+str(int(1e3*yd))+'_100eps'+str(int(100*epsilon))+ '_1e3lam'+str(int(1e3*spin_param))+'_NoEjection_red'+str(int(redshift))+'.txt', usecols=7)[-1])
    
    #LUV_z10_05=np.append(LUV_z10_05,np.loadtxt('/Users/lsommovigo/Desktop/Scripts/txt_files/JWST_dust_z10/Properties_10logMH'+str(int(10*np.log10(Mh_array[j])))+'_1e3yd'+str(int(1e3*yd))+'_100eps'+str(int(100*0.5))+ '_1e3lam'+str(int(1e3*spin_param))+'_NoEjection.txt', usecols=7)[-1])
    
    #LUV_z10_01=np.append(LUV_z10_01,np.loadtxt('/Users/lsommovigo/Desktop/Scripts/txt_files/JWST_dust_z10/Properties_10logMH'+str(int(10*np.log10(Mh_array[j])))+'_1e3yd'+str(int(1e3*yd))+'_100eps'+str(int(100*0.1))+ '_1e3lam'+str(int(1e3*spin_param))+'_NoEjection.txt', usecols=7)[-1])
    
    print('LUV [erg/s/Angstrom] -->', np.loadtxt('/Users/lsommovigo/Desktop/Scripts/txt_files/JWST_dust_z10/Properties_10logMH'+str(int(10*np.log10(Mh_array[j])))+'_1e3yd'+str(int(1e3*yd))+'_100eps'+str(int(100*epsilon))+ '_1e3lam'+str(int(1e3*spin_param))+'_NoEjection_red'+str(int(redshift))+'.txt', usecols=7)[-1])

if redshift>=10:
    for jj in range(len(names)):
        if log_Mstar_meas[jj]>0:
            plt.errorbar(log_Mstar_meas[jj],MUV_meas[jj],xerr=[[err_logMstar_low_meas[jj]],[err_logMstar_high_meas[jj]]],label=names[jj],ms=12.,marker='h',capsize=2.5,elinewidth=0.5,mec='black',alpha=0.8, mew=0.3,ls='none',color=custom_colormap((tau_v_meas[jj]-0.05)/(tau_v_meas.max()-0.05)))
            if tau_v_meas[jj]>0:
                plt.text(log_Mstar_meas[jj],1.01*MUV_meas[jj],str(round(tau_v_meas[jj],2)),fontsize=10,color=custom_colormap((tau_v_meas[jj]-0.05)/(tau_v_meas.max()-0.05)))

if redshift==7:
    for jj in range(len(REBELS_index)):
        ## adding data points for REBELS continuum detected galaxies
        plt.errorbar(logMstar_REB_npSFH[jj],MUV_REB[jj],xerr=[[errm_logMstar_REB_npSFH[jj]],[errp_logMstar_REB_npSFH[jj]]],label='REB-'+str(int(REBELS_index[jj])),ms=12.,marker='s',capsize=2.5,mec='black',elinewidth=0.5, alpha=0.8, mew=0.3,ls='none',color=custom_colormap((Av_REB_npSFH[jj]-0.05*1.086)/(Av_REB_npSFH.max()-0.05*1.086)))
    plt.xlim(logMstar_REB_npSFH.min()-1, logMstar_REB_npSFH.max()+1)

plt.plot(np.log10(Mstar_array),L1500_to_MUV_conv(LUV_z10),ms=8.,alpha=0.5,label='$\epsilon_{\star}=$'+str(epsilon),color='black',lw=2.)

#plt.plot(np.log10(Mstar_array_05),L1500_to_MUV_conv(LUV_z10_05),ms=8.,alpha=0.5,label='$\epsilon_{\star}=$'+str(0.5),color='silver',lw=2.)

#plt.plot(np.log10(Mstar_array_01),L1500_to_MUV_conv(LUV_z10_01),ms=8.,alpha=0.5,label='$\epsilon_{\star}=$'+str(0.1),color='grey',lw=2.)

plt.ylabel('$M_{UV}^{\\rm int}$')
plt.xlabel('$\log (M_{\star}/M_{\odot})$')
plt.legend(fontsize=11, loc='lower right',ncols=3,frameon=False)
plt.ylim(-13.8, -22.9)
plt.xlim(6.9, 9.7)
#plt.xlim(6.5,9.7)
plt.tight_layout()
plt.show()
print('[logMstar, MUV] for epsilon=',str(epsilon),'-->', str(np.log10(Mstar_array)),str(L1500_to_MUV_conv(LUV_z10)))



#------------------------------------------------------------------------------------------
### Study difference in L1500 from Kennicutt vs SB99 rescaled
#------------------------------------------------------------------------------------------
SFH, logMst_build, age =Build_SFH_funct(Mh_array[3],10.,1,0.1)
#print('SFH-->', SFH)
Lnu_SB99_arr=[]
for ind in range(len(age)):
    L1500_step = 0
    for t in range(ind + 1):
        age_delay = 1e6 * tstep * (ind - t)  # [yr]
        L1500_ssp = 10**np.interp(age_delay, time_yr_L1500, L1500_SB99)  # [erg/s/Å]
        mass_formed = SFH[t] * tstep * 1e6 # [Msun]
        L1500_step += L1500_ssp * (mass_formed/1e6) # normalize to SB 99 which is 1e6 Msun burst
    Lnu_SB99_arr = np.append(Lnu_SB99_arr, L1500_lambda_to_Lnu(L1500_step))
    
    #L1500_arr.append(L1500_step)
    #print('L1500 SB99:', L1500_step, 'erg/s/Å')
    #print('Lnu SB99:', Lnu_SB99, 'erg/s/Hz')
    #print('mass formed, L1500_step, tstep -->', mass_formed, L1500_step, tstep)
plt.scatter(age*1e6,Lnu_SB99_arr, color='royalblue', edgecolor='black', label='SB99 - rescaled with SFH',alpha=0.6)

plt.scatter(age*1e6, 7.14e27 * SFH, label='Kennicutt - 7.14e27 $\cdot$ SFH ', alpha=0.5, edgecolor='black',color='grey')

plt.scatter(age*1e6,SFH*1e26,label='SFH $\cdot$ 1e26', alpha=0.4, edgecolor='white',color='royalblue',marker='*',s=200)

plt.plot(time_yr_L1500, L1500_lambda_to_Lnu(10**L1500_SB99), label='SB99 - inst $10^6\ \mathrm{M_{\odot}}$', color='purple',lw=2.,alpha=0.5)
plt.plot(time_yr_L1500_cont, L1500_lambda_to_Lnu(10**L1500_SB99_cont), label='SB99 - continuous $1\ \mathrm{M_{\odot}/yr}$', ls='--', color='indigo',lw=2.,alpha=0.5)

#print('Kennicutt/SB99 ratio -->', 7.14e27 * SFH/Lnu_SB99_arr)
#print('end SB99 -->', L1500_SB99_cont[-1])
#print('in Lnu -->',  np.log10(L1500_lambda_to_Lnu(10**L1500_SB99_cont[-1])))
#print('in Lnu/ks -->',  np.log10(L1500_lambda_to_Lnu(10**L1500_SB99_cont[-1]))-np.log10(7.14e27))
plt.xlabel(r"$\mathrm{Galaxy\ Age [yr]}$")
plt.ylabel(r"$L^{int}_{UV}\,[\mathrm{erg\,s^{-1}\,Hz^{-1}}]$")
plt.xlim(1e5,2e8)
plt.ylim(3e25,5e29)
plt.xscale('log')
plt.yscale('log')
plt.legend(fontsize=14,ncols=2,loc='upper left')
plt.show()




## Showing the effect of adding the transmission function
plt.plot(np.linspace(1e-3,5,1000),T_1500_sphere(np.linspace(1e-3,5,1000)), label='our funct. $T_{\lambda}$')
plt.plot(np.linspace(1e-3,5,1000),T_1500_sphere_im(np.linspace(1e-3,5,1000)), label='our funct., mixed, $T_{\lambda}$')
plt.plot(np.linspace(1e-3,5,1000),np.exp(-np.linspace(1e-3,5,1000)),ls='--',color='grey',alpha=0.5, label=r"screen, $e^{-\tau}$")
plt.ylabel(r'$T_{1500}(\tau_{1500})$')
plt.xlabel(r'$\tau_{1500}$')
#plt.yscale('log')
plt.tight_layout()
plt.legend()
plt.show()





#------------------------------------------------------------------------------------------
### MUV–Mstar plot accounting for dust attenuation: two transmission functions
#------------------------------------------------------------------------------------------
arr_e  = np.array([0.1])            # You can add more values if needed
arr_yd = np.array([0.001, 0.01, 0.1])
plt.figure(figsize=(9, 7))

# Use the same colormap for points as in the intrinsic plot above
point_colormap = custom_colormap  
line_colormap  = cm.inferno

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=10):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap

line_colormap = truncate_colormap(line_colormap, 0., 0.7)

for epsilon in arr_e:
    for yd in arr_yd:
        # linestyle encodes dust yield
        if yd == 0.1:
            ls = ':'
        elif yd == 0.001:
            ls = '-.'
        elif yd == 0.01:
            ls = '--'

        Mstar_array = halo_to_stellar_mass(Mh_array, fb, epsilon)
        spin_param_distr = random.lognormal(mean=np.log(10**-1.5677),
                                            sigma=0.5390, size=1000)

        # arrays for intrinsic & attenuated magnitudes
        MUV_fin_array = []

        # mixed geometry (im)
        MUVatt_mix      = []
        MUVatt_mix_u    = []
        MUVatt_mix_l    = []

        # point–source geometry (ps)
        MUVatt_ps       = []
        MUVatt_ps_u     = []
        MUVatt_ps_l     = []

        for j in range(len(Mh_array)):
            SFH, logMst_build, age = Build_SFH_funct(Mh_array[j], redshift, tstep, epsilon)
            L1500_arr = compute_L1500_steps(age, tstep, SFH, time_yr_L1500, L1500_SB99)
            N_SN_arr, Md_arr = compute_Mdust_steps(age, tstep, SFH, time_yr, logSNr_yr, yd)

            # --- Dust attenuation ---
            tauUV = tau_pred(kUV, Md_arr[-1], Mh_array[j], spin_param_distr, redshift)

            # mixed geometry transmission
            T1500_mix = T_1500_sphere_im(tauUV)
            LUV_mix_arr = np.atleast_1d(np.asarray(T1500_mix) * L1500_arr[-1])
            MUV_mix_dist = L1500_to_MUV_conv(LUV_mix_arr)
            MUV_mix  = np.median(MUV_mix_dist)
            MUV_mix_u = np.percentile(MUV_mix_dist, 16)
            MUV_mix_l = np.percentile(MUV_mix_dist, 84)

            # point–source transmission
            T1500_ps = T_1500_sphere(tauUV)
            LUV_ps_arr = np.atleast_1d(np.asarray(T1500_ps) * L1500_arr[-1])
            MUV_ps_dist = L1500_to_MUV_conv(LUV_ps_arr)
            MUV_ps  = np.median(MUV_ps_dist)
            MUV_ps_u = np.percentile(MUV_ps_dist, 16)
            MUV_ps_l = np.percentile(MUV_ps_dist, 84)

            # intrinsic MUV (no attenuation)
            MUV_noatt = L1500_to_MUV_conv(L1500_arr[-1])

            MUV_fin_array.append(MUV_noatt)

            MUVatt_mix.append(MUV_mix)
            MUVatt_mix_u.append(MUV_mix_u)
            MUVatt_mix_l.append(MUV_mix_l)

            MUVatt_ps.append(MUV_ps)
            MUVatt_ps_u.append(MUV_ps_u)
            MUVatt_ps_l.append(MUV_ps_l)

        # --- Plotting lines with inferno colormap based on epsilon ---
        line_color = line_colormap(np.clip(epsilon + epsilon / 0.05 / 10, 0, 1))

        # mixed geometry (solid color)
        plt.plot(np.log10(Mstar_array), MUVatt_mix, ls=ls, lw=1.5,
                 label=fr"Attenuated (mix): $\epsilon={epsilon}$, $y_d={yd}$",
                 color=line_color, alpha=0.7)

        if yd == arr_yd[-1]:
            plt.fill_between(np.log10(Mstar_array), MUVatt_mix_l, MUVatt_mix_u,
                             color=line_color, alpha=0.05)
            #plt.text(9, -19.3,
            #        r"Mix ($T_{1500,mix})$", 
            #        rotation=20, color=line_color, fontsize=15)
            
            # point–source geometry only shaded for max yd to avoid clutter
            plt.plot(np.log10(Mstar_array), MUVatt_ps, ls=ls, lw=1.5, color='magenta', alpha=0.5)#, label=fr"ATT (ps): $\epsilon={epsilon}$, $y_d={yd}$")
            # Inline text label near the curve
            #plt.text(9,-16.15,
            #        r"Point Source ($T_{1500,ps})$", 
            #        rotation=-20, color='magenta', fontsize=15,alpha=0.5)
            #plt.fill_between(np.log10(Mstar_array), MUVatt_ps_l, MUVatt_ps_u,
            #                 edgecolor=darker, alpha=0.1,hatch='\\\\',lw=2.5,facecolor='None')

    # Plot intrinsic only once per epsilon
    plt.plot(np.log10(Mstar_array), MUV_fin_array,
             color=line_color, alpha=0.3, label='Intrinsic', lw=2., zorder=1000)

plt.legend(loc='upper left', fontsize=12,frameon=False)

# --- observational points (unchanged) ---
if redshift >= 10:
    for jj in range(len(names)):
        if log_Mstar_meas[jj] > 0:
            point_color = point_colormap((tau_v_meas[jj] - 0.05) /
                                         (tau_v_meas.max() - 0.05))
            plt.errorbar(log_Mstar_meas[jj], MUV_meas[jj],
                         xerr=[[err_logMstar_low_meas[jj]],
                               [err_logMstar_high_meas[jj]]],
                         label=names[jj], ms=12., marker='h', capsize=2.5,
                         mec='black', elinewidth=0.5, alpha=0.8, mew=0.3,
                         ls='none', color=point_color)
            if tau_v_meas[jj] > 0:
                plt.text(log_Mstar_meas[jj], 1.01 * MUV_meas[jj],
                         str(round(tau_v_meas[jj], 2)),
                         fontsize=10, color=point_color)

if redshift == 7:
    for jj in range(len(REBELS_index)):
        point_color = point_colormap((Av_REB_npSFH[jj] - 0.05*1.086) /
                                     (Av_REB_npSFH.max() - 0.05*1.086))
        
        #REBELS continuum detected galaxies
        plt.errorbar(logMstar_REB_npSFH[jj], MUV_REB[jj],
                     xerr=[[errm_logMstar_REB_npSFH[jj]],
                           [errp_logMstar_REB_npSFH[jj]]], ms=12.,
                     marker='s', capsize=2.5, elinewidth=0.5, mec='black',
                     alpha=0.8, mew=0.3, ls='none', color=point_color)
    plt.xlim(logMstar_REB_npSFH.min() - 1, logMstar_REB_npSFH.max() + 1)


plt.xlabel(r'$\log_{10}(M_\star/M_\odot)$')
plt.ylabel(r'$M_{\mathrm{UV}}$')
plt.gca().invert_yaxis()
plt.ylim(-13.8, -22.9)
plt.xlim(6.9, 9.7)
plt.tight_layout()
plt.show()




# --- figure ---
fig, ax = plt.subplots(figsize=(10,6))
# --- compute rd arrays ---
rd_arr, rd_arr_low, rd_arr_up = [], [], []
for j in range(len(Mh_array)):
    rd_arr.append(rd_kpc(redshift, Mh_array[j], spin_param_distr.mean()))
    rd_arr_low.append(rd_kpc(redshift, Mh_array[j], np.percentile(spin_param_distr,16)))
    rd_arr_up.append(rd_kpc(redshift, Mh_array[j], np.percentile(spin_param_distr,84)))

# --- points (color by Mh via your custom_colormap) ---
for j in range(len(Mh_array)):
    col = custom_colormap(float(j)/len(Mh_array))
    ax.scatter(MUV_fin_array[j],     rd_arr[j], color=col, edgecolor='none', s=80, zorder=5)
    ax.scatter(MUVatt_mix[j],  rd_arr[j], facecolors='none', edgecolors=col,
               linewidths=1.5, s=80, zorder=6)

# --- 16–84% hatched band (behind points) ---
ax.fill_between(MUV_fin_array, rd_arr_up, rd_arr_low,
                edgecolor='grey', hatch='//', facecolor='None', alpha=0.2,
                label=r'$M_{\rm UV,int}-r_d$ (16–84\%)', zorder=1)

# --- observations ---
ax.errorbar(MUV_meas, 1e-3*re_pc_meas,
            yerr=[1e-3*err_re_pc_low, 1e-3*err_re_pc_up],
            label='Obs., Blue Monsters $z>10$', color='dimgrey', mec='black',
            ms=12., marker='h', capsize=2.5, elinewidth=0.5, alpha=0.8, ls='None', zorder=7)

# --- legend handles for filled vs empty ---
ax.scatter([], [], color='grey', s=60, label='Intrinsic $M_{\\rm UV}$')
ax.scatter([], [], facecolors='none', edgecolors='grey', linewidths=1.5, s=60,
           label='Attenuated $M_{\\rm UV}$')

# ===== Inset color strip for halo mass (inside the main axes) =====
logMh = np.log10(Mh_array)
vmin, vmax = float(np.min(logMh)), float(np.max(logMh))
N = max(len(Mh_array), 2)
cmap_mh = ListedColormap([custom_colormap(i/(N-1)) for i in range(N)])

cax = inset_axes(ax, width="35%", height="3%", loc='upper right', borderpad=1.2)
bar = np.linspace(vmin, vmax, N)[None, :]
cax.imshow(bar, cmap=cmap_mh, extent=(vmin, vmax, 0, 1), aspect='auto', vmin=vmin, vmax=vmax)
cax.set_yticks([])
cax.set_xlabel(r'$\log_{10}(M_{h}/M_\odot)$', fontsize=11, labelpad=4)
cax.tick_params(axis='x', labelsize=10, pad=2)

# --- labels & axes (ON THE MAIN AXIS) ---
ax.set_xlabel(r'$M_{\mathrm{UV}}$')
ax.set_ylabel(r'$r_d$ [kpc]')
ax.set_yscale('log')
ax.invert_xaxis()
ax.legend(fontsize=14, loc='best', frameon=False)

# Let Matplotlib arrange things; inset stays inside ax so no clashes
plt.tight_layout()
plt.show()



