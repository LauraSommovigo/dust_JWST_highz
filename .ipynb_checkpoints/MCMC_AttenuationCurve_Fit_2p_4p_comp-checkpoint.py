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


################### FUNCTIONS ########################
###  Attenuation curves (Li+08 parametrization)
def Li_08(lam_micron, c1, c2, c3, c4, model):## Wavelength in microns
        
        if model == 'Calzetti':
            c1,c2,c3,c4 = 44.9, 7.56, 61.2, 0.
        if model == 'SMC':
            c1,c2,c3,c4 = 38.7, 3.83, 6.34, 0.
        if model== 'MW':
            c1,c2,c3,c4 = 14.4, 6.52, 2.04, 0.0519
        if model=='LMC':
            c1,c2,c3,c4 = 4.47, 2.39, -0.988, 0.0221
        #Attenuation curve normalized to Av
        A_lam_v = c1 / ((lam_micron/0.08)**c2 + (lam_micron/0.08)**-c2 + c3)  +   (233. * (1 - c1/(6.88**c2 + 0.145**c2 +c3) - c4/4.6)) / ((lam_micron/0.046)**2. + (lam_micron/0.046)**-2. + 90.)  +  c4/ ((lam_micron/0.2175)**2. + (lam_micron/0.2175)**-2. - 1.95)
        
        
        return A_lam_v


### Li+08 parametrized version of the attenuaton curves
def Li_08_fit_noratio(lam_micron, c1, c2, c3, c4):## Wavelength in microns
        #Attenuation curve normalized to Av
        A_lam_v = c1 / ((lam_micron/0.08)**c2 + (lam_micron/0.08)**-c2 + c3)  +   (233. * (1. - c1/(6.88**c2 + 0.145**c2 +c3) - c4/4.6)) / ((lam_micron/0.046)**2. + (lam_micron/0.046)**-2. + 90.) +  c4/ ((lam_micron/0.2175)**2. + (lam_micron/0.2175)**-2. - 1.95)
        return A_lam_v
'''
### Li+08 parametrized version of the attenuaton curves
def Li_08_fit(lam_micron, c1, c2, c3_c1_ratio, c4):## Wavelength in microns

        #Attenuation curve normalized to Av
        A_lam_v = 1. / ((lam_micron/0.08)**c2 / c1 + (lam_micron/0.08)**-c2 /c1 + c3_c1_ratio)  +   (233. * (1. - 1./(6.88**c2 /c1 + 0.145**c2 /c1 +c3_c1_ratio) - c4/4.6)) / ((lam_micron/0.046)**2. + (lam_micron/0.046)**-2. + 90.) +  c4/ ((lam_micron/0.2175)**2. + (lam_micron/0.2175)**-2. - 1.95)

        #Attenuation curve normalized to Av
        #A_lam_v = c1 / ((lam_micron/0.08)**c2 + (lam_micron/0.08)**-c2 + c3_c1_ratio*c1)  +   (233. * (1. - c1/(6.88**c2 + 0.145**c2 +c3_c1_ratio*c1) - c4/4.6)) / ((lam_micron/0.046)**2. + (lam_micron/0.046)**-2. + 90.) +  c4/ ((lam_micron/0.2175)**2. + (lam_micron/0.2175)**-2. - 1.95)
        return A_lam_v
'''

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=10):
    new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval), cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def power_law(x,a,b):
    return a * x**b

def rebin_two_arrays(array1, array2, bin_edges):
    """
    Rebin two arrays into specified bins, using the median value within each bin,
    along with the 16th and 84th percentiles. The second array is binned according to the bins of the first array.

    Parameters:
    array1 (np.ndarray): The primary array to determine the bins.
    array2 (np.ndarray): The secondary array to be binned according to array1.
    bin_edges (np.ndarray): The edges of the bins to rebin the arrays into.

    Returns:
    tuple: Six numpy arrays containing the rebinned medians, 16th percentiles, and 84th percentiles for both arrays.
    """
    
    if len(array1) != len(array2):
        raise ValueError("Both arrays must have the same length")
    
    # Initialize the rebinned arrays
    num_bins = len(bin_edges) - 1
    rebinned_medians1 = []
    rebinned_16th_percentiles1 = []
    rebinned_84th_percentiles1 = []
    
    rebinned_medians2 = np.zeros(num_bins)
    rebinned_16th_percentiles2 = np.zeros(num_bins)
    rebinned_84th_percentiles2 = np.zeros(num_bins)
    
    for i in range(num_bins):
        # Determine the start and end edges of the current bin
        start_edge = bin_edges[i]
        end_edge = bin_edges[i + 1]
        rebinned_medians1=np.append(rebinned_medians1,0.5*(bin_edges[i]+bin_edges[i+1]))
        rebinned_16th_percentiles1=np.append(rebinned_16th_percentiles1,bin_edges[i])
        rebinned_84th_percentiles1=np.append(rebinned_84th_percentiles1,bin_edges[i+1])
        
        # Get the values within the current bin for both arrays
        bin_mask = np.where((array1 >= start_edge) & (array1 < end_edge))
        bin_values1 = array1[bin_mask]
        bin_values2 = array2[bin_mask]
        
        if len(bin_values1) == 0:  # Handle empty bins
            rebinned_medians1[i] = np.nan
            rebinned_16th_percentiles1[i] = np.nan
            rebinned_84th_percentiles1[i] = np.nan
            rebinned_medians2[i] = np.nan
            rebinned_16th_percentiles2[i] = np.nan
            rebinned_84th_percentiles2[i] = np.nan
        else:
            # Compute the median, 16th, and 84th percentiles of the current bin for both arrays
            #rebinned_medians1[i] = np.median(bin_values1)
            #rebinned_16th_percentiles1[i] = np.percentile(bin_values1, 16)
            #rebinned_84th_percentiles1[i] = np.percentile(bin_values1, 84)
            
            rebinned_medians2[i] = np.mean(bin_values2)
            rebinned_16th_percentiles2[i] = np.percentile(bin_values2, 30)
            rebinned_84th_percentiles2[i] = np.percentile(bin_values2, 70)
    
    #print(rebinned_medians1, rebinned_medians2,'\n\n')
    
    return (rebinned_medians1, rebinned_16th_percentiles1, rebinned_84th_percentiles1,
            rebinned_medians2, rebinned_16th_percentiles2, rebinned_84th_percentiles2)



######################### EMCEE functions #########################

######################### Note for priors #########################
#      MW --> c1,c2,c3,c4 = 14.4, 6.52, 2.04, 0.0519
#Calzetti --> c1,c2,c3,c4 = 44.9, 7.56, 61.2, 0.
#     SMC --> c1,c2,c3,c4 = 38.7, 3.83, 6.34, 0.
#     LMC --> c1,c2,c3,c4 = 4.47, 2.39, -0.988, 0.0221
###################################################################

def log_likelihood(theta, x, y, yerr):
    c1,c2,c3,c4 =theta
    model = np.array(Li_08_fit_noratio(x,c1=theta[0],c2=theta[1],c3=theta[2], c4=theta[3]))
    sigma2 = yerr ** 2.0 + model ** 2.0
    return -1e3 * np.sum((y - model) ** 2 )#/ sigma2 + np.log(sigma2))

#"uninformative" priors
def log_prior(theta):
    c1,c2,c3,c4 = theta
    if (-1000. < c1 < 1000.  and 0.0 < c2 < 10. and -5e3 < c3 < 5e3 and -1.0 < c4 < 0.8):#if (0.1 < c1 < 500.  and 0.0 < c2 < 10. and -10. < c3_c1_ratio < 10. and -1.0 < c4 < 0.8):
        return (0.0)
    return (-np.inf)
    
def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)






def k_calzetti2000(wavelength):
    """Compute the Calzetti et al. (2000) A(λ)/E(B-V)∗

    Given a wavelength grid, this function computes the selective attenuation
    A(λ)/E(B-V)∗ using the formula from Calzetti at al. (2000). This formula
    is given for wavelengths between 120 nm and 2200 nm, but this function
    makes the computation outside.

    Parameters
    ----------
    wavelength: array of floats
        Wavelength grid in nm.

    Returns
    -------
    a numpy array of floats

    """
    wavelength = np.array(wavelength)
    result = np.zeros(len(wavelength))

    # Attenuation between 120 nm and 630 nm
    mask = (wavelength < 630)
    result[mask] = 2.659 * (-2.156 + 1.509e3 / wavelength[mask] -
                            0.198e6 / wavelength[mask] ** 2 +
                            0.011e9 / wavelength[mask] ** 3) + 4.05

    # Attenuation between 630 nm and 2200 nm
    mask = (wavelength >= 630)
    result[mask] = 2.659 * (-1.857 + 1.040e3 / wavelength[mask]) + 4.05

    return result


def k_leitherer2002(wavelength):
    """Compute the Leitherer et al. (2002) A(λ)/E(B-V)∗

    Given a wavelength grid, this function computes the selective attenuation
    A(λ)/E(B-V)∗ using the formula from Leitherer at al. (2002). This formula
    is given for wavelengths between 91.2 nm and 180 nm, but this function
    makes the computation outside.

    Parameters
    ----------
    wavelength: array of floats
        Wavelength grid in nm.

    Returns
    -------
    a numpy array of floats

    """
    wavelength = np.array(wavelength)
    result = (5.472 + 0.671e3 / wavelength -
              9.218e3 / wavelength ** 2 +
              2.620e6 / wavelength ** 3)

    return result


def uv_bump(wavelength, central_wave, gamma, ebump):
    """Compute the Lorentzian-like Drude profile.

    Parameters
    ----------
    wavelength: array of floats
        Wavelength grid in nm.
    central_wave: float
        Central wavelength of the bump in nm.
    gamma: float
        Width (FWHM) of the bump in nm.
    ebump: float
        Amplitude of the bump.

    Returns
    -------
    a numpy array of floats

    """
    return (ebump * wavelength ** 2 * gamma ** 2 /
            ((wavelength ** 2 - central_wave ** 2) ** 2 +
             wavelength ** 2 * gamma ** 2))


def power_law(wavelength, delta):
    """Power law 'centered' on 550 nm..

    Parameters
    ----------
    wavelength: array of floats
        The wavelength grid in nm.
    delta: float
        The slope of the power law.

    Returns
    -------
    array of floats

    """
    return (wavelength / 550) ** delta


def a_vs_ebv(wavelength, bump_wave, bump_width, bump_ampl, power_slope):
    """Compute the complete attenuation curve A(λ)/E(B-V)*

    The Leitherer et al. (2002) formula is used between 91.2 nm and 150 nm, and
    the Calzetti et al. (2000) formula is used after 150 (we do an
    extrapolation after 2200 nm). When the attenuation becomes negative, it is
    kept to 0. This continuum is multiplied by the power law and then the UV
    bump is added.

    Parameters
    ----------
    wavelength: array of floats
        The wavelength grid (in nm) to compute the attenuation curve on.
    bump_wave: float
        Central wavelength (in nm) of the UV bump.
    bump_width: float
        Width (FWHM, in nm) of the UV bump.
    bump_ampl: float
        Amplitude of the UV bump.
    power_slope: float
        Slope of the power law.

    Returns
    -------
    attenuation: array of floats
        The A(λ)/E(B-V)* attenuation at each wavelength of the grid.

    """
    attenuation = np.zeros(len(wavelength))

    # Leitherer et al.
    mask = (wavelength > 91.2) & (wavelength < 150)
    attenuation[mask] = k_leitherer2002(wavelength[mask])
    # Calzetti et al.
    mask = (wavelength >= 150)
    attenuation[mask] = k_calzetti2000(wavelength[mask])
    # We set attenuation to 0 where it becomes negative
    mask = (attenuation < 0)
    attenuation[mask] = 0
    # Power law
    attenuation *= power_law(wavelength, power_slope)

    # As the powerlaw slope changes E(B-V), we correct this so that the curve
    # always has the same E(B-V) as the starburst curve. This ensures that the
    # E(B-V) requested by the user is the actual E(B-V) of the curve.
    wl_BV = np.array([440., 550.])
    EBV_calz = (k_calzetti2000(wl_BV) * power_law(wl_BV, 0.)) # SS
    EBV = (k_calzetti2000(wl_BV) * power_law(wl_BV, power_slope)) # SS
    attenuation *= (EBV_calz[1]-EBV_calz[0]) / (EBV[1]-EBV[0])

    # UV bump
    attenuation += uv_bump(wavelength, bump_wave, bump_width, bump_ampl) # do bump at the end (SS)

    
    return attenuation


def Att_Curve_2param(lamda, B, delta):
    return a_vs_ebv(lamda/10., 217.5, 35., B, delta)/a_vs_ebv(550.*np.ones(2), 217.5, 35., B, delta)[0]


def log_likelihood_2p(theta, x, y, yerr):
    delta,bump =theta
    model = np.array(Att_Curve_2param(x, B=theta[0],delta=theta[1]))
    sigma2 = yerr ** 2.0 + model ** 2.0
    return -1e3 * np.sum((y - model) ** 2 )#/ sigma2 + np.log(sigma2))

#"uninformative" priors
def log_prior_2p(theta):
    bump,delta = theta
    if (-100 < delta < 100) and ( -100 < bump < 100):
        return (0.0)
    return (-np.inf)
    
def log_probability_2p(theta, x, y, yerr):
    lp = log_prior_2p(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_2p(theta, x, y, yerr)







##################### Reading TNG SKIRT Outputs #################
n_los = 51  # Number of lines of sight
colors = cm.get_cmap("magma", 51)  # Color map for plotting
colors_Av_TNG = cm.get_cmap("magma_r", 100)  # Color map for Av plotting

### Importing wavelengths from a text file
lam = np.loadtxt('/Users/lsommovigo/Desktop/Scripts/txt_files/LTU_att_curve/wavelengths.txt')

### Importing galaxy properties associated with a given snapshot
ids, mstar, gas_mass, sfr, sfr_compact, gas_Z, rstar, rgas = np.loadtxt(
    '/Users/lsommovigo/Desktop/Scripts/txt_files/LTU_att_curve/all_galaxy_properties_snap_93.dat',
    delimiter=',', unpack=True
)

# List to store galaxies missing attenuation curves
missing_att_curve = []





########################################################################################
######## Fitting the good attenuation curves with the parametric model: : FIG3 #########
#ii=10
where_end=2532

MW_par = np.array([12.8, 1.87, -0.11, 0.08])


for ii in range(13,247,300):#,247):
    print('\n', ii, ids[ii])
    
    #best_fit_par=open('//Users/lsommovigo/Desktop/Scripts/txt_files/LTU_att_curve/EMCEE/emcee_best_fit_id'+str(int(round(ids[ii],0)))+'.txt', "a+")
    #best_fit_par.write('#los    Av    c1    16th    84th    c2    16th    84th    c3    16th    84th    c4    16th    84th \n')

    try:
        ## rows== los, cols=A_lambda/A_V at given lambda
        Flux_ratio=np.loadtxt('//Users/lsommovigo/Desktop/Scripts/txt_files/LTU_att_curve/all_halos_output_snap93/output_snapnum93_shalo'+str(int(round(ids[ii],0)))+'.txt')
    except OSError:
        #best_fit_par.write('Attenuation Curve Not Found')
        #best_fit_par.close()
        continue
        missing_att_curve=np.append(missing_att_curve,ii)
    
    ll=1
    for ll in range(n_los):
        ## eliminating null values in the flux ratios
        Fr_arr=Flux_ratio[ll]
        Alam_arr=-2.5*np.log10(Fr_arr[Fr_arr>0])
        lam_arr=lam[Fr_arr>0]
        
        ### Normalizing ad A_lambda to Av
        v_index = find_nearest(lam_arr,0.551)
        Av = Alam_arr[v_index]
        Alam_Av_arr=Alam_arr/Av
        print('los, Av (los=', ll,') -->', -Av)
        
        ## Plot attenuation curve
        #plt.plot(lam_arr*1e4, Alam_Av_arr, color=colors[ll], lw=1.5,zorder=-100,alpha=0.8)
        
        ### Only consider wavelengths shorter than last SDSS band (0.9)
        lim=1.
        Alam_Av_cut=Alam_Av_arr[lam_arr<lim]#shorter than U-BAND
        Alam_cut=Alam_arr[lam_arr<lim]#shorter than U-BAND
        lam_cut=lam_arr[lam_arr<lim]
        #Alam_Av_cut=Li_08(lam_cut,0.,0.,0.,0.,'Calzetti')
        
        
        #############################
        ######## 4 PARAMETER
        #############################
        
        try:
            ### Fitting with Li+08 parametric function til V-band
            popt, pcov = curve_fit(Li_08_fit_noratio, lam_cut, Alam_Av_cut, bounds = ([-np.inf,-np.inf,-np.inf,0],[np.inf,np.inf,np.inf,1.]))
            ### Print best-fit params
            print('\n #### CURVE FIT -- No bounds (only on c3/c1, c4)')
            for par in range(len(popt)):
                print ("  ",popt[par],"pm",np.sqrt(pcov[par,par]))
            ### Plotting the best fit solution (Making sure it works)
            fit_nb = Li_08_fit_noratio(lam_cut,c1=popt[0],c2=popt[1],c3=popt[2], c4=popt[3])
            #plt.plot(lam_cut*1e4,fit_nb,ls=':', color=colors[ll],lw=4., alpha=0.2)
        
        except:
            #print('For this id,ll fit failed -> ', str(int(round(ids[ii],0))),',', ll)
            continue

        
        ##################### FITTING WITH EMCEE#########################
        ####### Input
        x=lam_cut
        y=Alam_Av_cut
        yerr=0.1*y

        nll = lambda *args: -log_likelihood(*args)
        initial = MW_par#popt+ 0.1 * np.random.randn(4)
        print('initial', initial)
        soln = minimize(nll, initial, args=(x, y, yerr))
        pos = soln.x + 1e-4 * np.random.randn(32, 4)
        nwalkers, ndim = pos.shape
        print ('\nEMCEE STARTS AT -->', initial)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr))
        sampler.run_mcmc(pos, 5000, progress=True);
        labels = ["$c_1$", "$c_2$", "$c_3$", "$c_4$"]

        #tau = sampler.get_autocorr_time()

        flat_samples_4p = sampler.get_chain(discard=200, thin=15, flat=True)
        #print(flat_samples.shape)

        fig, axes = pyplot.subplots(4, figsize=(10, 7), sharex=True)
        for i in range(ndim):
            ax = axes[i]
            ax.plot(flat_samples_4p[:, i], "k", alpha=0.6)
            ax.set_xlim(0, len(flat_samples_4p))
            ax.set_ylabel(labels[i],fontsize=18)
            ax.tick_params(axis='both', which='major', labelsize=16,pad=5)
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number");
        #pyplot.show()
        plt.close()
        fig = corner.corner(flat_samples_4p, labels=labels,
                               quantiles=[0.16, 0.5, 0.84],
                               show_titles=True, title_kwargs= {"fontsize": 15}, plot_contours=True, fill_contours=True,smooth=1.2, labelpad=0.2)

        print('\n #### CURVE FIT -- EMCEE ')
        for i in range(ndim):
            mcmc = np.percentile(flat_samples_4p[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            print(mcmc[1], q[0], q[1], '\n')
                
        # Extract the axes
        axes = numpy.array(fig.axes).reshape((ndim, ndim))

        i=0
        # Loop over the diagonal
        for i in range(ndim):
            ax = axes[i, i]
            ax.axvline(np.percentile(flat_samples_4p[:, i], [16, 50, 84])[1], color="black",linewidth=2.0)
            ## contraints from curve fit
            #ax.axvline(popt[i], color='teal', linewidth=2.5, alpha=0.35)
            #ax.axvline(popt[i]+np.sqrt(pcov[i,i]), color='teal', linewidth=2., ls='--', alpha=0.35)
            #ax.axvline(popt[i]-np.sqrt(pcov[i,i]), color='teal', linewidth=2., ls='--', alpha=0.35)
            #ax.axvline()
            ## MW value
            #ax.axvline(MW_par[i], color=colors_models[2], linewidth=2, alpha=0.4, ls='--')
            

        ##############################
        ### Output PLOT Att. curve
        ##############################
        att_axis = fig.add_axes([0.63, 0.64, 0.36, 0.35])

        ## Input attenuation curve
        att_axis.plot(x*1e4, y, color=colors_Av_TNG(-Av/0.7), lw=3., label='id='+str(int(ids[ii]))+', los='+str(ll), alpha=0.85, zorder=-1000)
        
        #att_axis.scatter(x*1e4, y, color='black', marker='|', zorder=-1000,alpha=0.5, s=70)

        ## Output curve-fit (no bounds apart from c4)
        #att_axis.scatter(lam_cut*1e4,fit_nb,marker='s', facecolor='None', alpha=0.55, label='curve fit', edgecolors='teal',linewidth=0.75,s=40,zorder=-1000)

        ## Output fit EMCEE
        y_fit_mcmc=Li_08_fit_noratio(x,c1=np.percentile(flat_samples_4p[:, 0], [50]),c2=np.percentile(flat_samples_4p[:, 1], [50]),c3=np.percentile(flat_samples_4p[:, 2], [50]), c4=np.percentile(flat_samples_4p[:, 3], [50]))
        y_fit_mcmc_u=Li_08_fit_noratio(x,c1=np.percentile(flat_samples_4p[:, 0], [16]),c2=np.percentile(flat_samples_4p[:, 1], [16]),c3=np.percentile(flat_samples_4p[:, 2], [16]), c4=np.percentile(flat_samples_4p[:, 3], [16]))
        y_fit_mcmc_l=Li_08_fit_noratio(x,c1=np.percentile(flat_samples_4p[:, 0], [84]),c2=np.percentile(flat_samples_4p[:, 1], [84]),c3=np.percentile(flat_samples_4p[:, 2], [84]), c4=np.percentile(flat_samples_4p[:, 3], [84]))
        
        ## saving best fit params
        #best_fit_par.write(str(ll)+', '+str(-Av)+', '+str(float(np.percentile(flat_samples_4p[:, 0], [50])))+', '+str(float(np.percentile(flat_samples_4p[:, 0], [16])))+', '+str(float(np.percentile(flat_samples_4p[:, 0], [84])))+', '+str(float(np.percentile(flat_samples_4p[:, 1], [50])))+', '+str(float(np.percentile(flat_samples_4p[:, 1], [16])))+', '+str(float(np.percentile(flat_samples_4p[:, 1], [84])))+', '+str( float(np.percentile(flat_samples_4p[:, 2], [50])))+', '+str(float(np.percentile(flat_samples_4p[:, 2], [16])))+', '+str( float(np.percentile(flat_samples_4p[:, 2], [84])))+', '+str(float(np.percentile(flat_samples_4p[:, 3], [50])))+', '+str( float(np.percentile(flat_samples_4p[:, 3], [16])))+', '+str(float(np.percentile(flat_samples_4p[:, 3], [84])))+'\n')
        
        
        ##
        #pyplot.scatter(x*1e4,y_fit_mcmc,color='black',label='Emcee',alpha=0.4, edgecolors='black',linewidth=0.8,marker='o',s=52)
        pyplot.fill_between(x*1e4,y_fit_mcmc_l,y_fit_mcmc_u,edgecolor=colors_Av_TNG(-Av/0.7),linewidth=2.,zorder=1000,label='EMCEE fit',hatch='\\',facecolor='none',ls=':')
        
        ### Plot details
        att_axis.set_xlabel(r"$\lambda [\dot {A}]$", fontsize=18)
        att_axis.set_ylabel(r"A$_{\lambda}/$A$_{V}$", fontsize=18)
        att_axis.grid(alpha=0.4,ls='--')
        att_axis.legend(fontsize=14,frameon=False)
        #Chosen ticks: Bump, poi FILTERS: UV: U;     Visible: G, R;     NIR: I, Z
        att_axis.set_xticks([2175, 3543., 4770., 6231., 7625.,9134.])
        att_axis.set_ylim(0,15)
        att_axis.set_xlim(1e3,9.2e3)
        att_axis.tick_params(axis='both', which='major', labelsize=15, pad=5)
        
        plt.subplots_adjust(wspace=0., hspace=0.,left=0.14,bottom=0.13)
        plt.savefig('/Users/lsommovigo/Desktop/PROJECT-LTU/attenuation_curves-TNG50/EMCEE_Corners/id'+str(int(round(ids[ii],0)))+'_los'+str(int(round(ll,0)))+'_new.png')
        plt.close()
        #plt.show()
        
        
        
        
        #############################
        ######## 2 PARAMETER
        #############################
        try:
            ### Fitting with 2-par parametric function til V-band
            popt, pcov = curve_fit(Att_Curve_2param, 1e4*lam_cut, Alam_Av_cut, bounds = ([-np.inf,-np.inf],[np.inf,np.inf]))
            ### Print best-fit params
            print('\n #### CURVE FIT -- No bounds')
            for par in range(len(popt)):
                print ("  ",popt[par],"pm",np.sqrt(pcov[par,par]))
            
            ### Plotting the best fit solution (Making sure it works)
            fit_nb = Att_Curve_2param(1e4*lam_cut,B=popt[0],delta=popt[1])
            #plt.plot(lam_cut*1e4,fit_nb,ls=':', color=colors[ll],lw=4., alpha=0.2)
        
        except:
            print('For this id,ll fit failed -> ', str(int(round(ids[ii],0))),',', ll)
            continue

        
        ##################### FITTING WITH EMCEE#########################
        ####### Input
        x=1e4*lam_cut
        y=Alam_Av_cut
        yerr=0.1*y

        nll = lambda *args: -log_likelihood_2p(*args)
        initial = np.array([1,1])#popt+ 0.1 * np.random.randn(4)
        print('initial', initial)
        soln = minimize(nll, initial, args=(x, y, yerr))
        pos = soln.x + 1e-4 * np.random.randn(32, 2)
        nwalkers, ndim = pos.shape
        print ('\nEMCEE STARTS AT -->', initial)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_2p, args=(x, y, yerr))
        sampler.run_mcmc(pos, 1000, progress=True);
        labels = ["$B$", "$\\delta$"]

        #tau = sampler.get_autocorr_time()

        flat_samples_2p = sampler.get_chain(discard=200, thin=15, flat=True)
        #print(flat_samples_2p.shape)

        fig, axes = pyplot.subplots(2, figsize=(14, 10), sharex=True)
        for i in range(ndim):
            ax = axes[i]
            ax.plot(flat_samples_2p[:, i], "k", alpha=0.6)
            ax.set_xlim(0, len(flat_samples_2p))
            ax.set_ylabel(labels[i],fontsize=18)
            ax.tick_params(axis='both', which='major', labelsize=16,pad=5)
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number");
        #pyplot.show()
        plt.close()
        fig = corner.corner(flat_samples_2p, labels=labels,
                               quantiles=[0.16, 0.5, 0.84],
                               show_titles=True, title_kwargs= {"fontsize": 15}, plot_contours=True, fill_contours=True,smooth=1.2, labelpad=0.03)

        print('\n #### CURVE FIT -- EMCEE ')
        for i in range(ndim):
            mcmc = np.percentile(flat_samples_2p[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            print(mcmc[1], q[0], q[1], '\n')
                
        # Extract the axes
        axes = numpy.array(fig.axes).reshape((ndim, ndim))

        i=0
        # Loop over the diagonal
        for i in range(ndim):
            ax = axes[i, i]
            ax.axvline(np.percentile(flat_samples_2p[:, i], [16, 50, 84])[1], color="black",linewidth=2.0)
            ## contraints from curve fit
            #ax.axvline(popt[i], color='teal', linewidth=2.5, alpha=0.35)
            #ax.axvline(popt[i]+np.sqrt(pcov[i,i]), color='teal', linewidth=2., ls='--', alpha=0.35)
            #ax.axvline(popt[i]-np.sqrt(pcov[i,i]), color='teal', linewidth=2., ls='--', alpha=0.35)
            #ax.axvline()
            ## MW value
            #ax.axvline(MW_par[i], color=colors_models[2], linewidth=2, alpha=0.4, ls='--')
            

        ##############################
        ### Output PLOT Att. curve
        ##############################
        att_axis = fig.add_axes([0.62, 0.66, 0.3, 0.3])

        ## Input attenuation curve
        att_axis.plot(x, y, color=colors_Av_TNG(-Av/0.7), lw=3., label='id='+str(int(ids[ii]))+', los='+str(ll), alpha=0.85, zorder=-1000)
        
        #att_axis.scatter(x*1e4, y, color='black', marker='|', zorder=-1000,alpha=0.5, s=70)

        ## Output curve-fit (no bounds apart from c4)
        #att_axis.scatter(lam_cut*1e4,fit_nb,marker='s', facecolor='None', alpha=0.55, label='curve fit', edgecolors='teal',linewidth=0.75,s=40,zorder=-1000)

        ## Output fit EMCEE
        y_fit_mcmc_2p=Att_Curve_2param(x,B=np.percentile(flat_samples_2p[:, 0], [50]),delta=np.percentile(flat_samples_2p[:, 1], [50]))
        y_fit_mcmc_u_2p=Att_Curve_2param(x,B=np.percentile(flat_samples_2p[:, 0], [16]),delta=np.percentile(flat_samples_2p[:, 1], [16]))
        y_fit_mcmc_l_2p=Att_Curve_2param(x,B=np.percentile(flat_samples_2p[:, 0], [84]),delta=np.percentile(flat_samples_2p[:, 1], [84]))
        
        ## saving best fit params
        #best_fit_par.write(str(ll)+', '+str(-Av)+', '+str(float(np.percentile(flat_samples_2p[:, 0], [50])))+', '+str(float(np.percentile(flat_samples_2p[:, 0], [16])))+', '+str(float(np.percentile(flat_samples_2p[:, 0], [84])))+', '+str(float(np.percentile(flat_samples_2p[:, 1], [50])))+', '+str(float(np.percentile(flat_samples_2p[:, 1], [16])))+', '+str(float(np.percentile(flat_samples_2p[:, 1], [84])))+', '+str( float(np.percentile(flat_samples_2p[:, 2], [50])))+', '+str(float(np.percentile(flat_samples_2p[:, 2], [16])))+', '+str( float(np.percentile(flat_samples_2p[:, 2], [84])))+', '+str(float(np.percentile(flat_samples_2p[:, 3], [50])))+', '+str( float(np.percentile(flat_samples_2p[:, 3], [16])))+', '+str(float(np.percentile(flat_samples_2p[:, 3], [84])))+'\n')
        
        
        ##
        #pyplot.scatter(x*1e4,y_fit_mcmc,color='black',label='Emcee',alpha=0.4, edgecolors='black',linewidth=0.8,marker='o',s=52)
        pyplot.fill_between(x,y_fit_mcmc_l_2p,y_fit_mcmc_u_2p,edgecolor='r',linewidth=2.,zorder=1000,label='EMCEE fit - 2 par',hatch='\\',facecolor='none',ls=':')
        
        pyplot.fill_between(x,y_fit_mcmc_l,y_fit_mcmc_u,edgecolor='b',linewidth=2.,zorder=1000,label='EMCEE fit - 4 par',hatch='\\',facecolor='none',ls=':')
        
        
        
        '''
        ### Plotting known attenuation curves
        att_axis.plot(lam*1e4, Li_08(lam, 0., 0., 0., 0., 'MW'), color='black', label='MW',lw=1.5,ls='--',zorder=10000000)
        att_axis.plot(lam*1e4, Li_08(lam, 0., 0., 0., 0., 'SMC'), color=colors_models[3], ls=':', label='SMC',lw=1.5,zorder=10000000)
        att_axis.plot(lam*1e4, Li_08(lam, 0., 0., 0., 0., 'LMC'), color=colors_models[1], ls='-.', label='LMC',lw=1.5,zorder=10000000)
        att_axis.plot(lam*1e4, Li_08(lam, 0., 0., 0., 0., 'Calzetti'), color=colors_models[2], dashes=[1.5,1.5], label='Calzetti',lw=1.5,zorder=10000000)
        '''
        if False:#ii==13 or ii==247:#show all los
            Av_arr=[]
            for pp in range(n_los):
                ## eliminating null values in the flux ratios
                Fr_arr=Flux_ratio[pp]
                Alam_arr=-2.5*np.log10(Fr_arr[Fr_arr>0])
                lam_arr=lam[Fr_arr>0]
                ### Normalizing ad A_lambda to Av
                v_index = find_nearest(lam_arr,0.551)
                Av = Alam_arr[v_index]
                Av_arr=np.append(Av_arr,-Av)
                Alam_Av_arr=Alam_arr/Av
                ## Plot attenuation curve
                att_axis.plot(lam_arr*1e4, Alam_Av_arr, color=colors_Av_TNG(-Av/0.7), lw=1.3,zorder=-100,alpha=0.9/np.sqrt(float(ii)))
                #att_axis.text(6000,2*float(pp)/51.+2.,str(round(-Av,1)),color=colors_Av_TNG(-Av),fontsize=10)
            
            ####### colorbar Av ########
            num=100
            print('\nRange of Av values for this source -->', Av_arr.min(),Av_arr.max())
            bar=np.zeros((1,num))
            bar[0][:]=np.linspace(0.01,0.7,num)
            cbaxis = fig.add_axes([0.77, 0.74, 0.16, 0.015])
            ext=(bar.min(), bar.max(),0.0,0.8)
            cbaxis.imshow(bar,cmap='magma_r',extent=ext,aspect='auto',vmin=bar.min(),vmax=bar.max(),alpha=0.8)
            cbaxis.set_xlabel('$A_v$',fontsize=16)
            cbaxis.tick_params(axis='both', which='major', labelsize=14, pad=5)
            cbaxis.set_yticks([])
            #cbaxis.set_xticks([0.6,2,6.52])
                        
        
        ### Plot details
        att_axis.set_xlabel(r"$\lambda [\dot {A}]$", fontsize=14)
        att_axis.set_ylabel(r"A$_{\lambda}/$A$_{V}$", fontsize=14)
        att_axis.grid(alpha=0.4,ls='--')
        att_axis.legend(fontsize=14,frameon=False)
        #Chosen ticks: Bump, poi FILTERS: UV: U;     Visible: G, R;     NIR: I, Z
        att_axis.set_xticks([2175, 4770., 7625.])
        att_axis.set_ylim(0,20)
        att_axis.set_xlim(1e3,9.2e3)
        att_axis.tick_params(axis='both', which='major', labelsize=13, pad=5)
        
        plt.subplots_adjust(wspace=0., hspace=0.)
        #plt.tight_layout()
        
        #plt.show()
        #plt.savefig('/Users/lsommovigo/Desktop/PROJECT-LTU/Overleaf_LTU_Att_Curves/Ref report/corner_id'+str(int(round(ids[ii],0)))+'_los'+str(int(round(ll,0)))+'.png')
        
        plt.close()
        
        
        
        ###########################
        ### Compute difference and save
        # At the end of the ll loop, store residuals and c3 values
        if ll == 0:
            residuals_4p_all = np.zeros((n_los, len(lam_cut)))
            residuals_2p_all = np.zeros((n_los, len(lam_cut)))
            c3_all = np.zeros(n_los)
            Av_all = np.zeros(n_los)
            A_lam_input=np.zeros((n_los, len(lam_cut)))
            
        # Store residuals and best-fit c3
        residuals_4p_all[ll, :] = y_fit_mcmc - Alam_Av_cut#/Alam_Av_cut
        residuals_2p_all[ll, :] = y_fit_mcmc_2p - Alam_Av_cut#/Alam_Av_cut
        A_lam_input[ll, :] = Alam_cut#/Alam_Av_cut
        c3_all[ll] = np.percentile(flat_samples_4p[:, 2], [50])
        Av_all[ll] = Av
        


# After all LoS (ll) are done, generate combined plot
fig, axs = plt.subplots(1, 2, figsize=(20, 8), sharex=True,sharey=True)
cmap = plt.cm.viridis
cmap_1 = plt.cm.Greys
norm = plt.Normalize(vmin=np.min(c3_all), vmax=np.max(c3_all))
for ll in range(n_los):
    axs[0].plot(lam_cut * 1e4, residuals_4p_all[ll], color=cmap(norm(c3_all[ll])), lw=2.5)
    axs[1].plot(lam_cut * 1e4, residuals_2p_all[ll], color=cmap(norm(c3_all[ll])), lw=2.5)
    #axs[1].plot(lam_cut * 1e4, A_lam_input[ll], color=cmap_1(norm(c3_all[ll])), lw=0.5, alpha=0.2,zorder=-100)
    #axs[0].plot(lam_cut * 1e4, A_lam_input[ll], color=cmap_1(norm(c3_all[ll])), lw=0.5, alpha=0.2, zorder=-100)
axs[0].set_ylabel(r"$(A_{\lambda}/A_V)_{fit} - (A_{\lambda}/A_V)_{inp}$", fontsize=20)
#axs[1].set_ylabel(r"$A_{2p} - A_{\mathrm{input}}$ tr", fontsize=20)
axs[1].set_xlabel(r"Wavelength [$\AA$]", fontsize=20)
axs[0].set_xlabel(r"Wavelength [$\AA$]", fontsize=20)
#axs[0].set_ylim(0.1,5.5)
#axs[0].set_ylim(0.1,5.5)
axs[1].text(6000,3, '2 Parameters (Salim+18)', fontsize=18)
axs[0].text(6000,3,'4 Parameters (Li+08)', fontsize=18)
axs[0].grid(True, alpha=0.3)
axs[1].grid(True, alpha=0.3)

for ax in axs:
    ax.grid(True, alpha=0.3)

# Create a new axis for the colorbar to the right of the subplots
from mpl_toolkits.axes_grid1 import make_axes_locatable

divider = make_axes_locatable(axs[1])
cax = divider.append_axes("right", size="4.5%", pad=0.1)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, cax=cax)
cbar.set_label(r"Best-fit $c_3$", fontsize=18)

plt.tight_layout()
# Optional save:
# plt.savefig('/Users/lsommovigo/Desktop/PROJECT-LTU/attenuation_curves-TNG50/Residuals_vs_c3_combined.png')
plt.show()





# After all LoS (ll) are done, generate combined plot
fig, axs = plt.subplots(1, 2, figsize=(20, 8), sharex=True,sharey=True)
cmap = plt.cm.viridis
cmap_1 = plt.cm.Greys
norm = plt.Normalize(vmin=np.min(c3_all), vmax=np.max(c3_all))

for ll in range(n_los):
    axs[0].plot(lam_cut * 1e4, 10**(0.4*Av_all[ll]*residuals_4p_all[ll]), color=cmap(norm(c3_all[ll])), lw=2.5)
    axs[1].plot(lam_cut * 1e4, 10**(0.4*Av_all[ll]*residuals_2p_all[ll]), color=cmap(norm(c3_all[ll])), lw=2.5)
    #axs[1].plot(lam_cut * 1e4, A_lam_input[ll], color=cmap_1(norm(c3_all[ll])), lw=0.5, alpha=0.2,zorder=-100)
    #axs[0].plot(lam_cut * 1e4, A_lam_input[ll], color=cmap_1(norm(c3_all[ll])), lw=0.5, alpha=0.2, zorder=-100)

axs[0].set_ylabel(r"$10^{0.4\ (A_{\lambda, fit} - A_{\lambda,inp})}$", fontsize=20)
#axs[1].set_ylabel(r"$A_{2p} - A_{\mathrm{input}}$ tr", fontsize=20)
axs[1].set_xlabel(r"Wavelength [$\AA$]", fontsize=20)
axs[0].set_xlabel(r"Wavelength [$\AA$]", fontsize=20)
#axs[0].set_ylim(0.1,5.5)
#axs[0].set_ylim(0.1,5.5)
#axs[1].set_yscale('log')
#axs[0].set_yscale('log')
axs[1].text(6000,1.1, '2 Parameters (Salim+18)', fontsize=18)
axs[0].text(6000,1.1,'4 Parameters (Li+08)', fontsize=18)
axs[0].grid(True, alpha=0.3)
axs[1].grid(True, alpha=0.3)
for ax in axs:
    ax.grid(True, alpha=0.3)
# Create a new axis for the colorbar to the right of the subplots
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(axs[1])
cax = divider.append_axes("right", size="4.5%", pad=0.1)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, cax=cax)
cbar.set_label(r"$c_3$", fontsize=18)
plt.tight_layout()
# Optional save:
# plt.savefig('/Users/lsommovigo/Desktop/PROJECT-LTU/attenuation_curves-TNG50/Residuals_vs_c3_combined.png')
plt.show()






# After all LoS (ll) are done, generate combined plot
fig, ax = plt.subplots(figsize=(20, 8))
cmap = plt.cm.viridis
cmap_1 = plt.cm.Greys
norm = plt.Normalize(vmin=np.min(c3_all), vmax=np.max(c3_all))

for ll in range(n_los):
    ax.plot(lam_cut * 1e4, residuals_4p_all[ll]/residuals_2p_all[ll], color=cmap(norm(c3_all[ll])), lw=2.5)
ax.set_ylabel(r"$(A_{\lambda, fit} - A_{\lambda,inp})_{4p}/(A_{\lambda, fit} - A_{\lambda,inp})_{2p}$", fontsize=17)
#axs[1].set_ylabel(r"$A_{2p} - A_{\mathrm{input}}$ tr", fontsize=20)
ax.set_xlabel(r"Wavelength [$\AA$]", fontsize=17)
#ax.set_yscale('log')
ax.grid(True, alpha=0.3)
# Create a new axis for the colorbar to the right of the subplots
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="4.5%", pad=0.1)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, cax=cax)
cbar.set_label(r"$c_3$", fontsize=18)
plt.tight_layout()
# Optional save:
# plt.savefig('/Users/lsommovigo/Desktop/PROJECT-LTU/attenuation_curves-TNG50/Residuals_vs_c3_combined.png')
plt.show()
