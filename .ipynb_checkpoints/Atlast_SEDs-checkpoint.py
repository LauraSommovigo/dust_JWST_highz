# Import necessary modules for scientific calculations and plotting
from scipy.integrate import quad
import numpy as np
import scipy
from matplotlib import pyplot as plt
from astropy.cosmology import Planck15 as cosmo
from matplotlib import gridspec

# Useful constants defined in CGS units
Msol = 1.99e33  # Solar mass in grams
Lsol = 3.9e33  # Solar luminosity in erg/s
pc = 3.09e18  # Parsec in centimeters
mp = 1.66e-24  # Proton mass in grams
kb = 1.38064852e-16  # Boltzmann constant in erg/K
h = 6.63e-27  # Planck constant in erg*s
c = 3e10  # Speed of light in cm/s
To = 2.725  # Cosmic Microwave Background temperature in Kelvin

# Dust model parameters for Milky Way dust based on Draine (2003)
bd = 2.0  # Power-law index for dust opacity
kabs = 10.41  # Dust absorption coefficient
nu_abs = 1.9e12  # Reference frequency for dust opacity in Hz

################### FUNCTIONS ########################

def T_CMB_corr(Td, redshift):
    # Applies correction due to CMB heating (from da Cunha+13)
    # Corrects dust temperature due to heating by the Cosmic Microwave Background
    return (Td**(4 + bd) + (To**(4 + bd)) * (((1.0 + redshift)**(4 + bd)) - 1.0))**(1.0 / (4.0 + bd))

def freq_convert(lamda):
    # Converts wavelength from microns to frequency in Hz
    return c * 1e4 / lamda

def temp_convert(nu):
    # Converts frequency (Hz) to effective temperature for Planck function
    return h * nu / kb

def kabs_beta(nu, betad):
    # Calculates dust opacity as a power-law dependence on frequency, based on Draine (2003)
    return kabs * (nu / nu_abs)**betad

def T_CMB(redshift):
    # Returns the CMB temperature at a given redshift
    return To * (1 + redshift)

def B_nu(T_obs, T):
    # Blackbody function for dust emission
    return 1.0 / (np.exp(T_obs / T) - 1.0)

def F_nu(nu, Td, logMdust, betad, redshift):
    # Computes the infrared flux in milliJansky (mJy) unit, given dust temperature, mass, and redshift
    dL = cosmo.luminosity_distance(redshift).value  # Luminosity distance in Mpc
    const = (2 * h * (nu**3) / c**2) * Msol * 1e26 / (pc * 1e6)**2.0  # Pre-factor for flux calculation
    T_obs = temp_convert(nu)  # Convert frequency to temperature
    # Calculate flux accounting for dust emission and CMB correction
    F_nu = ((1.0 + redshift) * (10**logMdust) * kabs_beta(nu, bd) * const *
            (B_nu(T_obs, T_CMB_corr(Td, redshift)) - B_nu(T_obs, T_CMB(redshift))) / (dL**2.0))
    return F_nu

def tdepl_Tacconi20(redshift):
    # Depletion time as a function of redshift, based on Tacconi et al. (2020)
    return 1.6 * (0.7 + 0.3 * (1 + redshift)**3.)**(-0.5)  # in Gyr

def Td_z(redshift):
    # Computes dust temperature evolution with redshift, model from Sommovigo+22a,b
    # Eq. 10, here we assume solar metallicity and tau_eff= -ln T_1500 = 1
    tau_eff=1.#This corresponds to a transmissivity at 1500 Angstrom of 36%
    gas_metall=1.#Zsun units
    return 29.7 * ((1 - np.exp(-tau_eff)) * tdepl_Tacconi20(redshift)/gas_metall)**(-1. / 6.)

def logL_IR(logMdust, bd, Td):
    # Computes the infrared luminosity in log(Lsun) units
    Teta = ((np.pi * 8.0 / c**2) * (kabs / freq_convert(158.)**bd) *
            ((kb**(4.0 + bd)) / (h**(3.0 + bd))) * scipy.special.zeta(4.0 + bd) *
            scipy.special.gamma(4.0 + bd))
    return logMdust + np.log10(Teta) + (4.0 + bd) * np.log10(Td) + np.log10(Msol / Lsol)

##### Dust model parameters (Weingartner & Draine 2001, Milky Way dust)
bd = 2.03  # Updated power-law index
kabs = 10.41  # Updated absorption coefficient
lamda_abs = 158.0  # Reference wavelength in microns

########### SOURCE properties ###########
redshift = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.3, 8.0, 9.0])  # Array of redshifts
T_dust = Td_z(redshift)  # Fixed dust temperature (can also use Td_z function for evolution with redshift)
logMd = 7.0  # Logarithm of dust mass in solar masses
logLIR = logL_IR(logMd, bd, T_dust)  # Calculate infrared luminosity
print('log (LIR/Lsun) = ', logLIR)

TCMB = To * (1.0 + redshift)  # CMB temperature at each redshift

##########################################
### Plot: same SED at different redshifts
##########################################
f = plt.figure(figsize=(9, 7))
gs = gridspec.GridSpec(2, 1, hspace=0.0)
IRSED_z = f.add_subplot(gs[0, 0])

# IR SEDs: single-temperature grey body for each redshift
lambda_obs = np.logspace(0.4, 4.2, 100)  # Observed wavelength range

for i in range(len(redshift)):
    nu_rf = freq_convert(lambda_obs / (1. + redshift[i]))  # Convert observed lambda to rest-frame frequency
    plt.plot(np.log10(lambda_obs), np.log10(F_nu(nu_rf, T_dust[i], logMd, bd, redshift[i])),
                label='z=' + str(int(redshift[i])))  # Plot SED for each redshift

######## ALMA bands ########
plt.fill_betweenx(np.linspace(-4, 4.3, 100), np.log10(1.1e3) * np.ones(100), np.log10(1.4e3) * np.ones(100), color='gainsboro', alpha=0.2, zorder=-100)
plt.text(np.log10(1.19e3), -2.6, '6', fontsize=14, alpha=0.6)
plt.fill_betweenx(np.linspace(-4,4.3,100), np.log10(1.1e3)*np.ones(100),np.log10(1.4e3)*np.ones(100), color='gainsboro',alpha=0.2,zorder=-100)
plt.text(np.log10(1.19e3),-2.6,'6',fontsize=14,alpha=0.6)
plt.fill_betweenx(np.linspace(-4,4.3,100), np.log10(0.8e3)*np.ones(100),np.log10(1.1e3)*np.ones(100), color='silver',alpha=0.2,zorder=-100)
plt.text(np.log10(0.9e3),-2.6,'7',fontsize=14,alpha=0.6)
plt.fill_betweenx(np.linspace(-4,4.3,100), np.log10(0.6e3)*np.ones(100),np.log10(0.8e3)*np.ones(100), color='grey',alpha=0.2,zorder=-100)
plt.text(np.log10(0.65e3),-2.6,'8',fontsize=14,alpha=0.6)
plt.fill_betweenx(np.linspace(-4,4.3,100), np.log10(0.4e3)*np.ones(100),np.log10(0.5e3)*np.ones(100), color='dimgrey',alpha=0.3,zorder=-100)
plt.text(np.log10(0.41e3),-2.6,'9',fontsize=14,alpha=0.6)
## JWST NIRCAM range ##
plt.text(0.56, -2.6, 'NIRCAM', fontsize=14, alpha=0.7, color='indigo', rotation=90)
plt.fill_betweenx(np.log10(np.linspace(1e-7, 1e4, 100)), np.log10(0.6) * np.ones(100), np.log10(5.3) * np.ones(100), color='indigo', alpha=0.1, zorder=-100)

# Plot labels and legend
plt.xlabel('$\log\ (\lambda_{obs}/ \mathrm{\mu m})$')
plt.ylabel('$\log\ (F_{\\nu}/\mathrm{mJy})$')
plt.ylim(-3, 1.5)
plt.legend(loc='upper left', fontsize=12)

##########################################
### Plot: Different T_dust, same redshift
##########################################
IRSED_LIR = f.add_subplot(gs[1, 0], sharex=IRSED_z)

# Fixed redshift, different dust temperatures
redshift = 7.
T_dust=Td_z(redshift)
nu_rf = freq_convert(lambda_obs / (1. + redshift))

print('Td assumed -->', Td_z(redshift))

plt.plot(np.log10(lambda_obs), np.log10(F_nu(nu_rf, T_dust, logMd, bd, redshift)),
            label='$\log (L_{\\rm IR}/L_{\odot})=$' + str(int(logL_IR(logMd, bd, T_dust))))
plt.plot(np.log10(lambda_obs), np.log10(F_nu(nu_rf, 2. * T_dust, logMd, bd, redshift)),
            label='$\log (L_{\\rm IR}/L_{\odot})=$' + str(int(logL_IR(logMd, bd, 2 * T_dust))))

# Plot labels and legend
plt.xlabel('$\log\ (\lambda_{obs}/ \mathrm{\mu m})$')
plt.ylabel('$\log\ (F_{\\nu}/\mathrm{mJy})$')
plt.ylim(-3, 2.5)
plt.text(4.0, 1.8, 'z=7', fontsize=16)


######## ALMA bands ########
plt.fill_betweenx(np.linspace(-4, 4.3, 100), np.log10(1.1e3) * np.ones(100), np.log10(1.4e3) * np.ones(100), color='gainsboro', alpha=0.2, zorder=-100)
plt.text(np.log10(1.19e3), -2.6, '6', fontsize=14, alpha=0.6)
plt.fill_betweenx(np.linspace(-4,4.3,100), np.log10(1.1e3)*np.ones(100),np.log10(1.4e3)*np.ones(100), color='gainsboro',alpha=0.2,zorder=-100)
plt.text(np.log10(1.19e3),-2.6,'6',fontsize=14,alpha=0.6)
plt.fill_betweenx(np.linspace(-4,4.3,100), np.log10(0.8e3)*np.ones(100),np.log10(1.1e3)*np.ones(100), color='silver',alpha=0.2,zorder=-100)
plt.text(np.log10(0.9e3),-2.6,'7',fontsize=14,alpha=0.6)
plt.fill_betweenx(np.linspace(-4,4.3,100), np.log10(0.6e3)*np.ones(100),np.log10(0.8e3)*np.ones(100), color='grey',alpha=0.2,zorder=-100)
plt.text(np.log10(0.65e3),-2.6,'8',fontsize=14,alpha=0.6)
plt.fill_betweenx(np.linspace(-4,4.3,100), np.log10(0.4e3)*np.ones(100),np.log10(0.5e3)*np.ones(100), color='dimgrey',alpha=0.3,zorder=-100)
plt.text(np.log10(0.41e3),-2.6,'9',fontsize=14,alpha=0.6)
## JWST NIRCAM range ##
plt.text(0.56, -2.6, 'NIRCAM', fontsize=14, alpha=0.7, color='indigo', rotation=90)
plt.fill_betweenx(np.log10(np.linspace(1e-7, 1e4, 100)), np.log10(0.6) * np.ones(100), np.log10(5.3) * np.ones(100), color='indigo', alpha=0.1, zorder=-100)

### REBELS data, z=7
reds_REB=np.array([6.49632623080003, 6.74949312222826, 7.34591349960637, 7.08424192680435, 7.67499921115377, 7.37010028069009, 7.30651374725317, 7.08975543485707, 6.68474280897578, 6.72902239593933, 6.57701183413199, 6.84488626164266, 7.36495347204348])
##log Mdust
logMd_REB=np.array([7.178553842550373, 7.222072062056873, 7.302253265513234, 7.016965485868306, 7.308617989378732, 7.232033622294462, 7.555296005147596, 7.138938465954253, 7.107044369891009, 7.216636655590892, 7.462844752786296, 7.197760252551811, 7.0777444193759385])
## dust temperatures
Td_REB =np.array([41.19473069695478, 47.72080464438527, 44.143263869734, 48.13426947613128, 38.62655724127967, 43.5887546733149, 44., 40.9330560250899, 41.728466054254284, 39.1468047356576, 45.63654800581859, 44.85008744958079, 43.11543190136595])

print('158 micron flux REBELS [microJy]-->', 1e3*F_nu(1900e9, Td_REB, logMd_REB, 2.03, reds_REB))
plt.errorbar(np.log10(158*(reds_REB+1.))*np.ones(len(logMd_REB)),np.log10(F_nu(1900e9, Td_REB, logMd_REB, 2.03, reds_REB)), alpha=0.8, mec='black',mew=0.1,label='REBELS, Obs.', ls='none',zorder=1000,marker='*',ms=10, color='midnightblue')
###



## ALPINE, z=5
reds_ALP=np.array([4.4105, 4.5424, 5.6700, 4.4381, 5.6704, 4.5134, 5.5448, 5.1818, 5.5420, 5.2931, 4.5777, 4.5205, 4.5785, 4.5802, 4.5701, 4.4298, 5.5016, 4.5613, 4.5457, 4.5739, 4.5296])
## dust masses
logMd_ALP=np.array([7.2, 7.75, 7.33, 7.24,  7.38, 7.25, 7.37, 7.13, 7.43, 7.86, 7.49, 7.2, 7.33, 6.77, 7.53, 6.97, 7.79, 8.04,  7.88,  7.4, 8.41])
## dust temperatures
Td_ALP=np.array([51., 37., 48., 54., 53., 48., 47., 55., 49., 39., 47., 50., 52., 60., 42., 48., 37., 45., 59., 59., 25])

plt.errorbar(np.log10(158*(reds_ALP+1.))*np.ones(len(logMd_ALP)),np.log10(F_nu(1900e9, Td_ALP, logMd_ALP, 2.03, reds_ALP)), alpha=0.8, mec='black',mew=0.1,label='ALPINE, Obs.', ls='none',zorder=1000,marker='*',ms=10, color='forestgreen')
###

plt.legend(loc='upper left', fontsize=12)
plt.tight_layout()
plt.show()


'''
def IRX_beta_Calzetti(beta):
    IRX= 1.67 * (10**(0.4 * (2.13 * beta + 5.57)) -1)
    return IRX

def IRX_beta_SMC(beta):
    IRX= 1.79 * (10**(0.4 * (1.07 * beta + 2.79)) -1)
    return IRX
    
def IRX_beta_Reddy(beta):
    IRX= 1.79 * (10**(0.4 * (1.82 * beta + 4.77)) -1)
    return IRX

beta_UV_arr=np.linspace(-5,1.5, 1000)
plt.plot(beta_UV_arr, IRX_beta_Calzetti(beta_UV_arr), label='Calzetti')
plt.plot(beta_UV_arr,IRX_beta_SMC(beta_UV_arr), label='SMC')
plt.plot(beta_UV_arr, IRX_beta_Reddy(beta_UV_arr), label='Reddy')
plt.yscale('log')
plt.legend()
plt.show()
'''
