import numpy as np
from scipy.optimize import root_scalar
from numpy.typing import NDArray


def stellar_mass_to_metallicity(
        model: str, 
        stellar_mass: float | NDArray[np.floating],
    ) -> float | NDArray[np.floating]:
    
    """
    Parameters
    ----------
    model: string
        Model to use for the stellar mass-metallicity relation, either "FIRE2" or "SC"
    stellar_mass: float
        Stellar mass of the galaxy [log10(M/M_sun)]
        

    Returns
    -------
    metallicity: float
        Metallicity of the galaxy [log10(Z/Z_sun)]
        

    Notes
    -----
    Using the scaling relation from the FIRE-2 simulations or the Santa Cruz SAM

    References
    ----------
    Marszewsi et al. (2024), Somerville et al. (2015)

    """

    if model == "FIRE2":
        a = 0.37
        b = -4.3

    if model == "SC":
        a = 0.45
        b = -4.7

    return a * stellar_mass + b



def mc_evolving_IMF(
        redshift: float, 
        metallicity: float, 
        imf_minimum_mass: float, 
        imf_maximum_mass: float, 
) -> float:
    
    """
    Parameters
    ----------
    redshift: float
        Redshift of the galaxy
    metallicity: float
        Metallicity of the galaxy [log10(Z/Z_sun)]
    imf_minimum_mass: float
        Minimum initial mass of stars (M_sun)
    imf_maximum_mass: float
        Maximum initial mass of stars (M_sun)

        
    Returns
    -------
    Mc: float
        Cutoff mass, defined as the initial stellar mass of the IMF slope change (M_sun)
        

    Notes
    -----
    Calculation of the cutoff mass for a two-component IMF: 
    - first component: salpeter slope (2.35)
    - second component: log-flat (1)
    The cutoff mass depends on redshift and stellar metallicity

    
    References
    ---------
    Evolving IMF recipe from Cueto et al. (2024), based on Chon et al. (2022) simulation results


    """
    
    # fraction of stellar mass in the log-flat IMF component, depends on redshift and metallicity
    x = 1 + metallicity 
    f_massive =  1.07*(1-(2**x)) + 0.04*redshift*(2.67**x)
        
    # imposing 0 < f_massive < 1
    if f_massive <= 0: 
        f_massive = 1e-6 #Mc=100
    elif f_massive >= 1: 
       f_massive = 1 - 1e-6 #Mc=0.1
       
    Mi = imf_minimum_mass
    Mf = imf_maximum_mass

    # cutoff mass estimation from imposing the fraction of stellar mass in the log-flat IMF component
    def equation_for_Mc(Mc):
         # integral of first component
         I1 = (Mi**-0.35 - Mc**-0.35) / 0.35 
         # integral of second component, imposing continuity between the two 
         I2 = Mc**-1.35 * (Mf - Mc) 
         return I2 / (I1 + I2) - f_massive
    
    sol = root_scalar(equation_for_Mc, bracket=[Mi + 1e-6, Mf - 1e-6])
    
    return sol.root



def select_SB99_tables(
        metallicity: float, 
        imf_cutoff_mass: float, 
        imf_maximum_mass: float, 
        SN_maximum_mass: float, 
) -> tuple[str, str, str, str]:

    """
    Parameters
    ----------
    metallicity: float
        Metallicity of the stellar population [log10(Z/Zsun)]
    imf_cutoff_mass: float
        Cutoff mass, defined as the initial mass of the IMF slope change (M_sun)
    imf_maximum_mass: float
        Maximum initial mass of stars (M_sun)
    SN_maximum_mass: float
        Maximum initial mass of stars that explode as supernovae (M_sun)

    Returns
    -------
    filename_spectra: string
        Name of the SB99 table with the spectrum fluxes
    filename_wavelengths: string
        Name of the SB99 table with the wavelengths
    filename_Ni: string
        Name of the SB99 table with the number of ionizing photons
    filename_snr: string
        Name of the SB99 table with the supernova rate

    Notes
    -----
    Selects filenames of pyStarburst99 (SB99) tables for input parameters

    """

    directory = "pySB99_models/" 
    Zsun = 0.02 #Anders&Grevesse89, used in FIRE-2
    abs_Z = 10**(metallicity) *Zsun

    # SB99 tables parameters 
    sb99_Z = np.array([1e-5, 4e-4, 2e-3, 6e-3, 1.4e-2, 1])
    Z_labels = ["1e-5", "4e-4", "2e-3", "6e-3", "1.4e-2", "1"]
    sb99_Mc_by_Z = {
        "1e-5":  np.array([2, 5, 10]),
        "4e-4":  np.array([2, 5, 10, 15]),
        "2e-3":  np.array([5, 10, 15, 20, 30, 50]),
        "6e-3":  np.array([10, 20, 30, 50, 70, 100]),
        "1.4e-2": np.array([10, 20, 30, 50, 70, 100]),
        "1":     np.array([10, 20, 30, 50, 70, 100]),
    }

    # Find closest match in the SB99 tables 
    Z = Z_labels[np.argmin(np.abs(sb99_Z - abs_Z))]  
    sb99_Mc = sb99_Mc_by_Z[Z]                                   
    Mc = int(sb99_Mc[np.argmin(np.abs(sb99_Mc - imf_cutoff_mass))])
    SN = int(SN_maximum_mass)

    directory = f"pySB99_models/pySB99_mc{Mc}_z{Z}_sn{SN}"   
    
    filename_spectra = f"{directory}/pySB_SED_stellar.npy"
    filename_wavelengths = f"{directory}/SED_wavelength.txt"
    filename_Ni = f"{directory}/ion_flux_HI.txt"
    filename_snr = f"{directory}/SNrate.txt"

    return filename_spectra, filename_wavelengths, filename_Ni, filename_snr




def compute_L1500(
        wavelength: NDArray[np.floating],
        spectra: NDArray[np.floating],
        wave_center = 1500.0, wave_window=25.0, log_flux=True
    ) -> NDArray[np.floating]:
    
    """
    Parameters
    ----------
    wavelength: array
        Wavelengths (A)
    spectra: array
        Spectra fluxes (erg/s/A)
    wave_center: float
        Central wavelength of the window to compute the UV luminosity (A)
    wave_window: float
        Width of the window to compute the UV luminosity (A)
    log_flux: bool
        if True, spectra are in log units

    Returns
    -------
    L1500: array
        UV luminosity at 1500A (erg/s)

    Notes
    -----
    Computes the UV luminosity at 1500A from the SB99 spectra

    """
    
    wave_mask = (wavelength >= wave_center - wave_window) & (wavelength <= wave_center + wave_window)
    flux = 10**spectra if log_flux else spectra
    l1500 = np.trapezoid(flux[:, wave_mask], wavelength[wave_mask], axis=1)

    return l1500 / (wave_center + wave_window - (wave_center - wave_window))


def evolving_SFE(
        halo_mass: float | NDArray[np.floating], 
        redshift:float
    ) -> float | NDArray[np.floating]:
    
    """
    Parameters
    ----------
    halo_mass: float
        Halo mass of the galaxy (M_sun)
    redshift: float
        Redshift of the galaxy

    Returns
    -------
    SFE: float
        Star formation efficiency

    Notes
    -----
    Formula from Yung et al. (2025)
    """

    if redshift == 14: 
            e0 = 0.40
            Mo = 10**10.40
            alpha = 0.97
            beta = 0.39
    else:
            e0 = 0.46
            Mo = 10**10.94
            alpha = 1.17
            beta = 0.41

    SFE = (2*e0)/( (halo_mass/Mo)**(-alpha) + (halo_mass/Mo)**(beta) )

    return SFE