import astropy.units as apu
import numpy as np
from astropy.cosmology import z_at_value
from numpy.typing import NDArray

from .cosmology import cosmo
from .halo import halo_mass_accretion_rate


def halo_to_stellar_mass(
    halo_mass: float | NDArray[np.floating],
    baryon_fraction: float,
    epsilon: float,
    alpha: float = 0.0,
) -> float | NDArray[np.floating]:
    """Convert halo mass to stellar mass with optional power-law scaling.

    Parameters
    ----------
    halo_mass : float or ndarray
        Halo mass in solar masses (M_sun).
    baryon_fraction : float
        Baryon fraction f_b = Omega_b / Omega_m at the given redshift.
    epsilon : float
        Star formation efficiency parameter (dimensionless).
    alpha : float, optional
        Power-law index for mass-dependent efficiency. Default is 0.0,
        which gives simple linear scaling. Non-zero values apply
        mass-dependent efficiency scaling.

    Returns
    -------
    float or ndarray
        Stellar mass in solar masses (M_sun).

    Notes
    -----
    The stellar mass is computed as:
        M_star = epsilon * f_b * M_halo * (M_halo / 10^10)^alpha

    When alpha = 0 (default), this reduces to simple linear scaling.
    For alpha > 0, efficiency increases for more massive halos.
    For alpha < 0, efficiency decreases for more massive halos.

    """
    return epsilon * baryon_fraction * halo_mass * (halo_mass / 1e10) ** alpha


def star_formation_history(
    halo_mass: float,
    redshift: float,
    time_step: float,
    eps: float,
    alpha: float = 0.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Build star formation history by integrating halo accretion until target stellar mass is reached.

    Parameters
    ----------
    halo_mass : float
        Halo mass in solar masses (M_sun).
    redshift : float
        Final redshift at which the stellar mass is evaluated.
    time_step : float
        Time step size in Myr for the integration.
    eps : float
        Star formation efficiency parameter (dimensionless).
    alpha : float, optional
        Power-law index for mass-dependent efficiency. Default is 0.0.

    Returns
    -------
    sfr_t : ndarray
        Star formation rate history [M_sun/yr] at each time step,
        ordered from earliest to latest time.
    log_mst : ndarray
        Log10 of cumulative stellar mass [M_sun] at each time step.
    time_sfh : ndarray
        Time grid [Myr] corresponding to the SFH.

    Notes
    -----
    This function integrates backwards in time from the given redshift,
    using the halo mass accretion rate from `dMhdt_GUREFT` to compute
    the star formation rate at each step. The integration continues until
    the cumulative stellar mass reaches the target stellar mass computed
    from the halo mass at the final redshift.

    The star formation rate at each step is computed as:
        SFR = epsilon * f_b * dM_h/dt * (M_h / 10^10)^alpha

    where f_b is the baryon fraction at that redshift.

    """
    i = 0
    log_mst = [0.0]
    sfh_z = []
    reds_arr = []
    # Mstar at the given z
    fb = cosmo.Ob(redshift) / cosmo.Om(redshift)
    stellar_mass = halo_to_stellar_mass(halo_mass, fb, eps, alpha)
    while log_mst[-1] < np.log10(stellar_mass):
        reds = z_at_value(
            cosmo.age, (cosmo.age(redshift).value - 1e-3 * time_step * i) * apu.Gyr, method="bounded"
        ).value
        reds_arr.append(reds)
        sfr_val = eps * fb * halo_mass_accretion_rate(halo_mass, reds, method="GUREFT") * (halo_mass / 1e10) ** alpha
        sfh_z.append(sfr_val)
        log_mst.append(np.log10(time_step * 1e6 * np.sum(sfh_z)))
        i += 1
    time_sfh = np.linspace(0, i * time_step, i)
    sfr_t = np.flip(np.array(sfh_z))
    log_mst = np.array(log_mst[1:])
    return sfr_t, log_mst, time_sfh
