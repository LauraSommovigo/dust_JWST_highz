from typing import Literal

import astropy.constants as apc
import numpy as np
from numpy.typing import NDArray


def compute_l1500_steps_kss(sfh: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute the 1500 Å monochromatic luminosity using the KS+98 scaling.

    This is a direct conversion with no convolution with SB99 models.
    L_1500 [erg/s/Hz] = 7.14e27 * SFR [Msun/yr]

    Parameters
    ----------
    sfh : ndarray
        Star formation rate [M_sun/yr] at each time step.

    Returns
    -------
    ndarray
        UV luminosity at 1500 Å in [erg/s/Hz].

    Notes
    -----
    This conversion is from Kennicutt (1998) for continuous star formation.

    """
    l1500 = 7.14e27 * sfh  # [erg/s/Hz], L_nu
    return np.array(l1500)


def l1500_to_muv_conv(l_1500_nu: float | NDArray[np.floating]) -> float | NDArray[np.floating]:
    """Convert L_1500 [erg/s/Hz] (L_nu) to absolute AB magnitude at 1500 Å.

    Parameters
    ----------
    l_1500_nu : float or ndarray
        Luminosity in [erg/s/Hz].

    Returns
    -------
    float or ndarray
        Absolute AB magnitude at 1500 Å.

    """
    return -2.5 * np.log10(l_1500_nu) + 51.63


def l1500_lambda_to_lnu(l1500_ang: float | NDArray[np.floating]) -> float | NDArray[np.floating]:
    """Convert L_1500 [erg/s/Å] (L_lambda) to L_nu [erg/s/Hz] at 1500 Å.

    Parameters
    ----------
    l1500_ang : float or ndarray
        Luminosity per unit wavelength in [erg/s/Å].

    Returns
    -------
    float or ndarray
        Luminosity per unit frequency in [erg/s/Hz].

    """
    lambda_ang = 1500.0  # wavelength in Angstrom
    return l1500_ang * lambda_ang**2 * 1e-8 / apc.c


def _compute_l1500_steps_ks98(sfh: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute the 1500 Å monochromatic luminosity using the KS+98 scaling.

    This is a direct conversion with no convolution with SB99 models.
    L_1500 [erg/s/Hz] = 7.14e27 * SFR [Msun/yr]

    Parameters
    ----------
    sfh : ndarray
        Star formation rate [M_sun/yr] at each time step.

    Returns
    -------
    ndarray
        UV luminosity at 1500 Å in [erg/s/Hz].

    Notes
    -----
    This conversion is from Kennicutt (1998) for continuous star formation.

    """
    l1500 = 7.14e27 * sfh  # [erg/s/Hz], L_nu
    return l1500


def _compute_l1500_steps_sb99(
    l1500: NDArray[np.floating],
    age: NDArray[np.floating],
    tstep: float,
    sfh: NDArray[np.floating],
    time_yr_l1500: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute the monochromatic luminosity at 1500 Å (L_1500) [erg/s/Hz] at each time step.

    Convolves the star formation history with the SB99 single stellar population (SSP) output.
    For each time step, sum the contribution from all previous bursts, accounting for their age.

    Parameters
    ----------
    age : ndarray
        Age grid in Myr.
    tstep : float
        Time step size in Myr.
    sfh : ndarray
        Star formation rate history [M_sun/yr] at each time step.
    time_yr_l1500 : ndarray
        SSP times in years corresponding to the tabulated L1500 values.
    l1500 : ndarray
        log10(L_1500) [erg/s/Å] for an instantaneous burst of 1e6 M_sun.

    Returns
    -------
    ndarray
        Total L1500 [erg/s/Hz] at each age step.

    """
    l1500_arr = []
    for ind in range(len(age)):
        l1500_step = 0.0
        for t in range(ind + 1):
            age_delay = 1e6 * tstep * (ind - t)  # time since burst, in years
            # interpolate to get L_1500 for this delay (per 1e6 Msun burst)
            l1500_ssp = 10 ** np.interp(age_delay, time_yr_l1500, l1500)
            mass_formed = sfh[t] * tstep * 1e6  # Msun formed in this burst (tstep in Myr!)
            l1500_step += l1500_ssp * (mass_formed / 1e6)  # rescale to burst mass
        l1500_arr.append(l1500_step)
    return l1500_lambda_to_lnu(np.array(l1500_arr))


def compute_l1500_steps(
    l1500: NDArray[np.floating],
    *args,
    method: Literal["SB99", "KS98"] = "SB99",
    **kwargs,
) -> NDArray[np.floating]:
    """Compute the 1500 Å monochromatic luminosity using the specified method.

    This is a dispatcher function that calls the appropriate implementation
    based on the chosen method.

    Parameters
    ----------
    l1500 : ndarray
        For SB99 method: log10(L_1500) [erg/s/Å] for an instantaneous burst of 1e6 M_sun.
        For KS98 method: Star formation rate [M_sun/yr] at each time step.
    *args
        Additional positional arguments passed to the underlying method.
        For SB99: age, tstep, sfh, time_yr_l1500
    method : {"SB99", "KS98"}
        Method for computing L1500:
        - "SB99": Convolve SFH with Starburst99 single stellar population models
        - "KS98": Use Kennicutt (1998) direct scaling relation
    **kwargs
        Additional keyword arguments passed to the underlying method.

    Returns
    -------
    ndarray
        UV luminosity at 1500 Å in [erg/s/Hz].


    See Also
    --------
    _compute_l1500_steps_sb99 : SB99-based computation
    _compute_l1500_steps_ks98 : KS98-based computation

    """
    if method == "SB99":
        return _compute_l1500_steps_sb99(l1500, *args, **kwargs)
    elif method == "KS98":
        return _compute_l1500_steps_ks98(l1500)
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_dotnion_steps(age, tstep, sfh, time_yr_nion, log_dot_nion):
    """Computes the instantaneous ionizing photon production rate (dot_Nion) at each age step.

    Parameters
    ----------
    age : array-like
        Age grid (in Myr).
    tstep : float
        Time step size (in Myr).
    sfh : array-like
        Star formation rate history (Msun/yr) at each age step.
    time_yr_nion : array-like
        SSP times (in yr) corresponding to the tabulated log_dot_nion.
    log_dot_nion : array-like
        log10 of the ionizing photon rate [photons/s] from stellar population models.

    Returns
    -------
    dotNion_arr : np.ndarray
        Array of instantaneous ionizing photon rate [photons/s] at each age step.

    """
    dotnion_arr = []
    for ind in range(len(age)):
        dotnion_step = 0.0
        for t in range(ind + 1):
            # Interpolate the SSP ionizing rate for the elapsed time since formation (in yr)
            dt_yr = 1e6 * (tstep * (ind - t))
            dotnion_ssp = 10 ** np.interp(dt_yr, time_yr_nion, log_dot_nion)
            dotnion_step += dotnion_ssp * sfh[t] * tstep
        dotnion_arr.append(dotnion_step)
    return np.array(dotnion_arr)
