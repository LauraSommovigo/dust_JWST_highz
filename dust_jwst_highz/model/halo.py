from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .cosmology import cosmo


def growth_suppression(z, method: Literal["RP16", "GUREFT"] = "RP16"):
    """Calculate growth suppression factor g(z) for structure formation.

    Parameters
    ----------
    z : float or ndarray
        Redshift.
    method : {"RP16", "GUREFT"}, optional
        Method for computing growth suppression:
        - "RP16": Rodriguez-Puebla et al. (2016), equation 28
        - "GUREFT": Yung et al. (2023) formulation
        Default is "RP16".

    Returns
    -------
    float or ndarray
        Growth suppression factor g(z).

    Notes
    -----
    The growth suppression factor modifies the linear growth of density
    perturbations in a ΛCDM cosmology, accounting for the effects of
    dark energy on structure formation.

    RP16 formula (eq. 28):
        g(z) = 2.5 * Ωm(z) * a(z) / [Ωm(z) - ΩΛ(z) + (1 + Ωm(z)/2) / (1 + ΩΛ(z)/70)]

    GUREFT formula:
        g(z) = 0.4 * Ωm(z) * a(z) / [Ωm(z)^(4/7) - ΩΛ(z) + (1 + Ωm(z)/2)(1 + ΩΛ(z)/70)]

    References
    ----------
    Rodriguez-Puebla, A., et al. 2016, MNRAS, 462, 893 (arXiv:1602.04813)
    Yung, L. Y. A., et al. 2024, MNRAS, 527, 5929 (arXiv:2309.14408)

    """
    a = cosmo.scale_factor(z)
    om = cosmo.Om(z)
    ol = cosmo.Ode(z)

    if method == "GUREFT":
        # GUREFT: includes Ωm^(4/7) term in denominator
        denominator = om ** (4.0 / 7.0) - ol + (1.0 + om / 2.0) * (1.0 + ol / 70.0)
        numerator = 0.4 * om * a
    elif method == "RP16":
        # RP16: simpler denominator without Ωm^(4/7) term
        denominator = (om - ol) + (1.0 + om / 2.0) / (1.0 + ol / 70.0)
        numerator = 2.5 * om * a
    else:
        raise ValueError(f"Invalid method '{method}'. Choose 'RP16' or 'GUREFT'.")

    return numerator / denominator


def linear_growth_factor(
    redshift: float | NDArray[np.floating],
    **kwargs,
) -> float | NDArray[np.floating]:
    """Calculate linear growth factor D(z) normalized to present day (z=0).

    Parameters
    ----------
    redshift : float or ndarray
        Redshift at which to evaluate the growth factor.
    **kwargs
        Additional keyword arguments passed to `growth_suppression`.
        See `growth_suppression` for available options (e.g., method).

    Returns
    -------
    float or ndarray
        Linear growth factor D(z)/D(0), normalized to unity at z=0.

    """
    return growth_suppression(redshift, **kwargs) / growth_suppression(0.0, **kwargs)


def halo_mass_accretion_rate(
    halo_mass: float | NDArray[np.floating],
    redshift: float | NDArray[np.floating],
) -> float | NDArray[np.floating]:
    """Calculate halo mass accretion rate from numerical simulations.

    Computes the rate of halo mass growth using the fitting formula from
    Sommovigo et al. (2022), calibrated to numerical simulations.

    Parameters
    ----------
    halo_mass : float or ndarray
        Halo mass in solar masses (M_sun).
    redshift : float or ndarray
        Redshift at which to evaluate the accretion rate.

    Returns
    -------
    float or ndarray
        Halo mass accretion rate in M_sun/yr.

    Notes
    -----
    The accretion rate is computed as::

        dMh/dt = 69.3 * f(z) * E(z) * (M_h / 10¹²)

    where f(z) = -0.24 + 0.75(1+z) and E(z) = H(z)/H₀.

    References
    ----------
    Sommovigo, L., et al. 2022, MNRAS, 513, 3122
    "The ALMA REBELS Survey: cosmic dust temperature evolution out to z~7"

    See Also
    --------
    dMhdt_GUREFT : Alternative accretion rate formula

    """
    dmdt = 69.3 * f(redshift) * cosmo.efunc(redshift) * halo_mass / 1e12  # Msun/yr
    return dmdt  # Msun/yr


def f(z: float | NDArray[np.floating]) -> float | NDArray[np.floating]:
    """Calculate the redshift-dependent scaling factor for halo accretion.

    Parameters
    ----------
    z : float or ndarray
        Redshift.

    Returns
    -------
    float or ndarray
        Scaling factor f(z) = -0.24 + 0.75(1+z) (dimensionless).

    Notes
    -----
    This is an auxiliary function for `dMhdt_num_sims`, representing the
    redshift evolution of halo accretion rates from simulations.

    References
    ----------
    Sommovigo, L., et al. 2022, MNRAS, 513, 3122

    See Also
    --------
    dMhdt_num_sims : Halo mass accretion rate from simulations

    """
    return -0.24 + 0.75 * (1.0 + z)


def virial_radius(
    z: float | NDArray[np.floating],
    halo_mass: float | NDArray[np.floating],
) -> float | NDArray[np.floating]:
    """Calculate the virial radius of a dark matter halo.

    Computes the virial radius using the spherical collapse model in a ΛCDM
    cosmology, following the formalism of Barkana & Loeb (2001).

    Parameters
    ----------
    z : float or ndarray
        Redshift at which to evaluate the virial radius.
    halo_mass : float or ndarray
        Halo mass in solar masses (M_sun).

    Returns
    -------
    float or ndarray
        Virial radius in kiloparsecs (kpc).

    Notes
    -----
    The virial radius is computed using the critical overdensity for collapse::

        Δc(z) = 18π² + 82[Ωm(z) - 1] - 39[Ωm(z) - 1]²

    The virial radius scales with halo mass and redshift as::

        R_vir ∝ M_h^(1/3) * (1+z)^(-1) * [Ωm(z) * Δc(z)]^(-1/3)

    References
    ----------
    Barkana, R. & Loeb, A. 2001, Phys. Rep., 349, 125
    "In the beginning: the first sources of light and the reionization of
    the universe"

    """
    deltac = 18 * np.pi**2 + 82 * (cosmo.Om(z) - 1.0) - 39.0 * (cosmo.Om(z) - 1.0) ** 2
    return (
        0.784
        / cosmo.h
        * (halo_mass / (1e8 / cosmo.h)) ** (1.0 / 3.0)
        * (cosmo.Om0 * deltac / (cosmo.Om(z) * 18 * np.pi**2)) ** (-1.0 / 3.0)
        / ((1 + z) / 10.0)
    )
