"""Interstellar medium (ISM) structure and turbulence models.

This module contains functions for modeling the structure of the interstellar
medium, including turbulent density distributions and clumpiness in high-redshift
galaxies.

"""

import numpy as np
from numpy.typing import NDArray


def density_compression_ratio(mach: float | NDArray[np.floating], alpha: float = 2.5) -> float | NDArray[np.floating]:
    r"""Compute the density compression ratio for turbulent supersonic gas.

    Calculates the compression ratio R from Equation (14) of Thompson & Krumholz (2016),
    which describes the density contrast in clumpy, turbulent interstellar medium.

    Parameters
    ----------
    mach : float or ndarray
        Turbulent Mach number (dimensionless). Must be positive and not equal to 1.
    alpha : float, optional
        Power-law index for the density-Mach number relation. Default is 2.5,
        appropriate for supersonic turbulence in molecular clouds.

    Returns
    -------
    float or ndarray
        Compression ratio R (dimensionless), representing the ratio of maximum
        to minimum density in the turbulent medium.

    Raises
    ------
    ValueError
        If the denominator becomes zero for any input Mach number.

    Notes
    -----
    The compression ratio describes the density contrast in turbulent gas
    and is computed as:

        R = 0.5 * (3 - α) / (2 - α) * [1 - M ^ (2(2 - α))] / [1 - M ^ (2(3 - α))]

    where M is the Mach number and α is the power-law index.

    References
    ----------
    Thompson, T. A. & Krumholz, M. R. 2016, MNRAS, 455, 334
    "Sub-Eddington star-forming regions are super-Eddington: momentum-driven
    outflows from supersonic turbulence"

    """
    numerator = 1 - mach ** (2 * (2 - alpha))
    denominator = 1 - mach ** (2 * (3 - alpha))
    if np.any(denominator == 0):
        raise ValueError("Denominator goes to zero; choose a different Mach number.")

    prefactor = 0.5 * (3 - alpha) / (2 - alpha)
    ratio = prefactor * (numerator / denominator)
    return ratio


def sample_surface_density(
    mu_sigma: float,
    mach: float,
    nsamples: int = 10000,
) -> NDArray[np.floating]:
    """Generate log-normal distribution of gas surface densities from turbulent properties.

    Draws samples from a log-normal distribution with median mu_sigma and variance
    determined by turbulence, following Equation (15) of Thompson & Krumholz (2016).

    Parameters
    ----------
    mu_sigma : float
        Median gas surface density in arbitrary units (e.g., M_sun/pc^2).
    mach : float
        Turbulent Mach number (dimensionless). Must be positive.
    nsamples : int, optional
        Number of samples to draw from the distribution. Default is 10,000.

    Returns
    -------
    ndarray
        Array of gas surface density values drawn from the log-normal distribution,
        in the same units as mu_sigma.

    Notes
    -----
    The variance of the log-normal distribution is given by::

        σ²_ln(Σ) = ln(1 + R * M² / 4)

    where R is the compression ratio from `density_compression_ratio` and M is
    the Mach number. This parameterization captures the density structure of
    turbulent, star-forming gas.

    References
    ----------
    Thompson, T. A. & Krumholz, M. R. 2016, MNRAS, 455, 334
    "Sub-Eddington star-forming regions are super-Eddington: momentum-driven
    outflows from supersonic turbulence"

    """
    compression_ratio = density_compression_ratio(mach)
    variance_ln = np.log(1.0 + (compression_ratio * mach**2) / 4.0)
    sigma_ln = np.sqrt(variance_ln)
    mu_ln = np.log(mu_sigma)

    return np.random.lognormal(mean=mu_ln, sigma=sigma_ln, size=nsamples)


def lognormal_variance_from_mach(mach: float | NDArray[np.floating]) -> float | NDArray[np.floating]:
    """Compute log-normal variance (sigma) from turbulent Mach number.

    Parameters
    ----------
    mach : float or ndarray
        Mach number (dimensionless). Must be positive.

    Returns
    -------
    float or ndarray
        Standard deviation σ_ln for the log-normal distribution (dimensionless).

    Notes
    -----
    The variance is computed as::

        σ_ln = sqrt(ln(1 + R * M² / 4))

    where R is the compression ratio from `density_compression_ratio` and M is
    the Mach number.

    """
    compression_ratio = density_compression_ratio(mach)
    variance_ln = np.log(1.0 + (compression_ratio * mach**2) / 4.0)
    return np.sqrt(variance_ln)
