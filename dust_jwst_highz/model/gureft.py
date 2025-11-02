import astropy.constants as apc
import numpy as np
from numpy.typing import NDArray

from ..utils import chi
from .cosmology import cosmo
from .halo import linear_growth_factor


# --- Halo Mass Function Fitting: Yung+23 (Appendix A)
def rms_perturbation_amplitude(halo_mass: float | NDArray[np.floating]) -> float | NDArray[np.floating]:
    """Calculate the RMS density fluctuation amplitude sigma(M_vir) for given halo mass.

    Parameters
    ----------
    halo_mass : float or ndarray
        Halo mass in solar masses (M_sun).

    Returns
    -------
    float or ndarray
        RMS density fluctuation amplitude sigma(M) (dimensionless).

    Notes
    -----
    This fitting function is from Yung et al. (2023), Appendix A

    References
    ----------
    Yung, L. Y. A., et al. 2024, MNRAS, 527, 5929 (arXiv:2309.14408)
    "Characterising ultra-high-redshift dark matter halo demographics and
    assembly histories with the GUREFT simulations"

    """
    y = 1e12 / halo_mass
    return 26.80004233 * y**0.40695158 / (1.0 + 6.18130098 * y**0.23076433 + 4.64104008 * y**0.36760939)


def f_sigma(
    halo_mass: float | NDArray[np.floating],
    redshift: float | NDArray[np.floating],
) -> float | NDArray[np.floating]:
    """Calculate the halo mass function shape parameter f(σ) at redshift z.

    Computes the multiplicity function f(σ) using the Yung et al. (2023) fitting
    formula, which describes the number density of dark matter halos as a function
    of mass and redshift.

    Parameters
    ----------
    halo_mass : float or ndarray
        Halo mass in solar masses (M_sun).
    redshift : float or ndarray
        Redshift at which to evaluate the mass function.

    Returns
    -------
    float or ndarray
        Multiplicity function f(σ) (dimensionless), representing the mass
        function shape at the given mass and redshift.

    Notes
    -----
    The multiplicity function is computed as::

        f(σ) = A * [(σ/b)^(-a) + 1] * exp(-c/σ²)

    where σ is the RMS density fluctuation amplitude and A, a, b, c are
    redshift-dependent parameters from the GUREFT fitting formula.

    References
    ----------
    Yung, L. Y. A., et al. 2024, MNRAS, 527, 5929 (arXiv:2309.14408)
    "Characterising ultra-high-redshift dark matter halo demographics and
    assembly histories with the GUREFT simulations"

    """
    sigma = rms_perturbation_amplitude(halo_mass) * linear_growth_factor(redshift)
    a_gft = chi(0.13765772, -0.01003821, 0.00102964, redshift)
    alpha_gft = chi(1.06641384, 0.02475576, -0.00283342, redshift)
    b_gft = chi(4.86693806, 0.09212356, -0.01426283, redshift)
    c_gft = chi(1.19837952, -0.00142967, -0.00033074, redshift)
    return a_gft * ((sigma / b_gft) ** -alpha_gft + 1.0) * np.exp(-c_gft / sigma**2.0)


def log_halo_mass_function(
    log10_halo_mass: NDArray[np.floating],
    redshift: float,
) -> NDArray[np.floating]:
    """Calculate the halo mass function dn/dlogMh using the GUREFT fitting formula.

    Computes the number density of dark matter halos per unit logarithmic mass
    interval at a given redshift, following Yung et al. (2023).

    Parameters
    ----------
    log10_halo_mass : ndarray
        Base-10 logarithm of halo mass in solar masses, log₁₀(M_h/M_sun).
    redshift : float
        Redshift at which to evaluate the mass function.

    Returns
    -------
    ndarray
        Number density in units of Mpc⁻³ dex⁻¹ (comoving).

    Notes
    -----
    The mass function is computed using the Press-Schechter formalism::

        dn/dlog(Mh) = ln(10) * f(σ) * (ρ_m/M_h) * |d ln σ / d ln M_h|

    where:
    - f(σ) is the multiplicity function from `f_sigma`
    - ρ_m is the mean matter density at z=0
    - σ(M) is the RMS density fluctuation amplitude

    The chain rule is applied using numerical gradients to convert from
    σ(M) to M_h.

    References
    ----------
    Yung, L. Y. A., et al. 2024, MNRAS, 527, 5929 (arXiv:2309.14408)

    See Also
    --------
    f_sigma : Multiplicity function
    dn_dMh_GUREFT : Mass function per unit linear mass

    """
    rho_m = cosmo.Om(0) * cosmo.critical_density(0).value / cosmo.h**2
    mh = 10**log10_halo_mass
    # Chain rule for converting between sigma(M) and Mh
    d_log_sigma = np.gradient(-np.log(rms_perturbation_amplitude(mh)), mh)
    d_log_mh = np.gradient(np.log(mh), mh)
    return (
        np.log(10)
        * f_sigma(mh, redshift)
        * rho_m
        / (mh * apc.M_sun.cgs)
        * np.abs(d_log_sigma / d_log_mh)
        * apc.Mpc.cgs**3
    )


def halo_mass_function(
    halo_mass: NDArray[np.floating],
    redshift: float,
) -> NDArray[np.floating]:
    """Calculate the halo mass function dn/dMh per unit linear mass.

    Computes the number density of dark matter halos per unit linear mass
    interval at a given redshift, following Yung et al. (2023).

    Parameters
    ----------
    halo_mass : ndarray
        Halo mass in solar masses (M_sun).
    redshift : float
        Redshift at which to evaluate the mass function.

    Returns
    -------
    ndarray
        Number density in units of Mpc⁻³ M_sun⁻¹ (comoving).

    Notes
    -----
    The mass function is computed as::

        dn/dMh = f(σ) * (ρ_m/M_h²) * |d ln σ / d ln M_h|

    This is the differential form per unit linear mass, related to the
    logarithmic form by::

        dn/dMh = (dn/dlog₁₀Mh) / (ln(10) * Mh)

    References
    ----------
    Yung, L. Y. A., et al. 2024, MNRAS, 527, 5929 (arXiv:2309.14408)

    See Also
    --------
    dn_dlogMh_GUREFT : Mass function per unit logarithmic mass
    f_sigma : Multiplicity function

    """
    rho_m = cosmo.Om(0) * cosmo.critical_density(0).value / cosmo.h**2
    d_log_sigma = np.gradient(-np.log(rms_perturbation_amplitude(halo_mass)), halo_mass)
    d_log_mh = np.gradient(np.log(halo_mass), halo_mass)
    return (
        f_sigma(halo_mass, redshift)
        * rho_m
        / halo_mass**2
        * np.abs(d_log_sigma / d_log_mh)
        * apc.Mpc.cgs**3
        / apc.M_sun.cgs
    )


def halo_mass_accretion_rate(
    halo_mass: float | NDArray[np.floating],
    redshift: float | NDArray[np.floating],
) -> float | NDArray[np.floating]:
    """Calculate the halo mass accretion rate dMh/dt using the GUREFT formula.

    Computes the rate of halo mass growth from cosmic accretion at a given
    redshift, following Yung et al. (2023).

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

        dMh/dt = 10^β(z) * (10⁻¹² * M_h * E(z))^α(z)

    where:
    - α(z) and β(z) are redshift-dependent parameters
    - E(z) = H(z)/H₀ is the normalized Hubble parameter

    This formula captures the mass growth of dark matter halos from
    smooth accretion and mergers.

    References
    ----------
    Yung, L. Y. A., et al. 2024, MNRAS, 527, 5929

    See Also
    --------
    alpha_funct : Power-law index α(z)
    logbeta_funct : Normalization parameter log₁₀β(z)

    """
    return 10 ** logbeta_funct(redshift) * (1e-12 * halo_mass * cosmo.efunc(redshift)) ** alpha_funct(
        redshift
    )  # Msun/yr


def alpha_funct(z: float | NDArray[np.floating]) -> float | NDArray[np.floating]:
    """Calculate the power-law index α(z) for halo accretion rate.

    Parameters
    ----------
    z : float or ndarray
        Redshift.

    Returns
    -------
    float or ndarray
        Power-law index α(z) (dimensionless).

    Notes
    -----
    This is a redshift-dependent parameter used in the GUREFT halo accretion
    rate formula. It is expressed as a quadratic function of scale factor a = 1/(1+z).

    References
    ----------
    Yung, L. Y. A., et al. 2024, MNRAS, 527, 5929

    See Also
    --------
    dMhdt_GUREFT : Halo mass accretion rate

    """
    a = 1.0 / (1.0 + z)
    return chi(0.858, 1.554, -1.176, a)


def logbeta_funct(z: float | NDArray[np.floating]) -> float | NDArray[np.floating]:
    """Calculate log₁₀β(z), the normalization parameter for halo accretion rate.

    Parameters
    ----------
    z : float or ndarray
        Redshift.

    Returns
    -------
    float or ndarray
        Base-10 logarithm of the normalization parameter β(z) (dimensionless).

    Notes
    -----
    This is a redshift-dependent parameter used in the GUREFT halo accretion
    rate formula. It is expressed as a quadratic function of scale factor a = 1/(1+z).

    References
    ----------
    Yung, L. Y. A., et al. 2024, MNRAS, 527, 5929

    See Also
    --------
    dMhdt_GUREFT : Halo mass accretion rate

    """
    a = 1.0 / (1.0 + z)
    return chi(2.578, -0.989, -1.545, a)
