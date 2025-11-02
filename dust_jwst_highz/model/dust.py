import astropy.constants as apc
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad

from .cosmology import cosmo
from .halo import virial_radius


def grain_mass_cgs(
    radius: float | NDArray[np.floating], density: float | NDArray[np.floating] = 3.5
) -> float | NDArray[np.floating]:
    """Calculate the mass of a spherical dust grain.

    Parameters
    ----------
    radius : float or ndarray
        Grain radius in cm.
    density : float or ndarray, optional
        Grain density in g/cm^3. Default is the assumed silicate density of 3.5 g/cm^3.

    Returns
    -------
    float or ndarray
        Mass of the grain in grams.

    """
    return (4.0 / 3.0) * np.pi * radius**3 * density


def grain_size_dist(radius: float) -> float:
    """Calculate grain size distribution from Draine+03 for MW-like silicates.

    Parameters
    ----------
    radius : float
        Grain size in cm.

    Returns
    -------
    float
        Number density of grains per unit size interval at size radius.

    Notes
    -----
    This implements the grain size distribution from Draine & Li (2003)
    for Milky Way-like silicate dust grains. The distribution has a power-law
    form for small grains (radius ≤ ats) and an exponential cutoff for larger grains.

    References
    ----------
    Draine, B. T., & Li, A. 2003, ApJ, 598, 1017

    """
    cs = 1.02e-12
    ats = 0.172e-4  # cm
    alphas = -1.48
    bs = -9.34
    acs = 0.1e-4  # cm

    power_dist = (cs / ats**alphas) * radius ** (alphas - 1) / (1 - bs * radius / ats)
    if radius <= ats:
        return power_dist
    else:
        return power_dist * np.exp(-(((radius - ats) / acs) ** 3))


def mass_weighted_grain_size_dist(radius: float) -> float:
    """Calculate mass-weighted grain size distribution.

    Parameters
    ----------
    radius : float
        Grain radius in cm.

    Returns
    -------
    float
        Mass-weighted grain size distribution, dn/da x mass of grain.

    Notes
    -----
    This is the product of the number density distribution dn_da(a) and
    the mass of a grain of size a.

    """
    return grain_size_dist(radius) * grain_mass_cgs(radius)


def normed_number_weighted_grain_dist(radius: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return normalized number-weighted grain distribution in bins of radius.

    Parameters
    ----------
    radius : ndarray
        Array of grain radius bin edges in cm. Must have at least 2 elements.

    Returns
    -------
    ndarray
        Normalized number-weighted grain distribution with length len(radius) - 1.
        Each element represents the fraction of grains in the corresponding size bin.

    Notes
    -----
    The distribution is computed by integrating `grain_size_dist` over each bin and then
    normalizing so that the sum equals 1.

    """
    weights = np.zeros(len(radius) - 1)
    for s in range(1, len(radius)):
        weights[s - 1] = quad(grain_size_dist, radius[s - 1], radius[s])[0]
    return weights / np.sum(weights)


def normed_mass_weighted_grain_size_dist(radius: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return normalized mass-weighted grain distribution in bins of radius.

    Parameters
    ----------
    radius : ndarray
        Array of grain size bin edges in cm. Must have at least 2 elements.

    Returns
    -------
    ndarray
        Normalized mass-weighted grain distribution with length len(a) - 1.
        Each element represents the mass fraction of grains in the corresponding size bin.

    Notes
    -----
    The distribution is computed by integrating dn_da_massg over each bin and then
    normalizing so that the sum equals 1. This gives the mass fraction in each bin
    rather than the number fraction.

    """
    weights = np.zeros(len(radius) - 1)
    for s in range(1, len(radius)):
        weights[s - 1] = quad(mass_weighted_grain_size_dist, radius[s - 1], radius[s])[0]
    return weights / np.sum(weights)


def dust_temp_cmb_corrected(
    dust_temp: float | NDArray[np.floating],
    redshift: float,
    emissivity: float = 2.03,
) -> float | NDArray[np.floating]:
    """Correct dust temperature for CMB heating at high redshift.

    Parameters
    ----------
    dust_temp : float or ndarray
        Intrinsic dust temperature in Kelvin (without CMB heating).
    redshift : float
        Redshift at which to evaluate the CMB correction.
    emissivity : float, optional
        Dust emissivity index (beta). Default is 2.03.

    Returns
    -------
    float or ndarray
        Effective dust temperature in Kelvin, corrected for CMB heating.

    Notes
    -----
    At high redshift, the cosmic microwave background (CMB) is hotter and
    can significantly heat dust grains, affecting their observed temperature.
    This function computes the effective dust temperature accounting for
    both intrinsic heating and CMB heating.

    The corrected temperature is given by:
        T_eff = [T_d^(4+β) + T_CMB(z)^(4+β) * ((1+z)^(4+β) - 1)]^(1/(4+β))

    where T_d is the intrinsic dust temperature, T_CMB(z) is the CMB
    temperature at redshift z, and β is the emissivity index.

    References
    ----------
    da Cunha, E., et al. 2013, ApJ, 766, 13
    "On the Effect of the Cosmic Microwave Background in High-Redshift
    (Sub-)Millimeter Observations"

    """
    exponent = 4.0 + emissivity
    tcmb_z = cosmo.Tcmb(redshift).value  # CMB temperature at redshift z [K]

    temp_corrected = (dust_temp**exponent + tcmb_z**exponent * ((1.0 + redshift) ** exponent - 1.0)) ** (1.0 / exponent)

    return temp_corrected


def compute_mdust_steps(
    age: NDArray[np.floating],
    tstep: float,
    sfh: NDArray[np.floating],
    time_yr: NDArray[np.floating],
    log_snr_yr: NDArray[np.floating],
    yd: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute cumulative supernova count and dust mass at each time step.

    Convolves the star formation history with supernova rate models to calculate
    the cumulative number of supernovae and resulting dust mass produced at each
    age step, accounting for delayed supernova explosions after star formation.

    Parameters
    ----------
    age : ndarray
        Array of time steps in Myr at which to evaluate dust mass.
    tstep : float
        Time step size in Myr between consecutive age values.
    sfh : ndarray
        Star formation rate history in M_sun/yr at each time step.
    time_yr : ndarray
        Time array in years for the supernova rate template from stellar
        population models (e.g., Starburst99).
    log_snr_yr : ndarray
        log10 of supernova rate per year [yr^-1] for a single stellar
        population burst, corresponding to time_yr.
    yd : float
        Dust yield per supernova in M_sun. Typical values range from
        ~0.01 to 0.5 M_sun depending on progenitor mass and dust formation
        efficiency.

    Returns
    -------
    n_sn_arr : ndarray
        Cumulative number of supernovae at each age step.
    mdust_arr : ndarray
        Total dust mass in M_sun at each age step, computed as
        n_sn_arr * yd.

    """
    n_sn_cumulative = 0
    n_sn_arr = []
    for ind in range(len(age)):
        n_sn_step = 0
        for t in range(ind + 1):
            # Interpolate SN rate as a function of time delay (in yr)
            time_delay_yr = 1e6 * tstep * (ind - t)
            snr_interp = 10 ** np.interp(time_delay_yr, time_yr, log_snr_yr)
            n_sn_step += snr_interp * 1e6 * tstep * sfh[t] * tstep
        n_sn_cumulative += n_sn_step
        n_sn_arr.append(n_sn_cumulative)
    n_sn_arr = np.array(n_sn_arr)
    mdust_arr = n_sn_arr * yd
    return n_sn_arr, mdust_arr


def optical_depth(
    opacity: float | NDArray[np.floating],
    dust_mass: float | NDArray[np.floating],
    halo_mass: float | NDArray[np.floating],
    spin: float | NDArray[np.floating],
    z: float | NDArray[np.floating],
) -> float | NDArray[np.floating]:
    """Calculate dust optical depth for a given geometry.

    Computes the optical depth τ_λ at wavelength λ for dust distributed
    in a spherical geometry with mixed stars and dust, similar to the
    Milky Way.

    Parameters
    ----------
    opacity : float or ndarray
        Dust opacity (mass absorption coefficient) at wavelength λ
        in units of [cm^2/g].
    dust_mass : float or ndarray
        Total dust mass in solar masses (M_sun).
    halo_mass : float or ndarray
        Halo mass in solar masses (M_sun).
    spin : float or ndarray
        Dimensionless halo spin parameter (λ), typically ~0.03-0.05.
    z : float or ndarray
        Redshift at which to evaluate the optical depth.

    Returns
    -------
    float or ndarray
        Optical depth τ_λ (dimensionless).

    Notes
    -----
    The optical depth is computed as::

        τ_λ = κ_λ * M_dust * f_μ / (π * R_d ^ 2)

    where:
    - κ_λ is the dust opacity at wavelength λ
    - M_dust is the total dust mass
    - R_d is the disk scale length
    - f_μ is a geometry factor (4/3 for spherical mixed geometry,
      ~0.841 for slab geometry)

    For spherical geometry with mixed stars and dust, f_μ = 4/3.

    """
    geometry_factor = 4.0 / 3.0  # sphere, mixed geometry
    # Alternative: geometry_factor = 0.841 for slab geometry
    scale_length = disk_scale_length(z, halo_mass, spin)

    return (
        opacity * dust_mass / (geometry_factor * np.pi * scale_length**2) * apc.M_sun.cgs.value / apc.kpc.cgs.value**2
    )


# --- Radiative Transfer: UV Transmission Functions ---


def _int_w_l(t: float, x: float) -> float:
    """Integrand for the W_l scattering integral.

    Helper function computing the integrand for the scattering kernel W_l
    used in slab geometry radiative transfer with anisotropic scattering.

    Parameters
    ----------
    t : float
        Integration variable from 0 to 1 (dimensionless).
    x : float
        Scaled optical depth parameter (dimensionless).

    Returns
    -------
    float
        Value of the integrand (dimensionless).

    Notes
    -----
    The integrand is given by::

        (1 - t) ^ (χ - 1) * cosh(x * t)

    where χ = 2 corresponds to a specific angular distribution of the
    scattering phase function.

    See Also
    --------
    w_l : The full integral using this integrand
    transmission_slab : Slab transmission function using W_l

    """
    chi = 2.0
    return (1.0 - t) ** (chi - 1) * np.cosh(x * t)


def w_l(x: float) -> float:
    """Calculate scattering kernel W_l for slab geometry radiative transfer.

    Computes the scattering kernel integral used in calculating UV transmission
    through a dusty slab with anisotropic scattering.

    Parameters
    ----------
    x : float
        Scaled optical depth parameter: x = (1 - ω) * τ / (2μ), where
        ω is the albedo, τ is the optical depth, and μ is the cosine of
        the viewing angle (dimensionless).

    Returns
    -------
    float
        Value of the scattering kernel W_l(x) (dimensionless).

    Notes
    -----
    The scattering kernel is computed as::

        W_l(x) = χ * ∫₀¹ (1-t)^(χ-1) * cosh(x*t) dt

    where χ = 2 for the assumed phase function. This integral accounts for
    multiple scattering effects in the slab geometry.

    """
    chi = 2.0
    return chi * quad(_int_w_l, 0, 1, args=x)[0]


def transmission_slab(tau: float, mu: float, omega: float = 0.3807) -> float:
    """Calculate UV transmission for slab geometry with scattering.

    Computes the transmission factor at 1500 Å for a plane-parallel slab
    geometry, accounting for both absorption and anisotropic scattering.

    Parameters
    ----------
    tau : float
        Optical depth .
    mu : float
        Cosine of the viewing angle relative to the slab normal (dimensionless).
        Must be between 0 and 1, where 1 is face-on and 0 is edge-on.
    omega : float, optional
        Albedo . Default is 0.3807 for Milky Way dust.

    Returns
    -------
    float
        Transmission factor T_1500 (dimensionless), representing the fraction
        of UV light that escapes the slab at the given viewing angle.

    Notes
    -----
    The transmission is computed as::

        T = (1/μ) * exp(-(1-ω)*τ/(2μ)) * W_l((1-ω)*τ/(2μ))

    where:
    - ω is the dust albedo at 1500 Å
    - W_l is the scattering kernel from the integral equation
    - μ is the cosine of the viewing angle

    This formulation includes both direct attenuation and multiple scattering
    contributions for a plane-parallel slab geometry.

    See Also
    --------
    w_l : Scattering kernel function
    transmission_sphere : Spherical geometry transmission

    """
    transmission = (1.0 / mu) * np.exp(-(1.0 - omega) * tau / (2.0 * mu)) * w_l((1.0 - omega) * tau / (2.0 * mu))
    return transmission


def transmission_sphere(
    tau: float | NDArray[np.floating],
    omega: float = 0.3807,
    g: float = 0.6633,
) -> float | NDArray[np.floating]:
    """Calculate transmission for spherical geometry with central point source.

    Computes the transmission factor for a spherical dust distribution
    with a central point source, accounting for both absorption and anisotropic
    scattering following Code (1973).

    Parameters
    ----------
    tau : float or ndarray
        Radial optical depth at from the center to the edge of the
        sphere (dimensionless).
    omega : float, optional
        Albedo. Default is 0.3807 for Milky Way dust.
    g : float, optional
        Scattering asymmetry parameter. Default is 0.6633,
        indicating forward scattering for Milky Way dust.

    Returns
    -------
    float or ndarray
        Transmission factor, representing the fraction
        of light that escapes the dusty sphere.

    Notes
    -----
    The transmission is computed as::

        T = 2 / [(1 + η) * exp(ψ * τ) + (1 - η) * exp(-ψ * τ)]

    where:
    - η = sqrt[(1-ω)/(1-ω*g)]
    - ψ = sqrt[(1-ω)*(1-ω*g)]

    This formulation accounts for anisotropic scattering in spherical geometry
    and is appropriate for compact sources embedded in dusty envelopes.

    References
    ----------
    Code, A. D. 1973, in Interstellar Dust and Related Topics, ed. J. M. Greenberg
        & H. C. van de Hulst, IAU Symposium, Vol. 52, 505

    """
    eta = np.sqrt((1.0 - omega) / (1.0 - omega * g))
    psi = np.sqrt((1.0 - omega) * (1.0 - omega * g))
    transmission = 2.0 / ((1.0 + eta) * np.exp(psi * tau) + (1.0 - eta) * np.exp(-psi * tau))
    return transmission


def transmission_sphere_mixed(
    tau: float | NDArray[np.floating],
) -> float | NDArray[np.floating]:
    """Calculate UV transmission for spherical geometry with mixed stars and dust.

    Computes the transmission factor at 1500 Å for a spherical distribution
    where stars and dust are spatially mixed, following Osterbrock (1989).
    This accounts for geometric dilution effects only, not scattering.

    Parameters
    ----------
    tau : float or ndarray
        Optical depth at 1500 Å (dimensionless).

    Returns
    -------
    float or ndarray
        Transmission factor T_1500 (dimensionless), representing the fraction
        of UV light that escapes the dusty sphere.

    Notes
    -----
    The transmission factor is computed as::

        T = (3 / 4τ) * [1 - 1/(2τ²) + (1/τ + 1/(2τ²)) * exp(-2τ)]

    This formula applies when stars and dust are uniformly mixed throughout
    a spherical volume, providing a more realistic geometry than a simple
    slab model for galaxies.

    References
    ----------
    Osterbrock, D. E. 1989, Astrophysics of Gaseous Nebulae and Active
    Galactic Nuclei (University Science Books)

    """
    return 3.0 / (4.0 * tau) * (1.0 - 1.0 / (2 * tau**2) + (1.0 / tau + 1.0 / (2 * tau**2)) * np.exp(-2.0 * tau))


# ? There is probbaly a better place for this function
def disk_scale_length(
    z: float | NDArray[np.floating],
    halo_mass: float | NDArray[np.floating],
    spin: float | NDArray[np.floating],
) -> float | NDArray[np.floating]:
    """Calculate the exponential disk scale length from halo properties.

    Computes the characteristic scale length of an exponential disk formed
    within a dark matter halo, assuming the disk extends to a fixed fraction
    of the virial radius determined by the halo spin parameter.

    Parameters
    ----------
    z : float or ndarray
        Redshift at which to evaluate the disk scale length.
    halo_mass : float or ndarray
        Halo mass in solar masses (M_sun).
    spin : float or ndarray
        Dimensionless halo spin parameter (λ), typically ~0.03-0.05.

    Returns
    -------
    float or ndarray
        Disk scale length in kiloparsecs (kpc).

    Notes
    -----
    The disk scale length is computed as::

        R_d = 4.5 * λ * R_vir

    where λ is the spin parameter and R_vir is the virial radius. This
    assumes the disk forms with angular momentum following the distribution
    of the dark matter halo.

    See Also
    --------
    virial_radius : Calculate the virial radius of a halo

    """
    return 4.5 * spin * virial_radius(z, halo_mass)
