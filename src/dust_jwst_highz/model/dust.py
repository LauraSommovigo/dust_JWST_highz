from __future__ import annotations
from functools import partial
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.integrate import quad
from scipy.special import erf

from .. import constants as const
from .cosmology import cosmo
from .halo import virial_radius


def stellar_grain_size_dist(a_cm, a0_cm=1e-5, sigma=0.47):
    """Lognormal stellar grain size distribution.

    φ(a) such that

        ∫ (4π/3) a^3 φ(a) da = 1  (mass-normalized).

    Uses scipy.stats.lognorm for the underlying distribution.

    Parameters
    ----------
    a_cm : array_like
        Grain radius in cm.
    a0_cm : float
        Central grain radius in cm (default 0.1 μm = 1e-5 cm).
    sigma : float
        Lognormal width (standard deviation of log(a)).

    Returns
    -------
    phi : ndarray
        φ(a) with the same shape as a_cm.
        Units: [1 / (cm^4)] so that (4π/3) ∫ a^3 φ(a) da is dimensionless.

    """
    a_cm = np.asarray(a_cm, dtype=float)

    # scipy lognorm: PDF(x) = 1/(x*sigma*sqrt(2*pi)) * exp(-(ln(x) - ln(scale))^2 / (2*sigma^2))
    # We set scale=a0_cm so the distribution is centered at a0_cm
    lognorm_pdf = stats.lognorm.pdf(a_cm, s=sigma, scale=a0_cm)

    # Normalize so that ∫ (4π/3) a^3 φ(a) da = 1
    # The normalization constant accounts for the mass weighting
    volume_factor = (4.0 / 3.0) * np.pi * a0_cm**3
    correction_factor = np.exp(4.5 * sigma**2)  # from ∫ a^3 lognorm(a) da

    return lognorm_pdf / (volume_factor * correction_factor)


def small_carbonaceous_grain_dist(
    radius: float | NDArray[np.floating], bc: float, sigma: float = 0.4
) -> NDArray[np.floating]:
    """Calculate very small carbonaceous grain size distribution.

    Computes the number density per unit size interval for very small carbonaceous
    grains (VSGs), including polycyclic aromatic hydrocarbons (PAHs), following
    Weingartner & Draine (2001) equations (2)-(3). The distribution is a sum of
    two log-normal components centered at 3.5 Å and 30 Å.

    Parameters
    ----------
    radius : float or ndarray
        Grain radius in cm. Must be positive.
    bc : float
        Carbon abundance parameter in very small grains (dimensionless).
        Typical values range from 0 (no VSGs) to 6e-5 (Milky Way).
        This sets the overall amplitude of the VSG population.
    sigma : float, optional
        Width parameter for the log-normal distributions (dimensionless).
        Default is 0.4, as adopted by Weingartner & Draine (2001).

    Returns
    -------
    float or ndarray
        Very small grain size distribution D(a) = (1/n_H) dn/da in cm⁻¹,
        where n_H is the hydrogen number density. Same shape as input radius.
        Returns 0 for radii below 3.5 Å.

    Notes
    -----
    The distribution is given by a sum of two log-normal components:

        D(a) = Σᵢ (Bᵢ/a) exp[-½((ln(a/a₀ᵢ))/σ)²]

    where:
    - a₀₁ = 3.5 Å and a₀₂ = 30 Å are the centers of the two components
    - bc is split as bc₁ = 0.75×bc and bc₂ = 0.25×bc between the components
    - Bᵢ is a normalization factor computed from bc and grain properties

    The distribution only applies for a > 3.5 Å; for smaller radii, D(a) = 0.

    References
    ----------
    Weingartner, J. C., & Draine, B. T. 2001, ApJ, 548, 296
    "Dust Grain-Size Distributions and Extinction in the Milky Way,
    Large Magellanic Cloud, and Small Magellanic Cloud"

    See Also
    --------
    grain_size_dist : Full grain size distribution including large grains


    """
    mp = 1.67262192e-24
    mc = 12.0 * mp
    rho_graphite = 2.24  # g cm^-3

    a = np.asarray(radius, dtype=float)
    # log-normal centers in cm
    a0_1 = 3.5 * const.angstrom
    a0_2 = 30.0 * const.angstrom
    a0 = np.array([a0_1, a0_2])

    # split bc into two components
    bc_i = np.array([0.75 * bc, 0.25 * bc])

    pref = 3.0 / ((2.0 * np.pi) ** 1.5)

    x0 = a0 / (3.5 * const.angstrom)
    arg = 3.0 * sigma / np.sqrt(2.0) + np.log(x0) / (sigma * np.sqrt(2.0))
    denom = 1.0 + erf(arg)

    bi = pref * np.exp(-4.5 * sigma**2) / (rho_graphite * a0**3 * sigma) * (bc_i * mc / denom)

    dist = np.zeros_like(a)
    mask = a > 3.5 * const.angstrom
    a_pos = a[mask]
    if a_pos.size > 0:
        a_ratio = np.log(a_pos[None, :] / a0[:, None]) / sigma
        gauss = np.exp(-0.5 * a_ratio**2)
        dist[mask] = np.sum(bi[:, None] / a_pos[None, :] * gauss, axis=0)

    return dist


def kappa_lambda(
    radius: NDArray[np.floating],
    q_abs_table: NDArray[np.floating],
    dn_da_on_grid: NDArray[np.floating],
    mu: float = const.mean_mol_weight,
    mh: float = const.m_p,
    dust_ratio: float = const.dust_ratio_mw,
) -> NDArray[np.floating]:
    """Compute mass absorption coefficient κ_λ from grain properties and size distribution.

    Calculates the wavelength-dependent mass absorption coefficient by integrating
    the product of grain cross sections, absorption efficiencies, and the grain
    size distribution over all grain sizes.

    Parameters
    ----------
    radius : ndarray of shape (Na,)
        Grain radii in cm for this component (same grid as q_abs_table).
    q_abs_table : ndarray of shape (Na, Nλ)
        Absorption efficiency Q_abs(a_i, λ_j) at each grain size and wavelength.
    dn_da_on_grid : ndarray of shape (Na,)
        Grain size distribution (1/n_H) dn/da evaluated at radius grid points in cm^-1.
        This is the number of grains per unit size per hydrogen atom.
    mu : float, optional
        Mean molecular weight of the gas (dimensionless). Default is 1.22 for typical ISM.
    mh : float, optional
        Proton mass in grams. Default is the CGS value from astropy.constants.
    dust_ratio : float, optional
        Dust-to-gas mass ratio (dimensionless). Default is 1/162 for Milky Way.

    Returns
    -------
    ndarray of shape (Nλ,)
        Mass absorption coefficient κ_λ in cm^2 g^-1 of dust.

    Notes
    -----
    The mass absorption coefficient is computed as::

        κ_λ = [∫ π a² Q_abs(a,λ) (dn/da) da] / (μ * m_H * D)

    where the numerator is the integral over grain sizes of the absorption cross
    section times the grain size distribution, and the denominator normalizes to
    dust mass per unit volume.

    """
    # cell widths Δa (cm)
    da = np.gradient(radius)

    # numerator: sum_j ∫ π a^2 Q_abs(a,λ) (dn/da) da
    # → vectorised over λ
    # (Na, Nλ) * (Na,1) * (Na,1)
    integrand = (
        np.pi
        * radius[:, None] ** 2
        * q_abs_table  # (Na, Nλ)
        * dn_da_on_grid[:, None]
    )  # (Na,1)

    num = np.sum(integrand * da[:, None], axis=0)  # (Nλ,)

    denom = mu * mh * dust_ratio

    return num / denom


def mass_absorption_coefficient(
    radius: float | NDArray[np.floating],
    q_abs: float | NDArray[np.floating],
    density: float | NDArray[np.floating] = 3.5,
) -> float | NDArray[np.floating]:
    """Calculate the mass absorption coefficient (kappa) for dust grains.

    Parameters
    ----------
    radius : float or ndarray
        Grain radius in cm.
    q_abs : float or ndarray
        Absorption efficiency factor (dimensionless).
    density : float or ndarray, optional
        Grain density in g/cm^3. Default is 3.5 g/cm^3 (silicate).

    Returns
    -------
    float or ndarray
        Mass absorption coefficient in cm^2/g.

    Notes
    -----
    The mass absorption coefficient is computed as::

        κ = (σ_abs) / m = (π * r^2 * Q_abs) / m

    where σ_abs is the absorption cross section, m is the grain mass,
    r is the grain radius, and Q_abs is the absorption efficiency.

    """
    mass = grain_mass(radius, density)
    sigma_abs = np.pi * radius**2 * q_abs
    return sigma_abs / mass


def grain_mass(
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


def grain_size_dist(
    radius: float | NDArray[np.floating],
    c: float,
    at: float,
    ac: float,
    alpha: float,
    beta: float,
    d_func: Callable | None = None,
) -> NDArray[np.floating]:
    """Calculate grain size distribution following Weingartner & Draine (2001).

    Computes the number density of dust grains per unit size interval, (1/n_H) dn/da,
    combining a power-law distribution for larger grains with an optional component
    for very small grains (e.g., PAHs). The large grain component includes curvature
    corrections and an exponential cutoff above the transition radius.

    Parameters
    ----------
    radius : float or ndarray
        Grain radius in cm.
    c : float
        Normalization constant C for the power-law component (dimensionless).
        This sets the overall amplitude of the large grain distribution.
    at : float
        Transition radius in cm where the exponential cutoff begins.
        Typical values are ~0.1 μm (1e-5 cm) for Milky Way dust.
    ac : float
        Cutoff width in cm controlling the sharpness of the exponential cutoff.
        Smaller values lead to a sharper cutoff above a_t.
    alpha : float
        Power-law index for the grain size distribution (dimensionless).
        The distribution scales as a^(alpha-1). Typical values range from
        -1.5 to -2.5, with more negative values favoring smaller grains.
    beta : float
        Curvature parameter modifying the power-law slope (dimensionless).
        Positive beta steepens the distribution for a > a_t, while negative
        beta flattens it. Set to 0 for no curvature correction.
    d_func : callable or None, optional
        Function to compute very small grain component D(a) (e.g., for PAHs).
        Must accept radius in cm and return (1/n_H) dn/da in cm^-1.
        If None, only the large grain component is computed. Default is None.

    Returns
    -------
    ndarray
        Grain size distribution (1/n_H) dn/da in cm^-1, where n_H is the
        hydrogen number density. Same shape as input radius.

    Notes
    -----
    The distribution is given by:

        dn/da = D(a) + (C/a_t^α) * a^(α-1) * F(a; β, a_t) * exp(-((a-a_t)/a_c)³)

    where:
    - D(a) is the very small grain component (if d_func is provided)
    - The second term is the power-law component with curvature F and exponential cutoff
    - The exponential cutoff only applies for a > a_t

    This formulation is equivalent to Weingartner & Draine (2001) equations (4)-(5),
    which describe the grain size distribution for carbonaceous and silicate grains
    in the Milky Way.

    References
    ----------
    Weingartner, J. C., & Draine, B. T. 2001, ApJ, 548, 296
    "Dust Grain-Size Distributions and Extinction in the Milky Way,
    Large Magellanic Cloud, and Small Magellanic Cloud"

    See Also
    --------
    f_curvature : Curvature correction function F(a; β, a_t)

    """
    radius = np.asarray(radius, dtype=float)
    if d_func is None:
        d_small = 0
    else:
        d_small = d_func(radius)

    curv = f_curvature(radius, beta, at)
    exp_cutoff = np.ones_like(radius)
    mask = radius > at
    exp_cutoff[mask] = np.exp(-(((radius[mask] - at) / ac) ** 3))

    d_big = (c / at**alpha) * radius ** (alpha - 1) * curv * exp_cutoff

    return d_small + d_big


def f_curvature(
    a: float | NDArray[np.floating],
    beta: float,
    a_t: float,
) -> float | NDArray[np.floating]:
    """Calculate curvature correction factor for grain size distribution.

    Computes the curvature term F(a; β, a_t) from Weingartner & Draine (2001)
    equation (6), which modifies the power-law slope of the grain size
    distribution as a function of grain radius.

    Parameters
    ----------
    a : float or ndarray
        Grain radius in cm.
    beta : float
        Curvature parameter (dimensionless). Controls how the power-law
        slope changes with grain size:
        - β > 0: Distribution steepens for a > a_t (favors smaller grains)
        - β = 0: No curvature correction (pure power law)
        - β < 0: Distribution flattens for a > a_t (favors larger grains)
        Typical values for Milky Way dust range from -0.1 to 0.1.
    a_t : float
        Transition radius in cm where the curvature correction is normalized.
        This is typically the same as the exponential cutoff radius.

    Returns
    -------
    float or ndarray
        Curvature correction factor F(a; β, a_t) (dimensionless).
        Same shape as input `a`. Values are typically close to 1.

    Notes
    -----
    The curvature factor is defined as:

    - For β ≥ 0:  F(a; β, a_t) = 1 + β * (a / a_t)
    - For β < 0:  F(a; β, a_t) = 1 / [1 - β * (a / a_t)]

    This correction allows the grain size distribution to deviate from a
    simple power law, providing better fits to observed interstellar
    extinction curves.

    References
    ----------
    Weingartner, J. C., & Draine, B. T. 2001, ApJ, 548, 296
    "Dust Grain-Size Distributions and Extinction in the Milky Way,
    Large Magellanic Cloud, and Small Magellanic Cloud"

    See Also
    --------
    grain_size_distribution : Full grain size distribution using this correction

    """
    a = np.asarray(a, dtype=float)
    f = np.ones_like(a)
    if beta >= 0.0:
        f += beta * a / a_t
    else:
        f /= 1.0 - beta * a / a_t
    return f


def mass_weighted_grain_size_dist(
    radius: float, gsd_kwargs: dict | None = None, gm_kwargs: dict | None = None
) -> float:
    """Calculate mass-weighted grain size distribution.

    Parameters
    ----------
    radius : float
        Grain radius in cm.
    gsd_kwargs : dict or None, optional
        Keyword arguments to pass to `grain_size_dist`. Default is None.
    gm_kwargs : dict or None, optional
        Keyword arguments to pass to `grain_mass`. Default is None.

    Returns
    -------
    float
        Mass-weighted grain size distribution, dn/da x mass of grain.

    Notes
    -----
    This is the product of the number density distribution dn_da(a) and
    the mass of a grain of size a.

    """
    return grain_size_dist(radius, **gsd_kwargs) * grain_mass(radius, **gm_kwargs)


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
    tcmb_0 = cosmo.Tcmb(0).value  # CMB temperature today [K] (da Cunha+13 eq. 9)

    temp_corrected = (dust_temp**exponent + tcmb_0**exponent * ((1.0 + redshift) ** exponent - 1.0)) ** (1.0 / exponent)

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

    return opacity * dust_mass / (geometry_factor * np.pi * scale_length**2) * const.M_sun / const.kpc**2


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
        Optical depth (dimensionless).
    mu : float
        Cosine of the viewing angle relative to the slab normal (dimensionless).
        Must be between 0 and 1, where 1 is face-on and 0 is edge-on.
    omega : float, optional
        Single-scattering albedo (dimensionless). Default is 0.3807 for Milky Way dust.

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
        Radial optical depth from the center to the edge of the
        sphere (dimensionless).
    omega : float, optional
        Single-scattering albedo (dimensionless). Default is 0.3807 for Milky Way dust.
    g : float, optional
        Scattering asymmetry parameter (dimensionless). Default is 0.6633,
        indicating forward scattering for Milky Way dust.

    Returns
    -------
    float or ndarray
        Transmission factor (dimensionless), representing the fraction
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


# ? There is probably a better place for this function
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


def compute_g_lambda(
    radii_um: NDArray[np.floating],
    qsca_table: NDArray[np.floating],
    g_table: NDArray[np.floating],
    dn_da: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute scatter-weighted asymmetry parameter g(λ) averaged over grain sizes.

    Parameters
    ----------
    radii_um : ndarray, shape (Na,)
        Grain radii in microns.
    qsca_table : ndarray, shape (Na, Nλ)
        Scattering efficiency Q_sca(a, λ).
    g_table : ndarray, shape (Na, Nλ)
        Asymmetry parameter for each (a, λ).
    dn_da : ndarray, shape (Na,)
        Grain size distribution (any normalization; cancels in ratio).

    Returns
    -------
    ndarray, shape (Nλ,)
        Scatter-weighted asymmetry parameter at each wavelength.

    """
    a_cm = radii_um * 1e-4
    da = np.gradient(a_cm)
    weight = np.pi * a_cm[:, None] ** 2 * qsca_table * dn_da[:, None]
    num = np.sum(weight * g_table * da[:, None], axis=0)
    denom = np.sum(weight * da[:, None], axis=0)
    return num / denom


def attenuation_curve_li08(
    lam_um: NDArray[np.floating],
    c1: float,
    c2: float,
    c3: float,
    c4: float,
    model: str | None = None,
) -> NDArray[np.floating]:
    """Compute A(λ)/A_V following the Li+2008 parameterization.

    Parameters
    ----------
    lam_um : ndarray
        Wavelength(s) in microns.
    c1, c2, c3, c4 : float
        Free parameters controlling the curve shape (ignored when *model*
        selects a named template).
    model : str or None
        Named preset: ``'Calzetti'``, ``'SMC'``, ``'MW'``, or ``'LMC'``.
        If ``None`` (default), the supplied c1–c4 values are used directly.

    Returns
    -------
    ndarray
        A(λ)/A_V at each input wavelength.

    References
    ----------
    Li, A. & Draine, B. T. 2008

    """
    presets = {
        "Calzetti": (44.9, 7.56, 61.2, 0.0),
        "SMC": (38.7, 3.83, 6.34, 0.0),
        "MW": (14.4, 6.52, 2.04, 0.0519),
        "LMC": (4.47, 2.39, -0.988, 0.0221),
    }
    if model in presets:
        c1, c2, c3, c4 = presets[model]

    lam_um = np.asarray(lam_um, dtype=float)
    return (
        c1 / ((lam_um / 0.08) ** c2 + (lam_um / 0.08) ** -c2 + c3)
        + (233.0 * (1.0 - c1 / (6.88**c2 + 0.145**c2 + c3) - c4 / 4.6))
        / ((lam_um / 0.046) ** 2.0 + (lam_um / 0.046) ** -2.0 + 90.0)
        + c4 / ((lam_um / 0.2175) ** 2.0 + (lam_um / 0.2175) ** -2.0 - 1.95)
    )


def attenuation_curve_rt(
    kappa_ext: NDArray[np.floating],
    omega_lam: NDArray[np.floating],
    g_lam: NDArray[np.floating],
    sigmad: float,
    geometry: str = "sphere_central",
    mu: float = 0.6,
) -> NDArray[np.floating]:
    """Compute A(λ) [mag] for a given dust column and RT geometry.

    Parameters
    ----------
    kappa_ext : ndarray, shape (Nλ,)
        Total extinction opacity [cm² g⁻¹].
    omega_lam : ndarray, shape (Nλ,)
        Single-scattering albedo.
    g_lam : ndarray, shape (Nλ,)
        Henyey-Greenstein asymmetry parameter.
    sigmad : float
        Dust surface density [g cm⁻²].
    geometry : str
        One of ``'sphere_central'``, ``'sphere_mixed'``, or ``'slab'``.
    mu : float
        Cosine of inclination angle (only used for ``geometry='slab'``).

    Returns
    -------
    ndarray
        Attenuation A(λ) in magnitudes.

    """
    tau = kappa_ext * sigmad

    if geometry == "sphere_central":
        t_lam = transmission_sphere(tau, omega_lam, g_lam)
    elif geometry == "sphere_mixed":
        t_lam = transmission_sphere_mixed(tau)
    elif geometry == "slab":
        t_lam = transmission_slab(tau, mu=mu, omega=omega_lam)
    else:
        raise ValueError(f"Unknown geometry '{geometry}'. Choose 'sphere_central', 'sphere_mixed', or 'slab'.")

    return -2.5 * np.log10(t_lam)


def attenuation_curve_sommovigo25(
    lam_um: NDArray[np.floating],
    a_v: float,
) -> NDArray[np.floating]:
    """Parametric attenuation curve from Sommovigo+2025 (TNG+RT calibration).

    Uses a modified Li+2008 functional form with A_V-dependent coefficients
    calibrated against TNG cosmological simulations with radiative transfer.

    Parameters
    ----------
    lam_um : ndarray
        Wavelengths in microns.
    a_v : float
        V-band attenuation A_V [mag].

    Returns
    -------
    ndarray
        A(λ)/A_V at each input wavelength.

    References
    ----------
    Sommovigo et al. 2025

    """
    log_av = np.log10(a_v)
    log_c1 = -0.37 * log_av + 0.75
    c1 = 10.0**log_c1
    c2 = 1.88
    c3 = 1.21 * log_c1 - 1.33
    log_c4 = -0.59 * log_av - 1.42
    c4 = 10.0**log_c4

    return attenuation_curve_li08(lam_um, c1, c2, c3, c4)


def greybody_dust_temp(
    log_mdust: float | NDArray[np.floating],
    log_ldust: float | NDArray[np.floating],
    kabs_158: float,
    emissivity: float = 2.03,
) -> float | NDArray[np.floating]:
    """Compute single-temperature greybody T_dust from luminosity and mass.

    Inverts the modified blackbody integral:

      L_IR = M_d * kappa_158 * (8π/c²) * (ν_0^{-β}) *
             (k_B/h)^{4+β} * Γ(4+β) * ζ(4+β) * T_d^{4+β}

    Parameters
    ----------
    log_mdust : float or ndarray
        log10(M_dust / M_sun).
    log_ldust : float or ndarray
        log10(L_IR / L_sun).
    kabs_158 : float
        Dust absorption opacity at 158 µm [cm² g⁻¹].
    emissivity : float, optional
        Dust emissivity index β. Default 2.03 (Draine+03 silicates).

    Returns
    -------
    float or ndarray
        Dust temperature T_d [K].

    References
    ----------
    Sommovigo et al. (2022), Draine (2003).

    """
    import scipy.special

    nu_158 = const.c * 1e4 / 158.0  # Hz
    teta = (
        (8.0 * np.pi / const.c**2)
        * (kabs_158 / nu_158**emissivity)
        * (const.k_B**(4.0 + emissivity) / const.h**(3.0 + emissivity))
        * scipy.special.zeta(4.0 + emissivity)
        * scipy.special.gamma(4.0 + emissivity)
    )
    log_td = (
        log_ldust - log_mdust - np.log10(teta) - np.log10(const.M_sun / const.L_sun)
    ) / (4.0 + emissivity)
    return 10.0**log_td


def dust_temp_from_lir(
    l_ir: float | NDArray[np.floating],
    m_dust: float | NDArray[np.floating],
    kabs_158: float,
    emissivity: float = 2.03,
) -> float | NDArray[np.floating]:
    """Wrapper: L_IR [erg/s] + M_dust [M_sun] → T_dust [K] via greybody inversion.

    Parameters
    ----------
    l_ir : float or ndarray
        Bolometric IR luminosity [erg/s].
    m_dust : float or ndarray
        Dust mass [M_sun].
    kabs_158 : float
        Dust absorption opacity at 158 µm [cm² g⁻¹].
    emissivity : float, optional
        Dust emissivity index β. Default 2.03.

    Returns
    -------
    float or ndarray
        Dust temperature T_d [K].

    """
    return greybody_dust_temp(
        np.log10(m_dust),
        np.log10(l_ir / const.L_sun),
        kabs_158,
        emissivity,
    )


def seedavg_lir(
    kabs_uv: float,
    mach: float,
    sigmad_arr: NDArray[np.floating],
    l_intr: float,
    k_spins: int = 13,
    k_gl: int = 24,
) -> float:
    """Seed-averaged IR luminosity from a turbulent lognormal Σ_d model.

    Mirrors the turbulent LF logic: draw K_SPINS quantile seeds from the
    spin-parameter Σ_d distribution, then integrate the absorbed fraction
    over a lognormal whose width is set by the Mach number.

    Parameters
    ----------
    kabs_uv : float
        Dust absorption opacity at 1500 Å [cm² g⁻¹].
    mach : float
        Turbulent Mach number (must be > 0).
    sigmad_arr : ndarray
        Dust surface density samples from the spin distribution [g cm⁻²].
    l_intr : float
        Intrinsic UV luminosity L_1500 [erg s⁻¹ Hz⁻¹].
    k_spins : int, optional
        Number of quantile seeds for the spin distribution. Default 13.
    k_gl : int, optional
        Number of Gauss–Legendre nodes for the Σ_d integral. Default 24.

    Returns
    -------
    float
        Bolometric IR luminosity L_IR [erg/s],
        L_IR ≈ L_1500 × f_abs × (c / λ_1500).

    """
    from scipy.stats import norm
    from .ism import lognormal_variance_from_mach

    u_l = (np.arange(1, k_spins // 2 + 1) - 0.5) / k_spins
    u_seeds = np.concatenate([u_l, [0.5], 1.0 - u_l[::-1]])
    mu_seeds = np.quantile(sigmad_arr, u_seeds)

    sig_ln = lognormal_variance_from_mach(mach)
    xu, wu = np.polynomial.legendre.leggauss(k_gl)
    u_n = np.clip(0.5 * (xu + 1.0), 1e-12, 1 - 1e-12)
    w_n = 0.5 * wu
    z_n = norm.ppf(u_n)

    x_n = np.exp(np.log(mu_seeds)[:, None] + sig_ln * z_n[None, :])
    a_n = 1.0 - transmission_sphere_mixed(kabs_uv * x_n)
    f_abs = np.sum(w_n[None, :] * a_n, axis=1).mean()

    return l_intr * f_abs * (const.c / 1500e-8)
