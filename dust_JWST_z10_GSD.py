
from scipy.integrate import quad
from matplotlib import cm
col_f = cm.get_cmap('gray')
from librerie import *
fold_in = '../'
from highz_gal_SAM import *
from general import name_and_save, increase_ticklabels, set_labels, do_minorticks, do_log, equal_axes, \
    set_colorbar_labels, set_ticklabels
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.gridspec as gridspec
from scipy.special import erf


##############################################
##########    INPUT ARRAYS         ###########
##############################################


def read_draine_table_PAH(filename):
    """
    Reads a modified Draine optical constants table containing multiple
    grain radii blocks, each formatted as:

        a_value = radius(micron) ...
        w(micron) Q_ext Q_abs Q_sca g
        <1201 numerical rows>
        <blank line>

    Returns
    -------
    radii : array
        Radii (micron), shape = (NRAD,)
    wavelength : array
        Wavelength grid (micron), shape = (NWAV,)
        Same for all radii.
    Qext, Qabs, Qsca, g : arrays
        Shape = (NRAD, NWAV)
    """

    radii = []
    wavelength = None
    Qext_list = []
    Qabs_list = []
    Qsca_list = []
    g_list    = []

    with open(filename, "r") as f:
        lines = f.readlines()

    i = 0
    N = len(lines)

    while i < N:
        line = lines[i].strip()

        # --- detect line starting with the radius
        if "=" in line and "radius" in line:
            # Extract radius
            a_val = float(line.split("=")[0])
            radii.append(a_val)

            # Next line is the header → skip it
            i += 2

            # Now read wavelength block
            wav_block = []
            Qext_block = []
            Qabs_block = []
            Qsca_block = []
            g_block    = []

            # Read until blank line or next radius
            while i < N and lines[i].strip() != "":
                cols = lines[i].split()
                if len(cols) >= 5:
                    wav_block.append(float(cols[0]))
                    Qext_block.append(float(cols[1]))
                    Qabs_block.append(float(cols[2]))
                    Qsca_block.append(float(cols[3]))
                    g_block.append(float(cols[4]))
                i += 1

            wav_block = np.array(wav_block)

            # First block defines global wavelength grid
            if wavelength is None:
                wavelength = wav_block
            else:
                # Consistency check
                if not np.allclose(wavelength, wav_block):
                    raise ValueError("Wavelength grid mismatch between radii!")

            Qext_list.append(np.array(Qext_block))
            Qabs_list.append(np.array(Qabs_block))
            Qsca_list.append(np.array(Qsca_block))
            g_list.append(np.array(g_block))

        i += 1  # advance to next line

    # Convert all lists to arrays
    radii = np.array(radii)
    Qext = np.array(Qext_list)
    Qabs = np.array(Qabs_list)
    Qsca = np.array(Qsca_list)
    g    = np.array(g_list)

    print(f"Loaded Draine table from {filename}:")
    print(f"  Number of radii: {len(radii)}")
    print(f"  Number of wavelengths: {len(wavelength)}")    

    return radii, wavelength, Qext, Qabs, Qsca, g



def read_draine_table_carb_sil(filename):
    """
    Reads a modified Draine optical constants table containing multiple
    grain radii blocks, each formatted as:

        a_value = radius(micron) ...
        w(micron) Q_ext Q_abs Q_sca g
        <1201 numerical rows>
        <blank line>

    Returns
    -------
    radii : array
        Radii (micron), shape = (NRAD,)
    wavelength : array
        Wavelength grid (micron), shape = (NWAV,)
        Same for all radii.
    Qabs, Qsca, g : arrays
        Shape = (NRAD, NWAV)
    """

    radii = []
    wavelength = None
    Qabs_list = []
    Qsca_list = []
    g_list    = []

    with open(filename, "r") as f:
        lines = f.readlines()

    i = 0
    N = len(lines)

    while i < N:
        line = lines[i].strip()

        # --- detect line starting with the radius
        if "=" in line and "radius" in line:
            # Extract radius
            a_val = float(line.split("=")[0])
            radii.append(a_val)

            # Next line is the header → skip it
            i += 2

            # Now read wavelength block
            wav_block = []
            Qabs_block = []
            Qsca_block = []
            g_block    = []

            # Read until blank line or next radius
            while i < N and lines[i].strip() != "":
                cols = lines[i].split()
                if len(cols) >= 3:
                    wav_block.append(float(cols[0]))
                    Qabs_block.append(float(cols[1]))
                    Qsca_block.append(float(cols[2]))
                    g_block.append(float(cols[3]))
                i += 1

            wav_block = np.array(wav_block)

            # First block defines global wavelength grid
            if wavelength is None:
                wavelength = wav_block
            else:
                # Consistency check
                if not np.allclose(wavelength, wav_block):
                    raise ValueError("Wavelength grid mismatch between radii!")

            Qabs_list.append(np.array(Qabs_block))
            Qsca_list.append(np.array(Qsca_block))
            g_list.append(np.array(g_block))

        i += 1  # advance to next line

    # Convert all lists to arrays
    radii = np.array(radii)
    Qabs = np.array(Qabs_list)
    Qsca = np.array(Qsca_list)
    g    = np.array(g_list)

    print(f"Loaded Draine table from {filename}:")
    print(f"  Number of radii: {len(radii)}")
    print(f"  Number of wavelengths: {len(wavelength)}")    

    return radii, wavelength, Qabs, Qsca, g




##--- Load Draine Q tables for carbon and silicate dust ---#
#- PAH
radii_PAH, wavelength_PAH, Qext_PAH, Qabs_PAH, Qsca_PAH, g_PAH = read_draine_table_PAH(
    "/Users/lsommovigo/Desktop/Scripts/txt_files/Dust_Draine/PAH_dust.txt"
)

#- Carbonaceous grains
radii_c, wavelength_c, Qabs_c, Qsca_c, g_c = read_draine_table_carb_sil(
    "/Users/lsommovigo/Desktop/Scripts/txt_files/Dust_Draine/graphite_dust.txt"
)   

#- Silicate grains
radii_s, wavelength_s, Qabs_s, Qsca_s, g_s = read_draine_table_carb_sil(
    "/Users/lsommovigo/Desktop/Scripts/txt_files/Dust_Draine/silicate_dust.txt"
)


#-- NOW: show large grains are less UV absorbing than small grains 
#-- (depending on grain size vs wavelength, specifically if a<<lambda or not)


# -----------------------------
# 1. Grain mass + kappa
# -----------------------------
def grain_mass(a_micron, rho):
    a_cm = a_micron * 1e-4
    return (4/3) * np.pi * a_cm**3 * rho

def kappa_abs(a_micron, Qabs_arr, wavelength_um, rho):
    a_cm = a_micron * 1e-4
    m = grain_mass(a_micron, rho)
    sigma_abs = np.pi * a_cm**2 * Qabs_arr
    return sigma_abs / m   # cm²/g


# -----------------------------
# 2. Choose two grain sizes
# -----------------------------
idx_small = 0          # smallest radius in table
idx_big   = 11#-1         # radius ~ 0.3 micron, where Hirashita young dust centred

a_small_c = radii_c[idx_small]
a_big_c   = radii_c[idx_big]

a_small_s = radii_s[idx_small]
a_big_s   = radii_s[idx_big]

# Densities
rho_carb = 2.24   # graphite
rho_sil  = 3.5    # silicate


# -----------------------------
# 3. Compute kappa_abs
# -----------------------------

# Carbonaceous
kappa_c_small = kappa_abs(a_small_c, Qabs_c[idx_small], wavelength_c, rho_carb)
kappa_c_big   = kappa_abs(a_big_c,   Qabs_c[idx_big],   wavelength_c, rho_carb)

# Silicate
kappa_s_small = kappa_abs(a_small_s, Qabs_s[idx_small], wavelength_s, rho_sil)
kappa_s_big   = kappa_abs(a_big_s,   Qabs_s[idx_big],   wavelength_s, rho_sil)


# -----------------------------
# 4. Plot
# -----------------------------
plt.figure(figsize=(8,6))

# Carbonaceous: same color, different linestyle
plt.loglog(wavelength_c, kappa_c_small,
           color="indigo", ls="-",
           label=fr"Carbonaceous, $a={a_small_c:.2e}\,\mu$m")

plt.loglog(wavelength_c, kappa_c_big,
           color="indigo", ls="--",
           label=fr"Carbonaceous, $a={a_big_c:.2e}\,\mu$m")

# Silicate: same color, different linestyle
plt.loglog(wavelength_s, kappa_s_small,
           color="darkorange", ls="-",
           label=fr"Silicate, $a={a_small_s:.2e}\,\mu$m")

plt.loglog(wavelength_s, kappa_s_big,
           color="darkorange", ls="--",
           label=fr"Silicate, $a={a_big_s:.2e}\,\mu$m")

# Mark transition from Rayleigh to geometric (a ~ lambda), a smalla nd big are the same for silicate and carbon
#plt.axvline(a_small_c, color="grey",  alpha=0.5,lw=3, ls='-')
#plt.text(a_small_c*1.1, 10, r'$\lambda=a_{\rm small}$', rotation=90, color='grey', alpha=0.7, fontsize=12)
plt.axvline(a_big_c,   color="grey", alpha=0.6,lw=1.5, ls='--')
plt.text(a_big_c*1.1, 200, r'$\lambda=a_{\rm big}$', rotation=90, color='grey', alpha=0.7, fontsize=12)

# Asymptotic scalings
# geometric-limit κ = 3/(4 ρ a)   (convert a[µm]→cm: a*1e-4)
k_geo_s = 3/(4*rho_carb*(a_small_c*1e-4))
k_geo_b = 3/(4*rho_carb*(a_big_c*1e-4))

# Rayleigh scaling, normalized to meet at λ = a_s
k_ray = k_geo_s * (a_small_c / wavelength_c)

plt.loglog(wavelength_c, k_ray, lw=3, label=r'Rayleigh $\propto \lambda^{-1}$', color='dimgrey', alpha=0.3)
#plt.hlines(k_geo_s, 1e-4, 1e3, color='grey', lw=2, label=r'geom (small)')
plt.hlines(k_geo_b, 1e-4, 1e3,  lw=3, label=r'geom (big)', color='dimgrey', ls='--', alpha=0.3)
plt.axvline(0.1500, lw=6, color='indigo', alpha=0.1)


# Check power law behabiour in the FIR
lam_fir = np.linspace(20, 1000, 100)

plt.loglog(lam_fir, 12.85127581*(lam_fir/158.)**(-2), color='r', ls=':', lw=2, label=r'$\lambda^{-2}$ in FIR')


# -----------------------------
# 5. Formatting
# -----------------------------
plt.xlabel(r"$\lambda$ [µm]")
plt.ylabel(r"$\kappa_{\rm abs}\,$ [cm$^2$ g$^{-1}$]")
plt.legend(frameon=False, fontsize=13, loc='upper right')
#plt.grid(alpha=0.25)
plt.xlim(0.050, 30)
plt.ylim(1e2, 0.6e7)
plt.tight_layout()
plt.show()



# -------------------------------------------------------------------
#----- Towards implementing WD01 GSD functions -----#
# -------------------------------------------------------------------
ANGSTROM = 1e-8  # cm

#-- Very small grain D(a) function, WD01 eqs. (2)-(3) --#
def D_of_a_MW(a, bC=6.0e-5, sigma=0.4):
    """
    WD01 very-small carbonaceous grain population D(a) (per H atom),
    eqs. (2)–(3). a in cm. Returns (1/n_H) (dn_gr/da)_vsg.
    """

    mC = 12.0 * mp
    rho_graphite = 2.24  # g cm^-3

    a = np.asarray(a, dtype=float)

    # log-normal centers in cm
    a0_1 = 3.5 * ANGSTROM
    a0_2 = 30.0 * ANGSTROM
    a0 = np.array([a0_1, a0_2])

    # split bC into two components
    bC_1 = 0.75 * bC
    bC_2 = 0.25 * bC
    bC_i = np.array([bC_1, bC_2])

    pref = 3.0 / ((2.0 * np.pi)**1.5)

    x0 = a0 / (3.5 * ANGSTROM)
    arg = 3.0 * sigma / np.sqrt(2.0) + np.log(x0) / (sigma * np.sqrt(2.0))
    denom = 1.0 + erf(arg)

    Bi = (pref * np.exp(-4.5 * sigma**2) /
          (rho_graphite * a0**3 * sigma) *
          (bC_i * mC / denom))

    D = np.zeros_like(a)
    mask = a > 3.5 * ANGSTROM
    a_pos = a[mask]
    if a_pos.size > 0:
        a_ratio = np.log(a_pos[None, :] / a0[:, None]) / sigma
        gauss = np.exp(-0.5 * a_ratio**2)
        D[mask] = np.sum(Bi[:, None] / a_pos[None, :] * gauss, axis=0)

    return D

# --- curvature term from WD01 eq. (6) ---
def F_curvature(a, beta, a_t):
    """
    WD01 F(a; beta, a_t).  a, a_t in cm.
    """
    a = np.asarray(a, dtype=float)
    F = np.ones_like(a)
    if beta >= 0.0:
        F += beta * a / a_t
    else:
        F /= (1.0 - beta * a / a_t)
    return F


# -------------------------------------------------------------------
# Full carbonaceous dn/da (very small + larger grains), WD01 eq. (4)
# -------------------------------------------------------------------
def dn_da_carbon_WD01(a, D_func,
                      C_g, a_t_g, a_c_g, alpha_g, beta_g):
    """
    (1/n_H) dn_gr/da for carbonaceous grains, WD01 eq. (4).
    All radii in cm.
    """
    a = np.asarray(a, dtype=float)

    # very small grains term
    D_small = D_func(a)

    # curvature + exponential cutoff
    Fg = F_curvature(a, beta_g, a_t_g)
    cutoff = np.ones_like(a)
    mask = (a > a_t_g)
    cutoff[mask] = np.exp(-((a[mask] - a_t_g) / a_c_g)**3)

    big = C_g / a * (a / a_t_g)**alpha_g * Fg * cutoff
    return D_small + big


# -------------------------------------------------------------------
# Silicate dn/da, WD01 eq. (5)
# -------------------------------------------------------------------
def dn_da_silicate_WD01(a,
                        C_s, a_t_s, a_c_s, alpha_s, beta_s):
    """
    (1/n_H) dn_gr/da for silicate grains, WD01 eq. (5).
    All radii in cm.
    """
    a = np.asarray(a, dtype=float)

    Fs = F_curvature(a, beta_s, a_t_s)
    cutoff = np.ones_like(a)
    mask = (a > a_t_s)
    cutoff[mask] = np.exp(-((a[mask] - a_t_s) / a_c_s)**3)

    return C_s / a * (a / a_t_s)**alpha_s * Fs * cutoff



# choose MW model: R_V = 3.1 (normal), case A, b_C = 2.0 entry
p = WD01_MW_PARAMS[3.1]["A"][2]

# convert micron -> cm
a_t_g_cm = p['at_g'] * 1e-4
a_c_g_cm = p['ac_g'] * 1e-4
a_t_s_cm = p['at_s'] * 1e-4
a_c_s_cm = 0.1e-4   # WD01 adopt a_c,s = 0.1 µm for MW

C_g     = p['Cg']
alpha_g = p['alpha_g']
beta_g  = p['beta_g']

C_s     = p['Cs']
alpha_s = p['alpha_s']
beta_s  = p['beta_s']

# grain-size grid in cm
a_grid = np.logspace(np.log10(3.5e-8), np.log10(1e-4), 500)  # 3.5 Å – 1 µm

# your very–small–grain D(a) (per H); must accept a in cm
# e.g. D_of_a_MW(a_grid)
dn_da_C  = dn_da_carbon_WD01(a_grid, D_of_a_MW,
                             C_g, a_t_g_cm, a_c_g_cm, alpha_g, beta_g)

dn_da_Si = dn_da_silicate_WD01(a_grid,
                               C_s, a_t_s_cm, a_c_s_cm, alpha_s, beta_s)

Yc = 1e29 * (a_grid**4) * dn_da_C    # for carbonaceous
Ys = 1e29 * (a_grid**4) * dn_da_Si   # for silicates

plt.figure(figsize=(7,5))
plt.loglog(a_grid*1e4, Yc, color='indigo', label='Carbonaceous')
plt.loglog(a_grid*1e4, Ys, color='darkorange', label='Silicate')

plt.xlabel(r'Grain radius $a\,[\mu{\rm m}]$')
plt.ylabel(r'$10^{29}\, a^{4} (n_{\rm H}^{-1} dn/da)$ [cm$^{3}$]')
plt.legend(frameon=False)
plt.grid(alpha=0.3)
plt.ylim(1e-2, 1e2)
plt.tight_layout()
plt.show()



# -------------------------------------------------------------------
#--- ok let's now compute opacity per unit dust mass
# -------------------------------------------------------------------

def compute_kappa_lambda(wavelength_um,
                          radii_um,
                          Qabs_table,
                          dn_da_on_grid,
                          mu=mu_gas,
                          mH=mp,
                          D=D_MW):
    """
    Compute κ_λ [cm^2/g_dust] from Q_abs(λ,a) and (1/n_H) dn/da(a).

    Parameters
    ----------
    wavelength_um : (Nλ,) array
        Wavelength grid (µm) – same as in the Draine tables.
    radii_um : (Na,) array
        Grain radii (µm) for this component (same grid as Qabs_table).
    Qabs_table : (Na, Nλ) array
        Q_abs(a_i, λ_j) from Draine.
    dn_da_on_grid : (Na,) array
        (1/n_H) dn/da evaluated on radii_um **in cm** (your WD01 dn/da).
    mu, mH, D : floats
        Gas μ, proton mass, and dust-to-gas ratio used for normalisation.

    Returns
    -------
    kappa_lambda : (Nλ,) array
        Mass absorption coefficient κ_λ [cm^2 g^-1 of dust].
    """
    # convert radii to cm
    a_cm = radii_um * 1e-4

    # cell widths Δa (cm)
    da = np.gradient(a_cm)

    # numerator: sum_j ∫ π a^2 Q_abs(a,λ) (dn/da) da
    # → vectorised over λ
    # (Na, Nλ) * (Na,1) * (Na,1)
    integrand = (np.pi * a_cm[:, None]**2 *
                 Qabs_table *                     # (Na, Nλ)
                 dn_da_on_grid[:, None])          # (Na,1)

    num = np.sum(integrand * da[:, None], axis=0)   # (Nλ,)

    denom = mu * mp * D

    return num / denom


# size grids in cm for dn/da
a_c_cm = radii_c * 1e-4
a_s_cm = radii_s * 1e-4

dn_da_C  = dn_da_carbon_WD01(a_c_cm, D_of_a_MW,
                             C_g, a_t_g_cm, a_c_g_cm, alpha_g, beta_g)

dn_da_Si = dn_da_silicate_WD01(a_s_cm,
                               C_s, a_t_s_cm, a_c_s_cm, alpha_s, beta_s)

# carbonaceous dn/da and silicate dn/da already computed:
# dn_da_C (radii_c in cm), dn_da_Si (radii_s in cm)

# ---- Carbon ----
kappa_c_abs = compute_kappa_lambda(wavelength_c, radii_c, Qabs_c, dn_da_C)
kappa_c_sca = compute_kappa_lambda(wavelength_c, radii_c, Qsca_c, dn_da_C)

# ---- Silicate ----
kappa_s_abs = compute_kappa_lambda(wavelength_s, radii_s, Qabs_s, dn_da_Si)
kappa_s_sca = compute_kappa_lambda(wavelength_s, radii_s, Qsca_s, dn_da_Si)

# ---- Total MW mixture (C+Si) ----
kappa_abs_tot = (1./11.) * kappa_c_abs + (10./11.) * kappa_s_abs
kappa_sca_tot = (1./11.) * kappa_c_sca + (10./11.) * kappa_s_sca

kappa_ext_tot = kappa_abs_tot + kappa_sca_tot          # true extinction
omega_tot     = kappa_sca_tot / kappa_ext_tot          # albedo ω(λ)




lam_A = wavelength_c * 1e4   # µm → Å


plt.figure(figsize=(7,5))

plt.loglog(lam_A, kappa_c_abs,  color='indigo',     alpha=0.6, label='Carbonaceous')
plt.loglog(lam_A, kappa_s_abs,  color='darkorange', alpha=0.6, label='Silicate')
plt.loglog(lam_A, kappa_abs_tot,color='black',      lw=2,      label='Total MW')
plt.plot(lam_A, kappa_ext_tot,   color='black',      lw=2, ls='--', label='Total MW (with scattering)')



print('kext_1500 [cm^2/g] -->', kappa_ext_tot[lam_A==1585])   # tweak as needed
kUV_drn=kappa_ext_tot[lam_A==1585]

plt.xlabel(r'Wavelength $\lambda$ [\AA]', fontsize=14)
plt.ylabel(r'$\kappa_{\rm abs}(\lambda)\ [{\rm cm}^2\,{\rm g}^{-1}]$', fontsize=14)
plt.legend(frameon=False, fontsize=12)
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()




#------ Ok now we add the case of the readily produced from stars dust (Hirashita=19) GSD -----#
#--- lognormal GSD centred at 0.3 micron with sigma=0.1


def C_phi_lognormal(a0_cm=1e-5, sigma=0.47):
    """
    Normalization constant C_phi for the lognormal grain size
    distribution of stellar dust (eq. 5 in your screenshot).

    a0_cm : float
        Central grain radius a0 in cm (0.1 micron → 1e-5 cm).
    sigma : float
        Lognormal width (dimensionless).
    """
    pref = (4.0/3.0) * np.pi * a0_cm**3
    return 1.0 / (pref * np.sqrt(2.0*np.pi) * sigma *
                  np.exp(0.5 * 9.0 * sigma**2))


def phi_stellar(a_cm, a0_cm=1e-5, sigma=0.47):
    """
    Lognormal stellar grain size distribution φ(a) such that

        ∫ (4π/3) a^3 φ(a) da = 1  (mass-normalized).

    Parameters
    ----------
    a_cm : array_like
        Grain radius in cm.
    a0_cm : float
        Central grain radius in cm (default 0.1 μm = 1e-5 cm).
    sigma : float
        Lognormal width.

    Returns
    -------
    phi : ndarray
        φ(a) with the same shape as a_cm.
        Units: [1 / (cm^4)] so that (4π/3) ∫ a^3 φ(a) da is dimensionless.
    """
    a_cm = np.asarray(a_cm, dtype=float)
    Cphi = C_phi_lognormal(a0_cm=a0_cm, sigma=sigma)

    x = np.log(a_cm / a0_cm)
    return Cphi / a_cm * np.exp(-x**2 / (2.0 * sigma**2))


a_grid_cm = np.logspace(-7, -4, 500)   # 0.001–1 μm
phi = phi_stellar(a_grid_cm, a0_cm=1e-5, sigma=0.47)

# Check normalization numerically:
mass_int = (4*np.pi/3) * np.trapz(a_grid_cm**3 * phi, a_grid_cm)
#print("should be 1 (cause norm) -->", mass_int)  



#--- Now compare WD01 GSD with stellar lognormal GSD ---#
# stellar lognormal φ(a) on those grids (mass-normalized)
phi_C_star  = phi_stellar(a_c_cm, a0_cm=1e-5, sigma=0.47)  # a0 = 0.1 μm
phi_Si_star = phi_stellar(a_s_cm, a0_cm=1e-5, sigma=0.47)

# convert φ(a) → (1/n_H) dn/da with the same D as MW
factor_C  = mu_gas * mp * D_MW / ((4.0*np.pi/3.0) * rho_carb)
factor_Si = mu_gas * mp * D_MW / ((4.0*np.pi/3.0) * rho_sil)

dn_da_C_star  = factor_C  * phi_C_star   # same units as dn_da_C
dn_da_Si_star = factor_Si * phi_Si_star  # same units as dn_da_Si

# -------- Draine-style comparison plot --------
plt.figure(figsize=(7,5))

# WD01 carbonaceous
plt.loglog(a_c_cm*1e4, 1e29 * a_c_cm**4 * dn_da_C,
           color='indigo', lw=2, label='WD01 - Carbonaceous')
# WD01 silicate
plt.loglog(a_s_cm*1e4, 1e29 * a_s_cm**4 * dn_da_Si,
           color='darkorange', lw=2, label='WD01 - Silicate')

# Stellar lognormal carbonaceous with same D (this is independent from species)
plt.loglog(a_c_cm*1e4, 1e29 * a_c_cm**4 * dn_da_C_star,
           color='indigo', ls='--', lw=2, label='Stellar ')


plt.xlabel(r'$a\ [\mu{\rm m}]$', fontsize=14)
plt.ylabel(r'$10^{29}\,a^4\,{\rm d}n/{\rm d}a\ [{\rm cm}^3]$', fontsize=14)
plt.legend(frameon=False)
plt.grid(alpha=0.3)
plt.ylim(1e-2, 1e2)
plt.tight_layout()
plt.show()


#--- Finally, compute κ_λ from stellar lognormal φ(a) -----#
def kappa_from_phi(wavelength_um, radii_um, Qabs_table, phi_a, rho_gr):
    """
    κ_λ from mass-normalized φ(a).

    wavelength_um : (Nλ,)
    radii_um      : (Na,)
    Qabs_table    : (Na, Nλ) 
    phi_a         : (Na,)     
    rho_gr        : float     -- grain material density [g/cm^3]
    """
    a_cm = radii_um * 1e-4
    da   = np.gradient(a_cm)

    # numerator: ∑ π a^2 Q_abs φ(a) da  (vectorised over λ)
    integrand = np.pi * a_cm[:, None]**2 * Qabs_table * phi_a[:, None]
    num = np.sum(integrand * da[:, None], axis=0)   # (Nλ,)

    denom = (4.0*np.pi/3.0) * rho_gr   # from the normalization condition
    return num / denom                 # κ_λ [cm^2/g_dust]

dn_da_C_star  = factor_C  * phi_C_star     # (1/n_H) dn/da [cm^-1]
dn_da_Si_star = factor_Si * phi_Si_star

# ------------------------------------------------------------------
# 2. κ_abs and κ_sca for stellar dust
# ------------------------------------------------------------------
# --- Carbonaceous ---
kappa_C_abs_star = compute_kappa_lambda(wavelength_c, radii_c, Qabs_c,
                                        dn_da_C_star)
kappa_C_sca_star = compute_kappa_lambda(wavelength_c, radii_c, Qsca_c,
                                        dn_da_C_star)

# --- Silicate ---
kappa_Si_abs_star = compute_kappa_lambda(wavelength_s, radii_s, Qabs_s,
                                         dn_da_Si_star)
kappa_Si_sca_star = compute_kappa_lambda(wavelength_s, radii_s, Qsca_s,
                                         dn_da_Si_star)

# put both components on the same λ grid (they *should* match; if not, interp)
lam_um = wavelength_c  # = wavelength_s
# Total stellar mixture
kappa_abs_star_tot = kappa_C_abs_star + kappa_Si_abs_star
kappa_sca_star_tot = kappa_C_sca_star + kappa_Si_sca_star

kappa_ext_star_tot = kappa_abs_star_tot + kappa_sca_star_tot
omega_star_tot     = kappa_sca_star_tot / kappa_ext_star_tot   # albedo



print('kext_1500_stellar [cm^2/g] -->', kappa_ext_star_tot[lam_A==1585])   
kUV_hir=kappa_ext_star_tot[lam_A==1585]

# ------------------------------------------------------------------
# 3. (Optional) compare stellar vs WD01 MW
# ------------------------------------------------------------------
lam_A = lam_um * 1e4  # µm → Å

plt.figure(figsize=(7,5))
plt.loglog(lam_A, kappa_abs_tot,      'k--',   lw=1, label='MW WD01 (abs)')
plt.loglog(lam_A, kappa_ext_tot,      'k', lw=1.5, label='MW WD01 (tot)')
plt.loglog(lam_A, kappa_abs_star_tot, 'r--', lw=1, label='Stellar (abs)')
plt.loglog(lam_A, kappa_ext_star_tot, 'r',  lw=1.5, label='Stellar (tot)')
plt.xlim(923, 2e4)
plt.ylim(1e2, 1e6)
plt.xlabel(r'$\lambda\ [\AA]$')
plt.ylabel(r'$\kappa_{\lambda}\ [{\rm cm}^2\,{\rm g}^{-1}]$', fontsize=14)
plt.legend(frameon=False,fontsize=14)
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

#-- Ok now that I ma not retarded anymore I can redo Fig. 5 in the paper ----#

# ==========================================================
# Compare MW WD01 vs stellar dust: κ_λ + a^4 dn/da inset
# ==========================================================
lam_um = wavelength_c                     # = wavelength_s
lam_A  = lam_um * 1e4                     # µm → Å

fig, ax = plt.subplots(figsize=(5, 5))

# --- MAIN PANEL: κ_ext(λ) ---
ax.plot(lam_A, kappa_ext_tot,
        lw=2.5, alpha=0.5, color='crimson',
        label='MW dust (WD01)')
ax.plot(lam_A, kappa_ext_star_tot,
        lw=2.0, alpha=0.8, color='teal',
        label='Stellar dust (H19)')
#-- abs
ax.plot(lam_A, kappa_abs_tot,
        lw=2.5, alpha=0.5, color='crimson', ls=':')
ax.plot(lam_A, kappa_abs_star_tot,
        lw=2.0, alpha=0.8, color='teal', ls=':')


print("\n==== κ_abs(1500 Å) results ====")
print(f"MW WD01 dust (ext)     : {kUV_drn} cm^2 g^-1")
print(f"Stellar dust model (ext): {kUV_hir} cm^2 g^-1")
print(f"WD01 dust (abs)     : {kappa_abs_tot[lam_A==1.585e3]} cm^2 g^-1")
print(f"Stellar dust model (abs): {kappa_abs_star_tot[lam_A==1.585e3]} cm^2 g^-1")
print("================================\n")

# --- Plot Points at 1500 Å ---
ax.scatter([1500], [kUV_drn],
           color='crimson', edgecolor='black',
           s=70, marker='D', alpha=0.4,
           label=r'$\kappa_{1500}=$'+str(int(kUV_drn))+' $cm^2\,g^{-1}$ (MW WD01)')

ax.scatter([1500], [kUV_hir],
           color='teal', edgecolor='black',
           s=70, marker='D', alpha=0.4,
           label=r'$\kappa_{1500}=$'+str(int(kUV_hir))+' $cm^2\,g^{-1}$ (Stellar)')

# vertical line at 1500 Å
ax.axvline(1500., color='gray', linestyle='--', lw=1)

ax.text(1550., ax.get_ylim()[0]*1.2,
        r'$\lambda=1500\,$Å', fontsize=12, color='gray')



ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(25, 2e5) 
ax.set_xlim(1e3, 2e6)               # 1000 Å – 200 µm
ax.set_xlabel(r'$\lambda\ [\mathrm{\AA}]$')
ax.set_ylabel(r'$\kappa_{\lambda}\ [{\rm cm}^2\,{\rm g}^{-1}]$')
ax.grid(True, alpha=0.2, lw=0.5)
ax.legend(frameon=False, fontsize=16, loc='lower left')

# ==========================================================
# INSET: 1e29 * a^4 dn/da for WD01 vs stellar dust
# ==========================================================
inset_ax = ax.inset_axes([0.57, 0.57, 0.4, 0.4])

# build a common size grid (cm) for total (C+Si)
a_c_cm = radii_c * 1e-4
a_s_cm = radii_s * 1e-4
a_all  = np.sort(np.unique(np.concatenate([a_c_cm, a_s_cm])))

# interpolate WD01 dn/da (per H) onto common grid
dn_da_WD  = np.interp(a_all, a_c_cm, dn_da_C,      left=0.0, right=0.0)
dn_da_WD += np.interp(a_all, a_s_cm, dn_da_Si,     left=0.0, right=0.0)

# interpolate stellar dn/da onto same grid
dn_da_star  = np.interp(a_all, a_c_cm, dn_da_C_star,  left=0.0, right=0.0)
dn_da_star += np.interp(a_all, a_s_cm, dn_da_Si_star, left=0.0, right=0.0)

# quantity to plot: 1e29 * a^4 dn/da   [cm^3]
y_WD   = 1e29 * a_all**4 * dn_da_WD
y_star = 1e29 * a_all**4 * dn_da_star

# convert a to µm for x–axis
a_um_all = a_all * 1e4

inset_ax.loglog(a_um_all, y_WD,
                color='crimson', alpha=0.7, lw=2,
                label='MW dust')
inset_ax.loglog(a_um_all, y_star,
                color='teal', alpha=0.8, lw=2,
                label='Stellar dust (H19)')

inset_ax.set_xscale('log')
inset_ax.set_yscale('log')
inset_ax.set_xlim(1e-3, 1.0)
# pick a sensible y–range; can tweak after first run
inset_ax.set_ylim(1e-3, 1e3)

inset_ax.set_xlabel(r'$a\ [\mu{\rm m}]$', fontsize=14)
inset_ax.set_ylabel(r'$10^{29}\,a^4\,{\rm d}n/{\rm d}a\ [{\rm cm}^3]$',
                    fontsize=14)
inset_ax.grid(True, which='both', ls=':', alpha=0.2)

plt.tight_layout()
plt.show()

kIR_drn=kappa_abs_tot[lam_A==1.585e6] 
kIR_hir=kappa_abs_star_tot[lam_A==1.585e6]


print("\n==== κ_abs(158 micron) results (for dust emission) ====")
print(f"MW WD01 dust (abs)     : {kIR_drn} cm^2 g^-1")
print(f"Stellar dust model (abs): {kIR_hir} cm^2 g^-1")
print("=========================================================\n")

kv_drn=kappa_ext_tot[lam_A==5.623e+03 ] 
kv_hir=kappa_ext_star_tot[lam_A==5.623e+03 ]

print("\n==== κ_tot(V band) results ====")
print(f"MW WD01 dust (ext)     : {kv_drn} cm^2 g^-1")
print(f"Stellar dust model (ext): {kv_hir} cm^2 g^-1")
print("=========================================================\n")