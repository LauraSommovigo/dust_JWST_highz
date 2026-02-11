# dust_JWST_highz

Semi-Analytic Model (SAM) of dust formation and radiative transfer in high-redshift (z ≳ 7–12) galaxies, with applications to JWST observations and UV/IR luminosity functions.

This repository accompanies the paper draft: https://www.overleaf.com/read/rsnzrzvdkyvm

---

## Scientific Goal

This project models:

- Halo growth and star-formation histories (SFHs)
- Dust mass build-up from supernovae
- Grain size distributions (MW-like vs stellar dust)
- Radiative transfer attenuation (multiple geometries)
- Turbulent ISM effects
- UV and IR luminosity functions at high redshift

The goal is to connect physical dust modelling to JWST observables in a self-consistent framework.

---

# Repository Structure

## 1️⃣ Core Library

### `highz_gal_SAM.py`

Core physical engine of the model. Imported by all other scripts.

Contains:

- Cosmology (Planck18)
- Halo mass function (Yung+23 / GUREFT)
- Star-formation history builder
- SB99 convolution (L1500, SN rate, Nion)
- Dust mass build-up
- Radiative transfer transmission functions:
  - `T_slab`
  - `T_sphere_central`
  - `T_sphere_mixed`
- Attenuation curve parameterisation (Li+08 modified)
- WD01 grain size distribution parameters
- Turbulent ISM lognormal formalism
- Observational data loaders (JWST z>10, REBELS)
- UV LF data compilation

This file is the physical backbone of the SAM.

---

## 2️⃣ Dust Opacity & Grain Size Distribution

### `dust_JWST_z10_GSD.py`

Computes dust opacities from Draine optical tables.

### Sections:

1. Load Draine (2003) optical properties
2. Single-grain opacity regimes (Rayleigh vs geometric)
3. Weingartner & Draine (2001) MW grain size distribution
4. Integrated κ(λ) over WD01 GSD
5. Stellar dust lognormal GSD (Hirashita+19)
6. Total κ, albedo, g for MW vs stellar dust
7. Radiative transfer attenuation curves
8. Save dust constants to text file

The resulting opacity constants are saved and imported by `highz_gal_SAM.py`.

---

## 3️⃣ Characteristic Halo Exploration

### `dust_JWST_z10_exploring_params_last.py`

Explores SAM predictions for individual halo masses.

Outputs include:

- SFR vs Mh
- SFH, Mstar, Mdust evolution
- Optical depth build-up
- MUV–Mstar relation
- Dust attenuation effects
- Half-light radius vs stellar mass

Used for diagnostic plots and model intuition.

---

## 4️⃣ Turbulent ISM Effects

### `dust_JWST_z10_clumpy_ISM.py`

Implements turbulent dust surface density distributions.

- Lognormal Σ_d (Fischera & Dopita 2004)
- Mach-number dependent scatter
- UV magnitude PDFs
- FIR SED comparisons

Demonstrates how turbulence makes attenuation greyer and broadens MUV distributions.

---

## 5️⃣ Population Luminosity Functions

### Uniform ISM
`dust_JWST_z10_population_LF_draw_sample_uniform.py`

Computes UV and IR luminosity functions assuming a uniform dust shell.

Method:
- Map halo mass → L1500, Md
- Draw spin distribution
- Compute attenuation
- Apply Jacobian transformation

---

### Turbulent ISM
`dust_JWST_z10_population_LF_turbulent.py`

Includes ISM turbulence in LF computation.

Key differences:
- Geometry + turbulent scatter in Σ_d
- Fraction of sightlines brighter than MUV
- "Lachlan method" integration
- Gauss-Legendre quadrature for IR LF

Produces UV and IR LFs for varying Mach numbers.

---

# Physical Ingredients

- Cosmology: Planck18
- Halo MF: Yung+23 (GUREFT fit)
- Stellar synthesis: SB99 (can be changed with your SSP of choice)
- Grain model: WD01 (MW) + Hirashita+19 (stellar dust)
- Optical constants: Draine (2003)
- Radiative transfer: slab + sphere geometries
- Turbulence: lognormal Σ_d (Fischera & Dopita 2004)

---

# How to Run

1. First compute dust opacities:
   ```
   python dust_JWST_z10_GSD.py
   ```

2. Then run one of:

   - Characteristic halo exploration:
     ```
     python dust_JWST_z10_exploring_params_last.py
     ```

   - Clumpy ISM exploration (turbulent dust):
     ```
     python dust_JWST_z10_clumpy_ISM.py
     ```
     
   - Uniform LF:
     ```
     python dust_JWST_z10_population_LF_draw_sample_uniform.py
     ```

   - Turbulent LF:
     ```
     python dust_JWST_z10_population_LF_turbulent.py
     ```

---

# Dependencies

- numpy
- scipy
- matplotlib
- astropy
- (optional) emcee / corner (if parameter exploration used)

---

# Citation

If using this model, please cite:

Sommovigo et al. (2025), in preparation.

