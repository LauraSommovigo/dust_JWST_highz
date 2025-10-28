# -- High-z Galaxy Formation and Dust Attenuation Models

This repository contains a set of Python scripts used to model the formation and dust properties of high-redshift galaxies, focusing on their UV and IR luminosities and how these are affected by the structure and turbulence of the interstellar medium (ISM).

Current paper draft: https://www.overleaf.com/read/kccngwhxtzfh#15f262

## -- Repository Structure

- highz_gal_SAM.py
Core module containing all functions used to populate the halo mass function (HMF) with galaxies.
It defines the basic semi-analytic relations connecting halo mass, star formation, and dust production.

- dust_JWST_z10_exploring_params_last.py
Explores individual redshift ranges (e.g., z ≈ 10) and computes galaxy properties such as stellar mass (M_star),
star formation rate (SFR), optical depth, intrinsic and dust-attenuated UV magnitudes.

- dust_JWST_z10_population_LF_last.py
Uses the SAM infrastructure to model galaxy populations and predict obscured fractions as a function of stellar mass,
accounting for the halo spin distribution.
It then produces UV and IR luminosity functions (LFs) as a function of the two main model parameters:
(1) star formation efficiency and (2) dust yield.

- dust_JWST_z10_clumpy_ISM.py
Investigates the impact of ISM turbulence on dust morphology and the resulting emergent luminosities.
By varying the Mach number, it derives the dust surface density distribution (σ_d) and its effect on UV and IR emission
for a given halo mass and redshift.
The resulting σ_d distributions are later fed into the population module.

## -- Modeling Logic

highz_gal_SAM.py → Populate halos with galaxies.

dust_JWST_z10_exploring_params_last.py → Explore individual galaxy properties at fixed redshift.

dust_JWST_z10_population_LF_last.py → Predict population-level observables (UV/IR LFs, obscured fractions).

dust_JWST_z10_clumpy_ISM.py → Incorporate ISM turbulence and clumpiness effects into UV/IR predictions.

## -- Physical Treatment Summary

For UV emission, each dust surface density (σ_d) value represents a different line of sight, allowing us to quantify
sightline-to-sightline attenuation variations.

For IR emission, assumed isotropic, the luminosity is integrated over the dust surface density distribution
(modeled as a log-normal PDF).

This setup yields self-consistent predictions of the UV and IR luminosity functions that naturally incorporate the
effects of ISM clumpiness and turbulence.
