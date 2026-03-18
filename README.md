# High-z Galaxy Formation and Dust Attenuation Models

This repository models high-redshift galaxies with a semi-analytic framework,
focusing on how dust microphysics, ISM geometry, and turbulence shape UV and IR observables.

Current paper draft: https://www.overleaf.com/read/rsnzrzvdkyvm#bbf2a2

## What This Project Does

- Builds star-formation histories from halo growth.
- Tracks dust mass build-up from SN yields.
- Computes dust opacities and attenuation curves from grain models.
- Predicts intrinsic/attenuated UV luminosities and UV/IR luminosity functions.
- Includes both uniform and clumpy/turbulent ISM treatments.

## Current Repository Layout

- `src/dust_jwst_highz/`
  Core Python package with reusable physics and plotting utilities.
  - `model/`
    - `halo.py`: halo growth, virial radius, halo mass function utilities.
    - `star_formation.py`: stellar-mass mapping, SFR, SFH construction.
    - `dust.py`: grain distributions, kappa calculations, optical depth, attenuation/transmission models.
    - `ism.py`: turbulent ISM and lognormal surface-density models.
    - `luminosity.py`: UV luminosity conversions and SB99/KS98-based luminosity calculations.
    - `cosmology.py`: cosmology helpers.
  - `data.py`: data-loading helpers (e.g., Hirashita and Draine interpolation functions).
  - `visualization.py`: observational LF plotting and colormap utilities.
  - `constants.py`, `phys_utils.py`, `utils.py`: constants and shared numerics.
- `notebooks/main.ipynb`
  Main end-to-end analysis notebook (figures, parameter exploration, and LF workflows).
- `data/`
  Input tables (now CSV-based, plus one YAML parameter file).
- `outputs/`
  Generated model products and figure outputs.

## Physical Modeling Summary

- **Dust models**:
  - MW-like WD01 grain populations.
  - Stellar-dust-inspired distributions (Hirashita-style lognormal grains).
- **Geometry**:
  - Point-source and mixed-sphere attenuation treatments.
- **Clumpiness/turbulence**:
  - Line-of-sight dust surface densities sampled from lognormal distributions.
  - Width controlled by turbulent Mach number.
- **Luminosity functions**:
  - Sampling-based UV/IR LF estimates.
  - Cumulative-distribution style UV LF mapping in turbulent workflows.

## Setup

```bash
uv sync --group dev
```


## Notes

- The project requires Python 3.11+.
- Data files previously stored as plain text tables have been migrated to CSV for consistency.
