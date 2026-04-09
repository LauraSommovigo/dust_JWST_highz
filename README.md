# High-z Galaxy Formation and Dust Attenuation Models

This repository models high-redshift galaxies with a semi-analytic framework,
focusing on how dust microphysics, ISM geometry, and turbulence shape UV and IR observables.

Paper: [Sommovigo et al. 2026](https://arxiv.org/abs/2602.18556)

## What This Project Does

- Builds star-formation histories from halo growth.
- Tracks dust mass build-up from SN yields.
- Computes dust opacities and attenuation curves from grain models.
- Predicts intrinsic/attenuated UV luminosities and UV/IR luminosity functions.
- Computes greybody FIR SEDs with CMB corrections for ALMA observability.
- Includes both uniform and clumpy/turbulent ISM treatments.

## Repository Layout

- `src/dust_jwst_highz/`
  Core Python package with reusable physics and plotting utilities.
  - `model/`
    - `halo.py`: halo growth, virial radius, halo mass function utilities.
    - `star_formation.py`: stellar-mass mapping, SFR, SFH construction.
    - `dust.py`: grain distributions, kappa calculations, optical depth, attenuation/transmission models, greybody dust temperatures.
    - `ism.py`: turbulent ISM and lognormal surface-density models.
    - `luminosity.py`: UV luminosity conversions, SB99-based luminosity calculations, greybody FIR flux.
    - `cosmology.py`: cosmology helpers.
  - `data.py`: data-loading helpers (dust constants I/O).
  - `visualization.py`: observational LF plotting and colormap utilities.
  - `constants.py`, `utils.py`: constants and shared numerics.
- `notebooks/`
  - `main.py` / `main.ipynb`: main end-to-end analysis notebook (all paper figures).
  - `notebook_dust_lf_cph26_z10.ipynb`: tutorial notebook for Copenhagen 2026 conference (z=10), using a simplified analytical model (Park+18 halo growth).
  - `notebook_dust_lf_cph26_z7.ipynb`: companion tutorial notebook (z=7), including IR LF and greybody SED with REBELS data.
- `data/`
  Input tables (CSV-based, plus one YAML parameter file). All files include source references and DOIs in their headers.
- `outputs/`
  Generated model products and figure outputs.

## Physical Modeling Summary

- **Dust models**:
  - MW-like WD01 grain populations (Weingartner & Draine 2001).
  - Stellar-dust-inspired distributions (Hirashita & Aoyama 2019 lognormal grains).
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

## Running the Analysis

The main analysis lives in `notebooks/main.py` (kept in sync with `notebooks/main.ipynb` via jupytext).

**Important:** the notebook has a two-step dependency. The grain-size distribution section must be run first — it computes dust opacity constants and saves them to `data/dust_constants.txt`. Subsequent sections (optical depth, attenuation, LF figures) load constants from that file. If `dust_constants.txt` is missing you will get a `FileNotFoundError` with a reminder to run the grain-size section first.

To run the full script end-to-end from the project root:

```bash
uv run python notebooks/main.py
```

Or open `notebooks/main.ipynb` in VS Code / JupyterLab and run all cells in order.

Figure outputs are saved under `outputs/z{redshift}/`.

## Citation

If you use this code, please cite:

```bibtex
@ARTICLE{Sommovigo2026,
       author = {{Sommovigo}, Laura and {Lancaster}, Lachlan and {Menon}, Shyam H. and {O'Leary}, Joseph A. and {Somerville}, Rachel S. and {Bryan}, Greg L.},
        title = "{Blue Monsters and Dusty Descendants: Reconciling UV and IR Emission from Galaxies from z=7, up to z= 14}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics of Galaxies},
         year = 2026,
        month = feb,
          eid = {arXiv:2602.18556},
        pages = {arXiv:2602.18556},
          doi = {10.48550/arXiv.2602.18556},
archivePrefix = {arXiv},
       eprint = {2602.18556},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2026arXiv260218556S},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}


```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Notes

- The project requires Python 3.11+.
- Data files include source references and DOIs in their headers.
- `data/dust_constants.txt` is a generated file — do not edit by hand.
