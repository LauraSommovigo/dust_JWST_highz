-- High-z Galaxy Formation and Dust Attenuation Models

This repository contains a set of Python scripts used to model the formation and dust properties of high-redshift galaxies, focusing on their UV and IR luminosities and how these are affected by the structure, geometry, and turbulence of the interstellar medium (ISM).

Current paper draft: https://www.overleaf.com/read/kccngwhxtzfh#15f262

-- Repository Structure

* dust_JWST_z10_GSD.py
Computes absorption, extinction etc., starting from grain size distribution (silicates, carbon and PAH dust). values derived here are then used in the rest of the scripts wherever dust attenuation/emission is computed

* highz_gal_SAM.py
Core module containing all the functions used to populate the halo mass function (HMF) with galaxies.
It defines the basic semi-analytic relations connecting halo mass, star formation, and dust production.

* dust_JWST_z10_exploring_params_last.py
Explores individual redshift ranges (typically $z \gtrsim 7$–10) and computes galaxy properties such as stellar mass ($M_\star$), star formation rate (SFR), optical depth, and both intrinsic and dust-attenuated UV magnitudes.

* dust_JWST_z10_clumpy_ISM
This version also looks at individual galaxy properties like the previous code, but includes the effect of clumpy ISM and thus dust distribution, as well as the changing grain size distribution.

* dust_JWST_z10_population_LF_last.py
Uses the SAM infrastructure to model galaxy populations and predict the obscured fraction as a function of stellar mass, accounting for the halo spin parameter distribution.
It computes UV and IR luminosity functions (LFs) for different combinations of the two main model parameters: the star formation efficiency ($\epsilon_\star$) and the dust yield ($y_d$).
The resulting LFs are derived from the mapping between halo mass and intrinsic or dust-attenuated UV luminosity, ensuring consistency across different parameter sets.

* dust_JWST_z10_population_LF_draw_sample.py
This is the most up-to-date version of the luminosity function module and should be preferred over dust_JWST_z10_population_LF_last.py.
It extends the population modeling to include turbulent and clumpy ISM effects.
By varying the Mach number ($\mathcal{M}$), it generates a log-normal distribution of dust surface densities ($\Sigma_d$), capturing the impact of turbulence on both UV attenuation and IR re-emission.
For each halo mass, UV emission is computed along multiple lines of sight (providing 30th, 50th, and 70th percentile curves of $M_{UV}$), while IR emission is computed isotropically by integrating over the full $\Sigma_d$ probability distribution.
A slope-based masking and monotonic redistribution algorithm ensures that the resulting UV and IR luminosity functions are numerically stable and free from artificial spikes or turnovers.

-- Modeling Logic

highz_gal_SAM.py → populate halos with galaxies
dust_JWST_z10_exploring_params_last.py → explore individual galaxy properties and ISM effects
dust_JWST_z10_population_LF_last.py → build galaxy populations and predict UV/IR luminosity functions
dust_JWST_z10_population_LF_draw_sample.py → updated version including ISM clumpiness and Mach-driven turbulence effects

-- Physical Treatment Summary

Star formation and dust buildup:
Computed self-consistently from the time-resolved star formation history (SFH) of each halo, with dust yields from supernovae and no dust ejection assumed.

UV emission:
Computed along multiple random lines of sight per halo to capture attenuation variability due to ISM geometry and orientation.
In the clumpy ISM version, this accounts for the log-normal scatter in dust surface density driven by turbulence (parameterized by the Mach number).

IR emission:
Treated as isotropic and obtained by integrating the absorbed UV fraction over the full log-normal distribution of $\Sigma_d$.

Luminosity functions:
Derived from percentile-based curves (typically 30th, 50th, and 70th percentiles) of $M_{UV}$ and the corresponding number densities, using a slope-filtering and monotonic redistribution scheme to ensure smooth and physical behavior.
The newest version (dust_JWST_z10_population_LF_draw_sample.py) uses a stable, sampling-based method to compute UV and IR LFs that naturally reflect the effects of ISM turbulence and orientation.
