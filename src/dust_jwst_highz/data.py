from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"
import pandas as pd


@lru_cache(maxsize=8)
def hirashita19_attenuation_curve(
    file_path: str | Path = DEFAULT_DATA_DIR / "Hirashita_dense_01Gyr_AttCurve.csv",
) -> interp1d:
    """Load Hirashita+2019 attenuation curve and return interpolation function.

    Parameters
    ----------
    file_path : str or Path
        Path to Hirashita attenuation curve data file.
        Default uses the packaged data file.

    Returns
    -------
    interp1d
        Interpolation function for A(λ)/A_V as a function of wavelength in Angstroms.

    """
    data = pd.read_csv(file_path)

    wavelength_ang = 1e4 / data["inv_lam_um"].to_numpy()  # Convert wavenumber (1/micron) to wavelength (Angstrom)
    attenuation = data["A_lam_over_AV"].to_numpy()  # A(λ)/A_V

    return interp1d(
        wavelength_ang,
        attenuation,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )


@lru_cache(maxsize=8)
def draine03_dust_model(
    file_path: str | Path = DEFAULT_DATA_DIR / "Draine_MWDustRv31_Optical_prop.csv",
) -> interp1d:
    """Load Draine+2003 Milky Way dust model and return interpolation function for kUV.

    Parameters
    ----------
    file_path : str or Path
        Path to Draine+2003 dust optical properties file.
        Default uses the MW dust RV=3.1 model.

    Returns
    -------
    interp1d
        Interpolation function for normalized kUV (absorption coefficient)
        as a function of wavelength in Angstroms.

    Notes
    -----
    The absorption coefficient is corrected for scattering (using albedo)
    and normalized at 5540 Å. The 90/163 factor converts from the standard
    gas-to-dust ratio to the adopted value.

    """
    data = pd.read_csv(file_path, comment="#")

    wavelength_ang = data.iloc[:, 0] * 1e4  # Convert microns to Angstrom
    albedo = data.iloc[:, 1]
    kabs = data.iloc[:, 4]
    # Correct for scattering and gas-to-dust ratio
    kabs_corrected = (kabs * 90 / 163) / (1 - albedo)

    # Normalize at V-band (5540 Å)
    normalization_idx = np.argmin(np.abs(wavelength_ang - 5540))
    kabs_corrected /= kabs_corrected[normalization_idx]

    return interp1d(
        wavelength_ang,
        kabs_corrected,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
