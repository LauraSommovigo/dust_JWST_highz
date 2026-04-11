from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.colors import Colormap, LinearSegmentedColormap

# Plotting styles organized by data source
# Each source has default styling parameters that can be overridden in Plot_LF_Data
PLOTTING_STYLES = {
    "Harikane23": {
        "marker": "h",
        "color": "teal",
        "label": {10: r"spec, Harikane+23, $z=9$"},
        "ms": 12,
        "alpha": 0.5,
        "capsize": 5,
    },
    "Donnan24": {
        "marker": "o",
        "color": "black",
        "label": {10: r"phot, Donnan+24, $z=10$", 12: r"Donnan+24, $z=11.5-12.5$", 14: r"Donnan+24, $z=14.5$"},
        "ms": {10: 8, 12: 10, 14: 9},
        "alpha": 0.4,
        "capsize": {10: 5, 12: 4, 14: 5},
    },
    "Whitler25": {
        "marker": "*",
        "color": "dimgrey",
        "label": {10: r"Whitler+25, $z_{\rm med}=9.8$", 12: r"Whitler+25, $z=12.8$", 14: r"Whitler+25, $z=14.3$"},
        "ms": 10,
        "alpha": 0.6,
        "capsize": {10: 5, 12: 4, 14: 4},
        "mew": {10: 1.5, 12: 1.2, 14: 1.2},
    },
    "McLeod23": {
        "marker": "d",
        "color": "grey",
        "label": {12: r"McLeod+23, $z=9.5-12.5$", 14: r"McLeod+23, $12.5 < z < 14.5$"},
        "ms": {12: 8, 14: 9},
        "alpha": {12: 0.5, 14: 0.6},
        "capsize": {12: 5, 14: 4},
        "mew": {12: 1.5, 14: 1.2},
    },
    "CEERS_Finkelstein23": {
        "marker": "x",
        "color": "grey",
        "label": {12: r"CEERS, $z=9.7-13$", 14: r"CEERS, $z>13$"},
        "ms": 10,
        "alpha": 0.6,
        "capsize": 4,
    },
    "Casey23": {
        "marker": "D",
        "color": "silver",
        "label": {12: r"Casey+23, $z=9.5-12.5$", 14: r"Casey+23, $z=13-15$"},
        "ms": 10,
        "alpha": {12: 0.6, 14: 0.4},
        "capsize": 4,
    },
    "Leung23": {
        "marker": "s",
        "color": "grey",
        "label": {12: r"Leung+23, $z=11$"},
        "ms": 10,
        "alpha": 0.5,
        "capsize": 4,
        "mew": 1.2,
    },
    "Robertson24": {
        "marker": "+",
        "color": "grey",
        "label": {12: r"Robertson+24, $11.5 < z < 13.5$", 14: r"Robertson+24, $13.5 < z < 15$"},
        "ms": 9,
        "alpha": 0.6,
        "capsize": 5,
        "mew": 1.2,
    },
    "Oesch18": {
        "marker": "s",
        "color": "grey",
        "label": {10: r"phot, Oesch+18, $z\sim 10$"},
        "ms": 8,
        "alpha": 0.5,
    },
}


def truncate_colormap(cmap: Colormap, minval: float = 0.0, maxval: float = 1.0, n: int = 10) -> LinearSegmentedColormap:
    """Truncate a colormap to a specified range.

    Parameters
    ----------
    cmap : Colormap
        The colormap to truncate.
    minval : float, optional
        The minimum value of the colormap range (default is 0.0).
    maxval : float, optional
        The maximum value of the colormap range (default is 1.0).
    n : int, optional
        The number of discrete colors in the truncated colormap (default is 10).

    Returns
    -------
    LinearSegmentedColormap
        A new colormap that spans from minval to maxval of the original colormap.

    """
    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval), cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap


def plot_lf_data(z, ax, data_dir=None, style_overrides=None):
    """Plot observed UV luminosity function (LF) data from various literature sources for a given redshift.

    Parameters
    ----------
    z : float or int
        Redshift at which to plot the LF data.
    ax : matplotlib.axes.Axes
        Matplotlib axis object on which to plot the LF data.
    data_dir : Path or str, optional
        Path to data directory. If None, uses ../data relative to this file.
    style_overrides : dict, optional
        Dictionary to override default plotting styles from PLOTTING_STYLES.
        Format: {"SourceName": {"param": value, ...}, ...}
        Example: {"Donnan24": {"color": "red", "alpha": 0.8}}

    Returns
    -------
    None
        The function adds data points and error bars to the provided axis but does not return any value.

    Notes
    -----
    Plotting styles are defined in the PLOTTING_STYLES dictionary and can be overridden
    per-source using the style_overrides parameter. Data is loaded from uv_lf_observations.csv.

    """
    # Setup data directory
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / "data"
    else:
        data_dir = Path(data_dir)

    # Setup style overrides
    if style_overrides is None:
        style_overrides = {}

    def _get_style_param(source, param, redshift=None):
        """Get a style parameter for a source, checking overrides first, then defaults.

        If param value is a dict keyed by redshift, return the value for this redshift.
        """
        # Check override first
        if source in style_overrides and param in style_overrides[source]:
            value = style_overrides[source][param]
        # Then check default
        elif source in PLOTTING_STYLES and param in PLOTTING_STYLES[source]:
            value = PLOTTING_STYLES[source][param]
        else:
            return None

        # If value is a dict, look up by redshift
        if isinstance(value, dict) and redshift is not None:
            return value.get(redshift, None)
        return value

    # Common plot styling defaults
    DEFAULT_STYLE = {  # noqa N806
        "ls": "None",
        "capsize": 4,
        "elinewidth": 0.8,
        "mew": 1.5,
        "mec": "black",
    }

    def _plot_detections(muv, phi, yerr, marker, color, label, ms=10, alpha=0.7, **kwargs):
        """Helper to plot detections with consistent styling."""
        style = {**DEFAULT_STYLE, **kwargs}
        ax.errorbar(muv, phi, yerr=yerr, marker=marker, ms=ms, color=color, alpha=alpha, label=label, **style)

    def _plot_upper_limits(muv, phi, marker, color, label=None, ms=10, alpha=0.7, arrow_frac=0.4):
        """Helper to plot upper limits with downward arrows."""
        style = DEFAULT_STYLE.copy()
        ax.errorbar(
            muv,
            phi,
            yerr=arrow_frac * np.asarray(phi),
            uplims=True,
            marker=marker,
            ms=ms,
            color=color,
            alpha=alpha,
            label=label,
            **style,
        )

    # ===== Load observational data from CSV =====
    obs_data_file = data_dir / "uv_lf_observations.csv"
    if obs_data_file.exists():
        df = pd.read_csv(obs_data_file, comment="#")
        df_z = df[df["redshift"] == z].copy()

        if len(df_z) > 0:
            # Group by source to plot each dataset
            for source, group in df_z.groupby("source", sort=False):
                detections = group[group["is_upper_limit"] == 0]
                upper_limits = group[group["is_upper_limit"] == 1]

                # Get style parameters for this source
                marker = _get_style_param(source, "marker")
                color = _get_style_param(source, "color")
                label = _get_style_param(source, "label", z)
                ms = _get_style_param(source, "ms", z) or 10
                alpha = _get_style_param(source, "alpha", z) or 0.7

                # Optional parameters
                kwargs = {}
                capsize = _get_style_param(source, "capsize", z)
                if capsize is not None:
                    kwargs["capsize"] = capsize
                mew = _get_style_param(source, "mew", z)
                if mew is not None:
                    kwargs["mew"] = mew

                # Plot detections
                if len(detections) > 0:
                    yerr = [
                        detections["phi_err_low"].values,
                        detections["phi_err_up"].values,
                    ]

                    _plot_detections(
                        detections["muv"].values,
                        detections["phi"].values,
                        yerr,
                        marker,
                        color,
                        label,
                        ms=ms,
                        alpha=alpha,
                        **kwargs,
                    )

                # Plot upper limits (no label on upper limits)
                if len(upper_limits) > 0:
                    _plot_upper_limits(
                        upper_limits["muv"].values,
                        upper_limits["phi"].values,
                        marker,
                        color,
                        ms=ms,
                        alpha=alpha,
                    )

    # ===== z < 10: Bouwens+21 (from CSV file) =====
    if z < 10:
        obs_data = pd.read_csv(data_dir / "Bouwens21_z2-9.csv", comment="#")

        mask = obs_data["redshift"] == z
        m_uv = obs_data["MUV"][mask].to_numpy(dtype=float)
        phi = obs_data["phi"][mask].to_numpy(dtype=float)
        err_phi_low = obs_data["err_phi_low"][mask].to_numpy(dtype=float)
        err_phi_up = obs_data["err_phi_up"][mask].to_numpy(dtype=float)

        # Identify upper limits: missing or huge upper error
        is_upper_limit = np.isnan(err_phi_up) | (err_phi_up > phi)
        is_detection = ~is_upper_limit

        if np.any(is_detection):
            err_phi_low_det = np.minimum(err_phi_low[is_detection], 0.99 * phi[is_detection])
            err_phi_up_det = np.minimum(err_phi_up[is_detection], 0.99 * phi[is_detection])
            _plot_detections(
                m_uv[is_detection],
                phi[is_detection],
                [err_phi_low_det, err_phi_up_det],
                "s",
                "grey",
                f"Obs, Bouwens+21, $z={int(z)}$",
            )

        if np.any(is_upper_limit):
            _plot_upper_limits(m_uv[is_upper_limit], phi[is_upper_limit], "s", "grey", arrow_frac=0.2)
