import numpy as np


def chi(chi_0, chi_1, chi_2, z):
    """Evaluate quadratic redshift dependence using Horner's method."""
    return chi_0 + z * (chi_1 + chi_2 * z)


def enforce_monotonic(x_array, y_array, threshold=0.05, direction="increasing", verbose=False):
    """Enforce monotonicity by detecting reversals and redistributing values.

    Uses np.searchsorted and np.add.at to efficiently redistribute non-monotonic
    tail values back into the monotonic region.

    Parameters
    ----------
    x_array : array-like
        Independent variable (e.g., magnitude, wavelength)
    y_array : array-like
        Dependent variable (e.g., probability density, flux)
    threshold : float, optional
        Threshold for detecting monotonicity breaks. Default: 0.05
    direction : {'increasing', 'decreasing'}, optional
        Expected direction of monotonicity. Default: 'increasing'
    verbose : bool, optional
        Whether to print diagnostic messages. Default: False

    Returns
    -------
    x_mono : ndarray
        Monotonic x values (trimmed at break point)
    y_mono : ndarray
        Corrected y values with non-monotonic tail redistributed

    Notes
    -----
    This function detects where x_array reverses direction (violating
    monotonicity), then redistributes the y values from the non-monotonic
    tail back into matching bins in the monotonic region. This preserves
    the total integral of y while ensuring x is strictly monotonic.

    For 'increasing' direction: detects where x starts decreasing (diffs < -threshold)
    For 'decreasing' direction: detects where x starts increasing (diffs > threshold)

    Examples
    --------
    >>> # MUV naturally increases (bright→faint), then folds back
    >>> x = np.array([-22, -21, -20, -19, -18.5, -19, -19.5, -20])
    >>> y = np.array([0.001, 0.01, 0.1, 0.5, 2.0, 3.0, 4.0, 5.0])
    >>> x_mono, y_mono = enforce_monotonic(x, y)
    >>> # Returns: x_mono = [-22, -21, -20, -19, -18.5]
    >>> #          y_mono with fold-back values redistributed

    """
    x_array = np.asarray(x_array)
    y_array = np.asarray(y_array)

    # Compute differences
    diffs = np.diff(x_array)

    # Detect break: where monotonicity is VIOLATED
    if direction == "increasing":
        # x should keep increasing; break if it decreases
        break_mask = diffs <= -threshold
    elif direction == "decreasing":
        # x should keep decreasing; break if it increases
        break_mask = diffs >= threshold
    else:
        raise ValueError(f"direction must be 'increasing' or 'decreasing', got '{direction}'")

    break_idx = np.argmax(break_mask) + 1 if np.any(break_mask) else len(x_array)

    if break_idx < len(x_array):
        if verbose:
            print(f"Monotonicity breaks at index {break_idx} (x = {x_array[break_idx]:.3f})")

        # Use the monotonic part as bin edges
        bins = x_array[:break_idx]

        # Find which bin each non-monotonic point belongs to
        # (closest value in monotonic region)
        indices = np.searchsorted(bins, x_array[break_idx:], side="left")
        indices = np.clip(indices, 0, len(bins) - 1)  # keep in bounds

        # Accumulate y values using np.add.at (vectorized, no loops!)
        y_corr = y_array[:break_idx].copy()
        np.add.at(y_corr, indices, y_array[break_idx:])

        return bins, y_corr
    else:
        if verbose:
            print("Array is monotonic — no redistribution needed.")
        return x_array, y_array
