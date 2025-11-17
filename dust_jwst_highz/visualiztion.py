import numpy as np
from matplotlib import colors
from matplotlib.colors import Colormap, LinearSegmentedColormap


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
