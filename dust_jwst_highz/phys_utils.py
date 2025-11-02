import numpy as np
from scipy.integrate import quad


def planck_funct(y, bd):
    """Planck function integrand for computing the total emitted power."""
    return (y ** (3 + bd)) / (np.exp(y) - 1)


def planck_integral(bd):
    """Integral of the Planck function needed for computing grain temperatures."""
    return quad(planck_funct, 0.0, 35.0, args=(bd))[0]
