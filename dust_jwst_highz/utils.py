def chi(chi_0, chi_1, chi_2, z):
    """Evaluate quadratic redshift dependence using Horner's method."""
    return chi_0 + z * (chi_1 + chi_2 * z)
