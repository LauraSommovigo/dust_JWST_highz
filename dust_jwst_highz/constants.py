from astropy import constants as _const
from astropy import units as _units

angstrom = 1e-8  # angstrom to cm
micron = 1e-4  # micron to cm
proton_mass = 1.67262192e-24  # Proton mass [g]
mean_mol_weight = 1.22  # Mean molecular weight (typical ISM)
dust_ratio_mw = 1.0 / 162.0  # Milky Way dust-to-gas ratio
Z_sun = 0.0142  # Solar metallicity
M_sun = _const.M_sun.cgs.value
m_p = _const.m_p.cgs.value
c = _const.c.cgs.value
Mpc = _units.Mpc.to("cm")
kpc = _units.kpc.to("cm")
L_sun = _const.L_sun.cgs.value
k_B = _const.k_B.cgs.value  # noqa N816
h = _const.h.cgs.value
