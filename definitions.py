import sys, os, re, math, itertools, matplotlib
from importlib import reload

import numpy as np
from scipy import optimize, stats, integrate
from scipy.interpolate import interp1d
from scipy.integrate import quad

# Importing dataclasses might not work in CASA environment
try:
    from dataclasses import dataclass
except:
    pass

from copy import deepcopy

from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.pyplot import cm
from matplotlib.patches import Circle, Ellipse, Rectangle
from matplotlib.colors import to_rgba
from matplotlib.collections import LineCollection
from matplotlib import gridspec
import matplotlib.ticker as ticker

from astropy import units as u
from astropy import constants as const
from astropy.coordinates import SkyCoord, Angle, match_coordinates_sky
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy import wcs
from astropy.io import fits, ascii
from astropy.table import Table, Column, vstack, hstack
from astropy.nddata import Cutout2D
from astropy.modeling.functional_models import Gaussian2D
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.stats import sigma_clipped_stats

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70.0*u.km/u.s/u.Mpc, Om0=0.3, Tcmb0=2.725*u.K)


# Set some default plotting parameters
rc('font',**{'family': 'sans-serif', 'sans-serif': ['STIX']})
rc('mathtext', fontset='stixsans')
rc('text.latex', preamble=r'\usepackage{cmbright}')
rc('text', usetex=True)
rc('xtick', labelsize=18) 
rc('ytick', labelsize=18) 

matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amsfonts}'
matplotlib.rcParams['contour.negative_linestyle']= 'dashed'
matplotlib.rcParams.update({'font.size': 20})
matplotlib.rcParams['figure.figsize'] = (8,5)

# Create a custom colormap
mycolors   = ['C0', 'C9', 'C4', 'C3', 'salmon', 'C1', '#76d138'] * 4 # repeat colors
almacolors = ['cornflowerblue', 'royalblue', 'slateblue', 'navy', 'cadetblue', 'darkcyan', 'deepskyblue'] * 4

mycmap   = matplotlib.colors.ListedColormap(mycolors)
mycmap.set_under("navy")
mycmap.set_over("gold")

# Some default parameters
wave_lyalpha_rest = 1215.67 * u.AA

# Commomly used emission lines observable with ALMA at high-z; data from splatalogue
linedict = {'CO(1-0)': 115.27120180 * u.GHz, 
            'CO(2-1)': 230.53800000 * u.GHz,
            'CO(3-2)': 345.79598990 * u.GHz,
            'CO(4-3)': 461.04076820 * u.GHz,
            'CO(5-4)': 576.26793050 * u.GHz,
            'CO(6-5)': 691.47307630 * u.GHz,
            'CO(7-6)': 806.65180600 * u.GHz,
            '[CI](2-1)': 809.34197 * u.GHz, 
            'CO(8-7)': 921.79970000 * u.GHz,
            'CO(9-8)': 1036.91239300 * u.GHz,
            'CO(10-9)': 1151.98545200 * u.GHz,
            'CO(11-10)': 1267.01448600 * u.GHz,
            'CO(12-11)': 1381.99510500 * u.GHz,
			'NII205': 1461.131 * u.GHz, 
            'CO(13-12)': 1496.92290900 * u.GHz,
            'CO(14-13)': 1611.79351800 * u.GHz,
            'CO(15-14)': 1726.6025057 * u.GHz,
            'CO(16-15)': 1841.34550600 * u.GHz,
            'CII158': 1900.53690000 * u.GHz, 
            'CO(17-16)': 1956.01813900 * u.GHz,
            'OI145':  2060.06909000 * u.GHz,
            'NII122': 2459.09 * u.GHz,
            'OIII88': 3393.00624400 * u.GHz, 
            'OI63':   4744.77749000 * u.GHz,
            'NIII57': 5240.21 * u.GHz, # not from spalatalogue; less accurate rest-wave of 57.21um
            'OIII52': 5785.87958900 * u.GHz}

# Common optical emisison lines; generally listed in VACUUM
linedict_opt = {'Lya': 1215.67 * u.AA,
                'CIV': 1548.2 * u.AA, # part of a doublet at 1548.2, 1550.8 Angstrom
                '[OII]3727': 3727.092 * u.AA, # part of a doublet; 3727 Angstrom
                '[OII]3729': 3729.875 * u.AA, # part of doublet; 3729 Angstrom
                '[NeIII]3869': 3869.0 * u.AA,
                'Hd': 4101.73 * u.AA,
                'Hg': 4341.691 * u.AA,
                '[OIII]4363': 4364.436 * u.AA,
                'Hb': 4862.721 * u.AA,
                '[OIII]4959': 4960.295 * u.AA,
                '[OIII]5007': 5008.239 * u.AA,
                '[NII]6548': 6549.86 * u.AA, # doublet; 6548 & 6583 Angstrom
                'Ha': 6564.614 * u.AA,
                '[NII]6583': 6585.27 * u.AA,
                '[OII]7320': 7319.99 * u.AA,
                '[OII]7331': 7330.73 * u.AA}

