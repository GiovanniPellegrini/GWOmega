# define useful constants
from astropy.cosmology import Planck18
import astropy.units as u
import numpy as np


H0 = Planck18.H0.to(1/ u.s).value
Omega_r0_hh = 4.2e-5
c_g = 0.39
cs_value = 1/np.sqrt(3)
