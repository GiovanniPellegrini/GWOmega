from gwbird.overlap import Response
import numpy as np, inspect
import constants
from scipy.interpolate import interp1d

def S0(f):
    return 3*(constants.H0**2)/(10*(np.pi**2)*(f**3))

def gamma_2L(f, shift_angle=0):
   """
   Compute the overlap reduction functions for a 2L detector configuration.
   Parameters:
    f : array_like
         Frequency array.
    shift_angle : float
         Shift angle between the two L-shaped detectors in radians.
   """
   return Response.overlap(det1="ET L1", det2="ET L2",f=f, pol="t",psi=0, shift_angle=shift_angle) 

def N_auto_interp(filename, Nanfill=0.0):
    """
    Load the auto-correlation noise amplitude from a file and create an interpolation function.
    Parameters:
     filename : str
          Path to the file containing frequency and noise amplitude data.
    """
    data = np.loadtxt(filename)
    f_data = data[:, 0]
    N_auto_data = data[:, 1]
    N_auto_function = interp1d(f_data, N_auto_data, bounds_error=False, fill_value="extrapolate")
    def N_auto(f):
        N_values = N_auto_function(f)
        N_values[np.isnan(N_values)] = Nanfill
        return N_values**2
    return N_auto

