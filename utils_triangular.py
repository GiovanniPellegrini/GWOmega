from gwbird.overlap import Response
import numpy as np, inspect
import constants
from scipy.interpolate import interp1d

def S0(f):
    return 3*(constants.H0**2)/(10*(np.pi**2)*(f**3))

def noise_power_law(f, f_pivot, N_amplitude,r, n_noise):
    """
    Compute the noise correlated power spectral density using a power-law model.
    Parameters:
     f : array_like
          Frequency array.
     f_pivot : float
          Pivot frequency.
     N_noise : float
          Noise amplitude at the pivot frequency.
     r : float
          correlation factor.
     n_noise : float
          Spectral index.
    """
    noise = N_amplitude*r * (f / f_pivot) ** n_noise
    return noise

def gamma_aet(f):
   """
   Compute the overlap reduction functions for the A and E channels of the Einstein Telescope.
   Parameters:
    f : array_like
         Frequency array.
   """

   gamma_aet=np.zeros((3, len(f)))
   gamma_aet[0]=Response.overlap(det1="ET A", det2="ET A",f=f, pol="t",psi=0) # A channel
   gamma_aet[1]=Response.overlap(det1="ET E", det2="ET E",f=f, pol="t",psi=0) # E channel
   gamma_aet[2]=Response.overlap(det1="ET T", det2="ET T",f=f, pol="t",psi=0) # T channel
   return gamma_aet

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

def N_aet(f, N_auto, f_pivot, N_amplitude, r, n_noise):
    """
    Compute the noise matrix for the A and E channels of the Einstein Telescope.
    Parameters:
     N_auto : array_like
          Auto-correlation noise amplitudes for A and E channels.
    """
    N=np.zeros((3, len(f)))
    N_corr=noise_power_law(f, f_pivot, N_amplitude, r, n_noise)
    N[0]=N_auto+ N_corr 
    N[1]=N_auto+ N_corr
    N[2]=N_auto-2*N_corr

    return N   


def constrain(f, N_auto, f_pivot, N_amplitude, parameters):
    """
    Function to include in the prior constraints on the noise parameters.
    """    
    res = []
    r = parameters["r"]
    n_noise = parameters["n_noise"]
    factor = 1 + (1 - np.sign(r)) / 2
    
    for i in range(len(f)):
        N_corr_i = N_amplitude * r * (f[i] / f_pivot) ** n_noise
        res.append(N_auto[i] - factor * np.abs(N_corr_i))
    
    parameters['min_res'] = float(np.min(res))
    if 'eta_R' in parameters and 'k_max' in parameters:
        # Calcoliamo il prodotto che vogliamo vincolare
        prod= parameters['eta_R'] * parameters['k_max']
        prod_scalar = np.asarray(prod).item() if np.asarray(prod).size == 1 else prod
        parameters['product_eta_k'] = prod_scalar
    return parameters

class ConstrainWrapper:
    """
    Wrapper class for the constrain function to make it picklable for multiprocessing.
    """
    def __init__(self, f, N_auto, f_pivot, N_amplitude):
        self.f = f
        self.N_auto = N_auto
        self.f_pivot = f_pivot
        self.N_amplitude = N_amplitude
    
    def __call__(self, parameters):
        return constrain(self.f, self.N_auto, self.f_pivot, self.N_amplitude, parameters)