from gwbird.overlap import Response
import numpy as np
import constants

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

   gamma_aet=np.zeros(3, len(f))
   gamma_aet[0]=Response.overlap(det1="ET A", det2="ET A",f=f, pol="t",psi=0) # A channel
   gamma_aet[1]=Response.overlap(det1="ET E", det2="ET E",f=f, pol="t",psi=0) # E channel
   gamma_aet[2]=Response.overlap(det1="ET T", det2="ET T",f=f, pol="t",psi=0) # T channel
   return gamma_aet

#capire come io avrò accesso ai dati di frequenza, e come questi mi danno N_autocorrelation 
def N_f(f):
   ...

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

def signal_aet(f, Omega_gw, T_seg, N_seg):
    """
    Compute the signal matrix for the A and E channels of the Einstein Telescope.
    Parameters:
     f : array_like
            Frequency array.
     T_seg : float
            Segment time duration.
     N_seg : int
            Number of segments.
    """

    gamma_aet= gamma_aet(f)
    h= np.zeros((2, len(f)))
    for i in range(2):
         h[i]=np.sqrt(0.5*T_seg*S0(f)*gamma_aet[i]*Omega_gw)

    h_re = np.random.normal(0, 1/(np.sqrt(2)), (N_seg, 2, len(f))) * h
    h_im = np.random.normal(0, 1/(np.sqrt(2)), (N_seg, 2, len(f))) * h
    return h_re + 1j * h_im

def noise_aet(f, N_auto, f_pivot, N_amplitude, r, n_noise, T_seg, N_seg):
    """
    Compute the noise matrix for the A and E channels of the Einstein Telescope.
    Parameters:
     f : array_like
            Frequency array.
     N_auto : float
            Auto-correlation noise amplitude.
     T_seg : float
            Segment duration.
     N_seg : int
            Number of segments.
    """

    n=np.sqrt(0.5*T_seg*N_aet(f, N_auto, f_pivot, N_amplitude, r, n_noise)[:2])
    n_re = np.random.normal(0, 1/(np.sqrt(2)), (N_seg, 2, len(f))) * n
    n_im = np.random.normal(0, 1/(np.sqrt(2)), (N_seg, 2, len(f))) * n 
    return n_re + 1j * n_im


def constrain(f, N_auto, f_pivot, N_amplitude, parameters):
    """
    Function to include in the prior constraints on the noise parameters, in particula -0.5<= (N_corr/N_auto) <=1.

    """    
    
    N_corr = noise_power_law(f, f_pivot, N_amplitude, parameters["r"],parameters["n_noise"]) 
    factor = 1 + (1 - np.sign(parameters["r"])) / 2
    res = N_auto - factor * np.abs(N_corr)
    parameters['min_res'] = np.min(res, axis=0)
    return parameters
    
       