import numpy as np
from utils_triangular import S0, N_aet, gamma_aet


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

    gamma_matrix= gamma_aet(f)
    h= np.zeros((2, len(f)))
    for i in range(2):
         h[i]=np.sqrt(0.5*T_seg*S0(f)*gamma_matrix[i]*Omega_gw)

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