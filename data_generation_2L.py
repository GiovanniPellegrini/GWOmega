import numpy as np
from utils_2L import gamma_2L, N_auto_interp, S0
import h5py
from Omega import P_k_lognormal, compute_Omega_RD_today_fast, compute_Omega_eMD_today_fast, P_theta_vec
import constants
from interpolation import ScaledInterpolatorEMD
import tqdm as tqdm


"""
Data generation functions for the Einstein Telescope 2L configuration. We consider an uncorrelated noise model.
"""

def signal_2L(f, Omega_gw, T_seg, N_seg, shift_angle, filename=None):
    """
    Compute the signal matrix for the 2L configuration of the Einstein Telescope.
    Parameters:
     f : array_like
            Frequency array.
     T_seg : float
            Segment time duration.
     N_seg : int
            Number of segments.
     shift_angle : float
            Shift angle between the two L-shaped detectors in radians.
    """
    rng=np.random.default_rng()
    gamma= gamma_2L(f, shift_angle)
    gamma_matrix=np.array([[np.ones(len(f)), gamma], [gamma, np.ones(len(f))]])
    h=np.sqrt(0.5*T_seg*S0(f)*Omega_gw)

    h_re =  np.zeros((N_seg, 2, len(f)))
    h_im =  np.zeros((N_seg, 2, len(f)))
    for i in range(len(f)):
        h_re[:,:,i]=rng.multivariate_normal(np.zeros(2), 0.5*gamma_matrix[:,:,i], N_seg, method='cholesky')*h[i]    
        h_im[:,:,i]=rng.multivariate_normal(np.zeros(2), 0.5*gamma_matrix[:,:,i], N_seg, method='cholesky')*h[i]
    
    signal = h_re + 1j * h_im
    #print("Signal data generated")
    if filename is not None:
        with h5py.File(filename, "w") as hf:
            hf.create_dataset("signal", data=signal, compression="gzip")
        print (f"Signal data saved to {filename}")
        print("Signal shape:", signal.shape)
    return signal
    

def noise_2L(f, N_auto, T_seg, N_seg, filename=None):
    """
    Compute the noise matrix for the 2L configuration of the Einstein Telescope.
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
    

    n=np.sqrt(0.5*T_seg*N_auto)
    n_re = np.random.normal(0, 1/(np.sqrt(2)), (N_seg, 2, len(f))) * n
    n_im = np.random.normal(0, 1/(np.sqrt(2)), (N_seg, 2, len(f))) * n
    noise = n_re + 1j * n_im
    #print("Noise data generated")
    if filename is not None:
        with h5py.File(filename, "w") as hf:
            hf.create_dataset("noise", data=noise, compression="gzip")
        print (f"Noise data saved to {filename}")
        print("Noise shape:", noise.shape)
    return noise


def create_data_files_2L_RD(A_s, k_peak, sigma, T_obs, N_seg, shift_angle=0, C_hat_filename="data/C_hat_2L_RD.h5", parameters_filename="data/parameters_2L_RD.txt"):
    """
    Create data files for the 2L configuration of the Einstein Telescope with correlated noise in RD
    """
    
    T_seg= T_obs / N_seg
    df= 1 / T_seg
    f_values= np.arange(1, 200+df, df)
    k_values= f_values * 2 * np.pi

    N_func= N_auto_interp("data/ET_15km_ASD.txt")
    N_auto=N_func(f_values)
    P_func= lambda k: P_k_lognormal(k, k_peak, sigma, A_s)
    Omega_gw = [
    compute_Omega_RD_today_fast(float(k), constants.cs_value, P_func)    for k in k_values
    ]
    n=noise_2L(f_values, N_auto, T_seg, N_seg)
    h=signal_2L(f_values, Omega_gw, T_seg, N_seg, shift_angle)
    d=h+n
    C_hat=(2/(T_seg*S0(f_values)))*np.real(d[:,0,:]*np.conj(d[:,1,:]))
    C_hat=np.average(C_hat, axis=0)
    with h5py.File(C_hat_filename, "w") as hf:
        hf.create_dataset("C_hat", data=C_hat, compression="gzip")
    print(f"C_hat data saved to {C_hat_filename}")
    print("C_hat shape:", C_hat.shape)
    parameters= {
        "A_s": A_s,
        "k_peak": k_peak,
        "sigma": sigma,
        "T_obs": T_obs,
        "N_seg": N_seg,
        "T_seg": T_seg,
        "shift_angle": shift_angle
    }
    write_parameters_to_file(parameters_filename, parameters)
    print("Data files created")


def create_data_files_2L_eMD(A_s, k_max, eta_R, T_obs, N_seg, shift_angle=0, C_hat_filename="data/2L/C_hat_2L_eMD.h5", parameters_filename="data/parameters_2L_eMD.txt"):
    """
    Create data files for the 2L configuration of the Einstein Telescope with correlated noise in eMD
    """
    T_seg= T_obs / N_seg
    df= 1 / T_seg
    f_values= np.arange(1, 200+df, df)
    k_values= f_values * 2 * np.pi

    N_func= N_auto_interp("data/ET_15km_ASD.txt")
    N_auto=N_func(f_values)
    P_func= lambda k: P_theta_vec(k, k_max, A_s)
    Omega_gw = [
    compute_Omega_eMD_today_fast(k, eta_R, k_max, P_func)    for k in k_values
    ]
    n=noise_2L(f_values, N_auto, T_seg, N_seg)
    h=signal_2L(f_values, Omega_gw, T_seg, N_seg, shift_angle)
    d=h+n
    C_hat=(2/(T_seg*S0(f_values)))*np.real(d[:,0,:]*np.conj(d[:,1,:]))
    C_hat=np.average(C_hat, axis=0)
    with h5py.File(C_hat_filename, "w") as hf:
        hf.create_dataset("C_hat", data=C_hat, compression="gzip")
    print(f"C_hat data saved to {C_hat_filename}")
    print("C_hat shape:", C_hat.shape)
    parameters= {
        "A_s": A_s,
        "k_max": k_max,
        "eta_R": eta_R,
        "T_obs": T_obs,
        "N_seg": N_seg,
        "T_seg": T_seg,
        "shift_angle": shift_angle
    }
    write_parameters_to_file(parameters_filename, parameters)
    print("Data files created")



def create_C_hat_2L_RD(A_s, k_peak, sigma, T_obs, N_seg, shift_angle=0, C_hat_filename="data/2L/C_hat_2L_RD.h5", parameters_filename="data/2L/parameters_2L_RD.txt"):
    """
    Create data files for the 2L configuration of the Einstein Telescope without correlated noise in RD
    We compute C_hat directly from the theoretical mean and variance, without generating individual signal and noise realizations.
    """
    T_seg= T_obs / N_seg
    df= 1 / T_seg
    f_values= np.arange(1, 200+df, df)
    k_values= f_values * 2 * np.pi

    rng=np.random.default_rng()
    P_func= lambda k: P_k_lognormal(k, k_peak, sigma, A_s)
    Omega_gw = [
    compute_Omega_RD_today_fast(float(k), constants.cs_value, P_func)    for k in k_values
    ]
    
    gamma= gamma_2L(f_values, shift_angle)
    N_func= N_auto_interp("data/ET_15km_ASD.txt")
    N_auto=N_func(f_values)
    S_0= S0(f_values)

    mean= gamma*Omega_gw
    var= (mean**2 + (N_auto/S_0 + Omega_gw)**2)/(2*N_seg)
    C_hat=rng.normal(mean, np.sqrt(var))

    with h5py.File(C_hat_filename, "w") as hf:
        hf.create_dataset("C_hat", data=C_hat, compression="gzip")
    print(f"C_hat data saved to {C_hat_filename}")
    print("C_hat shape:", C_hat.shape)
    parameters= {
        "A_s": A_s,
        "k_peak": k_peak,
        "sigma": sigma,
        "T_obs": T_obs,

        "N_seg": N_seg,
        "T_seg": T_seg,
        "shift_angle": shift_angle
    }
    write_parameters_to_file(parameters_filename, parameters)
    print("Data files created")
    

def create_C_hat_2L_eMD(A_s, k_max, eta_R, T_obs, N_seg, shift_angle=0, C_hat_filename="data/2L/C_hat_2L_eMD.h5", parameters_filename="data/2L/parameters_2L_eMD.txt"):
    """
    Create data files for the 2L configuration of the Einstein Telescope without correlated noise in eMD
    We compute C_hat directly from the theoretical mean and variance, without generating individual signal and noise realizations.
    """
    T_seg= T_obs / N_seg
    df= 1 / T_seg
    f_values= np.arange(1, 200+df, df)
    k_values= f_values * 2 * np.pi

    rng=np.random.default_rng()
    P_func= lambda k: P_theta_vec(k, k_max, A_s)
    Omega_gw = [
    compute_Omega_eMD_today_fast(k, eta_R, k_max, P_func)    for k in k_values
    ]    
    
    #interpolator=ScaledInterpolatorEMD("omega_grid_eMD_v8.pkl")
    #x_R=k_max*eta_R
    #Omega_gw=interpolator(k_max, x_R, A_s)
   
    gamma= gamma_2L(f_values, shift_angle)
    N_func= N_auto_interp("data/ET_15km_ASD.txt")
    N_auto=N_func(f_values)
    S_0= S0(f_values)

    mean= gamma*Omega_gw
    var= (mean**2 + (N_auto/S_0 + Omega_gw)**2)/(2*N_seg)
    C_hat=rng.normal(mean, np.sqrt(var))

    with h5py.File(C_hat_filename, "w") as hf:
        hf.create_dataset("C_hat", data=C_hat, compression="gzip")
    print(f"C_hat data saved to {C_hat_filename}")
    print("C_hat shape:", C_hat.shape)
    parameters= {
        "A_s": A_s,
        "k_max": k_max,
        "eta_R": eta_R,
        "T_obs": T_obs,
        "N_seg": N_seg,
        "T_seg": T_seg,
        "shift_angle": shift_angle
    }
    write_parameters_to_file(parameters_filename, parameters)
    print("Data files created")

    

def load_C_hat(filename):
    with h5py.File(filename, "r") as hf:
        C_hat = hf["C_hat"][:]
    print (f"C_hat data loaded from {filename}")
    print("C_hat shape:", C_hat.shape)
    return C_hat



def write_parameters_to_file(filename, params):
    with open(filename, 'w') as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
    print(f"Parameters written to {filename}")

def load_parameters_from_file(filename):
    params = {}
    with open(filename, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ')
            try:
                params[key] = float(value)
            except ValueError:
                params[key] = value
    print(f"Parameters loaded from {filename}")
    return params

def check_parameters(params, filename):
    loaded_params = load_parameters_from_file(filename)
    for key in params:
        if key not in loaded_params:
            print(f"Parameter {key} not found in file.")
            return False
        if params[key] != loaded_params[key]:
            print(f"Parameter {key} mismatch: {params[key]} != {loaded_params[key]}")
            return False
    print("All parameters match.")
    return True

