from scipy.interpolate import RegularGridInterpolator
import numpy as np
from Omega import compute_Omega_vectorized, precompute_geometry, compute_Omega_eMD_today_fast, P_theta_vec
import constants
import pickle
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
"""
Functions to interpolate Omega_RD_today values for a given set of parameters, to be used in the likelihood function.
"""


def _compute_row_RD(args):
    i, j, k_peak, sigma, k_values = args
    geometry_data = precompute_geometry()  # ogni worker ha la sua copia
    Omega_values = compute_Omega_vectorized(k_values, k_peak, sigma, geometry_data, A_s=1)
    return i, j, Omega_values


def compute_integral_grid_RD(k_values, k_peak_values, sigma_values, max_workers=8):
    integral_grid = np.zeros((len(k_peak_values), len(sigma_values), len(k_values)))

    tasks = []
    for i, k_peak in enumerate(k_peak_values):
        for j, sigma in enumerate(sigma_values):
            tasks.append((i, j, k_peak, sigma, k_values))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_compute_row_RD, t): t for t in tasks}

        with tqdm(total=len(tasks), desc="Computing Omega grid RD") as pbar:
            for future in as_completed(futures):
                i, j, Omega_values = future.result()
                integral_grid[i, j, :] = Omega_values
                pbar.update(1)

    print("Max Omega value in the grid is:", np.max(integral_grid))
    return integral_grid

def save_grid_RD(filename, k_peak_values, sigma_values, Omega_grid):
    data={'k_peak_values': k_peak_values, 'sigma_values': sigma_values, 'Omega_grid': Omega_grid}
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_grid_RD(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return (data['k_peak_values'], data['sigma_values']), data['Omega_grid']


def create_interpolator_RD(filename):
    grid_data, Omega_grid = load_grid_RD(filename)
    interpolator = RegularGridInterpolator(grid_data, Omega_grid,  bounds_error=False, fill_value=None)
    return interpolator


class ScaledInterpolatorRD:
    """
    Class to handle interpolation of Omega_GW values scaled by A_s^2.
    """
    def __init__(self, filename):
        self.interpolator = create_interpolator_RD(filename)

    def __call__(self, k_peak, sigma, A_s):
        Omega_scaled = self.interpolator((k_peak, sigma)) * (A_s ** 2)
        return Omega_scaled
    




# Funzione worker da eseguire in parallelo (deve essere top-level per pickle)
def _compute_row(args):
    i, j, k_max, x_R, k_values = args
    eta_R=x_R/ k_max
    P_func = lambda k: P_theta_vec(k, k_max, A_s=1.0)
    Omega = np.zeros(len(k_values))
    for idx, k in enumerate(k_values):
        Omega[idx] = compute_Omega_eMD_today_fast(k, eta_R, k_max, P_func,N_s_1=30, N_s_2=500, N_t_1=300, N_t_3=100, N_t_2=0)

    return i, j, Omega



def compute_integral_grid_eMD(k_values, k_max_values, x_R_values, 
                               max_workers=8, checkpoint_file='emd_grid_checkpoint.pkl'):
    if os.path.exists(checkpoint_file):
        print("Checkpoint trovato, riprendo...")
        with open(checkpoint_file, 'rb') as f:
            data = pickle.load(f)
        integral_grid = data['grid']
        done_pairs    = data['done']
        print(f"Già completati: {len(done_pairs)} punti")
    else:
        integral_grid = np.zeros((len(k_max_values), len(x_R_values), len(k_values)))
        done_pairs    = set()
    tasks = []
    for i, k_max in enumerate(k_max_values):
        for j, x_R in enumerate(x_R_values):
            if (i, j) in done_pairs:
                continue  # Salta se già fatto
            tasks.append((i, j, k_max, x_R, k_values))

    CHECKPOINT_EVERY = 250

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_compute_row, t): t for t in tasks}
        with tqdm(total=len(tasks), desc="Computing Omega grid eMD") as pbar:
            for  count, future in enumerate(as_completed(futures)):
                i, j, Omega = future.result()
                integral_grid[i, j, :] = Omega
                done_pairs.add((i, j))
                pbar.update(1)

                if count % CHECKPOINT_EVERY == 0:
                    with open(checkpoint_file, 'wb') as f:
                        pickle.dump({'grid': integral_grid, 'done': done_pairs}, f)

    # Salva finale e rimuovi checkpoint
    os.remove(checkpoint_file)
    print("Max Omega:", np.nanmax(integral_grid))

    return integral_grid



"""
def save_grid_eMD(filename, k_max_values, eta_R_values, Omega_grid):
    data = {
        'k_max_values': k_max_values,
        'eta_R_values': eta_R_values,      
        'Omega_grid': Omega_grid
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_grid_eMD(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    Omega_grid = data['Omega_grid']    
    Omega_grid = np.nan_to_num(Omega_grid, nan=0.0)

    return (data['k_max_values'], data['eta_R_values']), Omega_grid


def create_interpolator_eMD(filename):
    grid_data, Omega_grid = load_grid_eMD(filename)
    interpolator = RegularGridInterpolator(grid_data, Omega_grid, method='linear', bounds_error=True)
    return interpolator


class ScaledInterpolatorEMD:
   
    def __init__(self, filename):
        self.interpolator = create_interpolator_eMD(filename)

    def __call__(self, k_max, eta_R, A_s):    
        Omega_scaled = self.interpolator((k_max, eta_R)) * (A_s ** 2)
        return Omega_scaled
    
"""


def save_grid_eMD(filename, k_max_values, x_R_values, Omega_grid):
    data = {
        'k_max_values': k_max_values,
        'x_R_values': x_R_values,      
        'Omega_grid': Omega_grid
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_grid_eMD(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    Omega_grid = data['Omega_grid']    
    Omega_grid = np.nan_to_num(Omega_grid, nan=0.0)

    return (data['k_max_values'], data['x_R_values']), Omega_grid


def create_interpolator_eMD(filename):
    grid_data, Omega_grid = load_grid_eMD(filename)
    log_Omega = np.log(np.maximum(Omega_grid, 1e-100))

    interpolator = RegularGridInterpolator(grid_data, log_Omega, method='linear', bounds_error=False,
        fill_value=np.log(1e-100))
    return interpolator


class ScaledInterpolatorEMD:


    def __init__(self, filename):
        self.interpolator = create_interpolator_eMD(filename)

    def __call__(self, k_max, x_R, A_s):
       
        log_Omega = self.interpolator((k_max, x_R))
        Omega     = np.exp(log_Omega)        
        Omega[Omega <= 1e-99] = 0.0
        
        Omega_scaled = Omega * (A_s ** 2)
        return Omega_scaled # restituisci solo i primi 106 valori (fino a ~11.5 Hz)
  