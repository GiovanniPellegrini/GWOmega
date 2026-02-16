from scipy.interpolate import RegularGridInterpolator
import numpy as np
from Omega import compute_Omega_vectorized, precompute_geometry, compute_Omega_eMD_today_fast, P_theta_vec
import constants
import pickle
from tqdm import tqdm
"""
Functions to interpolate Omega_RD_today values for a given set of parameters, to be used in the likelihood function.
"""


def compute_integral_grid_RD(k_values, k_peak_values, sigma_values):
    geometry_data= precompute_geometry()
    integral_grid = np.zeros((len(k_peak_values), len(sigma_values), len(k_values)))
    total=len(k_peak_values)*len(sigma_values)
    with tqdm(total=total, desc="Computing Omega grid") as pbar:
        for i, k_peak in enumerate(k_peak_values):
            for j, sigma in enumerate(sigma_values):
                Omega_values = compute_Omega_vectorized(k_values, k_peak, sigma, geometry_data, A_s=2*1e-9)
                integral_grid[i, j, :] = Omega_values
                pbar.update(1)
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
    interpolator = RegularGridInterpolator(grid_data, Omega_grid, bounds_error=False, fill_value=None)
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
    


def compute_integral_grid_eMD(k_values, k_max_values, eta_R_values):
    integral_grid=np.zeros((len(k_max_values), len(eta_R_values), len(k_values)))
    
    with tqdm(total=len(k_max_values)*len(eta_R_values), desc="Computing Omega grid for eMD") as pbar:
        skips_count=0
        for i, k_max in enumerate(k_max_values):
            P_func=lambda k: P_theta_vec(k, k_max, A_s=1.0)
            for j, eta_R in enumerate(eta_R_values):
                #if eta_R*k_max<50 or eta_R*k_max>200:
                 #   skips_count += 1
                  #  continue
                Omega=np.zeros(len(k_values))    
                for idx, k in enumerate(k_values):
                    Omega_values=compute_Omega_eMD_today_fast(k, eta_R, k_max, P_func)
                    Omega[idx]=Omega_values
                    #print("Max Omega value for k =", k, "is:", np.max(Omega))
                integral_grid[i,j,:]=Omega
                pbar.update(1)
        print(" Max Omega value in the grid is:", np.max(integral_grid))
        print(f"Skipped {skips_count} combinations of (k_max, eta_R) due to being outside the valid range.")        
    return integral_grid

def save_grid_eMD(filename, k_max_values, eta_R_values, Omega_grid):
    data={'k_max_values': k_max_values, 'eta_R_values': eta_R_values, 'Omega_grid': Omega_grid}
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_grid_eMD(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return (data['k_max_values'], data['eta_R_values']), data['Omega_grid']

def create_interpolator_eMD(filename):
    grid_data, Omega_grid = load_grid_eMD(filename)
    interpolator = RegularGridInterpolator(grid_data, Omega_grid, bounds_error=False, fill_value=None)
    return interpolator

class ScaledInterpolatorEMD:
    """
    Class to handle interpolation of Omega_GW values for eMD scenario.
    """
    def __init__(self, filename):
        self.interpolator = create_interpolator_eMD(filename)

    def __call__(self, k_max, eta_R, A_s):
        Omega_scaled = self.interpolator((k_max, eta_R)) * (A_s ** 2)
        return Omega_scaled