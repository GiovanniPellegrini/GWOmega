from scipy.interpolate import RegularGridInterpolator
import numpy as np
from Omega import compute_Omega_vectorized, precompute_geometry
import constants
import pickle
from tqdm import tqdm
"""
Functions to interpolate Omega_RD_today values for a given set of parameters, to be used in the likelihood function.
"""


def compute_integral_grid(k_values, k_peak_values, sigma_values):
    geometry_data= precompute_geometry()
    integral_grid = np.zeros((len(k_peak_values), len(sigma_values), len(k_values)))
    total=len(k_peak_values)*len(sigma_values)
    with tqdm(total=total, desc="Computing Omega grid") as pbar:
        for i, k_peak in enumerate(k_peak_values):
            for j, sigma in enumerate(sigma_values):
                Omega_values = compute_Omega_vectorized(k_values, k_peak, sigma, geometry_data, A_s=1.0)
                integral_grid[i, j, :] = Omega_values
                pbar.update(1)
    return integral_grid

def save_grid(filename, k_peak_values, sigma_values, Omega_grid):
    data={'k_peak_values': k_peak_values, 'sigma_values': sigma_values, 'Omega_grid': Omega_grid}
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_grid(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return (data['k_peak_values'], data['sigma_values']), data['Omega_grid']


def create_interpolator(filename):
    grid_data, Omega_grid = load_grid(filename)
    interpolator = RegularGridInterpolator(grid_data, Omega_grid, bounds_error=False, fill_value=None)
    return interpolator


class ScaledInterpolator:
    """
    Class to handle interpolation of Omega_GW values scaled by A_s^2.
    """
    def __init__(self, filename):
        self.interpolator = create_interpolator(filename)

    def __call__(self, k_peak, sigma, A_s):
        Omega_scaled = self.interpolator((k_peak, sigma)) * (A_s ** 2)
        return Omega_scaled
    
