from transfer_function import analytical_kernel_RD, kernel_eMD_resonant, kernel_eMD_large_v
import numpy as np
from scipy import integrate
from scipy.special import erfc

"""
    This section computes the Omega_GW using the analytical kernel for radiation domination and early matter domination.
"""



def tranform_to_uv(s,t):
    #Transform variables from (s,t) to (u,v)
    u=(t+s+1)/2
    v=(t-s+1)/2
    return u,v



def P_k_lognormal(k, k_p, sigma, A):
    """
    Log-normal power spectrum
     P(k) = A / (sqrt(2*pi) * sigma) * exp(- (log(k/k_p))^2 / (2*sigma^2))
    paameters:
    k : Wavenumber
    k_p : Peak wavenumber
    sigma : Width of the log-normal distribution
    A : Amplitude of the power spectrum
    """

    log_ratio = np.log(k / k_p)
    normalization = 1.0 / (np.sqrt(2 * np.pi) * sigma)
    exponential = np.exp(-log_ratio**2 / (2 * sigma**2))
    
    return A* normalization * exponential


def integrand_RD(s,t,k, P_func,cs_value):
    """Integrand for Omega_GW computation using analytical kernel in RD."""
    u,v=tranform_to_uv(s,t)
    prefactor=(t * (t + 2) * (1 - s**2)/((1 + t - s) * (1 + t + s)))**2
    I=analytical_kernel_RD(u,v,x_val=1,cs_value=cs_value)

    k1=k * (1 + t + s)/2
    k2=k * (1 + t - s)/2

    P_ku=P_func(k1)
    P_kv=P_func(k2)

    return 4*prefactor * I * P_ku * P_kv

def compute_Omega_RD(k, cs_value, P_func):

    result, error = integrate.dblquad(
        lambda s, t: integrand_RD(s, t, k, P_func, cs_value),
        0, np.inf,         
        lambda t: 0,          
        lambda t: 1,         
    )
    return result, error
        

def compute_Omega_RD_today(k, cs_value, P_func, Omega_r0_hh,c_g):
    """ Compute Omega_GW today during radiation domination phase
    new parameters:
    Omega_r0_hh : Radiation density parameter today times h^2
    c_g : Transfer function factor
    """
    Omega_RD, error= compute_Omega_RD(k, cs_value, P_func)
    Omega_GW = (1/24) * Omega_RD
    Omega_GW_today= Omega_GW*Omega_r0_hh*c_g
    return Omega_GW_today, error


""" 
    Functions for computing Omega_GW during early matter domination (eMD) phase."""
def P_theta(k, k_cut, A_s):
    if k < k_cut:
        return A_s
    else:
        return 0.0
    

def integrand_eMD_resonant(s,t,k, eta_R, P_func, Y=2.3):
    """Integrand for Omega_GW computation using eMD resonant kernel."""
    
    if (1 + t - s) <= 0 or (1 + t + s) <= 0 or t <= 0:
        return 0.0
    prefactor=(t * (t + 2) * (1 - s**2)/((1 + t - s) * (1 + t + s)))**2
    I=kernel_eMD_resonant(s, t, k, eta_R, Y)

    k1=k * (1 + t + s)/2
    k2=k * (1 + t - s)/2

    P_ku=P_func(k1)
    P_kv=P_func(k2)

    return 4* I * P_ku * P_kv * prefactor

def integrand_eMD_large_v(s,t,k,k_max,eta_R, P_func):
    """Integrand for Omega_GW computation using eMD large v kernel."""
    if (1 + t - s) <= 0 or (1 + t + s) <= 0 or t <= 0:
        return 0.0
    
    prefactor=(t * (t + 2) * (1 - s**2)/((1 + t - s) * (1 + t + s)))**2
    I=kernel_eMD_large_v(s, t, k, eta_R)
    k1=k * (1 + t + s)/2
    k2=k * (1 + t - s)/2

    P_ku=P_func(k1)
    P_kv=P_func(k2)

    return 4* I * P_ku * P_kv*prefactor

def compute_Omega_eMD_resonant(k, eta_R, k_max, P_func, Y=2.3):
    k_thr = 2*k_max/np.sqrt(3)
    if k > k_thr:
        return 0.0, 0.0
    limit_t_upper = lambda s: -s + 2*k_max/k - 1
    limit_t_lower = lambda s: 0  
    result, error = integrate.dblquad(
        lambda t, s: integrand_eMD_resonant(s, t, k, eta_R, P_func, Y),
        0, 1,     # limiti per s
        limit_t_lower,          # Limite inferiore interno (t)
        limit_t_upper,         # Limite superiore interno (t)
        epsabs=1e-12,epsrel=1e-6
    )
    return result, error

def compute_Omega_eMD_large_v(k, eta_R, k_max, P_func):
    
    limit_t_upper = lambda s: (-s + 2*k_max/k - 1)
    limit_t_lower = lambda s: 0  
    result, error = integrate.dblquad(
        lambda t, s: integrand_eMD_large_v(s, t, k, k_max, eta_R,P_func),
        0, 1,                   # Limiti esterni (s)
        limit_t_lower,          # Limite inferiore interno (t)
        limit_t_upper,         # limiti per s
    )
    return result, error

def compute_Omega_eMD_total(k, eta_R, k_max, P_func, Y=2.3):
    omega_resonant, error_resonant = compute_Omega_eMD_resonant(k, eta_R, k_max, P_func, Y)
    omega_large_v, error_large_v = compute_Omega_eMD_large_v(k, eta_R, k_max, P_func)
    
    total_omega = omega_resonant + omega_large_v
    total_error= np.sqrt(error_resonant**2 + error_large_v**2)

    return total_omega, total_error

def compute_Omega_eMD_today(k, eta_R, k_max, P_func, Omega_r0_hh,c_g):
    """ Compute Omega_GW today during radiation domination phase after eMD with sudden reheating.
    new parameters:
    Omega_r0_hh : Radiation density parameter today times h^2
    c_g : Transfer function factor
    """
    x_R= k * eta_R
    Omega_GW, error= compute_Omega_eMD_total(k, eta_R, k_max, P_func)
    Omega_GW = (1/24) * x_R**2 * Omega_GW
    Omega_GW_today= Omega_GW*Omega_r0_hh*c_g
    return Omega_GW_today, error

