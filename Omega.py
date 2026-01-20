from transfer_function import analytical_kernel_RD, kernel_eMD_resonant, kernel_eMD_large_v, analytical_kernel_RD_vec
import numpy as np
from scipy import integrate
from scipy.special import erfc

"""
    This section computes the Omega_GW using the analytical kernel for radiation domination and early matter domination.
    We refered to the following papers:
    - https://arxiv.org/abs/2501.11320
    - https://arxiv.org/abs/1804.08577
    - https://arxiv.org/abs/1904.12879
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
    parameters:
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
    """ Compute Omega_GW during radiation domination phase using analytical kernel.
    Integral limits: s in [0,1], t in [0, +inf)
    """
    result, error = integrate.dblquad(
        lambda s, t: integrand_RD(s, t, k, P_func, cs_value),
        0, np.inf,         
        lambda t: 0,          
        lambda t: 1,         
    )

    return result, error
    

def compute_Omega_RD_fast(k, cs_value, P_func, t_max=50, N_s=100, N_t_1=100, N_t_2=1000, N_t_3=100):
    """
    Here it used the simpson integration instead of dblquad. A grid of s-t values is constructed

    Parameters:
    t_max : Superior limit for the integral in t (the integrand decays quickly)
    N_s, N_t : Number of points in the grid for s and t
    """
    eps=1e-10 
    s = np.linspace(0, 1-eps, N_s)
    t_sing = (1.0 / cs_value) - 1.0  

    
    # Build the integration grid (we focus on the singularity of the integral in order to be more precise) 
    width = 0.05 
    t1 = np.linspace(1e-4, t_sing - width, N_t_1)
    t2 = np.linspace(t_sing - width, t_sing + width, N_t_2)
    t3 = np.linspace(t_sing + width, t_max, N_t_3)
    t = np.unique(np.concatenate([t1, t2, t3]))
    

    # constructing the integrand as in "integrand_RD" function
    S = s[:, None]  
    T = t[None, :]  
    U = (T + S + 1) / 2
    V = (T - S + 1) / 2
    
    
    prefactor_integrand = (T * (T + 2) * (1 - S**2)/((1 + T - S) * (1 + T + S)))**2
    I_val = analytical_kernel_RD_vec(U, V, x_val=1, cs_value=cs_value)
    k1 = k * (1 + T + S)/2
    k2 = k * (1 + T - S)/2
    P_ku = P_func(k1)
    P_kv = P_func(k2)
    Z = 4 * prefactor_integrand * I_val * P_ku * P_kv
    
    #integration with simpson's method
    integral_t = integrate.simpson(y=Z, x=t, axis=1)
    result = integrate.simpson(y=integral_t, x=s, axis=0)
    
    return result, 0.0 

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


def compute_Omega_RD_today_fast(k, cs_value, P_func,Omega_r0_hh,c_g, t_max=50, N_s=100, N_t_1=100, N_t_2=1000, N_t_3=100):
    """ Compute Omega_GW today during radiation domination phase using fast method
    new parameters:
    Omega_r0_hh : Radiation density parameter today times h^2
    c_g : Transfer function factor
    """
    Omega_RD, error= compute_Omega_RD_fast(k,cs_value,P_func,t_max,N_s,N_t_1,N_t_2,N_t_3)
    Omega_GW_today= (1/24)*Omega_RD*Omega_r0_hh*c_g,
    error_today= (1/24)*error*Omega_r0_hh*c_g
    return Omega_GW_today, error_today


""" 
    Functions for computing Omega_GW during early matter domination (eMD) phase.
"""

def P_theta(k, k_cut, A_s):
    """Theta power spectrum: P(k) = A_s for k < k_cut, 0 otherwise."""
    if k < k_cut:
        return A_s
    else:
        return 0.0

def P_theta_vec(k, k_cut, A_s):
    return np.where(k < k_cut, A_s, 0.0)
    

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
    """
    Compute Omega_GW during early matter domination phase using eMD resonant kernel.
    Integral limits: s in [0,1], t in [0, -s + 2*k_max/k - 1] and k < 2*k_max/sqrt(3)
    """ 
    k_thr = 2*k_max/np.sqrt(3)
    if k > k_thr:
        return 0.0, 0.0
    limit_t_upper = lambda s: -s + 2*k_max/k - 1
    limit_t_lower = lambda s: 0  
    result, error = integrate.dblquad(
        lambda t, s: integrand_eMD_resonant(s, t, k, eta_R, P_func, Y),
        0, 1,     
        limit_t_lower,         
        limit_t_upper,       
        epsabs=1e-12,epsrel=1e-6
    )
    return result, error

def compute_Omega_eMD_large_v(k, eta_R, k_max, P_func):
    """
    Compute Omega_GW during early matter domination phase using eMD large v kernel.
    Integral limits: s in [0,1], t in [0, -s + 2*k_max/k - 1]
    """
    limit_t_upper = lambda s: (-s + 2*k_max/k - 1)
    limit_t_lower = lambda s: 0  
    result, error = integrate.dblquad(
        lambda t, s: integrand_eMD_large_v(s, t, k, k_max, eta_R,P_func),
        0, 1,                  
        limit_t_lower,         
        limit_t_upper,        
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


def compute_Omega_eMD_resonant_fast(k, eta_R, k_max, P_func, Y=2.3, t_max=50, N_s=100, N_t_1=100, N_t_2=1000, N_t_3=100):
    """
    Here it used the simpson integration instead of dblquad. A grid of s-t values is constructed

    Parameters:
    t_max : Superior limit for the integral in t (the integrand decays quickly)
    N_s, N_t : Number of points in the grid for s and t
    """
    

    eps=1e-10
    s= np.linspace(0, 1-eps, N_s)
    t_sing=np.sqrt(3)-1
    t_max=2*k_max/k -1

    if t_sing > t_max:
        t= np.linspace(1e-4, t_max+1e-4, N_t_1+N_t_2+N_t_3)

    else:
        width=0.05
        t1= np.linspace(1e-4, t_sing - width, N_t_1)
        t2= np.linspace(t_sing - width, t_sing + width, N_t_2)
        t3= np.linspace(t_sing + width, t_max+1e-4, N_t_3)
        t= np.unique(np.concatenate([t1,t2,t3]))
    
    # constructing the integrand as in "integrand_eMD_resonant" function, using the mask to restrict the integration domain
    S = s[:, None]
    T = t[None, :]
    mask= (T<= -S + 2*k_max/k -1) & ( (1 + T - S) >0) 
    if not np.any(mask):
        return 0.0, 0.0
    
    if np.any(k > 2*k_max/np.sqrt(3)):
        return 0.0, 0.0
    
    
    prefactor_integrand = (T * (T + 2) * (1 - S**2)/((1 + T - S) * (1 + T + S)))**2
    I=kernel_eMD_resonant(S, T, k, eta_R, Y)
    k1 = k * (1 + T + S)/2
    k2 = k * (1 + T - S)/2
    P_ku = P_func(k1)
    P_kv = P_func(k2)
    Z = 4 * I * P_ku * P_kv * prefactor_integrand
    Z[~mask] = 0.0
    
    #integration with simpson's method
    integral_t = integrate.simpson(y=Z, x=t, axis=1)
    result = integrate.simpson(y=integral_t, x=s, axis=0)
    return result, 0.0

def compute_Omega_eMD_large_v_fast(k, eta_R, k_max, P_func, t_max=50, N_s=100,N_t=100):
    """
    Here it used the simpson integration instead of dblquad. A grid of s-t values is constructed

    Parameters:
    t_max : Limite superiore per l'integrale in t (l'integrando decade velocemente)
    N_s, N_t : Numero di punti della griglia per s e t
    """
    
    # Build the integration grid
    eps=1e-10
    s= np.linspace(0, 1-eps, N_s)
    t_max=2*k_max/k -1
    t= np.linspace(1e-4, t_max, N_t)
    S = s[:, None]
    T = t[None, :]
    # constructing the integrand as in "integrand_eMD_large_v" function, using the mask to restrict the integration domain
    mask= (T<= -S + 2*k_max/k -1) & ( (1 + T - S) >0) & ( (1 + T + S) >0) 
    if not np.any(mask):
        return 0.0, 0.0
    
    
    prefactor_integrand = (T * (T + 2) * (1 - S**2)/((1 + T - S) * (1 + T + S)))**2
    I=kernel_eMD_large_v(S, T, k, eta_R)
    k1 = k * (1 + T + S)/2
    k2 = k * (1 + T - S)/2
    P_ku = P_func(k1)
    P_kv = P_func(k2)
    Z = 4 * I * P_ku * P_kv * prefactor_integrand
    Z[~mask] = 0.0
    
    #integration with simpson's method
    integral_t = integrate.simpson(y=Z, x=t, axis=1)
    result = integrate.simpson(y=integral_t, x=s, axis=0)
    return result, 0.0

def compute_Omega_eMD_total_fast(k, eta_R, k_max, P_func, Y=2.3, t_max=50, N_s_1=100,N_t=100, N_s_2=100, N_t_1=100, N_t_2=1000, N_t_3=100):
    compute_Omega_eMD_large_v_fast_result, _ = compute_Omega_eMD_large_v_fast(k, eta_R, k_max, P_func, t_max, N_s_1, N_t)
    compute_Omega_eMD_resonant_fast_result, _ = compute_Omega_eMD_resonant_fast(k, eta_R, k_max, P_func, Y, 
                                                                                                                    t_max, N_s_2, N_t_1, N_t_2, N_t_3)
    total_omega = compute_Omega_eMD_large_v_fast_result + compute_Omega_eMD_resonant_fast_result
    total_error= 0.0
    return total_omega, total_error

def compute_Omega_eMD_today_fast(k, eta_R, k_max, P_func, Omega_r0_hh,c_g, t_max=50, N_s_1=100,N_t=100, N_s_2=100, N_t_1=100, N_t_2=1000, N_t_3=100):
    """ Compute Omega_GW today during radiation domination phase after eMD with sudden reheating using fast method.
    new parameters:
    Omega_r0_hh : Radiation density parameter today times h^2
    c_g : Transfer function factor
    """
    x_R= k * eta_R
    Omega_GW, error= compute_Omega_eMD_total_fast(k, eta_R, k_max, P_func, Y=2.3, t_max=t_max, N_s_1=N_s_1,N_t=N_t, N_s_2=N_s_2,
                                                   N_t_1=N_t_1, N_t_2=N_t_2, N_t_3=N_t_3)
    Omega_GW = (1/24) * x_R**2 * Omega_GW
    Omega_GW_today= Omega_GW*Omega_r0_hh*c_g
    error_today= 0.0
    return Omega_GW_today, error_today

                        