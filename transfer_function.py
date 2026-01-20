import sympy as sp
import numpy as np
from scipy import integrate
from scipy.special import sici


"""
    Kernel functions for radiation domination (RD) phase.
"""
cs,u,v,x,xb,z=sp.symbols('cs u v x xb z', real=True, positive=True)

def make_kernel_RD(cs_value=1/np.sqrt(3)):
    

    def T_RD(x):
        return (9/(x**2))*(sp.sin(cs*x)/(cs*x) - sp.cos(cs*x))    

    def build_transfer_function(u,v,x):

        Tz=T_RD(z)
        dTz=sp.diff(Tz, z)

        U=u*xb
        V=v*xb

        Tu=Tz.subs(z, U)
        Tv=Tz.subs(z, V)
        dT_u=dTz.subs(z, U)
        dT_v=dTz.subs(z, V)

        f=sp.Rational(1,9)*(12*Tu*Tv +4*(xb**2)*u*v*dT_u*dT_v +4*xb*(u*Tv*dT_u + v*Tu*dT_v))
        return f


    def green_function(u,v,x):
        return sp.sin(x-xb)

    integrand_expr = xb*build_transfer_function(u,v,x) * green_function(u,v,x)
    integrand_expr = integrand_expr.subs(cs, cs_value)
    integrand_np=sp.lambdify((u,v,x,xb), integrand_expr, "numpy")

    def I(u_val, v_val, x_val, **quad_kwargs):
        val, err = integrate.quad(
            lambda xb_val: integrand_np(u_val, v_val, x_val, xb_val),
            0.0, float(x_val),
            **quad_kwargs
        )
        return val/x_val

    return I


def analytical_kernel_RD(u_val, v_val, x_val=1, cs_value=1/np.sqrt(3)):
    """Analytical kernel for radiation domination.
    Parameters:
    u_val : integration variable
    v_val : integration variable
    x_val : conformal time variable (default is 1)
    cs_value : sound speed (default is 1/sqrt(3) for relativistic species)
    """

    
    prefactor = 1/2 * np.pow((3 * (u_val**2 + v_val**2 - 3)) / (4 * u_val**3 * v_val**3 * x_val), 2)
    # Second term
    num = 3 - (u_val + v_val)**2
    den = 3 - (u_val - v_val)**2
    log_arg = np.abs(num / den)
    term_log = (u_val**2 + v_val**2 - 3) * np.log(log_arg)
    second_term = -4 * u_val * v_val + term_log 
    
    # Third term
    theta = 1 if (u_val + v_val) >= 1/cs_value else 0
    third_term = theta * np.pi**2 * (u_val**2 + v_val**2 - 3)**2
    
    result = prefactor * (second_term**2 + third_term)
    return result

def analytical_kernel_RD_vec(u_val, v_val, x_val=1, cs_value=1/np.sqrt(3)):
    """Vectorized version of the analytical kernel for radiation domination, used for simpson integration."""
    
    
    denom = 4 * u_val**3 * v_val**3 * x_val
    term1 = (3 * (u_val**2 + v_val**2 - 3)) / denom
    prefactor = 0.5 * term1**2
    num = 3 - (u_val + v_val)**2
    den = 3 - (u_val - v_val)**2
    log_arg = np.abs(num / den) 
    term_log = (u_val**2 + v_val**2 - 3) * np.log(log_arg)
    second_term = -4 * u_val * v_val + term_log 
    
    theta = np.where((u_val + v_val) >= 1/cs_value, 1.0, 0.0)
    third_term = theta * np.pi**2 * (u_val**2 + v_val**2 - 3)**2
    
    result = prefactor * (second_term**2 + third_term)
    return result


"""
Kernel functions for early matter domination (eMD) phase.
"""

def kernel_eMD_resonant(s,t,k,eta_R, Y=2.3):
    """Kernel for early matter domination at resonance.
        Parameters:
        t : Integration variable
        s : Integration variable
        k : Wavenumber
        eta_R : Conformal time at reathing
        Y : Numerical factor (default is 2.3)
        """

    x_R= k * eta_R
    y=np.abs(t+1-np.sqrt(3)) * x_R / (2 * np.sqrt(3))
    _, ci_y= sici(y)
    numerator=Y * 9 * (-5 + s**2 + 2*t + t**2)**4 * x_R**8
    denominator=81920000 * (1 - s + t)**2 * (1 + s + t)**2

    return numerator / denominator * ci_y**2

def kernel_eMD_large_v(s,t,k,eta_R):
    """Kernel for early matter domination at large wavelength.
        Parameters:
        t : Integration variable
        s : Integration variable
        k : Wavenumber
        eta_R : Conformal time at reathing
        """
    x_R= k * eta_R
    Si_half, Ci_half= sici(x_R / 2)

    numerator= 9* t**4 * x_R**8 * (4 *Ci_half**2 + (np.pi - 2 * Si_half)**2 )
    denominator=2**17 * 5**4

    return numerator / denominator