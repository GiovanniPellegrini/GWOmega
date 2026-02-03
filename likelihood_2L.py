import bilby
from bilby.core.prior import PriorDict, Uniform, Constraint, LogUniform
from utils_2L import gamma_2L, S0, N_auto_interp
from data_generation import signal_2L, noise_2L
from Omega import P_theta_vec, compute_Omega_eMD_today_fast, P_k_lognormal, compute_Omega_RD_today_fast
import numpy as np
import functools
import json
import constants

class SigwEstimatorLikelihood_RD(bilby.Likelihood):
    """
    Likelihood class for the SGWB estimator in the Einstein Telescope in triangular configuration.
    Sigw are produced during Radiation Domination (RD).
    See Omega.py for the computation of Omega_gw in RD.
    See https://arxiv.org/abs/2501.09057 for details about the estimator method.

    Parameters
    C_hat: array shape (N_seg, 2, len(f))
        estimated cross-correlation spectrum from data.
    f: array shape (len(f),)
        frequency array.
    T_seg: float
        segment duration.
    T_obs: float
        total observation time.
    N_auto: function
        function that returns the auto-correlation noise amplitude given frequency.
    shift_angle: float, optional
        Shift angle between the two L-shaped detectors in radians. Default is 0.
    """
    def __init__(self, C_hat, f, T_seg, T_obs, N_auto, shift_angle=0):
        self.C_hat = C_hat
        self.f = f
        self.k = f * 2 * np.pi
        self.T_seg = T_seg
        self.T_obs = T_obs
        self.N_seg = int(T_obs / T_seg)
        self.gamma= gamma_2L(self.f, shift_angle)
        self.N_auto = N_auto
        self.S0= S0(self.f)
        super().__init__(parameters={"A_s": None, "sigma": None,  "k_peak": None})
        """
        Initial parameters
        A_s : float
            Amplitude of the primordial power spectrum.
        sigma : float
            Width of the log-normal primordial power spectrum.
        k_peak : float
            Peak wavenumber of the primordial power spectrum.
        """

    def log_likelihood(self):
        A_s= self.parameters["A_s"]
        sigma= self.parameters["sigma"]
        k_peak= self.parameters["k_peak"]

        P_func=lambda k: P_k_lognormal(k,k_peak, sigma, A_s)
        Omega_gw = [compute_Omega_RD_today_fast(k, constants.cs_value, P_func) for k in self.k]
        Omega_gw = np.array(Omega_gw)

        C_bar=Omega_gw * self.gamma
        sigma_bar=(C_bar**2 + (self.N_auto/self.S0+Omega_gw)**2)/(2*self.N_seg)
        logL=-0.5*np.sum((self.C_hat - C_bar)**2/sigma_bar + np.log(2*np.pi*sigma_bar))
        return logL
    
def run_pe_2L_RD(A_s, sigma, k_peak, T_obs, N_seg, shift_angle=0, outdir='output_pe_2L_RD'):
    T_seg= T_obs / N_seg
    df= 1 / T_seg
    f_values= np.arange(1, 200, df)
    k_values= f_values * 2 * np.pi

    N_func= N_auto_interp("data/ET_Sh_coba.txt")
    N_auto=N_func(f_values)
    P_func= lambda k: P_k_lognormal(k, k_peak, sigma, A_s)
    Omega_gw = [
    compute_Omega_RD_today_fast(float(k), constants.cs_value, P_func)    for k in k_values
    ]

    h=signal_2L(f_values, Omega_gw, T_seg, N_seg, shift_angle)
    n= noise_2L(f_values, N_auto, T_seg, N_seg)
    data=h+n


    C_hat=(2/(T_seg*S0(f_values)))*np.real(data[:,0,:]*np.conj(data[:,1,:]))
    C_hat=np.average(C_hat, axis=0)
    likelihood=SigwEstimatorLikelihood_RD(C_hat,f_values, T_seg,T_obs, N_auto, shift_angle)
    priors = PriorDict()
    priors['A_s']       = LogUniform(1e-4, 1e-1, '$A_s$') 
    priors['sigma']     = Uniform(0.3, 1, '$\sigma$') 
    priors['k_peak']    = Uniform(50, 150, '$k_{peak}$') 

    true_parameters = {'A_s': A_s, 'sigma': sigma, 'k_peak': k_peak}
    label = "estimator RD"+f'_A{A_s:.1f}_sigma{sigma:.1f}_kpeak{k_peak:.1f}_Tobs{T_obs/3600:.1f}hr_Tseg{T_seg:.1f}sec'
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler='dynesty',
        nlive=1000,
        n_pool=7,
        outdir=outdir,
        label=label,
        injection_parameters=true_parameters
    )
    result.plot_corner(truths=true_parameters)

    posterior_dict = {}
    for key in likelihood.parameters.keys():
        posterior_dict[key] = [result.get_one_dimensional_median_and_error_bar(key).median, result.get_one_dimensional_median_and_error_bar(key).plus, result.get_one_dimensional_median_and_error_bar(key).minus]

    json.dump(posterior_dict,open( outdir+'/'+label+'_stat.json', 'w'))  


    return result



class SigwEstimatorLikelihood_eMD(bilby.Likelihood):
    """
    Likelihood class for the SGWB estimator in the Einstein Telescope in triangular configuration.
    Sigw are produced during sudden transition between early Matter Domination (eMD) and Radiation Domination (RD).
    See Omega.py for the computation of Omega_gw in eMD.
    See https://arxiv.org/abs/2501.09057 for details about the estimator method.

    Parameters
    C_hat: array shape (N_seg, 2, len(f))
        estimated cross-correlation spectrum from data.
    f: array shape (len(f),)
        frequency array.
    T_seg: float
        segment duration.
    T_obs: float
        total observation time.
    N_auto: function
        function that returns the auto-correlation noise amplitude given frequency.
    shift_angle: float, optional
        Shift angle between the two L-shaped detectors in radians. Default is 0.
    """
    def __init__(self, C_hat, f, T_seg, T_obs, N_auto, shift_angle=0):
        self.C_hat = C_hat
        self.f = f
        self.k = f * 2 * np.pi
        self.T_seg = T_seg
        self.T_obs = T_obs
        self.N_seg = int(T_obs / T_seg)
        self.gamma= gamma_2L(self.f, shift_angle)
        self.N_auto = N_auto
        self.S0= S0(self.f)
        super().__init__(parameters={"A_s": None,  "k_max": None, "eta_R": None})
        """
        Initial parameters
        A_s : float
            Amplitude of the primordial power spectrum.
        sigma : float
            Width of the log-normal primordial power spectrum.
        k_peak : float
            Peak wavenumber of the primordial power spectrum.
        eta_R : float
            Conformal time at the end of eMD.
        """

    def log_likelihood(self):
        A_s= self.parameters["A_s"]
        k_max= self.parameters["k_max"]
        eta_R= self.parameters["eta_R"]

        P_func=lambda k: P_theta_vec(k, k_max, A_s)
        Omega_gw = [compute_Omega_eMD_today_fast(k, eta_R, k_max, P_func) for k in self.k]
        Omega_gw = np.array(Omega_gw)

        C_bar=Omega_gw * self.gamma
        sigma_bar=(C_bar**2 + (self.N_auto/self.S0+Omega_gw)**2)/(2*self.N_seg)
        logL=-0.5*np.sum((self.C_hat - C_bar)**2/sigma_bar + np.log(2*np.pi*sigma_bar))
        return logL
    
def run_pe_2L_eMD(A_s, k_max, eta_R, T_obs, N_seg, shift_angle=0,outdir='output_pe_2L_eMD'):
    T_seg= T_obs / N_seg
    df= 1 / T_seg
    f_values= np.arange(1, 200, df)
    k_values= f_values * 2 * np.pi

    N_func= N_auto_interp("data/ET_Sh_coba.txt")
    N_auto=N_func(f_values)
    P_func= lambda k: P_theta_vec(k, k_max, A_s)
    Omega_gw = [
    compute_Omega_eMD_today_fast(float(k), eta_R, k_max, P_func)    for k in k_values
    ]

    h=signal_2L(f_values, Omega_gw, T_seg, N_seg, shift_angle)
    n= noise_2L(f_values, N_auto, T_seg, N_seg)
    data=h+n


    C_hat=(2/(T_seg*S0(f_values)))*np.real(data[:,0,:]*np.conj(data[:,1,:]))
    C_hat=np.average(C_hat, axis=0)
    likelihood=SigwEstimatorLikelihood_eMD(C_hat,f_values, T_seg,T_obs, N_auto, shift_angle)
    priors = PriorDict()
    #priors['A_s']       = LogUniform(1e-4, 1e-1, '$A_s$') 
    priors['k_max']     = Uniform(10, 150, '$k_{max}$') 
    #priors['eta_R']     = LogUniform(1e-25, 1e-20, '$\eta_{R}$') 

    true_parameters = {'A_s': A_s, 'k_max': k_max, 'eta_R': eta_R}
    label = "estimator eMD"+f'_A{A_s:.1f}_kmax{k_max:.1f}_Tobs{T_obs/3600:.1f}hr_Tseg{T_seg:.1f}sec'
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler='dynesty',
        nlive=200,
        n_pool=7,
        outdir=outdir,
        label=label,
        injection_parameters=true_parameters
    )
    result.plot_corner(truths=true_parameters)

    posterior_dict = {}
    for key in likelihood.parameters.keys():
        posterior_dict[key] = [result.get_one_dimensional_median_and_error_bar(key).median, result.get_one_dimensional_median_and_error_bar(key).plus, result.get_one_dimensional_median_and_error_bar(key).minus]

    json.dump(posterior_dict,open( outdir+'/'+label+'_stat.json', 'w'))  
