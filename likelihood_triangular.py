import bilby
from bilby.core.prior import PriorDict, Uniform, Constraint, LogUniform
from utils_triangular import gamma_aet, N_aet, S0, signal_aet, noise_aet, S0, constrain
from Omega import P_theta_vec, compute_Omega_eMD_today_fast, P_k_lognormal, compute_Omega_RD_today
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
    """
    def __init__(self,C_hat,f, T_seg, T_obs, N_auto, f_pivot, N_amplitude):
        """
        Parameters
        C_hat : array_like
            Estimated cross-correlation spectrum.
        f : array_like
            Frequency array.
        T_seg : float
            Segment time duration.
        T_obs : float
            Total observation time.
        N_auto : float
            Auto-correlation noise amplitude.
        f_pivot : float
            Pivot frequency for the noise power law.
        N_amplitude : float
            Amplitude of the noise power law.
        """
        self.C_hat = C_hat
        self.f = f
        self.k=f*2*np.pi
        self.T_seg = T_seg
        self.T_obs = T_obs
        self.N_seg = int(T_obs / T_seg)
        self.N_auto = N_auto
        self.f_pivot = f_pivot
        self.N_amplitude = N_amplitude
        self.gamma_matrix= gamma_aet(self.f)[:2]
        self.S0=S0(self.f)
        
        super().__init__(parameters={"n_noise": None, "r": None, "A_s": None, "sigma": None,  "k_peak": None})
        """
        initial parameters:
        n_noise : float
            Noise spectral index.
            r : float
            Correlation factor.
            A_s : float
            Amplitude of the primordial power spectrum.
            sigma : float
            Width of the log-normal primordial power spectrum.
            k_peak : float
            Peak wavenumber of the primordial power spectrum.
        """
    def log_likelihood(self):
        n_noise = self.parameters["n_noise"]
        r = self.parameters["r"]
        A_s = self.parameters["A_s"]
        sigma = self.parameters["sigma"]
        k_peak = self.parameters["k_peak"]
        
        P_theta = P_k_lognormal(self.k, k_peak, sigma, A_s)
        Omega_gw,_ = compute_Omega_RD_today(self.k, cs_value=constants.cs_value, P_func=P_theta)
        

    
        N_matrix= N_aet(self.f, self.N_auto, self.f_pivot, self.N_amplitude, r, n_noise)[:2]
        C_bar=self.gamma_matrix*Omega_gw+N_matrix/self.S0
        sigma=C_bar**2 / self.N_seg
        log=-0.5*((self.C_hat - C_bar)**2 / sigma + np.log(2*np.pi*sigma))
        return np.sum(log)
    

def run_pe_RD(A_s, sigma, k_peak,r, n_noise, T_obs, N_seg, N_auto,outdir='output_pe_triangular_RD'):
    T_seg= T_obs / N_seg
    df= 1 / T_seg
    f_min = (0.001 * k_peak) / (2 * np.pi)
    f_max = (1000 * k_peak) / (2 * np.pi)
    f_values = np.arange(f_min, f_max+df, df)
    k_values= f_values * 2 * np.pi

    f_pivot = 2.75
    N_auto=...
    N_amplitude=...

    P_func=P_k_lognormal(k_values, k_peak, sigma, A_s)
    Omega_gw,_=compute_Omega_RD_today(k_values, constants.cs_value, P_func)
    h=signal_aet(f_values, Omega_gw,T_seg, N_seg)
    n= noise_aet(f_values, N_auto, f_pivot, N_amplitude, r ,n_noise)
    data=h+n
    C_hat=(2/(T_seg*S0(f_values)))*np.abs(data)**2
    C_hat=np.sum(C_hat, axis=0) / N_seg
    likelihood=SigwEstimatorLikelihood_RD(C_hat,f_values, T_seg,T_obs, N_auto, f_pivot, N_amplitude)
    
    priors = PriorDict(conversion_function=functools.partial(constrain, f_values, N_auto, f_pivot, N_amplitude))
    priors['r']       = Uniform(-0.5, 1, '$r$') 
    priors['n_noise'] = Uniform(-10, -5, '$n_{\\text{noise}}$')
    priors['A_s']     = LogUniform(1e-10, 1e-8, '$A_{\\text{s}}$')
    priors['sigma']   = LogUniform(0.1, 2, '$\\sigma$')
    priors['k_peak']  = LogUniform(1, 1e3, '$k_{\\text{peak}}$')
    priors['min_res'] = Constraint(minimum=0. , maximum=np.max(N_auto))
    
    true_parameters = {'r': r, 'n_noise': n_noise, 'A_s': A_s, 'sigma': sigma, 'k_peak': k_peak}
    label = "estimator RD"+f'_A{A_s:.1f}_kpeak{k_peak:.1f}_sigma{sigma:.2f}_r{r:.2f}_n{n_noise:.2f}_Tobs{T_obs/3600:.1f}hr_Tseg{T_seg:.1f}sec'
    
    result = bilby.run_sampler(likelihood = likelihood, 
                            priors = priors, 
                            sampler='dynesty',
                            label=label, 
                            outdir=outdir,
                            npool = 5,
                            nlive=2000,
                            injection_parameters = true_parameters,
                            )
    result.plot_corner(truths=true_parameters)
    
    posterior_dict = {}
    for key in likelihood.parameters.keys():
        posterior_dict[key] = [result.get_one_dimensional_median_and_error_bar(key).median, result.get_one_dimensional_median_and_error_bar(key).plus, result.get_one_dimensional_median_and_error_bar(key).minus]     
    
    json.dump(posterior_dict,open( outdir+'/'+label+'_stat.json', 'w'))

    




# Likelihood for eMD. Still to undestand if eta_R should be a parameter or fixed. In case it is a parameter, then should be a constrain between k_max and A_s.
class SigwEstimatorLikelihood_eMD(bilby.Likelihood):
    """
    Likelihood class for the SGWB estimator in the Einstein Telescope in triangular configuration.
    Sigw are produced during sudden transition between early Matter Domination (eMD) and Radiation Domination (RD).
    See Omega.py for the computation of Omega_gw in eMD.
    See https://arxiv.org/abs/2501.09057 for details about the estimator method.
    """
    def __init__(self,C_hat,f, T_seg, T_obs, N_auto, f_pivot, N_amplitude):
        """
        Parameters
        C_hat : array_like
            Estimated cross-correlation spectrum.
        f : array_like
            Frequency array.
        T_seg : float
            Segment time duration.
        T_obs : float
            Total observation time.
        N_auto : float
            Auto-correlation noise amplitude.
        f_pivot : float
            Pivot frequency for the noise power law.
        N_amplitude : float
            Amplitude of the noise power law.
        """
        self.C_hat = C_hat
        self.f = f
        self.k=f*2*np.pi
        self.T_seg = T_seg
        self.T_obs = T_obs
        self.N_seg = int(T_obs / T_seg)
        self.N_auto = N_auto
        self.f_pivot = f_pivot
        self.N_amplitude = N_amplitude
        self.gamma_matrix= gamma_aet(self.f)[:2]
        self.S0=S0(self.f)
        
        super().__init__(parameters={"n_noise": None, "r": None, "k_max": None, "A_s": None})
        """
        initial parameters:
        n_noise : float
            Noise spectral index.
            r : float
            Correlation factor.
            k_max : float
            Cutoff wavenumber for primordial power spectrum in eMD
            A_s : float
            Amplitude of the primordial power spectrum.
        """
    def log_likelihood(self):
        n_noise = self.parameters["n_noise"]
        r = self.parameters["r"]
        k_max = self.parameters["k_max"]
        A_s = self.parameters["A_s"]
        eta_R=120/k_max  #s
        P_theta = P_theta_vec(self.k, k_max, A_s)
        Omega_gw,_ = compute_Omega_eMD_today_fast(self.k, eta_R, k_max, P_theta)
        

    
        N_matrix= N_aet(self.f, self.N_auto, self.f_pivot, self.N_amplitude, r, n_noise)[:2]
        C_bar=self.gamma_matrix*Omega_gw+N_matrix/self.S0
        sigma=C_bar**2 / self.N_seg
        logL= -0.5*((self.C_hat - C_bar)**2 / sigma + np.log(2*np.pi*sigma))
        return np.sum(logL)
    



def run_pe_eMD(A_s, eta_R, k_max,r, n_noise, T_obs, N_seg, N_auto,outdir='output_pe_triangular_eMD'):
    k_thr = 2 * k_max / np.sqrt(3)
    T_seg= T_obs / N_seg
    f_min = (0.02 * k_max) / (2 * np.pi)
    f_max = (1.02 * k_thr) / (2 * np.pi)
    df= 1 / T_seg
    f_values = np.arange(f_min, f_max+df, df)
    k_values= f_values * 2 * np.pi

    f_pivot = 2.75
    N_auto=...
    N_amplitude=...
    
    P_func=P_theta_vec(k_values,k_max,A_s)
    Omega_gw,_=compute_Omega_eMD_today_fast(k_values,eta_R,k_max,P_func)
    h=signal_aet(f_values, Omega_gw,T_seg, N_seg)
    n= noise_aet(f_values, N_auto, f_pivot, N_amplitude, r ,n_noise)
    data=h+n
    C_hat=(2/(T_seg*S0(f_values)))*np.abs(data)**2
    C_hat=np.sum(C_hat, axis=0) / N_seg
    likelihood=SigwEstimatorLikelihood_eMD(C_hat,f_values, T_seg,T_obs, N_auto, f_pivot, N_amplitude)

    priors = PriorDict(conversion_function=functools.partial(constrain, f_values, N_auto, f_pivot, N_amplitude))
    priors['r']       = Uniform(-0.5, 1, '$r$') 
    priors['n_noise'] = Uniform(-10, -5, '$n_{\\text{noise}}$')
    priors['k_max']   = Uniform(50, 200, '$k_{\\text{max}}$')
    priors['A_s']     = LogUniform(1e-10, 1e-8, '$A_{\\text{s}}$')
    priors['min_res'] = Constraint(minimum=0. , maximum=np.max(N_auto))

    true_parameters = {'r': r, 'n_noise': n_noise, 'k_max': k_max, 'A_s': A_s}
    label = "estimator eMD"+f'_A{A_s:.1f}_kmax{k_max:.1f}_r{r:.2f}_n{n_noise:.2f}_Tobs{T_obs/3600:.1f}hr_Tseg{T_seg:.1f}sec'

    result = bilby.run_sampler(likelihood = likelihood, 
                            priors = priors, 
                            sampler='dynesty',
                            label=label, 
                            outdir=outdir,
                            npool = 5,
                            nlive=2000,
                            injection_parameters = true_parameters,
                            )  
    
    result.plot_corner(truths=true_parameters)
  
    posterior_dict = {}
    for key in likelihood.parameters.keys():
        posterior_dict[key] = [result.get_one_dimensional_median_and_error_bar(key).median, result.get_one_dimensional_median_and_error_bar(key).plus, result.get_one_dimensional_median_and_error_bar(key).minus]

    json.dump(posterior_dict,open( outdir+'/'+label+'_stat.json', 'w'))  





        
        

        
        
