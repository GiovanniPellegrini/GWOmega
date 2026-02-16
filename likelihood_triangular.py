import bilby
from bilby.core.prior import PriorDict, Uniform, Constraint, LogUniform
from utils_triangular import gamma_aet, N_aet,S0, constrain, N_auto_interp, ConstrainWrapper
from data_generation import signal_aet, noise_aet,load_parameters_from_file, check_parameters, load_C_hat
from gwbird import snr
from gwbird import snr
from interpolation import ScaledInterpolatorRD, ScaledInterpolatorEMD
from Omega import P_theta_vec, compute_Omega_eMD_today_fast, P_k_lognormal, compute_Omega_RD_today_fast
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import functools
import json
import constants

    
class SigwEstimatorLikelihood_triangular_RD(bilby.Likelihood):
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
        self.interpolator= ScaledInterpolatorRD('omega_grid_RD.pkl')
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
            
        Omega_gw=self.interpolator(k_peak, sigma, A_s)
        Omega_gw=np.array(Omega_gw)
        

    
        N_matrix= N_aet(self.f, self.N_auto, self.f_pivot, self.N_amplitude, r, n_noise)[:2]
        C_bar=self.gamma_matrix*Omega_gw+N_matrix/self.S0
        sigma=C_bar**2 / self.N_seg
        log=-0.5*((self.C_hat - C_bar)**2 / sigma + np.log(2*np.pi*sigma))
        return np.sum(log)


def run_pe_triangula_RD(A_s, sigma, k_peak,r, n_noise, T_obs, N_seg, C_hat_filename, parameters_filename, outdir='output_pe_triangular_RD'):
    T_seg= T_obs / N_seg
    df= 1 / T_seg
    f_values= np.arange(1, 200+df, df)
    parameters= {
        "A_s": A_s,
        "k_peak": k_peak,
        "sigma": sigma,
        "r": r,
        "n_noise": n_noise,
        "T_obs": T_obs,
        "N_seg": N_seg,
        "T_seg": T_seg
    }
    print("time for each segment:", T_seg)
    print("number of segments:", N_seg)
    print("total observation time (s):", T_obs)

    # SNR valuation (da rifare)
    det_list = ['ET L1', 'ET L2']
    def compute_Omega_RD_today_fast_f(f_values):
        P_func = lambda kk: P_k_lognormal(kk, k_peak, sigma, A_s)
        Omega =[compute_Omega_RD_today_fast(k, constants.cs_value, P_func, t_max=100, N_s=10, N_t_1=50, N_t_2=100, N_t_3=700) for k in f_values*2*np.pi]
        return Omega
    T_obs_years= T_obs / (3600*24*365)
    SNR=snr.SNR(T_obs_years, f_values, None, det_list, pol='t', psi=0, gw_spectrum_func=compute_Omega_RD_today_fast_f)
    print(f"SNR ET RD: {SNR:.2f}")

    N_func= N_auto_interp("data/ET_Sh_coba.txt")
    N_auto=N_func(f_values)
    f_pivot = 2.75
    N_amplitude=float(N_func(f_pivot))
    # Load C_hat and check parameters
    try: 
        if not check_parameters(parameters, parameters_filename):
            raise ValueError("The parameters in the file do not match the input parameters.")
        C_hat= load_C_hat(C_hat_filename)
    
    except Exception as e:
        raise ValueError(f"Error loading data files: {e}")


    likelihood=SigwEstimatorLikelihood_triangular_RD(C_hat,f_values, T_seg,T_obs, N_auto, f_pivot, N_amplitude)    
    priors = PriorDict(conversion_function=ConstrainWrapper(f_values, N_auto, f_pivot, N_amplitude))
    priors['r']       = Uniform(-0.5, 1, '$r$') 
    priors['n_noise'] = Uniform(-10, -5, '$n_{\\text{noise}}$')
    priors['A_s']     = LogUniform(1e-4, 5e-2, '$A_{\\text{s}}$')
    priors['sigma']   = Uniform(0.3, 1, '$\\sigma$')
    priors['k_peak']  = Uniform(30, 250, '$k_{\\text{peak}}$')
    priors['min_res'] = Constraint(minimum=0. , maximum=np.max(N_auto)) 

    true_parameters = {'r': r, 'n_noise': n_noise, 'A_s': A_s, 'sigma': sigma, 'k_peak': k_peak}
    label = "estimator RD"+f'_A{A_s:.1f}_kpeak{k_peak:.1f}_sigma{sigma:.2f}_r{r:.2f}_n{n_noise:.2f}_Tobs{T_obs/3600:.1f}hr_Tseg{T_seg:.1f}sec'
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler='dynesty',
        nlive=2000,
        n_pool=7,
        outdir=outdir,
        label=label,
        injection_parameters=true_parameters,
        check_point_plot=False,
    )
    
    
    # plot corner with custom formatting for the titles and axes
    try:
        import scienceplots  # Importa qui per assicurarsi che sia caricato
        plt.style.use(['science', 'no-latex'])
        plt.style.use('seaborn-v0_8-bright')
    except (ImportError, OSError) as e:
        print(f"Warning: Could not use 'science' style: {e}")
        print("Using default matplotlib style instead")
        plt.style.use('seaborn-v0_8-bright')  
    mpl.rcParams["figure.dpi"] = 300
    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams["font.size"] = 11
    mpl.rcParams["figure.dpi"] = 110
    mpl.rcParams["font.size"] = 10
    mpl.rcParams["axes.labelsize"] = 18
    mpl.rcParams["xtick.labelsize"] = 12
    mpl.rcParams["ytick.labelsize"] = 12

    cmap = mpl.colormaps['viridis']
    colors = [cmap(i/7) for i in range(7)]

    fig = result.plot_corner(
        truths=true_parameters,
        title_kwargs={"fontsize": 12},
        title_fmt='.6f',  # Cambiato da .2e a .3f per più decimali
        color=colors[3],
        save=False           
    )

    # Ottieni tutti gli assi
    axes = fig.get_axes()
    n_params = int(np.sqrt(len(axes)))  # Numero di parametri

    for idx, param in enumerate(result.search_parameter_keys):
        ax_idx = idx * n_params + idx  # Posizione sulla diagonale
        ax = axes[ax_idx]
        
        # Prendi i campioni e calcola quantili
        samples = result.posterior[param].values
        q_16, q_50, q_84 = np.percentile(samples, [16, 50, 84])
        err_plus = q_84 - q_50
        err_minus = q_50 - q_16
        
        if param == 'A_s':
        # Trova l'esponente del valore mediano
            exponent = int(np.floor(np.log10(abs(q_50))))
        # Scala tutti i valori con lo stesso esponente
            mantissa = q_50 / 10**exponent
            err_plus_scaled = err_plus / 10**exponent
            err_minus_scaled = err_minus / 10**exponent
        
            new_title = f"$({mantissa:.1f}_{{{-err_minus_scaled:.1f}}}^{{+{err_plus_scaled:.1f}}}) \\times 10^{{{exponent}}}$"
        else:
            new_title = f"${q_50:.2f}_{{{-err_minus:.2f}}}^{{+{err_plus:.2f}}}$"
    
        ax.set_title(new_title, fontsize=12)

    for i, ax in enumerate(axes):
    # Calcola riga e colonna dell'asse corrente
        row = i // n_params
        col = i % n_params
    
    # Identifica quale parametro corrisponde a questo asse
        param_x = result.search_parameter_keys[col] if col < len(result.search_parameter_keys) else None
        param_y = result.search_parameter_keys[row] if row < len(result.search_parameter_keys) else None
        
        # Formattazione asse X
        if param_x == 'A_s':
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(
                lambda x, p: f"${float(f'{x:.1e}'.split('e')[0])} \\times 10^{{{int(f'{x:.1e}'.split('e')[1])}}}$" if x != 0 else "$0$"
            ))
        else:
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
            ax.ticklabel_format(style='plain', axis='x')
        
        # Formattazione asse Y
        if param_y == 'A_s':
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(
                lambda y, p: f"${float(f'{y:.1e}'.split('e')[0])} \\times 10^{{{int(f'{y:.1e}'.split('e')[1])}}}$" if y != 0 else "$0$"
            ))
        else:
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
            ax.ticklabel_format(style='plain', axis='y')
        
        # Nascondi le etichette per gli assi interni
        if row < n_params - 1:
            ax.set_xticklabels([])
        else:
            ax.tick_params(axis='x', rotation=45)
            
        if col > 0:
            ax.set_yticklabels([])
        
        # Rimuovi offset text
        ax.xaxis.get_offset_text().set_visible(False)
        ax.yaxis.get_offset_text().set_visible(False)

    # Aggiusta il layout per evitare sovrapposizioni
    plt.subplots_adjust(top=0.95, hspace=0.1, wspace=0.1)

    plot_filename = f"{outdir}/{label}_corner.png"
    fig.savefig(plot_filename, bbox_inches='tight')
    plt.close(fig)

    posterior_dict = {}
    for key in likelihood.parameters.keys():
        posterior_dict[key] = [result.get_one_dimensional_median_and_error_bar(key).median, result.get_one_dimensional_median_and_error_bar(key).plus, result.get_one_dimensional_median_and_error_bar(key).minus]

    json.dump(posterior_dict,open( outdir+'/'+label+'_stat.json', 'w')) 

    return result


    




# Likelihood for eMD. Still to undestand if eta_R should be a parameter or fixed. In case it is a parameter, then should be a constrain between k_max and A_s.
class SigwEstimatorLikelihood_triangular_eMD(bilby.Likelihood):
    """
    Likelihood class for the SGWB estimator in the Einstein Telescope in triangular configuration.
    Sigw are produced during sudden transition between early Matter Domination (eMD) and Radiation Domination (RD).
    See Omega.py for the computation of Omega_gw in eMD.
    See https://arxiv.org/abs/2501.09057 for details about the estimator method.
    """
    def __init__(self,C_hat,f, T_seg, T_obs, N_auto, f_pivot, N_amplitude):
        """
        Parameters
        C_hat : array_like (2, len(f))
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
        self.interpolator= ScaledInterpolatorEMD('omega_grid_eMD_75x75.pkl')
        self.S0=S0(self.f)
        
        super().__init__(parameters={"n_noise": None, "r": None, "A_s": None,"k_max": None,"eta_R":None })
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
        A_s = self.parameters["A_s"]
        k_max = self.parameters["k_max"]
        eta_R=self.parameters["eta_R"]
    
        Omega_gw=self.interpolator(k_max,eta_R, A_s)
        Omega_gw=np.array(Omega_gw)
        N_matrix= N_aet(self.f, self.N_auto, self.f_pivot, self.N_amplitude, r, n_noise)[:2]
        C_bar=self.gamma_matrix*Omega_gw+N_matrix/self.S0
        sigma=C_bar**2 / self.N_seg
        logL= -0.5*((self.C_hat - C_bar)**2 / sigma + np.log(2*np.pi*sigma))
        return np.sum(logL)
    



def run_pe_triangular_eMD(A_s, k_max, eta_R,r, n_noise, T_obs, N_seg,C_hat_filename, parameters_filename, outdir='output_pe_triangular_eMD'):
    T_seg= T_obs / N_seg
    df= 1 / T_seg
    f_values= np.arange(1, 200+df, df)
    parameters= {
        "A_s": A_s,
        "k_max": k_max,
        "eta_R": eta_R,
        "T_obs": T_obs,
        "N_seg": N_seg,
        "T_seg": T_seg,
    }
    print("time for each segment:", T_seg)
    print("number of segments:", N_seg)
    print("total observation time (s):", T_obs)

    # SNR valuation (da rifare)
    det_list = ['ET L1', 'ET L2']
    def compute_Omega_eMD_today_fast_f(f_values):
        P_func = lambda kk: P_theta_vec(kk, k_max, A_s)
        Omega =[compute_Omega_eMD_today_fast(k, eta_R, k_max, P_func) for k in f_values*2*np.pi]
        return Omega
    T_obs_years= T_obs / (3600*24*365)
    SNR=snr.SNR(T_obs_years, f_values, None, det_list, pol='t', psi=0, gw_spectrum_func=compute_Omega_eMD_today_fast_f)
    print(f"SNR ET RD: {SNR:.2f}")

    N_func= N_auto_interp("data/ET_Sh_coba.txt")
    N_auto=N_func(f_values)
    f_pivot = 2.75
    N_amplitude=float(N_func(f_pivot))
    # Load C_hat and check parameters
    try: 
        if not check_parameters(parameters, parameters_filename):
            raise ValueError("The parameters in the file do not match the input parameters.")
        C_hat= load_C_hat(C_hat_filename)
    
    except Exception as e:
        raise ValueError(f"Error loading data files: {e}")


    likelihood=SigwEstimatorLikelihood_triangular_eMD(C_hat,f_values, T_seg,T_obs, N_auto, f_pivot, N_amplitude)    
    priors = PriorDict(conversion_function=functools.partial(constrain, f_values, N_auto, f_pivot, N_amplitude))
    priors= PriorDict()
    priors['r']       = Uniform(-0.5, 1, '$r$') 
    priors['n_noise'] = Uniform(-10, -5, '$n_{\\text{noise}}$')
    priors['A_s']     = LogUniform(1e-10, 1e-8, '$A_{\\text{s}}$')
    priors['k_max']   = Uniform(50, 250, '$k_{\\text{max}}$')
    priors['eta_R']   = Uniform(0.6, 5, '$\\eta_R$')
    priors['product_eta_k'] = Constraint(minimum=50., maximum=200.)
    priors['min_res'] = Constraint(minimum=0. , maximum=np.max(N_auto)) 

    print("Priors created successfully")
    print("Testing prior sampling...")
    test_sample = priors.sample()
    print(f"Test sample: {test_sample}")
    true_parameters = {'r': r, 'n_noise': n_noise, 'A_s': A_s, 'k_max': k_max, 'eta_R': eta_R}
    label = "estimator eMD"+f'_A{A_s:.1f}_kmax{k_max:.1f}_eta_R{eta_R:.2f}_r{r:.2f}_n{n_noise:.2f}_Tobs{T_obs/3600:.1f}hr_Tseg{T_seg:.1f}sec'
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler='dynesty',
        nlive=2000,
        n_pool=7,
        outdir=outdir,
        label=label,
        injection_parameters=true_parameters,
        check_point_plot=False,
    )
    
    # plot corner with custom formatting for the titles and axes
    try:
        import scienceplots  # Importa qui per assicurarsi che sia caricato
        plt.style.use(['science', 'no-latex'])
        plt.style.use('seaborn-v0_8-bright')
    except (ImportError, OSError) as e:
        print(f"Warning: Could not use 'science' style: {e}")
        print("Using default matplotlib style instead")
        plt.style.use('seaborn-v0_8-bright')  
    mpl.rcParams["figure.dpi"] = 300
    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams["font.size"] = 11
    mpl.rcParams["figure.dpi"] = 110
    mpl.rcParams["font.size"] = 10
    mpl.rcParams["axes.labelsize"] = 18
    mpl.rcParams["xtick.labelsize"] = 12
    mpl.rcParams["ytick.labelsize"] = 12

    cmap = mpl.colormaps['viridis']
    colors = [cmap(i/7) for i in range(7)]

    fig = result.plot_corner(
        truths=true_parameters,
        title_kwargs={"fontsize": 12},
        title_fmt='.6f',  # Cambiato da .2e a .3f per più decimali
        color=colors[3],
        save=False           
    )

    # Ottieni tutti gli assi
    axes = fig.get_axes()
    n_params = int(np.sqrt(len(axes)))  # Numero di parametri

    for idx, param in enumerate(result.search_parameter_keys):
        ax_idx = idx * n_params + idx  # Posizione sulla diagonale
        ax = axes[ax_idx]
        
        # Prendi i campioni e calcola quantili
        samples = result.posterior[param].values
        q_16, q_50, q_84 = np.percentile(samples, [16, 50, 84])
        err_plus = q_84 - q_50
        err_minus = q_50 - q_16
        
        if param == 'A_s':
        # Trova l'esponente del valore mediano
            exponent = int(np.floor(np.log10(abs(q_50))))
        # Scala tutti i valori con lo stesso esponente
            mantissa = q_50 / 10**exponent
            err_plus_scaled = err_plus / 10**exponent
            err_minus_scaled = err_minus / 10**exponent
        
            new_title = f"$({mantissa:.1f}_{{{-err_minus_scaled:.1f}}}^{{+{err_plus_scaled:.1f}}}) \\times 10^{{{exponent}}}$"
        else:
            new_title = f"${q_50:.2f}_{{{-err_minus:.2f}}}^{{+{err_plus:.2f}}}$"
    
        ax.set_title(new_title, fontsize=12)

    for i, ax in enumerate(axes):
    # Calcola riga e colonna dell'asse corrente
        row = i // n_params
        col = i % n_params
    
    # Identifica quale parametro corrisponde a questo asse
        param_x = result.search_parameter_keys[col] if col < len(result.search_parameter_keys) else None
        param_y = result.search_parameter_keys[row] if row < len(result.search_parameter_keys) else None
        
        # Formattazione asse X
        if param_x == 'A_s':
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(
                lambda x, p: f"${float(f'{x:.1e}'.split('e')[0])} \\times 10^{{{int(f'{x:.1e}'.split('e')[1])}}}$" if x != 0 else "$0$"
            ))
        else:
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
            ax.ticklabel_format(style='plain', axis='x')
        
        # Formattazione asse Y
        if param_y == 'A_s':
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(
                lambda y, p: f"${float(f'{y:.1e}'.split('e')[0])} \\times 10^{{{int(f'{y:.1e}'.split('e')[1])}}}$" if y != 0 else "$0$"
            ))
        else:
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
            ax.ticklabel_format(style='plain', axis='y')
        
        # Nascondi le etichette per gli assi interni
        if row < n_params - 1:
            ax.set_xticklabels([])
        else:
            ax.tick_params(axis='x', rotation=45)
            
        if col > 0:
            ax.set_yticklabels([])
        
        # Rimuovi offset text
        ax.xaxis.get_offset_text().set_visible(False)
        ax.yaxis.get_offset_text().set_visible(False)

    # Aggiusta il layout per evitare sovrapposizioni
    plt.subplots_adjust(top=0.95, hspace=0.1, wspace=0.1)

    plot_filename = f"{outdir}/{label}_corner.png"
    fig.savefig(plot_filename, bbox_inches='tight')
    plt.close(fig)

    posterior_dict = {}
    for key in likelihood.parameters.keys():
        posterior_dict[key] = [result.get_one_dimensional_median_and_error_bar(key).median, result.get_one_dimensional_median_and_error_bar(key).plus, result.get_one_dimensional_median_and_error_bar(key).minus]

    json.dump(posterior_dict,open( outdir+'/'+label+'_stat.json', 'w')) 

    return result
