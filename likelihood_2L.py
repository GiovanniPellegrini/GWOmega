import bilby
from bilby.core.prior import PriorDict, Uniform, Constraint, LogUniform
from utils_2L import gamma_2L, S0, N_auto_interp, constrain
from data_generation import signal_2L, noise_2L, write_parameters_to_file, load_parameters_from_file, check_parameters, load_C_hat
from gwbird import snr
from Omega import P_theta_vec, compute_Omega_eMD_today_fast, P_k_lognormal, compute_Omega_RD_today_fast
from interpolation import ScaledInterpolatorRD, ScaledInterpolatorEMD
import numpy as np
import functools
import json
import constants


import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker


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
        self.interpolator= ScaledInterpolatorRD('omega_grid_RD.pkl')
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

        Omega_gw = self.interpolator(k_peak, sigma, A_s)
        Omega_gw = np.array(Omega_gw)

        C_bar=Omega_gw * self.gamma
        sigma_bar=(C_bar**2 + (self.N_auto/self.S0+Omega_gw)**2)/(2*self.N_seg)
        logL=-0.5*np.sum((self.C_hat - C_bar)**2/sigma_bar + np.log(2*np.pi*sigma_bar))
        return logL
    

    
def run_pe_2L_RD(A_s, k_peak, sigma, T_obs, N_seg, C_hat_filename, parameters_filename, shift_angle=0, outdir='output_pe_2L_RD'):
    T_seg= T_obs / N_seg
    df= 1 / T_seg
    f_values= np.arange(1, 200+df, df)
    parameters= {
        "A_s": A_s,
        "k_peak": k_peak,
        "sigma": sigma,
        "T_obs": T_obs,
        "N_seg": N_seg,
        "T_seg": T_seg,
        "shift_angle": shift_angle
    }


    print("time for each segment:", T_seg)
    print("number of segments:", N_seg)
    print("total observation time (s):", T_obs)

    #SNR valuation    
    det_list = ['ET L1', 'ET L2']
    def compute_Omega_RD_today_fast_f(f_values):
        P_func = lambda kk: P_k_lognormal(kk, k_peak, sigma, A_s)
        Omega =[compute_Omega_RD_today_fast(k, constants.cs_value, P_func, t_max=100, N_s=10, N_t_1=50, N_t_2=100, N_t_3=700) for k in f_values*2*np.pi]
        return Omega
    T_obs_years= T_obs / (3600*24*365)
    SNR=snr.SNR(T_obs_years, f_values, None, det_list, pol='t', psi=0, gw_spectrum_func=compute_Omega_RD_today_fast_f)
    print(f"SNR ET RD: {SNR:.2f}")
    
    # Load C_hat and check parameters
    try: 
        if not check_parameters(parameters, parameters_filename):
            raise ValueError("The parameters in the file do not match the input parameters.")
        C_hat= load_C_hat(C_hat_filename)
    
    except Exception as e:
        raise ValueError(f"Error loading data files: {e}")

    N_func= N_auto_interp("data/ET_15km_ASD.txt")
    N_auto=N_func(f_values)

    # Run parameter estimation
    likelihood=SigwEstimatorLikelihood_RD(C_hat,f_values, T_seg,T_obs, N_auto, shift_angle)
    priors = PriorDict()
    priors['A_s']       = LogUniform(1e-4, 5*10**(-2), '$A_s$') 
    priors['sigma']     = Uniform(0.3, 1, '$\sigma$') 
    priors['k_peak']    = Uniform(30, 250, '$k_{peak}$') 

    true_parameters = {'A_s': A_s, 'sigma': sigma, 'k_peak': k_peak}
    label = "estimator RD"+f'_A{A_s:.4f}_sigma{sigma:.1f}_kpeak{k_peak:.1f}_Tobs{T_obs/3600:.1f}hr_Tseg{T_seg:.1f}sec'
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
        self.interpolator= ScaledInterpolatorEMD('omega_grid_eMD_75x75.pkl')
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

        Omega_gw = self.interpolator(k_max, eta_R, A_s)
        Omega_gw = np.array(Omega_gw)
        C_bar=Omega_gw * self.gamma
        sigma_bar=(C_bar**2 + (self.N_auto/self.S0+Omega_gw)**2)/(2*self.N_seg)
        logL=-0.5*np.sum((self.C_hat - C_bar)**2/sigma_bar + np.log(2*np.pi*sigma_bar))
        return logL
    
def run_pe_2L_eMD(A_s, k_max, eta_R, T_obs, N_seg, shift_angle=0,outdir='output_pe_2L_eMD', C_hat_filename='data/2L/C_hat_2L_eMD_10D.h5', parameters_filename='data/2L/parameters_2L_eMD_10D.txt'):
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
        "shift_angle": shift_angle
    }


    print("time for each segment:", T_seg)
    print("number of segments:", N_seg)
    print("total observation time (s):", T_obs)

    #SNR valuation    
    det_list = ['ET L1', 'ET L2']
    def compute_Omega_eMD_today_fast_f(f_values):
        P_func = lambda kk: P_theta_vec(kk, k_max, A_s)
        Omega =[compute_Omega_eMD_today_fast(k, eta_R,k_max, P_func) for k in f_values*2*np.pi]
        return Omega
    T_obs_years= T_obs / (3600*24*365)
    SNR=snr.SNR(T_obs_years, f_values, None, det_list, pol='t', psi=0, gw_spectrum_func=compute_Omega_eMD_today_fast_f)
    print(f"SNR ET eMD: {SNR:.2f}")
    
    # Load C_hat and check parameters
    try: 
        if not check_parameters(parameters, parameters_filename):
            raise ValueError("The parameters in the file do not match the input parameters.")
        C_hat= load_C_hat(C_hat_filename)
    
    except Exception as e:
        raise ValueError(f"Error loading data files: {e}")

    N_func= N_auto_interp("data/ET_15km_ASD.txt")
    N_auto=N_func(f_values)

    # Run parameter estimation
    likelihood=SigwEstimatorLikelihood_eMD(C_hat,f_values, T_seg,T_obs, N_auto, shift_angle)
    priors = PriorDict(conversion_function=constrain)
    priors['A_s']     = LogUniform(1e-10, 1e-8, '$A_{\\text{s}}$')
    priors['k_max']   = Uniform(50, 250, '$k_{\\text{max}}$')
    priors['eta_R']   = Uniform(0.6, 5, '$\\eta_R$')
    priors['product_eta_k'] = Constraint(minimum=50., maximum=200.)

    true_parameters = {'A_s': A_s, 'k_max': k_max, 'eta_R': eta_R}
    label = "estimator RD"+f'_A{A_s:.4f}_kmax{k_max:.1f}_etaR{eta_R:.1f}_Tobs{T_obs/3600:.1f}hr_Tseg{T_seg:.1f}sec'
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