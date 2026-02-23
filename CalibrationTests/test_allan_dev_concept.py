"""
Allan Deviation Concepy test
Authors: Tomás Suárez, Agustín Corazza, Rodrigo Pérez
University: Universidad Nacional de Cuyo
"""

import numpy as np
from ImuCalibrationModules import imu_calibration as imu
from ImuCalibrationModules.utils import extract_imu_data, show_time_data

spanish = False

# Number of samples
n_samples = 5000000

# Sampling frequency
sampling_freq = 100

# Synthetic parameters
R = 4.5676e-06
q = 5.3642e-11

# Synthetic data
syn_data = imu.simulate_sensor_data(n_samples, sampling_freq, R, q, 0)
syn_data = syn_data.reshape(-1,1)

# Compute Allan Deviation
syn_tau, syn_avar = imu.compute_allan_variance(syn_data, sampling_freq, m_steps='exponential')
syn_a_dev = np.sqrt(syn_avar)
syn_a_dev = syn_a_dev.reshape(-1)
syn_tau = syn_tau.reshape(-1)

if spanish:
    show_time_data(syn_data, sampling_freq, legend=["Datos"], xlabel="Tiempo [s]", ylabel="[u]", title="Datos sintéticos")
else:   
    show_time_data(syn_data, sampling_freq, legend=["Data"], xlabel="Time [s]", ylabel="[u]", title="Synthetic data")

if spanish:
    plot_titles = 'Desviación de Allan'
else:
    plot_titles = 'Allan deviation'  

# Estimate R and q values
R_est, q_est, tauwn_syn, taurw_syn = imu.auto_estimate_R_q_from_allan(syn_tau, syn_a_dev, sampling_freq, plot=True, u="u", title=plot_titles, spanish=spanish)

# Show results
if spanish:
    print(f">>> Número de muestras: {n_samples}")
    print(f">>> Varianza del ruido blanco de medición [u^2]: {R_est}")
    print(f">>> Intensidad de la caminata aleatoria del sesgo [u^2/s]: {q_est}")
else:
    print(f">>> Number of samples: {n_samples}")
    print(f">>> White measurement–noise variance [u^2]: {R_est}")
    print(f">>> Bias random–walk intensity [u^2/s]: {q_est}")



