"""
IMU accelerometer calibration test
Authors: Tomás Suárez, Agustín Corazza, Rodrigo Pérez
University: Universidad Nacional de Cuyo
Based on: "A Robust and Easy to Implement Method for IMU Calibration without External Equipments" - Tedaldi et al., 2014
"""

import matplotlib.pyplot as plt
import numpy as np

from ImuCalibrationModules import imu_calibration as imu
from ImuCalibrationModules.utils import extract_imu_data, show_time_data

plt.style.use("seaborn-whitegrid")
plt.rcParams['font.family'] = 'Times New Roman'

spanish = False

# Save data flag
save = False

# Gravity's acceleration
g = 9.80665

# Scale factor
scale_factor = g/16384

# Read data
file_name = "calibration data/example_data_calibration.csv"
params, data = extract_imu_data(file_name)
sampling_freq, _, t_init, t_wait = params
n_samples = data.shape[0]

# Raw accelerometer data
raw_accel_data = data[:,:3]

# Calibrate accelerometer
if spanish:
    print(">>> Calibración del acelerómetro en progreso...")
else:
    print(">>> Accelerometer calibration in progress...")

theta_opt_acc, (starts, ends) = imu.calibrate_accel_from_data(t_init, t_wait, raw_accel_data, sampling_freq)

if spanish:
    print(">>> Calibración del acelerómetro finalizada")
else:
    print(">>> Accelerometer calibration finished")

# Static acceleration data
raw_accel_avg_data = imu.compute_accel_averages(starts, ends, raw_accel_data)

# Optimization parameters
T_opt_acc = theta_opt_acc[:3]
k_opt_acc = theta_opt_acc[3:6]
b_opt_acc = theta_opt_acc[6:]

# Compute calibrated acceleration measurements
calibrated_accel_data = imu.apply_accel_calibration(theta_opt_acc, raw_accel_data)
calibrated_accel_avg_data = imu.apply_accel_calibration(theta_opt_acc, raw_accel_avg_data)

# Compute static acceleration magnitudes
raw_accel_magnitudes = np.linalg.norm(raw_accel_avg_data, axis=1)
cal_accel_magnitudes = np.linalg.norm(calibrated_accel_avg_data, axis=1)

# Save data if required
if save:
    np.savetxt("optmization result data/params_acc.csv", theta_opt_acc, delimiter=',')
    np.savetxt("optmization result data/calibrated_accel_avg_data.csv", calibrated_accel_avg_data, delimiter=',')
    np.savetxt("optmization result data/static_intervals.csv", (starts, ends), delimiter=',')

if spanish:
    print(f">>> Número de muestras en el archivo de mediciones para calibración: {n_samples}")
    print(f">>> Bias sistemático del acelerómetro optimizado: {b_opt_acc}")
    print(f">>> Factores de escala del acelerómetro optimizados: {k_opt_acc}")
    print(f">>> Desalineamientos del acelerómetro optimizados: {T_opt_acc}")
    show_time_data(np.vstack([scale_factor*raw_accel_magnitudes,cal_accel_magnitudes,g*np.ones_like(cal_accel_magnitudes)]).T, 1, ["Sensor sin calibrar","Sensor calibrado","Referencia (g = 9.81 m/s²)"], xlabel="Muestras promediadas", ylabel="Magnitud [m/s^2]", title="Medición estática del acelerómetro a lo largo de las muestras")

else:
    print(f">>> Number of samples in the calibration data file: {n_samples}")
    print(f">>> Accelerometer optimized sistematic bias: {b_opt_acc}")
    print(f">>> Accelerometer optimized scale factors: {k_opt_acc}")
    print(f">>> Accelerometer optimized missalignments: {T_opt_acc}")
    show_time_data(np.vstack([scale_factor*raw_accel_magnitudes,cal_accel_magnitudes,g*np.ones_like(cal_accel_magnitudes)]).T, 1, ["Uncalibrated sensor","Calibrated sensor","Reference (g = 9.81 m/s²)"], xlabel="Averaged samples", ylabel="Magnitude [m/s^2]", title="Static accelerometer measurement magnitude over samples")
