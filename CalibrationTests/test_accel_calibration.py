"""
IMU accelerometer calibration test
Authors: Tomás Suárez, Agustín Corazza, Rodrigo Pérez
University: Universidad Nacional de Cuyo
Based on: "A Robust and Easy to Implement Method for IMU Calibration without External Equipments" - Tedaldi et al., 2014
"""

import matplotlib.pyplot as plt
import numpy as np

from ImuCalibrationModules import imu_calibration as imu
from ImuCalibrationModules.utils import extract_imu_data

plt.style.use("seaborn-whitegrid")
plt.rcParams['font.family'] = 'Times New Roman'

spanish = True

# Save data flag
save = False

# Gravity's acceleration
g = 9.80665

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

theta_opt_acc, (starts, ends) = imu.calibrate_accel_from_data(t_init, t_wait,
                                                              raw_accel_data, sampling_freq)

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

# Plots
fig, ax1 = plt.subplots(figsize=(12, 7))
ax2 = ax1.twinx()

if spanish:
    print(f">>> Número de muestras en el archivo de mediciones para calibración: {n_samples}")
    print(f">>> Bias sistemático del acelerómetro optimizado: {b_opt_acc}")
    print(f">>> Factores de escala del acelerómetro optimizados: {k_opt_acc}")
    print(f">>> Desalineamientos del acelerómetro optimizados: {T_opt_acc}")
    ax1.set_xlabel("Muestras promediadas", fontsize=14)
    ax1.set_ylabel("Magnitud de la medida del acelerómetro sin calibrar [-]", fontsize=14)
    ax1.set_title("Magnitud de la medición del acelerómetro estática sin calibrar a lo largo del tiempo", fontsize=16)
    ax2.set_ylabel("Magitud de la medición del acelerómetro calibrada [m/s^2]", fontsize=14)
    labels_plot_1 = ["Sensor sin calibrar","Sensor calibrado","Referencia (g = 9.81 m/s²)"] 
    
else:
    print(f">>> Number of samples in the calibration data file: {n_samples}")
    print(f">>> Accelerometer optimized sistematic bias: {b_opt_acc}")
    print(f">>> Accelerometer optimized scale factors: {k_opt_acc}")
    print(f">>> Accelerometer optimized missalignments: {T_opt_acc}")
    ax1.set_xlabel("Averaged samples", fontsize=14)
    ax1.set_ylabel("Raw acceletometer measurement magnitude [-]", fontsize=14)
    ax1.set_title("Static accelerometer measurement magnitude over time", fontsize=16)
    ax2.set_ylabel("Calibrated accelerometer measurement magnitude [m/s^2]", fontsize=14)
    labels_plot_1 = ["Uncalibrated sensor","Calibrated sensor","Reference (g = 9.81 m/s²)"] 

ax1.plot(raw_accel_magnitudes, color="green", linewidth=1.8, label=labels_plot_1[0])
ax2.plot(cal_accel_magnitudes, color="red", linewidth=1.8, label=labels_plot_1[1])
ax2.axhline(g, color='k', linestyle='--', alpha=1, linewidth=1.5, label=labels_plot_1[2])
fig.legend(loc='upper right', framealpha=1, fontsize=10)
fig.tight_layout()

# Time vector
time_vector = np.arange(0, n_samples, 1) / sampling_freq

# Completed accelerometer calibrated data
fig2, ax3 = plt.subplots(figsize=(12, 7))
ax3.plot(time_vector, calibrated_accel_data)

if spanish:
    ax3.set_title("Evolución de la medición del acelerómetro calibrado a lo largo del tiempo")
    ax3.set_xlabel("Tiempo [s]")
    ax3.set_ylabel("Medición del acelerómetro calibrado [m/s^2]")
    ax3.legend(['Eje X','Eje Y','Eje Z'])
else:
    ax3.set_title("Calibrated accelerometer measurements evolution over time")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Calibrated Accelerometer Measurement [m/s^2]")
    ax3.legend(['X axis','Y axis','Z axis'])

plt.show()