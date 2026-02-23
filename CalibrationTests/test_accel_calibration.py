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

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'Times New Roman'

spanish = True

# Save data flag
save = True

# Gravity's acceleration
g = 9.80665

# Read data
file_name = "calibration data/example_data_calibration.csv"
params, data = extract_imu_data(file_name)
sampling_freq, _, t_init, t_wait = params
n_samples = data.shape[0]

# Raw accelerometer data
scale = 9.80665 / 16384
raw_accel_data = data[:,:3] * scale

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
#ax2 = ax1.twinx()

if spanish:
    print(f">>> Número de muestras en el archivo de mediciones para calibración: {n_samples}")
    print(f">>> Bias sistemático del acelerómetro optimizado: {b_opt_acc}")
    print(f">>> Factores de escala del acelerómetro optimizados: {k_opt_acc}")
    print(f">>> Desalineamientos del acelerómetro optimizados: {T_opt_acc}")
    ax1.set_xlabel("Número de medición estática", fontsize=14)
    ax1.set_ylabel("Aceleración medida [m/s^2]", fontsize=14)
    ax1.set_title("Magnitud de las mediciones estáticas del acelerómetro", fontsize=18, fontweight='bold', pad=15)
    ax1.tick_params(axis='both', labelsize=14)
    # ax2.set_ylabel("Magitud de la medición del acelerómetro calibrada [m/s^2]", fontsize=14)
    labels_plot_1 = ["Sensor sin calibrar","Sensor calibrado","Referencia (g = 9.80665 m/s²)"] 
    
else:
    print(f">>> Number of samples in the calibration data file: {n_samples}")
    print(f">>> Accelerometer optimized sistematic bias: {b_opt_acc}")
    print(f">>> Accelerometer optimized scale factors: {k_opt_acc}")
    print(f">>> Accelerometer optimized missalignments: {T_opt_acc}")
    ax1.set_xlabel("Averaged samples", fontsize=14)
    ax1.set_ylabel("Raw acceletometer measurement magnitude [-]", fontsize=14)
    ax1.set_title("Static accelerometer measurement magnitude over time", fontsize=18, fontweight='bold', pad=15)
    ax1.set_ylabel("Calibrated accelerometer measurement magnitude [m/s^2]", fontsize=14)
    labels_plot_1 = ["Uncalibrated sensor","Calibrated sensor","Reference (g = 9.80665 m/s²)"] 

ax1.plot(raw_accel_magnitudes, color="green", linewidth=1.8, label=labels_plot_1[0])
ax1.plot(cal_accel_magnitudes, color="red", linewidth=1.8, label=labels_plot_1[1])
ax1.axhline(g, color='k', linestyle='--', alpha=1, linewidth=1.5, label=labels_plot_1[2])
ax1.set_xlim(0, len(cal_accel_magnitudes)-1)
ax1.legend(fontsize=14, loc="best")

# Time vector
time_vector = np.arange(0, n_samples, 1) / sampling_freq

# Calcular módulo
acc__cal_magnitude = np.linalg.norm(calibrated_accel_data, axis=1)
acc__raw_magnitude = np.linalg.norm(raw_accel_data, axis=1)
# Plot
fig2, ax3 = plt.subplots(figsize=(12, 7))
ax3.plot(time_vector, acc__raw_magnitude, color="green", linewidth=1.8)
ax3.plot(time_vector, acc__cal_magnitude, color="red", linewidth=1.8)
ax3.axhline(g, color='k', linestyle='--', alpha=1, linewidth=1.5)

if spanish:
    ax3.set_title("Evolución del módulo de la aceleración a lo largo del tiempo",  fontsize=18, fontweight='bold', pad=15)
    ax3.set_xlabel("Tiempo [s]", fontsize=14)
    ax3.set_ylabel("Módulo de la aceleración [m/s^2]", fontsize=14)
    ax3.legend(['Sensor sin calibrar', 'Sensor calibrado', "Referencia (g = 9.80665 m/s²)"], fontsize=14, loc="best")
else:
    ax3.set_title("Calibrated acceleration magnitude over time",  fontsize=18, fontweight='bold', pad=15)
    ax3.set_xlabel("Time [s]", fontsize=14)
    ax3.set_ylabel("Acceleration magnitude [m/s^2]", fontsize=14)
    ax3.legend(["Uncalibrated sensor","Calibrated sensor","Reference (g = 9.80665 m/s²)"],  fontsize=14, loc="best")
ax3.tick_params(axis='both', labelsize=14)
ax3.set_xlim(time_vector[0], time_vector[-1])
plt.show()