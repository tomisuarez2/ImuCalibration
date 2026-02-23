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

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'Times New Roman'

spanish = True

# Save data flag
save = True

# Gravity's acceleration
g = 9.80665

# Manufacturer scale factor
scale_factor = g/16384

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

theta_opt_acc, (starts, ends) = imu.calibrate_accel_from_data(t_init, t_wait, raw_accel_data, sampling_freq)
#theta_opt_acc = np.array([1.043860522409845042e-05,2.098452816658057195e-06,2.005985606930172646e-06,6.020083400010406817e-04,5.968449488725015338e-04,5.859241382355084718e-04,-7.115163422327075295e+02,3.585054836108664063e+02,1.840128451085784491e+03])
#starts = np.array([150,4297,4934,5585,6205,7018,7570,8297,9094,9662])
#ends = np.array([3613,4299,4946,5690,6368,7046,7765,8457,9158,10095])

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
    xlabel_plot_1 = "M Muestras promediadas"
    ylabel_plot_1 = "Magnitud [m/s^2]"
    title_plot_1 = "Magnitud de la medición del acelerómetro estática a lo largo del tiempo"
    labels_plot_1 = ["Sensor sin calibrar","Sensor calibrado","Referencia (g = 9.81 m/s²)"] 
    
else:
    print(f">>> Number of samples in the calibration data file: {n_samples}")
    print(f">>> Accelerometer optimized sistematic bias: {b_opt_acc}")
    print(f">>> Accelerometer optimized scale factors: {k_opt_acc}")
    print(f">>> Accelerometer optimized missalignments: {T_opt_acc}")
    xlabel_plot_1 = "M Averaged samples"
    ylabel_plot_1 = "Magnitude [m/s^2]"
    title_plot_1 = "Static accelerometer measurement magnitude over time"
    labels_plot_1 = ["Uncalibrated sensor","Calibrated sensor","Reference (g = 9.81 m/s²)"] 

show_time_data(np.hstack([scale_factor*raw_accel_magnitudes.reshape(-1,1), cal_accel_magnitudes.reshape(-1,1), g*np.ones_like(cal_accel_magnitudes).reshape(-1,1)]), 
               1, legend=labels_plot_1, xlabel=xlabel_plot_1, ylabel=ylabel_plot_1, title=title_plot_1)

if spanish:
    title_plot_2 = "Evolución de la medición calibrada del acelerómetro lo largo del tiempo"
    xlabel_plot_2 = "Tiempo [s]"
    ylabel_plot_2 = "[m/s^2]"
    labels_plot_2 = ['Eje X','Eje Y','Eje Z']
else:
    title_plot_2 = "Calibrated accelerometer measurements evolution over time"
    xlabel_plot_2 = "Time [s]"
    ylabel_plot_2 = "[m/s^2]"
    labels_plot_2 = ['X axis','Y axis','Z axis']

# Completed accelerometer calibrated data
show_time_data(calibrated_accel_data, sampling_freq, 
               legend=labels_plot_2, xlabel=xlabel_plot_2, ylabel=ylabel_plot_2, title=title_plot_2)

plt.show()