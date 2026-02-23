"""
IMU gyroscope calibration test
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

# Manufacturer scale factor
scale_factor = 0.01745/131

# Gravity's acceleration
g = 9.80665

# Read data
file_name ="calibration data/example_data_calibration.csv"
params, data = extract_imu_data(file_name)
sampling_freq, _, t_init, t_wait = params
n_samples = data.shape[0]

# Calibrated static acceleration points
calibrated_accel_avg_data = np.loadtxt("optmization result data/calibrated_accel_avg_data.csv", delimiter=',')

# Static intervals indices
starts, ends = np.loadtxt("optmization result data/static_intervals.csv", delimiter=',').astype(int)

# Raw gyroscope data
raw_gyro_data = data[:,3:]

# Calibrate gyroscope
if spanish:
    print(">>> Calibración del giroscopio en progreso...")
else:
    print(">>> Gyroscope calibration in progress...")
#theta_opt_gyro = imu.calibrate_gyro_from_data(t_init, calibrated_accel_avg_data,
#                                              raw_gyro_data, sampling_freq, starts,
#                                              ends)
if spanish:
    print(">>> Calibración del giroscopio finalizada")
else:
    print(">>> Gyroscope calibration finished")
theta_opt_gyro = np.array([-0.00996739,0.00918384,-0.0029122,0.00723488,-0.00984196,0.00579592,0.00013737,0.00013292,0.00013394,-427.46176147,147.94793701,-80.72266388])
    
# Optimization parameters
T_opt_gyro= theta_opt_gyro[:6]
k_opt_gyro = theta_opt_gyro[6:9]
b_opt_gyro = theta_opt_gyro[9:]

# Show results
if spanish:
    print(f">>> Bias sistemático del giroscopio optimizado: {b_opt_gyro}")
    print(f">>> Factores de escala del giroscopio optimizados: {k_opt_gyro}")
    print(f">>> Desalineamientos del giroscopio optimizados: {T_opt_gyro}")
else:
    print(f">>> Gyroscope optimized sistematic bias: {b_opt_gyro}")
    print(f">>> Gyroscope optimized scale factors: {k_opt_gyro}")
    print(f">>> Gyroscope optimized missalignments: {T_opt_gyro}")

# Compute calibrated angular velocity measurements
calibrated_gyro_data = imu.apply_gyro_calibration(theta_opt_gyro[:9], raw_gyro_data - b_opt_gyro)

# Save data if required
if save:
    np.savetxt("optmization result data/params_gyro.csv", theta_opt_gyro, delimiter=',')

if spanish:
    xlabel_plot = "Tiempo [s]"
    legend_plot_2 = ['Eje X','Eje Y','Eje Z']
    ylabel_plot_2 = "[rad/s]"
    title_plot_3 = "Medición del giroscopio sin calibrar"
    title_plot_4 = "Medición del giroscopio calibrada"
else:
    xlabel_plot = "Time [s]"
    legend_plot_2 = ['X axis','Y axis','Z axis']
    ylabel_plot_2 = "[rad/s]"
    title_plot_3 = "Uncalibrated Gyroscope Measurement"
    title_plot_4 = "Calibrated Gyroscope Measurement"

# Completed gyroscope uncalibrated data
show_time_data(scale_factor*raw_gyro_data.reshape(-1,3), sampling_freq,
               legend=legend_plot_2, xlabel=xlabel_plot, ylabel=ylabel_plot_2, title=title_plot_3)

# Completed gyroscope calibrated data
show_time_data(calibrated_gyro_data.reshape(-1,3), sampling_freq,
                legend=legend_plot_2, xlabel=xlabel_plot, ylabel=ylabel_plot_2, title=title_plot_4)
