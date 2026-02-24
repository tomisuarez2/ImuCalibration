"""
IMU gyroscope calibration test
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
save = False

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
scale = 1.0 / 131.0                                # for ±250 °/s sensor
raw_gyro_data = data[:,3:] * scale * np.pi / 180.0 # in rad/s

# Calibrate gyroscope
if spanish:
    print(">>> Calibración del giroscopio en progreso...")
else:
    print(">>> Gyroscope calibration in progress...")
theta_opt_gyro = imu.calibrate_gyro_from_data(t_init, calibrated_accel_avg_data,
                                              raw_gyro_data, sampling_freq, starts,
                                              ends)
if spanish:
    print(">>> Calibración del giroscopio finalizada")
else:
    print(">>> Gyroscope calibration finished")
#theta_opt_gyro = np.array([-0.00996739,0.00918384,-0.0029122,0.00723488,-0.00984196,0.00579592,0.00013737,0.00013292,0.00013394,-427.46176147,147.94793701,-80.72266388])
    
# Optimization parameters
T_opt_gyro= theta_opt_gyro[:6]
k_opt_gyro = theta_opt_gyro[6:9]
b_opt_gyro = theta_opt_gyro[9:]

# Show results
print(f"Number of samples in the file: {n_samples}")
print(f"Gyroscope optimized bias: {b_opt_gyro}")
print(f"Gyroscope optimized scale factors: {k_opt_gyro}")
print(f"Gyroscope optimized missalignments: {T_opt_gyro}")

# Compute calibrated angular velocity measurements
calibrated_gyro_data = imu.apply_gyro_calibration(theta_opt_gyro[:9], raw_gyro_data - b_opt_gyro)

# Save data if required
if save:
    np.savetxt("optmization result data/params_gyro.csv", theta_opt_gyro, delimiter=',')

# Plots
# Time vector
time_vector = np.arange(0, n_samples, 1) / sampling_freq

# Completed accelerometer calibrated data
fig1, ax1 = plt.subplots(figsize=(12, 7))
ax1.plot(time_vector, raw_gyro_data)

if spanish:
    ax1.set_title("Evolución de la medición del giroscopio sin calibrar a lo largo del tiempo")
    ax1.set_xlabel("Tiempo [s]")
    ax1.set_ylabel("Medición del del giroscopio sin calibrar [-]")
    ax1.legend(['Eje X','Eje Y','Eje Z'])
else:
    ax1.set_title("Raw gyroscope measurements evolution over time")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Raw Gyroscope Measurement [-]")
    ax1.legend(['X axis','Y axis','Z axis'])

# Completed gyroscope calibrated data
fig2, ax2 = plt.subplots(figsize=(12, 7))
ax2.plot(time_vector, calibrated_gyro_data)
if spanish:
    ax2.set_title("Evolución de la medición del giroscopio calibrado a lo largo del tiempo")
    ax2.set_xlabel("Tiempo [s]")
    ax2.set_ylabel("Medición del del giroscopio calibrado [rad/s]")
    ax2.legend(['Eje X','Eje Y','Eje Z'])
else:
    ax2.set_title("Calibrated gyroscope measurements evolution over time")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Calibrated Gyroscope Measurement [rad/s]")
    ax2.legend(['X axis','Y axis','Z axis'])

plt.show()