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

# Save data flag
save = True

# Gravity's acceleration
g = 9.80665

# Read data
file_name = "calibration data/example_data_calibration.csv"
params, data = extract_imu_data(file_name)
sampling_freq, _, t_init, t_wait = params
n_samples = data.shape[0]
print(f"Number of samples in the file: {n_samples}")

# Raw accelerometer data
raw_accel_data = data[:,:3]

# Calibrate accelerometer
theta_opt_acc, (starts, ends) = imu.calibrate_accel_from_data(t_init, t_wait,
                                                              raw_accel_data, sampling_freq)

# Static acceleration data
raw_accel_avg_data = imu.compute_accel_averages(starts, ends, raw_accel_data)

# Optimization parameters
T_opt_acc = theta_opt_acc[:3]
k_opt_acc = theta_opt_acc[3:6]
b_opt_acc = theta_opt_acc[6:]

# Show results
print(f"Accelerometer optimized bias: {b_opt_acc}")
print(f"Accelerometer optimized scale factors: {k_opt_acc}")
print(f"Accelerometer optimized missalignments: {T_opt_acc}")

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

ax1.set_xlabel("Samples", fontsize=14)
ax1.set_ylabel("Raw acceleration magnitud [-]", fontsize=14)
ax1.set_title("Static acceleration magnitud over time", fontsize=16)
ax1.plot(raw_accel_magnitudes, color="green", linewidth=1.8, label="Uncalibrated sensor")

ax2 = ax1.twinx()
ax2.set_ylabel("Acceleration magnitud [m/s^2]", fontsize=14)
ax2.plot(cal_accel_magnitudes, color="red", linewidth=1.8, label="Calibrated sensor")

# Gravity reference
ax2.axhline(g, color='k', linestyle='--', alpha=1, linewidth=1.5, label="Reference (g = 9.81 m/s²)")
fig.legend(loc='upper right', framealpha=1, fontsize=13)
fig.tight_layout()

# Completed accelerometer calibrated data
fig2, ax3 = plt.subplots(figsize=(12, 7))

# Time vector
time_vector = np.arange(0, n_samples, 1) / sampling_freq

ax3.plot(time_vector, calibrated_accel_data)
ax3.set_xlabel("Time [s]")
ax3.set_ylabel("Calibrated Acceleration [m/s^2]")
ax3.legend(["ax","ay","az"])

plt.show()