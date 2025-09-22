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

# Save data flag
save = False

# Gravity's acceleration
g = 9.80665

# Read data
file_name ="calibration data/example_data_calibration.csv"
params, data = extract_imu_data(file_name)
sampling_freq, _, t_init, t_wait = params
n_samples = data.shape[0]
print(f"Number of samples in the file: {n_samples}")

# Calibrated static acceleration points
calibrated_accel_avg_data = np.loadtxt("optmization result data/calibrated_accel_avg_data.csv", delimiter=',')

# Static intervals indices
starts, ends = np.loadtxt("optmization result data/static_intervals.csv", delimiter=',').astype(int)

# Raw gyroscope data
raw_gyro_data = data[:,3:]

# Calibrate gyroscope
theta_opt_gyro = imu.calibrate_gyro_from_data(t_init, calibrated_accel_avg_data,
                                              raw_gyro_data, sampling_freq, starts,
                                              ends)

# Optimization parameters
T_opt_gyro= theta_opt_gyro[:6]
k_opt_gyro = theta_opt_gyro[6:9]
b_opt_gyro = theta_opt_gyro[9:]

# Show results
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
time_vector = np.arange(0,n_samples,1) / sampling_freq

# Completed raw gyroscope data
fig3, ax4 = plt.subplots(figsize=(12, 7))
ax4.plot(time_vector, raw_gyro_data)
ax4.set_xlabel("Time [s]")
ax4.set_ylabel("Raw Angular Velocity [-]")
ax4.legend(["wx","wy","wz"])

# Completed gyroscope calibrated data
fig4, ax5 = plt.subplots(figsize=(12, 7))
ax5.plot(time_vector, calibrated_gyro_data)
ax5.set_xlabel("Time [s]")
ax5.set_ylabel("Calibrated Angular Velocity [rad/s]")
ax5.legend(["wx","wy","wz"])

plt.show()