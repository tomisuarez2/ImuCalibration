"""
IMU complete calibration test
Authors: Tomás Suárez, Agustín Corazza, Rodrigo Pérez
University: Universidad Nacional de Cuyo
Based on: "A Robust and Easy to Implement Method for IMU Calibration without External Equipments" - Tedaldi et al., 2014
"""

import numpy as np

from ImuCalibrationModules import imu_calibration as imu
from ImuCalibrationModules.utils import extract_imu_data

# Gravity's acceleration
gravity = 9.80665

# IMU calibration data filename
data_filename = "calibration data/example_data_calibration.csv"

# Extract calibration data
params, data = extract_imu_data(data_filename)
samp_freq, _, t_init, t_wait = params
n_samples = data.shape[0]
print(f"Number of samples in the calibration data file: {n_samples}")

# Calibrate IMU
params_acc, params_gyro = imu.calibrate_imu_from_data(t_init, t_wait, data,
                                                      samp_freq, g=gravity,
                                                      show_data_flag=True, save_data_flag=True)






