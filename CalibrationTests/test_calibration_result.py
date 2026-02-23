"""
IMU calibration result test
Authors: Tomás Suárez, Agustín Corazza, Rodrigo Pérez
University: Universidad Nacional de Cuyo
Based on: "A Robust and Easy to Implement Method for IMU Calibration without External Equipments" - Tedaldi et al., 2014
"""

import numpy as np

from ImuCalibrationModules import imu_calibration as imu
from ImuCalibrationModules.utils import extract_imu_data

spanish = False

# Gravity's acceleration
g = 9.80665

# Manufacturer scale factors
acc_scale_factor = g/16384
gyro_scale_factor = 0.01745/131

# Read data
file_name = "results test data/imu_static_test.csv"
params, data = extract_imu_data(file_name)
fs, _ = params
params_acc = np.loadtxt("optmization result data/params_acc.csv", delimiter=',')
params_gyro = np.loadtxt("optmization result data/params_gyro.csv", delimiter=',')

# Show data
imu.show_data(data, fs, acc_scale_factor, gyro_scale_factor, params_acc, params_gyro, "static test", spanish=spanish)