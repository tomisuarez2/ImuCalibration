"""
IMU t_init calculation test
Authors: Tomás Suárez, Agustín Corazza, Rodrigo Pérez
University: Universidad Nacional de Cuyo
Based on: "A Robust and Easy to Implement Method for IMU Calibration without External Equipments" - Tedaldi et al., 2014
"""

import matplotlib.pyplot as plt
import numpy as np

from ImuCalibrationModules import imu_calibration as imu
from ImuCalibrationModules.utils import extract_imu_data

# IMU t_init calculation data filename
file_name = "calibration data/example_data_tinit_calc.csv" 

# Extract data
params, data = extract_imu_data(file_name)
samp_freq, _, = params
n_samples = data.shape[0]
print(f"Number of samples in the calibration data file: {n_samples}")

# Separate data
raw_gyro_data = data[:,3:6]

# Compute Allan Variance
tau, avar = imu.compute_allan_variance(raw_gyro_data, samp_freq)
avar_norm = np.linalg.norm(avar, axis=1)

# Time interval with the least variance norm
t_init = tau[np.argmin(avar_norm)]
print(f"Recommended time interval length for IMU initialization: {t_init} s")

# Visualization
plt.semilogx(tau, avar)
plt.xlabel("Interval Length [s]")
plt.ylabel("Allan Variance of Raw Gyroscope Data")
plt.legend(["Allan Variance X axis", "Allan Variance Y axis", "Allan Variance Z axis"])
plt.grid(True, which="both")
plt.show()