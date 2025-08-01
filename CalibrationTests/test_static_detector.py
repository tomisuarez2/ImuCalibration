"""
IMU static detector test
Authors: Tomás Suárez, Agustín Corazza, Rodrigo Pérez
University: Universidad Nacional de Cuyo
Based on: "A Robust and Easy to Implement Method for IMU Calibration without External Equipments" - Tedaldi et al., 2014
"""

import matplotlib.pyplot as plt
import numpy as np

from ImuCalibrationModules import imu_calibration as imu
from ImuCalibrationModules.utils import extract_imu_data

# Read data
file_name = "calibration data/example_data_calibration.csv" 
params, data = extract_imu_data(file_name)
sampling_freq, _, t_init, t_wait = params
n_samples = data.shape[0]
print(f"Number of samples in the file: {n_samples}")

# Accleration data
accel_data = data[:, :3]

# Initial acceleration variance calculation
t_init_steps = int(t_init * sampling_freq)
zitta_init_squared = np.sum(np.var(data[:t_init_steps-1, :3], axis=0) ** 2)
threshold = 35 * zitta_init_squared # As we change the integer multiple we change our static detection
(starts, ends), statics = imu.find_static_imu_intervals(accel_data, sampling_freq, t_wait, threshold, return_labels=True)
print(f"{len(starts)} static intervals have detected.")

# Time vector
time_vector = np.arange(0, n_samples, 1) / sampling_freq

# Plots
fig, ax1 = plt.subplots()

ax1.plot(time_vector, accel_data)
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Raw Acceleration")
ax1.legend(['raw ax','raw ay','raw az'])

ax2 = ax1.twinx()
ax2.set_ylabel("Static detector")
ax2.plot(time_vector, statics, color='black')

fig.tight_layout()
plt.show()



