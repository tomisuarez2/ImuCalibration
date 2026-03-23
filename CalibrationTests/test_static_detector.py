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

plt.style.use("seaborn-whitegrid")
plt.rcParams['font.family'] = 'Times New Roman'

spanish = True

# Read data
file_name = "calibration data/example_data_calibration.csv"
params, data = extract_imu_data(file_name)
sampling_freq, _, t_init, t_wait = params
n_samples = data.shape[0]

# Accleration data
accel_data = data[:, :3]

# Initial acceleration variance calculation
t_init_steps = int(t_init * sampling_freq)
zitta_init_squared = np.sum(np.var(data[:t_init_steps-1, :3], axis=0) ** 2)
threshold = 35 * zitta_init_squared # As we change the integer multiple we change our static detection
(starts, ends), statics = imu.find_static_imu_intervals(accel_data, sampling_freq, t_wait, threshold, return_labels=True)

# Time vector
time_vector = np.arange(0, n_samples, 1) / sampling_freq

# Language configuration
if spanish:
    labels_axes = ['Eje X', 'Eje Y', 'Eje Z']
    label_static = 'Estático'
    title = "Detector de Intervalos Estáticos"
    xlabel = "Tiempo [s]"
    ylabel = "Medición del acelerómetro sin calibrar"
    msg_samples = f">>> Número de muestras en el archivo de mediciones para calibración: {n_samples}"
    msg_detected = f">>> {len(starts)} intervalos estáticos han sido detectados."
else:
    labels_axes = ['X axis', 'Y axis', 'Z axis']
    label_static = 'Static'
    title = "Static Interval Detector"
    xlabel = "Time [s]"
    ylabel = "Raw accelerometer measurement"
    msg_samples = f">>> Number of samples in the calibration data file: {n_samples}"
    msg_detected = f">>> {len(starts)} static intervals have been detected."

# Print info
print(msg_samples)
print(msg_detected)

# Plot
fig, ax1 = plt.subplots()

# Plot signals
for i in range(3):
    ax1.plot(time_vector, accel_data[:, i], label=labels_axes[i])

ax1.set_xlim([time_vector[0], time_vector[-1]])

# Shade static intervals
for idx, (start, end) in enumerate(zip(starts, ends)):
    ax1.axvspan(time_vector[start], time_vector[end],
                color='green', alpha=0.2,
                label=label_static if idx == 0 else "")

# Labels and title
ax1.set_title(title)
ax1.set_xlabel(xlabel)
ax1.set_ylabel(ylabel)

# Legend (single call)
ax1.legend(frameon=True, fancybox=False, edgecolor='black')

fig.tight_layout()
plt.show()



