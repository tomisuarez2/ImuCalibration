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

spanish = False

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

# Plots
fig, ax1 = plt.subplots()
ax1.plot(time_vector, accel_data)
ax2 = ax1.twinx()
ax2.plot(time_vector, statics, color='black')
fig.tight_layout()

if spanish:
    print(f">>> Número de muestras en el archivo de mediciones para calibración: {n_samples}")
    print(f">>> {len(starts)} han sido detectados.")
    ax1.set_title("Detector de Intervalos Estáticos")
    ax1.set_xlabel("Tiempo [s]")
    ax1.set_ylabel("Medición del acelerómetro sin calibrar")
    ax1.legend(['Eje X','Eje Y','Eje Z'])
    ax2.set_ylabel("Detector de estaticidad")
else:
    print(f">>> Number of samples in the calibration data file: {n_samples}")
    print(f">>> {len(starts)} static intervals have detected.")
    ax1.set_title("Static Interval Detector")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Raw accelerometer measurement")
    ax1.legend(['X axis','Y axis','Z axis'])
    ax2.set_ylabel("Static detector")

plt.show()



