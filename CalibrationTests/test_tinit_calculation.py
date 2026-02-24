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

plt.style.use("seaborn-whitegrid")
plt.rcParams['font.family'] = 'Times New Roman'

spanish = False

# IMU t_init calculation data filename
file_name = "calibration data/example_data_tinit_calc.csv"

# Extract data
params, data = extract_imu_data(file_name)
samp_freq, _, = params
n_samples = data.shape[0]

# Separate data
raw_gyro_data = data[:,3:6]

# Compute Allan Variance
tau, avar = imu.compute_allan_variance(raw_gyro_data, samp_freq)
avar_norm = np.linalg.norm(avar, axis=1)

# Time interval with the least variance norm
t_init = tau[np.argmin(avar_norm)]

# Visualization
plt.semilogx(tau, avar)
if spanish:
    print(f">>> Número de muestras en el archivo de mediciones para calibración: {n_samples}")
    print(f">>> Longitud del intervalo de inicialización de la IMU recomendado: {t_init} s")
    plt.title("Varianza de Allan de la medición sin procesar del giroscopio")
    plt.xlabel("Longitud del intervalo temporal [s]")
    plt.ylabel("Varianza de Allan [-]")
    plt.legend(["Varianza de Allan medición eje X", "Varianza de Allan medición eje Y", "Varianza de Allan medición eje Z"])
else:
    print(f">>> Number of samples in the calibration data file: {n_samples}")
    print(f">>> Recommended time interval length for IMU initialization: {t_init} s")
    plt.title("Allan Variance of Raw Gyroscope Data")
    plt.xlabel("Interval Length [s]")
    plt.ylabel("Allan Variance [-]")
    plt.legend(["Allan Variance X axis", "Allan Variance Y axis", "Allan Variance Z axis"])
plt.grid(True, which="both")
plt.show()