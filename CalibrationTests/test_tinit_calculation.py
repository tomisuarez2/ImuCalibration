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
_, ax1 = plt.subplots(figsize=(12, 7))
lines = ax1.semilogx(tau, avar, linewidth=0.8)
if spanish:
    print(f">>> Número de muestras en el archivo de mediciones para calibración: {n_samples}")
    print(f">>> Longitud del intervalo de inicialización de la IMU recomendado: {t_init} s")
    title = "Varianza de Allan de la medición sin procesar del giroscopio"
    xlabel = "Longitud del intervalo temporal [s]"
    ylabel = "Varianza de Allan [-]"
    legend = ["Varianza de Allan medición eje X", "Varianza de Allan medición eje Y", "Varianza de Allan medición eje Z"]
else:
    print(f">>> Number of samples in the calibration data file: {n_samples}")
    print(f">>> Recommended time interval length for IMU initialization: {t_init} s")
    title = "Allan Variance of Raw Gyroscope Data"
    xlabel = "Interval Length [s]"
    ylabel = "Allan Variance [-]"
    legend = ["Allan Variance X axis", "Allan Variance Y axis", "Allan Variance Z axis"]
for line, label in zip(lines, legend):
        line.set_label(label)
ax1.grid(True, which="both")
ax1.set_xlabel(xlabel, fontsize=14)
ax1.set_ylabel(ylabel, fontsize=14)
ax1.set_title(title, fontsize=18, fontweight='bold', pad=15)
handles, _ = ax1.get_legend_handles_labels()
ax1.legend(
    handles=handles,
    fontsize=12,
    loc="best",
    frameon=True,        
    fancybox=False,     
    framealpha=1.0,      
    edgecolor='black',  
    facecolor='white'   
)

plt.show()