"""
IMU static detector test
Authors: Tomás Suárez, Agustín Corazza, Rodrigo Pérez
University: Universidad Nacional de Cuyo
Based on: "A Robust and Easy to Implement Method for IMU Calibration without External Equipments" - Tedaldi et al., 2014
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import numpy as np

from ImuCalibrationModules import imu_calibration as imu
from ImuCalibrationModules.utils import extract_imu_data

plt.style.use('seaborn-v0_8-whitegrid')
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
lines = ax1.plot(time_vector, accel_data)


if spanish:
    print(f">>> Número de muestras en el archivo de mediciones para calibración: {n_samples}")
    print(f">>> {len(starts)} han sido detectados.")

    labels_axes = ['Eje X', 'Eje Y', 'Eje Z']

    for line, label in zip(lines, labels_axes):
        line.set_label(label)

    # ax2 = ax1.twinx()
    # ax2.plot(time_vector, statics, color='black')
    #fig.tight_layout()

    # Detectar cambios en statics
    changes = np.diff(statics.astype(int))
    start_indices = np.where(changes == 1)[0] + 1   # pasa de 0 → 1
    end_indices   = np.where(changes == -1)[0] + 1  # pasa de 1 → 0

    # Casos borde
    if statics[0] == 1:
        start_indices = np.insert(start_indices, 0, 0)

    if statics[-1] == 1:
        end_indices = np.append(end_indices, len(statics)-1)

    # Pintar intervalos estáticos
    for start, end in zip(start_indices, end_indices):
        ax1.axvspan(time_vector[start],
                    time_vector[end],
                    alpha=0.2,
                    color='green')

    ax1.set_title("Detector de Intervalos Estáticos - Acelerómetro",
                fontsize=18, fontweight='bold', pad=15)
    ax1.set_xlabel("Tiempo [s]", fontsize=14)
    ax1.set_ylabel("Mediciones crudas [-]", fontsize=14)
    ax1.tick_params(axis='both', labelsize=14)

    static_patch = Patch(facecolor='green', alpha=0.2, label='Estático')
    handles, labels = ax1.get_legend_handles_labels()
    handles.append(static_patch)
    ax1.legend(
        handles=handles,
        fontsize=12,
        loc="best",
        frameon=True,        # activa el recuadro
        fancybox=False,      # caja rectangular (no redondeada)
        framealpha=1.0,      # opacidad completa
        edgecolor='black',   # color del borde
        facecolor='white'    # fondo blanco
    )
    ax1.set_xlim(time_vector[0], time_vector[-1])
    # ax2.set_ylabel("Detector de estaticidad")
else:
    print(f">>> Number of samples in the calibration data file: {n_samples}")
    print(f">>> {len(starts)} static intervals have detected.")

    labels_axes = ['X axis','Y axis','Z axis']

    for line, label in zip(lines, labels_axes):
        line.set_label(label)

    changes = np.diff(statics.astype(int))
    start_indices = np.where(changes == 1)[0] + 1   # pasa de 0 → 1
    end_indices   = np.where(changes == -1)[0] + 1  # pasa de 1 → 0

    # Casos borde
    if statics[0] == 1:
        start_indices = np.insert(start_indices, 0, 0)

    if statics[-1] == 1:
        end_indices = np.append(end_indices, len(statics)-1)

    # Pintar intervalos estáticos
    for start, end in zip(start_indices, end_indices):
        ax1.axvspan(time_vector[start],
                    time_vector[end],
                    alpha=0.2,
                    color='green')

    ax1.set_title("Static Interval Detector - Accelerometer",
                fontsize=18, fontweight='bold', pad=15)
    ax1.set_xlabel("Time [s]", fontsize=14)
    ax1.set_ylabel("Raw measurement [-]", fontsize=14)
    ax1.tick_params(axis='both', labelsize=14)

    static_patch = Patch(facecolor='green', alpha=0.2, label='Static')
    handles, labels = ax1.get_legend_handles_labels()
    handles.append(static_patch)
    ax1.legend(
        handles=handles,
        fontsize=12,
        loc="best",
        frameon=True,        # activa el recuadro
        fancybox=False,      # caja rectangular (no redondeada)
        framealpha=1.0,      # opacidad completa
        edgecolor='black',   # color del borde
        facecolor='white'    # fondo blanco
    )
    ax1.set_xlim(time_vector[0], time_vector[-1])

    # ax2.set_ylabel("Static detector")

plt.show()



