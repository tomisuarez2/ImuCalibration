"""
IMU gyroscope calibration test
Authors: Tomás Suárez, Agustín Corazza, Rodrigo Pérez
University: Universidad Nacional de Cuyo
Based on: "A Robust and Easy to Implement Method for IMU Calibration without External Equipments" - Tedaldi et al., 2014
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation

from ImuCalibrationModules import imu_calibration as imu
from ImuCalibrationModules.utils import compute_pitch_roll_from_acc, extract_imu_data, integrate_quaternion, show_time_data

plt.style.use("seaborn-whitegrid")
plt.rcParams['font.family'] = 'Times New Roman'

spanish = False

# Save data flag
save = False

# Manufacturer's scale factor
n = 16 # Your IMU A/D converter bits
y = 250.0 * np.pi / 180.0 # Your IMU gyroscope scale in rad/s
scale_factor = 2*y / (2 ** n - 1) 
 
# Gravity's acceleration
g = 9.80665

# Read data
file_name ="calibration data/example_data_calibration.csv"
params, data = extract_imu_data(file_name)
sampling_freq, _, t_init, t_wait = params
n_samples = data.shape[0]

# Calibrated static acceleration points
calibrated_accel_avg_data = np.loadtxt("optmization result data/calibrated_accel_avg_data.csv", delimiter=',')

# Static intervals indices
starts, ends = np.loadtxt("optmization result data/static_intervals.csv", delimiter=',').astype(int)

# Raw gyroscope data
raw_gyro_data = data[:,3:]

# Calibrate gyroscope
if spanish:
    print(">>> Calibración del giroscopio en progreso...")
else:
    print(">>> Gyroscope calibration in progress...")

#theta_opt_gyro = imu.calibrate_gyro_from_data(t_init, calibrated_accel_avg_data, raw_gyro_data, sampling_freq, starts, ends)

if spanish:
    print(">>> Calibración del giroscopio finalizada")
else:
    print(">>> Gyroscope calibration finished")

theta_opt_gyro = np.array([-0.00996739,0.00918384,-0.0029122,0.00723488,-0.00984196,0.00579592,0.00013737,0.00013292,0.00013394,-427.46176147,147.94793701,-80.72266388])
    
# Optimization parameters
T_opt_gyro= theta_opt_gyro[:6]
k_opt_gyro = theta_opt_gyro[6:9]
b_opt_gyro = theta_opt_gyro[9:]

# Compute calibrated angular velocity measurements
calibrated_gyro_data = imu.apply_gyro_calibration(theta_opt_gyro[:9], raw_gyro_data - b_opt_gyro)

# Save data if required
if save:
    np.savetxt("optmization result data/params_gyro.csv", theta_opt_gyro, delimiter=',')

# Compute pitch and roll angles for every static position by mean of giroscope data numerical integration
n_intervals = len(starts) - 1
time_vector = np.arange(0, n_samples, 1) / sampling_freq
dt = 1.0 / sampling_freq

# Preallocate arrays
norm_static_accel = np.empty((n_intervals, 3))
norm_integrated_cal_gyro = np.empty((n_intervals, 3))
norm_integrated_non_cal_gyro = np.empty((n_intervals, 3))

# Non and Calibrated gyro data for numerical integration
cal_gyro = interp1d(time_vector, calibrated_gyro_data, axis=0)
non_cal_gyro = interp1d(time_vector, scale_factor*raw_gyro_data, axis=0)

# Normalized calibrated static acceleration data
magnitudes = np.linalg.norm(calibrated_accel_avg_data, axis=1, keepdims=True)
norm_static_accel = calibrated_accel_avg_data / magnitudes

# Identity quaternion
q0 = np.array([1, 0, 0, 0])

for i in range(n_intervals):

    # Define angular velocity indices
    t0 = time_vector[ends[i]+1]
    tf = time_vector[starts[i+1]-1]

    # Integrate through motion interval
    q_result_cal_gyro = integrate_quaternion(cal_gyro, (t0, tf), q0=q0, dt=dt)
    q_result_non_cal_gyro = integrate_quaternion(non_cal_gyro, (t0, tf), q0=q0, dt=dt)

    # Compute final orientation
    rot_cal_gyro = Rotation.from_quat(np.roll(q_result_cal_gyro,-1)) # We need to use scalar last quaternion [x, y, z, w]
    norm_integrated_cal_gyro[i,:] = rot_cal_gyro.apply(norm_static_accel[i,:], inverse=True)
    rot_non_cal_gyro = Rotation.from_quat(np.roll(q_result_non_cal_gyro,-1)) # We need to use scalar last quaternion [x, y, z, w]
    norm_integrated_non_cal_gyro[i,:] = rot_non_cal_gyro.apply(norm_static_accel[i,:], inverse=True)

euler_angles_acc = compute_pitch_roll_from_acc(norm_static_accel)
pitch_acc = euler_angles_acc[1:,0].reshape(-1,1)
roll_acc = euler_angles_acc[1:,1].reshape(-1,1)
euler_angles_cal_gyro = compute_pitch_roll_from_acc(norm_integrated_cal_gyro)
pitch_cal_gyro = euler_angles_cal_gyro[:,0].reshape(-1,1)
roll_cal_gyro = euler_angles_cal_gyro[:,1].reshape(-1,1)
euler_angles_non_cal_gyro = compute_pitch_roll_from_acc(norm_integrated_non_cal_gyro)
pitch_non_cal_gyro = euler_angles_non_cal_gyro[:,0].reshape(-1,1)
roll_non_cal_gyro = euler_angles_non_cal_gyro[:,1].reshape(-1,1)
# Show results
if spanish:
    print(f">>> Numero de muestras en el archivo: {n_samples}")
    print(f">>> Bias optimizado del giroscopio: {b_opt_gyro}")
    print(f">>> Factores de escala optimizados del giroscopio: {k_opt_gyro}")
    print(f">>> Desalineaciones optimizadas del giroscopio: {T_opt_gyro}")
    show_time_data(np.hstack([pitch_acc, pitch_cal_gyro, pitch_non_cal_gyro]), 1, ["Acelerómetro","Giroscopio calibrado","Giroscopio sin calibrar"], "Muestras", "Grados [°]", "Ángulo de pitch de las muestras estáticas")
    show_time_data(np.hstack([roll_acc, roll_cal_gyro, roll_non_cal_gyro]), 1, ["Acelerómetro","Giroscopio calibrado","Giroscopio sin calibrar"], "Muestras", "Grados [°]", "Ángulo de roll de las muestras estáticas")
else:
    print(f">>> Number of samples in the file: {n_samples}")
    print(f">>> Gyroscope optimized bias: {b_opt_gyro}")
    print(f">>> Gyroscope optimized scale factors: {k_opt_gyro}")
    print(f">>> Gyroscope optimized missalignments: {T_opt_gyro}")
    show_time_data(np.hstack([pitch_acc, pitch_cal_gyro, pitch_non_cal_gyro]), 1, ["Accelerometer","Calibrated gyroscope","Non calibrated gyroscope"], "Samples", "Degrees [°]", "Static measurement pitch angle")
    show_time_data(np.hstack([roll_acc, roll_cal_gyro, roll_non_cal_gyro]), 1, ["Accelerometer","Calibrated gyroscope","Non calibrated gyroscope"], "Samples", "Degrees [°]", "Static measurement roll angle")
