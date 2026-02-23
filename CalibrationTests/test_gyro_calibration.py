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
from ImuCalibrationModules.utils import extract_imu_data, show_time_data, integrate_quaternion, compute_pitch_yaw_from_acc

plt.style.use("seaborn-whitegrid")
plt.rcParams['font.family'] = 'Times New Roman'

spanish = False

# Save data flag
save = False

# Manufacturer scale factor
scale_factor = 0.01745/131

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
theta_opt_gyro = imu.calibrate_gyro_from_data(t_init, calibrated_accel_avg_data,
                                              raw_gyro_data, sampling_freq, starts,
                                              ends)
if spanish:
    print(">>> Calibración del giroscopio finalizada")
else:
    print(">>> Gyroscope calibration finished")
#theta_opt_gyro = np.array([-0.00996739,0.00918384,-0.0029122,0.00723488,-0.00984196,0.00579592,0.00013737,0.00013292,0.00013394,-427.46176147,147.94793701,-80.72266388])
    
# Optimization parameters
T_opt_gyro= theta_opt_gyro[:6]
k_opt_gyro = theta_opt_gyro[6:9]
b_opt_gyro = theta_opt_gyro[9:]

# Show results
if spanish:
    print(f">>> Bias sistemático del giroscopio optimizado: {b_opt_gyro}")
    print(f">>> Factores de escala del giroscopio optimizados: {k_opt_gyro}")
    print(f">>> Desalineamientos del giroscopio optimizados: {T_opt_gyro}")
else:
    print(f">>> Gyroscope optimized sistematic bias: {b_opt_gyro}")
    print(f">>> Gyroscope optimized scale factors: {k_opt_gyro}")
    print(f">>> Gyroscope optimized missalignments: {T_opt_gyro}")

# Compute calibrated angular velocity measurements
calibrated_gyro_data = imu.apply_gyro_calibration(theta_opt_gyro[:9], raw_gyro_data - b_opt_gyro)

# Save data if required
if save:
    np.savetxt("optmization result data/params_gyro.csv", theta_opt_gyro, delimiter=',')

if spanish:
    xlabel_plot = "Tiempo [s]"
    legend_plot_2 = ['Eje X','Eje Y','Eje Z']
    ylabel_plot_2 = "[rad/s]"
    title_plot_3 = "Medición del giroscopio sin calibrar"
    title_plot_4 = "Medición del giroscopio calibrada"
else:
    xlabel_plot = "Time [s]"
    legend_plot_2 = ['X axis','Y axis','Z axis']
    ylabel_plot_2 = "[rad/s]"
    title_plot_3 = "Uncalibrated Gyroscope Measurement"
    title_plot_4 = "Calibrated Gyroscope Measurement"

# Completed gyroscope uncalibrated data
show_time_data(scale_factor*raw_gyro_data.reshape(-1,3), sampling_freq,
               legend=legend_plot_2, xlabel=xlabel_plot, ylabel=ylabel_plot_2, title=title_plot_3)

# Completed gyroscope calibrated data
show_time_data(calibrated_gyro_data.reshape(-1,3), sampling_freq,
                legend=legend_plot_2, xlabel=xlabel_plot, ylabel=ylabel_plot_2, title=title_plot_4)

# Check attitude angles
n_intervals = len(starts) - 1
time_vector = np.arange(0, n_samples, 1) / sampling_freq
dt = 1.0 / sampling_freq

# Preallocate arrays
norm_static_accel = np.empty((n_intervals, 3))
norm_integrated_accel_cal = np.empty((n_intervals, 3))
norm_integrated_accel_noncal = np.empty((n_intervals, 3))

# Calibrated gyro data for numerical integration
cal_gyro = interp1d(time_vector, calibrated_gyro_data, axis=0)
non_cal_gyro = interp1d(time_vector, scale_factor*raw_gyro_data, axis=0)

# Normalized calibrated static acceleration data
magnitudes = np.linalg.norm(calibrated_accel_avg_data, axis=1, keepdims=True)
norm_static_accel = calibrated_accel_avg_data / magnitudes

# Identity quaternion
q_0 = np.array([1, 0, 0, 0])

for i in range(n_intervals):

    # Define angular velocity indices
    t0 = time_vector[ends[i]+1]
    tf = time_vector[starts[i+1]-1]

    # Integrate through motion interval
    q_result_cal = integrate_quaternion(cal_gyro, (t0, tf), q0=q_0, dt=dt)
    q_result_noncal = integrate_quaternion(non_cal_gyro, (t0, tf), q0=q_0, dt=dt)

    # Compute final orientation
    rot_cal = Rotation.from_quat(np.roll(q_result_cal,-1)) # We need to use scalar last quaternion [x, y, z, w]
    norm_integrated_accel_cal[i,:] = rot_cal.apply(norm_static_accel[i,:], inverse=True)
    rot_noncal = Rotation.from_quat(np.roll(q_result_noncal,-1)) # We need to use scalar last quaternion [x, y, z, w]
    norm_integrated_accel_noncal[i,:] = rot_noncal.apply(norm_static_accel[i,:], inverse=True)

# Angles plots
rad_2_deg = 180/np.pi

# Averaged accelerometer measurements Euler Angles
euler_angles_avg_acc = compute_pitch_yaw_from_acc(norm_static_accel[1:,:])
pitch_avg_acc = euler_angles_avg_acc[:,0]
roll_avg_acc = euler_angles_avg_acc[:,1]

# Integreated calibrated gyroscope measurement Euler Angles
euler_angles_int_cal_gyro = compute_pitch_yaw_from_acc(norm_integrated_accel_cal)
pitch_int_cal_gyro = euler_angles_int_cal_gyro[:,0]
roll_int_cal_gyro= euler_angles_int_cal_gyro[:,1]

# Integreated non calibrated gyroscope measurement Euler Angles
euler_angles_int_noncal_gyro = compute_pitch_yaw_from_acc(norm_integrated_accel_noncal)
pitch_int_noncal_gyro = euler_angles_int_noncal_gyro[:,0]
roll_int_noncal_gyro= euler_angles_int_noncal_gyro[:,1]

if spanish:
    show_time_data(np.hstack([rad_2_deg*pitch_avg_acc.reshape(-1,1), rad_2_deg*pitch_int_cal_gyro.reshape(-1,1),  rad_2_deg*pitch_int_noncal_gyro.reshape(-1,1)]), 1, 
                ["Cabeceo promediado de la medición del acelerómetro","Cabeceo de la integración del giroscopio calibrado","Cabeceo de la integración del giroscopio no calibrado"], 
                xlabel="M Posiciones Estáticas", ylabel="Ángulo [°]", title="Cabeceo de la orientación de la IMU")
    show_time_data(np.hstack([rad_2_deg*roll_avg_acc.reshape(-1,1), rad_2_deg*roll_int_cal_gyro.reshape(-1,1),  rad_2_deg*roll_int_noncal_gyro.reshape(-1,1)]), 1, 
                ["Alabeo promediado de la medición del acelerómetro","Alabeo de la integración del giroscopio calibrado","Alabeo de la integración del giroscopio no calibrado"], 
                xlabel="M Posiciones Estáticas", ylabel="Ángulo [°]", title="Alabeo de la orientación de la IMU")
else:
    show_time_data(np.hstack([rad_2_deg*pitch_avg_acc.reshape(-1,1), rad_2_deg*pitch_int_cal_gyro.reshape(-1,1),  rad_2_deg*pitch_int_noncal_gyro.reshape(-1,1)]), 1, 
                ["Averaged accelerometer measurement pitch","Calibrated gyroscope pitch","Non calibrated gyroscope pitch"], 
                xlabel="M Static Position", ylabel="Angle [°]", title="Pitch of IMU orientation")
    show_time_data(np.hstack([rad_2_deg*roll_avg_acc.reshape(-1,1), rad_2_deg*roll_int_cal_gyro.reshape(-1,1),  rad_2_deg*roll_int_noncal_gyro.reshape(-1,1)]), 1, 
                ["Averaged accelerometer measurement roll","Calibrated gyroscope roll","Non calibrated gyroscope roll"], 
                xlabel="M Static Position", ylabel="Angle [°]", title="Roll of IMU orientation")