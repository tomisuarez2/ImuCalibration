"""
IMU signal characterization test
Authors: Tomás Suárez, Agustín Corazza, Rodrigo Pérez
University: Universidad Nacional de Cuyo
"""

import numpy as np
from ImuCalibrationModules import imu_calibration as imu
from ImuCalibrationModules.utils import extract_imu_data, show_time_data

# Use synthetic data
synthetic = False

# Save data flag
save = True

# Read data
file_name = "characterization data/imu_static_data_6h.csv" 
params, imu_data = extract_imu_data(file_name)
sampling_freq, t_init = params
n_samples = imu_data.shape[0]

print(f"Number of samples in the file: {n_samples}")

# Calibrate data
params_acc = np.loadtxt("optmization result data/params_acc.csv", delimiter=',')
params_gyro = np.loadtxt("optmization result data/params_gyro.csv", delimiter=',')

raw_accel_data = imu_data[:,:3]
raw_gyro_data = imu_data[:,3:]

cal_accel_data = imu.apply_accel_calibration(params_acc, raw_accel_data)
cal_gyro_data = imu.apply_gyro_calibration(params_gyro[:9], raw_gyro_data - params_gyro[9:])

# Recorded data
show_time_data(cal_accel_data, sampling_freq, ["ax", "ay", "az"], ylabel="Acceleration data [m2/s]", title="Accelerometer data")
show_time_data(cal_gyro_data, sampling_freq, ["gx", "gy", "gz"], ylabel="Angular velocity data [rad/s]", title="Gyroscope data")

time_vector = np.arange(0, n_samples, 1) / sampling_freq

# Compute Allan Variance
acc_tau, acc_avar = imu.compute_allan_variance(cal_accel_data, sampling_freq, m_steps='exponential')
acc_a_dev = np.sqrt(acc_avar)

gyro_tau, gyro_avar = imu.compute_allan_variance(cal_gyro_data, sampling_freq, m_steps='exponential')
gyro_a_dev = np.sqrt(gyro_avar)

# Estimate R and q values
R_ax, q_ax, tauwn_ax, taurw_ax = imu.auto_estimate_R_q_from_allan(acc_tau, acc_a_dev[:,0], sampling_freq, plot=True, u='m2/s', title='X Accel Allan Deviation')
R_ay, q_ay, tauwn_ay, taurw_ay = imu.auto_estimate_R_q_from_allan(acc_tau, acc_a_dev[:,1], sampling_freq, plot=True, u='m2/s', title='Y Accel Allan Deviation')
R_az, q_az, tauwn_az, taurw_az = imu.auto_estimate_R_q_from_allan(acc_tau, acc_a_dev[:,2], sampling_freq, plot=True, u='m2/s', title='Z Accel Allan Deviation')

R_gx, q_gx, tauwn_gx, taurw_gx = imu.auto_estimate_R_q_from_allan(gyro_tau, gyro_a_dev[:,0], sampling_freq, plot=True, u='rad/s', title='X Gyro Allan Deviation')
R_gy, q_gy, tauwn_gy, taurw_gy = imu.auto_estimate_R_q_from_allan(gyro_tau, gyro_a_dev[:,1], sampling_freq, plot=True, u='rad/s', title='Y Gyro Allan Deviation')
R_gz, q_gz, tauwn_gz, taurw_gz = imu.auto_estimate_R_q_from_allan(gyro_tau, gyro_a_dev[:,2], sampling_freq, plot=True, u='rad/s', title='Z Gyro Allan Deviation')

# Show results
print(f">>> X axis accelerometer white measurement–noise variance [m^4/s^2]: {R_ax}")
print(f">>> X axis accelerometer bias random–walk intensity [m^4/s^3]: {q_ax}")

print(f">>> Y axis accelerometer white measurement–noise variance [m^4/s^2]: {R_ay}")
print(f">>> Y axis accelerometer bias random–walk intensity [m^4/s^3]: {q_ay}")

print(f">>> Z axis accelerometer white measurement–noise variance [m^4/s^2]: {R_az}")
print(f">>> Z axis accelerometer bias random–walk intensity [m^4/s^3]: {q_az}")

print(f">>> X axis gyroscope white measurement–noise variance [rad^2/s^2]: {R_gx}")
print(f">>> X axis gyroscope bias random–walk intensity [rad^2/s^3]: {q_gx}")

print(f">>> Y axis gyroscope white measurement–noise variance [rad^2/s^2]: {R_gy}")
print(f">>> Y axis gyroscope bias random–walk intensity [rad^2/s^3]: {q_gy}")

print(f">>> Z axis gyroscope white measurement–noise variance [rad^2/s^2]: {R_gz}")
print(f">>> Z axis gyroscope bias random–walk intensity [rad^2/s^3]: {q_gz}")

# Save data if required
if save:
    np.savetxt("characterization result data/R_q_ax.csv", (R_ax, q_ax), delimiter=',')
    np.savetxt("characterization result data/R_q_ay.csv", (R_ay, q_ay), delimiter=',')
    np.savetxt("characterization result data/R_q_az.csv", (R_az, q_az), delimiter=',')
    np.savetxt("characterization result data/R_q_gx.csv", (R_gx, q_gx), delimiter=',')
    np.savetxt("characterization result data/R_q_gy.csv", (R_gy, q_gy), delimiter=',')
    np.savetxt("characterization result data/R_q_gz.csv", (R_gz, q_gz), delimiter=',')

# Show time data and simulated data.
sim_data_ax = imu.simulate_sensor_data(n_samples, sampling_freq, R_ax, q_ax, np.mean(cal_accel_data[:,0]))
sim_data_ay = imu.simulate_sensor_data(n_samples, sampling_freq, R_ay, q_ay, np.mean(cal_accel_data[:,1]))
sim_data_az = imu.simulate_sensor_data(n_samples, sampling_freq, R_az, q_az, np.mean(cal_accel_data[:,2]))
sim_data_gx = imu.simulate_sensor_data(n_samples, sampling_freq, R_gx, q_gx, np.mean(cal_gyro_data[:,0]))
sim_data_gy = imu.simulate_sensor_data(n_samples, sampling_freq, R_gy, q_gy, np.mean(cal_gyro_data[:,1]))
sim_data_gz = imu.simulate_sensor_data(n_samples, sampling_freq, R_gz, q_gz, np.mean(cal_gyro_data[:,2]))

show_time_data(np.vstack([cal_accel_data[:,0], sim_data_ax]).T, sampling_freq, ["Logged Ax Signal", "Simulated Ax Signal"], ylabel="Acceleration [m2/s]")
show_time_data(np.vstack([cal_accel_data[:,1], sim_data_ay]).T, sampling_freq, ["Logged Ay Signal", "Simulated Ay Signal"], ylabel="Acceleration [m2/s]")
show_time_data(np.vstack([cal_accel_data[:,2], sim_data_az]).T, sampling_freq, ["Logged Az Signal", "Simulated Az Signal"], ylabel="Acceleration [m2/s]")
show_time_data(np.vstack([cal_gyro_data[:,0], sim_data_gx]).T, sampling_freq, ["Logged Gx Signal", "Simulated Gx Signal"], ylabel="Angular velocity [rad/s]")
show_time_data(np.vstack([cal_gyro_data[:,1], sim_data_gy]).T, sampling_freq, ["Logged Gy Signal", "Simulated Gy Signal"], ylabel="Angular velocity [rad/s]")
show_time_data(np.vstack([cal_gyro_data[:,2], sim_data_gy]).T, sampling_freq, ["Logged Gz Signal", "Simulated Gz Signal"], ylabel="Angular velocity [rad/s]")



