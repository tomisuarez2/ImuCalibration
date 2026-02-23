"""
IMU signal characterization test
Authors: Tomás Suárez, Agustín Corazza, Rodrigo Pérez
University: Universidad Nacional de Cuyo
"""

import numpy as np
from ImuCalibrationModules import imu_calibration as imu
from ImuCalibrationModules.utils import extract_imu_data, show_time_data

spanish = False

# Save data flag
save = False

# Read data
file_name = "characterization data/imu_static_data_6h.csv" 
params, imu_data = extract_imu_data(file_name)
sampling_freq, t_init = params
n_samples = imu_data.shape[0]

# Calibrate data
params_acc = np.loadtxt("optmization result data/params_acc.csv", delimiter=',')
params_gyro = np.loadtxt("optmization result data/params_gyro.csv", delimiter=',')

raw_accel_data = imu_data[:,:3]
raw_gyro_data = imu_data[:,3:]

cal_accel_data = imu.apply_accel_calibration(params_acc, raw_accel_data)
cal_gyro_data = imu.apply_gyro_calibration(params_gyro[:9], raw_gyro_data - params_gyro[9:])

# Recorded data
if spanish:
    show_time_data(cal_accel_data, sampling_freq, ["Eje X", "Eje Y", "Eje Z"], xlabel="Tiempo [s]", ylabel="[m/s^2]", title="Datos del acelerómetro")
    show_time_data(cal_gyro_data, sampling_freq, ["Eje X", "Eje Y", "Eje Z"], xlabel="Tiempo [s]", ylabel="[rad/s]", title="Datos del giróscopo")
else:   
    show_time_data(cal_accel_data, sampling_freq, ["X axis", "Y axis", "Z axis"], ylabel="[m/s^2]", title="Accelerometer data")
    show_time_data(cal_gyro_data, sampling_freq, ["X axis", "Y axis", "Z axis"], ylabel="[rad/s]", title="Gyroscope data")

time_vector = np.arange(0, n_samples, 1) / sampling_freq

# Compute Allan Variance
acc_tau, acc_avar = imu.compute_allan_variance(cal_accel_data, sampling_freq, m_steps='exponential')
acc_a_dev = np.sqrt(acc_avar)

gyro_tau, gyro_avar = imu.compute_allan_variance(cal_gyro_data, sampling_freq, m_steps='exponential')
gyro_a_dev = np.sqrt(gyro_avar)

if spanish:
    plot_titles_acc = 'Desviación de Allan acelerometro'
    plot_titles_gyro = 'Desviación de Allan giroscopio'
else:
    plot_titles_acc = 'Allan deviation accelerometer'  
    plot_titles_gyro = 'Allan deviation gyroscope'  

# Estimate R and q values
R_ax, q_ax, tauwn_ax, taurw_ax = imu.auto_estimate_R_q_from_allan(acc_tau, acc_a_dev[:,0], sampling_freq, plot=True, u='m/s^2', title=plot_titles_acc+' X', spanish=spanish)
R_ay, q_ay, tauwn_ay, taurw_ay = imu.auto_estimate_R_q_from_allan(acc_tau, acc_a_dev[:,1], sampling_freq, plot=True, u='m/s^2', title=plot_titles_acc+' Y', spanish=spanish)
R_az, q_az, tauwn_az, taurw_az = imu.auto_estimate_R_q_from_allan(acc_tau, acc_a_dev[:,2], sampling_freq, plot=True, u='m/s^2', title=plot_titles_acc+' Z', spanish=spanish)

R_gx, q_gx, tauwn_gx, taurw_gx = imu.auto_estimate_R_q_from_allan(gyro_tau, gyro_a_dev[:,0], sampling_freq, plot=True, u='rad/s', title=plot_titles_gyro+' X', spanish=spanish)
R_gy, q_gy, tauwn_gy, taurw_gy = imu.auto_estimate_R_q_from_allan(gyro_tau, gyro_a_dev[:,1], sampling_freq, plot=True, u='rad/s', title=plot_titles_gyro+' Y', spanish=spanish)
R_gz, q_gz, tauwn_gz, taurw_gz = imu.auto_estimate_R_q_from_allan(gyro_tau, gyro_a_dev[:,2], sampling_freq, plot=True, u='rad/s', title=plot_titles_gyro+' Z', spanish=spanish)

# Show results
if spanish:
    print(f">>> Número de muestras en el archivo de mediciones para calibración: {n_samples}")
    print(f">>> Varianza del ruido blanco de medición del eje X del acelerómetro [m^2/s^4]: {R_ax}")
    print(f">>> Intensidad de la caminata aleatoria del sesgo del eje X del acelerómetro [m^2/s^5]: {q_ax}")
    print(f">>> Varianza del ruido blanco de medición del eje Y del acelerómetro [m^2/s^4]: {R_ay}")
    print(f">>> Intensidad de la caminata aleatoria del sesgo del eje Y del acelerómetro [m^2/s^5]: {q_ay}")
    print(f">>> Varianza del ruido blanco de medición del eje Z del acelerómetro [m^2/s^4]: {R_az}")
    print(f">>> Intensidad de la caminata aleatoria del sesgo del eje Z del acelerómetro [m^2/s^5]: {q_az}")
    print(f">>> Varianza del ruido blanco de medición del eje X del giroscopio [rad^2/s^2]: {R_gx}")
    print(f">>> Intensidad de la caminata aleatoria del sesgo del eje X del giroscopio [rad^2/s^3]: {q_gx}")
    print(f">>> Varianza del ruido blanco de medición del eje Y del giroscopio [rad^2/s^2]: {R_gy}")
    print(f">>> Intensidad de la caminata aleatoria del sesgo del eje Y del giroscopio [rad^2/s^3]: {q_gy}")
    print(f">>> Varianza del ruido blanco de medición del eje Z del giroscopio [rad^2/s^2]: {R_gz}")
    print(f">>> Intensidad de la caminata aleatoria del sesgo del eje Z del giroscopio [rad^2/s^3]: {q_gz}")
else:
    print(f">>> Number of samples in the calibration data file: {n_samples}")
    print(f">>> X axis accelerometer white measurement–noise variance [m^2/s^4]: {R_ax}")
    print(f">>> X axis accelerometer bias random–walk intensity [m^2/s^5]: {q_ax}")
    print(f">>> Y axis accelerometer white measurement–noise variance [m^2/s^4]: {R_ay}")
    print(f">>> Y axis accelerometer bias random–walk intensity [m^2/s^5]: {q_ay}")
    print(f">>> Z axis accelerometer white measurement–noise variance [m^2/s^4]: {R_az}")
    print(f">>> Z axis accelerometer bias random–walk intensity [m^2/s^5]: {q_az}")
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

if spanish:
    show_time_data(np.vstack([cal_accel_data[:,0], sim_data_ax]).T, sampling_freq, ["Señal Eje X medida", "Señal Eje X Simulada"], xlabel="Tiempo [s]", ylabel="Medicion acelerometro [m/s^2]", title="Datos del sensor")
    show_time_data(np.vstack([cal_accel_data[:,1], sim_data_ay]).T, sampling_freq, ["Señal Eje Y medida", "Señal Eje Y Simulada"], xlabel="Tiempo [s]", ylabel="Medicion acelerometro [m/s^2]", title="Datos del sensor")
    show_time_data(np.vstack([cal_accel_data[:,2], sim_data_az]).T, sampling_freq, ["Señal Eje Z medida", "Señal Eje Z Simulada"], xlabel="Tiempo [s]", ylabel="Medicion acelerometro [m/s^2]", title="Datos del sensor")
    show_time_data(np.vstack([cal_gyro_data[:,0], sim_data_gx]).T, sampling_freq, ["Señal Eje X medida", "Señal Eje X Simulada"], xlabel="Tiempo [s]", ylabel="Medicion giroscopio [rad/s]", title="Datos del sensor")
    show_time_data(np.vstack([cal_gyro_data[:,1], sim_data_gy]).T, sampling_freq, ["Señal Eje Y medida", "Señal Eje Y Simulada"], xlabel="Tiempo [s]", ylabel="Medicion giroscopio [rad/s]", title="Datos del sensor")
    show_time_data(np.vstack([cal_gyro_data[:,2], sim_data_gz]).T, sampling_freq, ["Señal Eje Z medida", "Señal Eje Z Simulada"], xlabel="Tiempo [s]", ylabel="Medicion giroscopio [rad/s]", title="Datos del sensor")
else:
    show_time_data(np.vstack([cal_accel_data[:,0], sim_data_ax]).T, sampling_freq, ["Logged X axis Signal", "Simulated X axis Signal"], ylabel="Accelerometer data [m/s^2]")
    show_time_data(np.vstack([cal_accel_data[:,1], sim_data_ay]).T, sampling_freq, ["Logged Y axis Signal", "Simulated Y axis Signal"], ylabel="Accelerometer data [m/s^2]")
    show_time_data(np.vstack([cal_accel_data[:,2], sim_data_az]).T, sampling_freq, ["Logged Z axis Signal", "Simulated Z axis Signal"], ylabel="Accelerometer data [m/s^2]")
    show_time_data(np.vstack([cal_gyro_data[:,0], sim_data_gx]).T, sampling_freq, ["Logged X axis Signal", "Simulated X axis Signal"], ylabel="Gyroscope data [rad/s]")
    show_time_data(np.vstack([cal_gyro_data[:,1], sim_data_gy]).T, sampling_freq, ["Logged Y axis Signal", "Simulated Y axis Signal"], ylabel="Gyroscope data [rad/s]")
    show_time_data(np.vstack([cal_gyro_data[:,2], sim_data_gz]).T, sampling_freq, ["Logged Z axis Signal", "Simulated Z axis Signal"], ylabel="Gyroscope data [rad/s]")



