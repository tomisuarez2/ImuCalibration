"""
IMU logging data test
Authors: Tomás Suárez, Agustín Corazza, Rodrigo Pérez
University: Universidad Nacional de Cuyo
Based on: "A Robust and Easy to Implement Method for IMU Calibration without External Equipments" - Tedaldi et al., 2014
For this example we use an arduino UNO connected with a MPU6050 imu as indicated in "connection.jpeg" with 
"MPU6050_raw.ino" code, both found in "arduiono code" folder.
"""

from ImuCalibrationModules.utils import log_data_from_imu

file_name = log_data_from_imu('COM7', 38400, 1, t_avar=60*60*6) # t init calculation data log
#file_name = log_data_from_imu('COM7', 38400, 0, t_init=36.5, t_wait=3, t_move=4, n_moves=9) # calibration data log
print(f"\nData has been saved in the following file: {file_name}")

