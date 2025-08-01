"""
utils module from IMU Calibration Module
Authors: Tomás Suárez, Agustín Corazza, Rodrigo Pérez
University: Universidad Nacional de Cuyo
Based on: "A Robust and Easy to Implement Method for IMU Calibration without External Equipments" - Tedaldi et al., 2014
"""

import csv
from datetime import datetime
import serial
import time
from typing import Union, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

def log_data_from_imu(
    port: str,
    baud_rate: int,
    log_type: int,
    t_init: Union[int, float, None] = None,
    t_wait: Union[int, float, None] = None,
    t_move: Union[int, float, None] = None,
    n_moves: int=10,
    t_avar: Union[int, float, None] = None
) -> str:
    """
    Log data from an MPU6050 IMU via UART communication. 
    You will need "MPU6050_raw.ino" code in Arduino UNO and connected as shown in "connections.jpeg",
    you can find both files in "arduino code" folder.
    Supports three logging modes:
    1. Data logging for IMU calibration (log_type=0)
    2. Data logging for TINIT computation using Allan Variance (log_type=1)
        - This mode can be also used as a free loggig data type, setting "t_avar" 
          as long as desired logging time

    Args:
        port: Serial port for communication (e.g., 'COM3')
        baud_rate: Baud rate for serial communication (e.g., 38400)
        log_type: Logging mode (0 for calibration, 1 for Allan Variance)
        t_init: Initial static time interval for IMU initialization (seconds)
        t_wait: Static time interval for calibration (required when log_type=0)
        t_move: Moving time interval for calibration (required when log_type=0)
        n_moves: Number of static positions for calibration (default=10)
        time_avar: Logging duration for Allan Variance (required when log_type=1)

    Returns:
        str: Filename of the generated CSV file containing logged data

    Raises:
        ValueError: If required parameters are missing based on log_type
        serial.SerialException: If serial communication fails

    Output CSV format:
        - First row: Fs, sampling_frequency (Hz)
        - Second row: Logging Type, log_type
        - Third row: initialization time, t_init (if log_type=0 else -1)
        - Fourth row: waiting time, t_wait (if log_type=0 else -1)
        - Fifth row: ax,ay,az,gx,gy,gz (column headers)
        - Subsequent rows: [ax,ay,az,gx,gy,gz] (sensor readings)

    Notes:
        - You must reset your Arduino UNO every time you want to use this function
    """
    # Validate arguments
    if log_type == 1 and t_avar is None:
        raise ValueError("t_avar is required for Allan Variance computation")
    elif log_type == 0 and None in (t_init, t_wait, t_move):
        raise ValueError("t_init, t_wait and t_move are required for calibration")

    error_data = 0 # Counter for corrupt data lines.

    def save_line(ser: serial.Serial, writer: csv.writer) -> None:
        """
        Read, validate, and save a single line of IMU data.

        Args:
            ser: Active serial connection
            writer: CSV writer object
        """
        nonlocal error_data
        try:
            line = ser.readline().decode('utf-8').strip()
            if line:
                data = [int(val) for val in line.split(',')]
                if len(data) == 6:
                    writer.writerow(data)   
        except (ValueError, UnicodeDecodeError) as e:
            error_data += 1
            if error_data % 100 == 0: # Only print periodic error to avoid flooding
                print(f"Corrupt data (total errors: {error_data}): {e}")

    # Generate output file with timestamp
    time_stamp = datetime.now().strftime("%d_%m_%Y_%H_%M")
    log_time = t_avar if log_type == 1 else t_init
    suffix = '_tinit.csv' if log_type else '_cal.csv'
    output_file = f"imu_data_{time_stamp}{suffix}"

    try:
        with serial.Serial(port, baud_rate, timeout=1) as ser, open(output_file, 'w', newline='') as csvfile:
            print(f"Connected to {port}.")
            print(">>> Waiting for IMU initialization...")

            # Wait for MPU6050 connection confirmation
            while True:
                response = ser.readline().decode('utf-8').strip()
                if response == "MPU6050 connection successful":
                    break
                elif response == "MPU6050 connection failed":
                    raise RuntimeError("MPU6050 connection failed")

            # Get sampling frecuency
            while True:
                response = ser.readline().decode('utf-8').strip()
                if response.startswith("Computing sample rate frecuency"):
                    sampling_freq = ser.readline().decode('utf-8').strip().split()[1]
                    break

            print(f">>> Sampling frecuency: {sampling_freq} Hz")

            # Initialize CSV file
            writer = csv.writer(csvfile)
            writer.writerow(["Fs", sampling_freq])
            writer.writerow(["Logging Type", log_type])
            if not log_type:
                writer.writerow(["Initialization time", t_init])
                writer.writerow(["Waiting time", t_wait])
            else:
                writer.writerow(["Initialization time", -1])
                writer.writerow(["Waiting time", -1])
            writer.writerow(["ax", "ay", "az", "gx", "gy", "gz"])

            print(f">>> Hold the IMU steady for {log_time + (t_wait if t_wait else 0)} seconds")
            print(">>> Press any letter to start: ")
            input() # Wait for user input
            ser.write(b' ') # Send any byte to start

            print(">>> Data logging process has started")
            # Wait for data collection to start
            while True:
                if ser.readline().decode('utf-8').strip() == "Getting raw data...":
                    break
            
            # Main data collection
            t_static_start = time.time()
            while time.time() - t_static_start < log_time:
                save_line(ser, writer)

            # Addtional calibration movements if in calibration mode
            if not log_type:
                for move_num in range(1, n_moves + 1):
                    print(f"\n>>> Move IMU to new position ({move_num}/{n_moves})")
                    
                    # Movement period
                    t_move_start = time.time()
                    while time.time() - t_move_start < t_move:
                        save_line(ser, writer)

                    # Static period (longer for last position)
                    static_time = t_wait * 2 if move_num == n_moves else t_wait
                    print(">>> Hold the IMU steady..")
                    t_static_start = time.time()
                    while time.time() - t_static_start < static_time:
                        save_line(ser, writer)

            print("\n>>> Capture completed")
            print(f"\n>>> Total corrupt data lines: {error_data}")

    except serial.SerialException as e:
        print(f"Serial communication error: {e}")
        raise

    return output_file

def extract_imu_data(
    file_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract IMU data and parameters from a CSV log file.

    Args:
        file_name: Path to the CSV file containing logged IMU data.

    Returns:
        A tuple containing:
        - params: Array of parameters:
            [fs, log_type, (t_init, t_wait if log_type==0)]
        - data: Array of shape (N, n) containing IMU measurements:
            [ax, ay, az, gx, gy, gz]

    Raises:
        ValueError: If the file format is invalid
        FileNotFoundError: If the file doesn't exist

    Notes:
        - First 4 rows must contain metadata:
          1. fs: sampling frequency
          2. log_type: 0 (calibration) or 1 (Allan variance)
          3. t_init: initial static time (if log_type=0)
          4. t_wait: static interval time (if log_type=0)
        - Subsequent rows contain IMU data
    """

    try:
        # Read and parse metadata in one pass
        with open(file_name, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            metadata = [next(reader) for _ in range(4)]
            
        # Convert metadata with proper error handling
        fs = float(metadata[0][1])
        log_type = int(metadata[1][1])
        
        params = [fs, log_type]
        if log_type == 0:
            params.extend([
                float(metadata[2][1]),  # t_init
                float(metadata[3][1])  # t_wait
            ])

        # Load data efficiently with numpy
        data = np.loadtxt(file_name, delimiter=',', skiprows=5, dtype=np.float32)

        return np.array(params, dtype=np.float32), data

    except (IndexError, ValueError) as e:
        raise ValueError(f"Invalid file format in {file_name}") from e
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {file_name}") from e

def quaternion_rate_of_change(
    quat: np.ndarray,
    ang_vel: np.ndarray,
) -> np.ndarray:
    """
    Computes the time derivative of a quaternion given current orientation and angular velocity.

    Args:
        quat: Current orientation quaternion [w, x, y, z] as array of shape (4,) 
        ang_vel: Angular velocity vector [w1, w2, w3] (rad/s) as array of shape (3,)

    Returns:
    |   Quaternion time derivative as array of shape (4,)

    Notes:
        - Implements the standard quaternion kinematics equation

    """    
    # Angular velocity components
    w1, w2, w3 = ang_vel

    B = np.array([[0 ,-w1,-w2,-w3],
                  [w1,  0, w3,-w2],
                  [w2,-w3,  0, w1],
                  [w3, w2,-w1, 0]])

    return 0.5 * B @ quat

def integrate_quaternion(
    angular_velocity: callable,
    t_span: Tuple[float, float],
    q0: np.ndarray,
    dt: float,
    return_sequence: bool=False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Integrates quaternion attitude using RK4 method with angular velocity.

    Args:
        angular_velocity: Callable returning angular velocity vector (rad/s) at time t
        t_span: Integration time range (t_start, t_end)
        q0: Initial orientation quaternion as array of shape (4,) 
        dt: step time
        return_sequence: Whether to return full integration history

    Returns:
        If return_sequence=False: Final quaternion (4,)
        If return_sequence=True: Tuple of (time_points, quaternion_sequence)
    """
    # Calculate number of steps and preallocate if needed
    t_points = np.arange(t_span[0], t_span[1] + dt, dt)
    
    if return_sequence:
        q_sequence = np.empty((len(t_points), 4))
        q_sequence[0] = q0

    q_current = q0

    # Numerical integration
    for i, t in enumerate(t_points):
        q = q_current
        k1 = quaternion_rate_of_change(q, angular_velocity(t))
        k2 = quaternion_rate_of_change(q + dt / 2 * k1, angular_velocity(t + dt / 2))
        k3 = quaternion_rate_of_change(q + dt / 2 * k2, angular_velocity(t + dt / 2))
        k4 = quaternion_rate_of_change(q + dt * k3, angular_velocity(t + dt))
        q_current = q + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        q_current /= np.linalg.norm(q_current) # Normalize quaternion

        if return_sequence:
            q_sequence[i,:] = q_current

    if return_sequence:
        return t_points, q_sequence
    return q_current
    

    



    

   