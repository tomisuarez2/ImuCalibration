"""
IMU Calibration Module
Authors: Tomás Suárez, Agustín Corazza, Rodrigo Pérez
University: Universidad Nacional de Cuyo
Based on: "A Robust and Easy to Implement Method for IMU Calibration without External Equipments" - Tedaldi et al., 2014
"""

from typing import Union, Literal, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from . import utils

def compute_allan_variance(
    data: np.ndarray,
    fs: Union[int,float],
    m_steps: Literal['linear', 'exponential'] = 'linear'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Allan Variance of the input data.

    Args:
        data: Input array of shape (N, n) where N is number of samples and n is number of signals.
        fs: Sampling rate in Hz.
        m_steps: Method for interval length variation:
                'linear' - linear spacing between intervals
                'exponential' - base-2 exponential spacing (default: 'linear')

    Returns:
        Tuple containing:
        - taus: Array of interval lengths in seconds
        - avar: Array of corresponding Allan Variance values

    Notes:
        - For 'linear', evaluates intervals from 2 samples to N//2 samples
        - For 'exponential', evaluates intervals as powers of 2 up to N//2
        - Minimum of 2 intervals required for variance calculation
    """
    n_samples = data.shape[0]

    # Generate interval lengths (tau in samples)
    if m_steps == 'linear':
        max_m = n_samples // 2
        taus = np.arange(2, max_m, dtype=int)
    elif m_steps == 'exponential':
        max_power = int(np.floor(np.log2(n_samples // 2)))
        taus = 2**np.arange(1, max_power + 1)
    else:
        raise ValueError("m_steps must be either 'linear' or 'exponential'")

    # Pre-allocate array for Allan Variance resuts
    avar = np.empty((len(taus), data.shape[1]))

    for i, tau in enumerate(taus):
        # Reshape data into intervals of length tau
        n_intervals = n_samples // tau
        reshaped = data[:n_intervals * tau].reshape(n_intervals, tau, -1)

        # Compute means and differences
        interval_means = reshaped.mean(axis=1)
        diffs = np.diff(interval_means, axis=0)

        # Compute the Allan Variance
        avar[i] = 0.5 * np.mean(diffs**2, axis=0)

    return taus / fs, avar

def find_static_intervals_indices(
    static_labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Indentifies contiguous static intervals from a binary indicator array.

    Args:
        static_labels: Binary array of shape (N,) where 1 indicates static periods and 0 indicates movement.
                       Must be a numpy array of integers or booleans.

    Returns:
        A tuple containing:
        - starts: Array of start indices for each static interval
        - ends: Array of ends indices for each static interval

    Example:
        >>> labels = np.array([0, 1, 1, 0, 1, 1, 1, 0])
        >>> starts, ends = find_static_intervals(labels)
        >>> starts
        array([1, 4])
        >>> ends
        array([3, 7])
    """
    # Ensure input is numpy array and convert int if boolean
    static_labels = np.asarray(static_labels, dtype=np.int8)

    # Find where the intervals change
    changes = np.diff(static_labels, prepend=0, append=0)

    # Start and end indices of 1-blocks
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    # Verify equal number of starts and ends
    if len(starts) != len(ends):
        raise ValueError("Mismatched start/end indices - invalid static_labels input")

    return starts, ends

def find_static_imu_intervals(
    accel_data: np.ndarray,
    fs: Union[int, float],
    t_wait: Union[int, float],
    threshold: Union[int, float],
    return_labels: bool=False
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]]:
    """
    Identify static interval in IMU data by evaluating acceleration data within sliding windows.

    Args:
        accel_data: Input acceleration array data of shape (N, 3) where N is number of samples.
        fs: Sampling rate in Hz.
        threshold: Maximum allowed variance magnitude squared for static classification.
        return_labels: If true returns static labels

    Returns:
        If return_labels=False:
            Tuple of (start_indices, end_indices)
        If return_labels=True:
            Tuple containing ((start_indices, end_indices), static_labels)
        Where:
            - starts: Array of start indices for each static interval
            - ends: Array of ends indices for each static interval
            - static_labels: Binary array of shape (N,) where 1 indicates static periods and 0 indicates movement

    Notes:
        - The first and last half-windows periods are automatically marked as non-static
        - Uses sliding window variance computation for efficiency
    """
    n_samples = accel_data.shape[0]
    half_window_size = int(t_wait * fs / 2)

    # Initialize output array (default non-static)
    static_intervals = np.zeros(n_samples, dtype=np.int8)

    # Early return if window size is too large for data
    if 2 * half_window_size >= n_samples:
        return static_intervals

    # Sliding window implementation
    for i in range(half_window_size, n_samples - half_window_size):
        window = slice(i - half_window_size, i + half_window_size)

        # Compute variance
        window_var = np.var(accel_data[window,:], axis=0)

        # Check threshold condition
        variance_magnitud_squared = np.sum(window_var ** 2)
        if variance_magnitud_squared < threshold:
            static_intervals[i] = 1

    if not return_labels:
        return find_static_intervals_indices(static_intervals)
    else:
        return find_static_intervals_indices(static_intervals), static_intervals

def compute_accel_averages(
    start_indices: np.ndarray,
    end_indices: np.ndarray,
    accel_data: np.ndarray,
) -> np.ndarray:
    """
    Computes static acceleration averages from an acceleration data array and static interval start and end indices.

    Args:
        start_indices: Array of start indices for each static interval
        end_indices: Array of ends indices for each static interval
        accel_data: Input acceleration array data of shape (N, 3) where N is number of samples.

    Returns:
        Numpy array of shape (M,) containing averages of consecutive blocks of data where static_intervals
        has consecutive 1s
    """
    # Initialize average vector
    averages = np.zeros((start_indices.shape[0],3))

    # Computes the averages for each block
    for i, (start, end) in enumerate(zip(start_indices, end_indices)):
        averages[i,:] = np.mean(accel_data[start:end,:], axis=0)

    return averages

def apply_accel_calibration(
    params: np.ndarray,
    raw_data: np.ndarray
) -> np.ndarray:
    """
    Applies the Tedaldi et al. (2014) IMU calibration model to raw acceleration data.

    Args:
        params: Calibration parameters array of shape (9,)
            [misalignment_yz, misalignment_zy, misalignment_zx,
             scale_x, scale_y, scale_z,
             bias_x, bias_y, bias_z]
        raw_data: Raw acceleration array of shape (N, 3)

    Returns:
        Calibrated acceleration data of shape (N, 3)

    Notes:
        - Implements the model from:
          "A Robust and Easy to Implement Method for IMU Calibration without External Equipments"
        - The calibration model is: T @ K @ (raw + bias)
          where T is the misalignment matrix and K is the scaling matrix
    """
    # Split parameters into components
    misalignment, scale, bias = np.split(params, [3,6])

    # Construct missalignment matrix (T)
    T = np.array([
        [1.0, -misalignment[0],  misalignment[1]],
        [0.0,              1.0, -misalignment[2]],
        [0.0,              0.0,              1.0]
    ], dtype=np.float32)

    # Construct scalling matrix (K) as diagonal
    K = np.diag(scale)

    # Pre-compute the combined transformation matrix
    transformation = T @ K

    # Apply calibration in vectorized form:
    # Equivalent to: (transformation @ (raw_data.T + bias)).T

    return (raw_data + bias) @ transformation.T

def apply_gyro_calibration(
    params: np.ndarray,
    raw_data: np.ndarray
) -> np.ndarray:
    """
    Applies the Tedaldi et al. (2014) IMU calibration model to raw angular velocity data.

    Args:
        params: Calibration parameters array of shape (9,)
            [misalignment_yz, misalignment_zy, misalignment_xz,
             misalignment_zx, misalignment_xy, misalignment_yx,
             scale_x, scale_y, scale_z]
        raw_data: Raw acceleration/angular velocity array of shape (N, 3)

    Returns:
        Calibrated angular velocity data of shape (N, 3)

    Notes:
        - Implements the model from:
          "A Robust and Easy to Implement Method for IMU Calibration without External Equipments"
        - The calibration model is: T @ K @ raw
          where T is the misalignment matrix and K is the scaling matrix
    """
    # Split parameters into components
    misalignment, scale = np.split(params, [6])

    # Construct missalignment matrix (T)
    T = np.array([
        [1.0, -misalignment[0], misalignment[1]],
        [misalignment[2], 1.0, -misalignment[3]],
        [-misalignment[4], misalignment[5], 1.0]
    ], dtype=np.float32)

    # Construct scalling matrix (K) as diagonal
    K = np.diag(scale)

    # Pre-compute the combined transformation matrix
    transformation = T @ K

    # Apply calibration in vectorized form:
    # Equivalent to: (transformation @ raw_data.T).T

    return raw_data @ transformation.T

def accel_residuals(
    params: np.ndarray,
    raw_accel_data: np.ndarray,
    g: float=9.80665
) -> np.ndarray:
    """
    Compute residuals between calibrated acceleration magnitudes and gravity reference according to
    Tedaldi et al. (2014).

    Args:
        params: Calibration parameters array of shape (9,)
            [misalignment_yz, misalignment_zy, misalignment_zx,
             scale_x, scale_y, scale_z,
             bias_x, bias_y, bias_z]
        raw_accel_data: Raw acceleration array of shape (M, 3)
        g: Reference gravity magnitude (default: 9.80665 m/s²)

    Returns:
        Array of residuals of shape (M,)
    """
    # Calibrate the acceleration data
    calibrated_accel_data = apply_accel_calibration(params, raw_accel_data)

    # Compute vector norms (magnitud) of calibrated data
    accel_norm = np.linalg.norm(calibrated_accel_data, axis=1)

    # Compute residuals
    return g - accel_norm

def gyro_residuals(
    params: np.ndarray,
    static_accel_data: np.ndarray,
    gyro_data: np.ndarray,
    start_idx: np.ndarray,
    end_idx: np.ndarray,
    fs: Union[int, float],
) -> np.ndarray:
    """
    Compute residuals between calibrated static acceleration measurements and
    gyro-integrated estimates, according to Tedaldi et al. (2014).

    Args:
        params: Calibration parameters array of shape (9,)
            [misalignment_yz, misalignment_zy, misalignment_xz,
             misalignment_zx, misalignment_xy, misalignment_yx,
             scale_x, scale_y, scale_z]
        static_accel_data: Calibrated static acceleration array of shape (M, 3)
        gyro_data: Bias free angular velocity data array of shape (N, 3)
        start_idx: Array of start indices for each static interval
        end_idx: Array of ends indices for each static interval
        fs: Sampling rate in Hz.

    Returns:
        Array of residuals of shape (3*(M-1),)
    """
    n_samples = len(gyro_data)
    n_intervals = len(start_idx) - 1
    time_vector = np.arange(0, n_samples, 1) / fs
    dt = 1.0 / fs

    # Preallocate arrays
    residuals = np.empty((n_intervals, 3))
    norm_static_accel = np.empty((n_intervals, 3))
    norm_integrated_accel = np.empty((n_intervals, 3))

    # Calibrated gyro data for numerical integration
    cal_gyro = interp1d(time_vector, apply_gyro_calibration(params, gyro_data), axis=0)

    # Normalized calibrated static acceleration data
    magnitudes = np.linalg.norm(static_accel_data, axis=1, keepdims=True)
    norm_static_accel = static_accel_data / magnitudes

    # Identity quaternion
    q0 = np.array([1, 0, 0, 0])

    for i in range(n_intervals):

        # Define angular velocity indices
        t0 = time_vector[end_idx[i]+1]
        tf = time_vector[start_idx[i+1]-1]

        # Integrate through motion interval
        q_result = utils.integrate_quaternion(cal_gyro, (t0, tf), q0=q0, dt=dt)

        # Compute final orientation
        rot = Rotation.from_quat(np.roll(q_result,-1)) # We need to use scalar last quaternion [x, y, z, w]
        norm_integrated_accel[i,:] = rot.apply(norm_static_accel[i,:], inverse=True)

        # Compute residuals
        residuals[i] = norm_static_accel[i+1,:] - norm_integrated_accel[i,:]

    return residuals.flatten()

def calibrate_accel_from_data(
    t_init: Union[int, float],
    t_wait: Union[int, float],
    raw_accel_data: np.ndarray,
    fs: Union[int, float],
    theta_init_acc: Optional[np.ndarray]=None,
    g: float=9.80665,
    n_iteration: int=200
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Optimized accelerometer calibration using iterative threshold adjustment as presented in
    Tedaldi et al. (2014).

    Args:
        t_init: Initial static time interval for IMU initialization (s)
        t_wait: Static time interval for calibration (s)
        raw_accel_data: Raw acceleration array of shape (N, 3)
        fs: Sampling rate in Hz.
        theta_init_acc: Initial calibration parameters (9,)
        g: Reference gravity magnitude (default: 9.80665 m/s²)
        n_iterations: Maximum iterations

    Returns:
        Tuple of:
        - Optimal calibration parameters (9,)
        - Tuple of (start_indices, end_indices) for static intervals
    """
    # t_init samples
    t_init_samples = int(t_init * fs)

    # Initial acceleration variance calculation
    zitta_init_sq = np.sum(np.var(raw_accel_data[:t_init_samples-1,:], axis=0) ** 2)

    # Initialize parameters if not provided
    if theta_init_acc is None:
            theta_init_acc = np.array([0,0,0,1,1,1,0,0,0]) # Ideal initial guess

    # Pre-allocate results storage
    Minf = []
    tol = 1e-2

    print(">>> Accelerometer calibration in progress...")

    k_real = -1 # Number of optimization iterations with valid static intervals
    for k in range(2, n_iteration):
        threshold = k * zitta_init_sq
        starts, ends = find_static_imu_intervals(raw_accel_data, fs, t_wait, threshold)
        avg_accel = compute_accel_averages(starts, ends, raw_accel_data)
        n_static_samples = avg_accel.shape[0]

        # Check for sufficient unique orientations
        if n_static_samples > 9:
            rel_diffs = np.abs(avg_accel[1:,:] - avg_accel[:-1,:])
            magnitudes = np.linalg.norm(avg_accel[1:,:], axis=1, keepdims=True)
            if not np.any(np.all(rel_diffs/magnitudes < tol, axis=1)):
                k_real += 1
                # Levenberg-Marquardt accelerometer parameters optimization
                result = least_squares(fun=accel_residuals, x0=theta_init_acc,
                                       args=(avg_accel, g), method='lm',
                                       max_nfev=2000)
                # Append to M matrix the results
                Minf.append((result.cost, result.x, threshold, (starts, ends)))

                # Early termination if no improvement
                if k_real > 50 and Minf[-1][0] > Minf[-2][0]:
                    break

    # Find optimal parameters
    if not Minf:
        raise RuntimeError("No valid static intervals found for calibration")

    optimal_idx = np.argmin(np.array([result[0] for result in Minf]))
    params_acc = Minf[optimal_idx][1]
    static_intervals = Minf[optimal_idx][3]

    print(">>> Accelerometer calibration finished")
    return params_acc, static_intervals

def calibrate_gyro_from_data(
    t_init: Union[int, float],
    static_accel_data: np.ndarray,
    raw_gyro_data: np.ndarray,
    fs: Union[int, float],
    start_indices: np.ndarray,
    end_indices: np.ndarray,
    theta_init_gyro: Optional[np.ndarray]=None,
) -> np.ndarray:
    """
    Optimized gyroscope calibration using calibrated static acceleration vectors as presented in
    Tedaldi et al. (2014).

    Args:
        t_init: Initial static time interval for IMU initialization (s)
        static_accel_data: Calibrated static acceleration array of shape (M, 3)
        raw_gyro_data: Raw angular velocity array of shape (N, 3)
        fs: Sampling rate in Hz.
        start_indices: Array of start indices for each static interval
        end_indices: Array of ends indices for each static interval
        theta_init_gyro: Initial calibration parameters (9,)

    Returns:
        Array:
        - Optimal calibration parameters (12,) (included bias)
    """
    # t_init samples
    t_init_samples = int(t_init * fs)

    # Average gyroscope signals over t_init
    bias = np.mean(raw_gyro_data[:t_init_samples-1,:], axis=0)

    # Bias free gyroscope data
    unbiased_gyro_data = raw_gyro_data - bias

    # Initialize parameters if not provided
    if theta_init_gyro is None:
        n = 16 # Your IMU A/D converter bits
        y = 250.0 * np.pi / 180.0 # Your IMU gyroscope scale in rad/s
        r = (2 ** n - 1) / 2 / y
        theta_init_gyro = np.array([0, 0, 0, 0, 0, 0 , 1/r, 1/r, 1/r]) # Ideal initial guess

    print(">>> Gyroscope calibration in progress")

    # Gyroscope parameters optimization
    result = least_squares(fun=gyro_residuals, x0=theta_init_gyro,
                           args=(static_accel_data, unbiased_gyro_data, start_indices, end_indices, fs),
                           method='lm', max_nfev=2000)

    print(">>> Gyroscope calibration finished")

    return np.hstack([result.x, bias])

def calibrate_imu_from_data(
    t_init: Union[int, float],
    t_wait: Union[int, float],
    data: np.ndarray,
    fs: Union[int, float],
    theta_init_acc: Optional[np.ndarray]=None,
    theta_init_gyro: Optional[np.ndarray]=None,
    g: float=9.80665,
    n_iteration: int=200,
    show_data_flag: bool=False,
    save_data_flag: bool=False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimized imu calibration (accelerometer and gyroscope) using iterative threshold adjustment
    for accelerometer and calibrated static acceleration vectors for gyroscope as presented in
    Tedaldi et al. (2014).

    Args:
        t_init: Initial static time interval for IMU initialization (s)
        t_wait: Static time interval for calibration (s)
        data: Raw acceleration and gyroscope data array of shape (N, 6)
        fs: Sampling rate in Hz.
        theta_init_acc: Initial calibration parameters (9,)
        theta_init_gyro: Initial calibration parameters (9,)
        g: Reference gravity magnitude (default: 9.80665 m/s²)
        n_iterations: Maximum iterations
        show_data_flag: Show data, both console and plots
        save_data_flag: Save data as csv files

    Returns:
        Tuple of:
        - Optimal accelerometer calibration parameters (9,)
            - [misalignment_yz, misalignment_zy, misalignment_zx,
               scale_x, scale_y, scale_z,
               bias_x, bias_y, bias_z]
        - Optimal gyroscope calibration parameters (12,)
            - [misalignment_yz, misalignment_zy, misalignment_xz,
               misalignment_zx, misalignment_xy, misalignment_yx,
               scale_x, scale_y, scale_z, bias_x, bias_y, bias_z]
    """
    # Separate data
    raw_accel_data = data[:,:3]
    raw_gyro_data = data[:,3:]

    # Optimized acceleromater calibration parameters
    params_acc, (starts, ends) = calibrate_accel_from_data(t_init, t_wait,
                                                           raw_accel_data, fs,
                                                           theta_init_acc, g,
                                                           n_iteration)

    # Compute calibrated static acceleration data
    static_accel_data = compute_accel_averages(starts, ends, raw_accel_data)
    calibrated_accel_avg_data = apply_accel_calibration(params_acc, static_accel_data)

    # Optimized gyroscope calibration parameters
    params_gyro = calibrate_gyro_from_data(t_init, calibrated_accel_avg_data,
                                           raw_gyro_data, fs, starts,
                                           ends, theta_init_gyro)

    if save_data_flag:
        np.savetxt("optmization result data/params_acc.csv", params_acc, delimiter=',')
        np.savetxt("optmization result data/calibrated_accel_avg_data.csv", calibrated_accel_avg_data, delimiter=',')
        np.savetxt("optmization result data/static_intervals.csv", (starts, ends), delimiter=',')
        np.savetxt("optmization result data/params_gyro.csv", params_gyro, delimiter=',')

    if show_data_flag:
        # Optimization parameters
        T_opt_acc = params_acc[:3]
        k_opt_acc = params_acc[3:6]
        b_opt_acc = params_acc[6:]

        # Show acelerometer calibration results
        print(f"Accelerometer optimized bias: {b_opt_acc}")
        print(f"Accelerometer optimized scale factors: {k_opt_acc}")
        print(f"Accelerometer optimized missalignments: {T_opt_acc}")

        # Optimization parameters
        T_opt_gyro= params_gyro[:6]
        k_opt_gyro = params_gyro[6:9]
        b_opt_gyro = params_gyro[9:]

        # Show gyroscope calibration results
        print(f"Gyroscope optimized bias: {b_opt_gyro}")
        print(f"Gyroscope optimized scale factors: {k_opt_gyro}")
        print(f"Gyroscope optimized missalignments: {T_opt_gyro}")

        # Show data plots
        show_data(data, fs, params_acc, params_gyro, "calibration")

    return params_acc, params_gyro

def show_data(
    data: np.ndarray,
    fs: Union[int,float],
    params_acc: np.ndarray,
    params_gyro: np.ndarray,
    title: str
) -> None:
    """
    Show IMU data in different plots of raw and calibrated data, applying calibration models as
    presented in Tedaldi et al. (2014).

    Args:
        data: Input array of shape (N, 6) where N is number of samples.
            - data[:,:3]: acceleration data
            - data[:,3:]: angular velocity data
        fs: Sampling rate in Hz.
        params_acc: Accelerometer calibration parameters (9,)
        params_gyro: Gyroscope calibration parameters (12,)
        title: Data title

    Returns:
        None
    """
    n_samples = data.shape[0]

    # Time vector
    time_vector = np.arange(0, n_samples, 1) / fs

    raw_accel_data = data[:,:3]
    raw_gyro_data = data[:,3:]

    cal_accel_data = apply_accel_calibration(params_acc, raw_accel_data)
    cal_gyro_data = apply_gyro_calibration(params_gyro[:9], raw_gyro_data - params_gyro[9:])

    # Completed accelerometer raw data
    _, ax1 = plt.subplots(figsize=(12, 7))
    ax1.plot(time_vector, raw_accel_data)
    ax1.plot(time_vector, np.linalg.norm(raw_accel_data, axis=1))
    ax1.grid(True)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Raw Acceleration [-]")
    ax1.set_title("Raw Acceleration of " + title + " data")
    ax1.legend(["ax","ay","az","|a|"])

    # Completed accelerometer calibrated data
    _, ax2 = plt.subplots(figsize=(12, 7))
    ax2.plot(time_vector, cal_accel_data)
    ax2.plot(time_vector, np.linalg.norm(cal_accel_data, axis=1))
    ax2.grid(True)
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Calibrated Acceleration [m/s^2]")
    ax2.set_title("Calibrated Acceleration of " + title + " data")
    ax2.legend(["ax","ay","az","|a|"])

    # Completed gyroscope raw data
    _, ax3 = plt.subplots(figsize=(12, 7))
    ax3.plot(time_vector, raw_gyro_data)
    ax3.grid(True)
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Raw Angular Velocity [-]")
    ax3.set_title("Raw Angular Velocity of " + title + " data")
    ax3.legend(["wx","wy","wz"])

    # Completed gyroscope calibrated data
    _, ax4 = plt.subplots(figsize=(12, 7))
    ax4.plot(time_vector, cal_gyro_data)
    ax4.grid(True)
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Calibrated Angular Velocity [rad/s]")
    ax4.set_title("Calibrated Angular Velocity of " + title + " data")
    ax4.legend(["wx","wy","wz"])

    plt.show()

