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
from scipy.stats import linregress

from . import utils

#===========================================================
#----------Functions for synthetic data generation----------
#===========================================================

def simulate_sensor_data(
    N: int, 
    fs: float, 
    R: float, 
    q: float, 
    mean: float,
) -> np.ndarray:
    """
    Simulate synthetic static sensor data with white noise and bias random walk.

    Args:
        N: Number of samples.
        fs: Sampling frequency [Hz].
        R: White noise variance (per sample).
        q: Random walk variance (per sample).
        mean: Data mean.

    Returns:
        y: Synthetic sensor measurement array of length N.
    """
    # Initialization
    y = np.zeros(N)

    # White noise
    v = 0
    if not np.isnan(R):
        v = np.random.normal(0, np.sqrt(R), size=N)

    # Random walk increments for bias
    w = 0
    if not np.isnan(q):
        w = np.random.normal(0, np.sqrt(q/fs), size=N)  
    u = np.cumsum(w) 

    y = u + v + mean

    return y

#===========================================================
#-----------Functions for signal characterization-----------
#===========================================================

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

def auto_estimate_R_q_from_allan(
    tau: np.ndarray, 
    sigma: np.ndarray, 
    fs: float,
    slope_tol: float=0.1, 
    min_points: int=4,
    plot: bool=False,
    u: Optional[str] = None,
    title: Optional[str] = None,
    spanish: bool=False
) -> Tuple[float, float, Tuple[int, int], Tuple[int, int]]:
    """
    Automatically estimate R and q from Allan deviation curve.
    It is assumed the standard 1-state random–walk + white-noise measurement model
    for a sensor signal.

    d_k = p_k + b_k + v_k ,    v_k ~ N(0, R)
    b_{k+1} = b_k + w_k ,      w_k ~ N(0, q·T_s)

    where:
        - d_k = sensor measurement 
        - p_k = true measurement
        - R   = white measurement–noise variance [u²]
        - b_k = barometer bias at step k
        - q   = bias random–walk intensity [u²/s]
        - T_s = sampling time [s]

    Args:
        tau: Array of interval lengths in seconds.
        sigma: Array of corresponding Allan Deviation values [u].
        fs: Sampling frecuency [Hz].
        slope_tol: Allowed deviation from ideal slopes (-0.5, +0.5).
        min_points: Minimum number of consecutive points to accept a region.
        plot: Plot flag.
        u: Plot units.
        title: Plot title.
        spanish: Spanish comments.

    Returns:
        R: Measurement noise variance [u^2].
        q: Random walk intensity [u^2/s].
        tau_white_region: (min_tau, max_tau) used for white noise fit.
        tau_rw_region: (min_tau, max_tau) used for random walk fit.
    """
    logtau = np.log10(tau)
    logsig = np.log10(sigma)

    # Local slopes between adjacent points
    slopes = np.diff(logsig) / np.diff(logtau)
    
    def find_region(
        target_slope: float
    ) -> Optional[Tuple[int, int]]:
        """"
        Find regions whit an desired slope.

        Args:
            target_slop: Desired slope.
        
        Returns:
            A tuple containing the indices of the region.
        """
        mask = np.abs(slopes - target_slope) < slope_tol
        # Group consecutive True values
        regions = []
        start = None
        for i, m in enumerate(mask):
            if m and start is None:
                start = i
            elif not m and start is not None:
                if i - start + 1 >= min_points:
                    regions.append((start, i))
                start = None
        if start is not None and len(slopes)-start >= min_points:
            regions.append((start, len(slopes)-1))
        if not regions:
            return None
        # Choose longest region
        region = max(regions, key=lambda r: r[1]-r[0])
        return region
    
    # White noise region 
    reg_w = find_region(-0.5)
    if reg_w:
        idx = range(reg_w[0], reg_w[1]+1)
        _, intercept_w, *_ = linregress(logtau[idx], logsig[idx])
        # Model: sigma = sqrt(R)/sqrt(tau) => log10(sigma) = -0.5*log10(tau) + log10(sqrt(R/fs))
        sqrtR_fs = 10**intercept_w
        R = (sqrtR_fs**2) * fs
        tau_white = (tau[idx[0]], tau[idx[-1]])
    else:
        R, tau_white = np.nan, None

    # Random walk region 
    reg_rw = find_region(0.5)
    if reg_rw:
        idx = range(reg_rw[0], reg_rw[1]+1)
        _, intercept_rw, *_ = linregress(logtau[idx], logsig[idx])
        # Model: sigma = sqrt(q/3) * sqrt(tau) => log10(sigma) = 0.5*log10(tau) + log10(sqrt(q/3))
        sqrt_q_over_3 = 10**intercept_rw
        q = 3 * (sqrt_q_over_3**2)
        tau_rw = (tau[idx[0]], tau[idx[-1]])
    else:
        q, tau_rw = np.nan, None

    if plot:
        if spanish:
            plot_legend = ["Desviación de Allan de la medición del sensor","Pendiente Ruido Blanco Gaussiano","Pendiente Deriva Aleatoria del Sesgo"]
            plot_xlabel = "Duración del intervalo [s]"
            plot_ylabel = f"Desviación de Allan de la señal del sensor [{u}]"
        else:   
            plot_legend = ["Sensor measurement Allan Dev.","White-Gaussian Noise slope","Random-Walk bias slope"]
            plot_xlabel = "Interval Length [s]"
            plot_ylabel = f"Sensor signal Allan deviation [{u}]"
        utils.show_loglog_data(tau, np.vstack([sigma,np.sqrt(R/fs)/np.sqrt(tau),np.sqrt(q/3)*np.sqrt(tau)]).T, 
                               legend=plot_legend, xlabel=plot_xlabel, ylabel=plot_ylabel, title=title)

    return R, q, tau_white, tau_rw

#==========================================================
#-----------Functions for sensor calibration---------------
#==========================================================

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

    # Gyroscope parameters optimization
    result = least_squares(fun=gyro_residuals, x0=theta_init_gyro,
                           args=(static_accel_data, unbiased_gyro_data, start_indices, end_indices, fs),
                           method='lm', max_nfev=2000)

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
    spanish: bool=False
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
        spanish: Spanish comments

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

    if spanish:
        print(">>> Calibración del acelerómetro en progreso...")
    else:
        print(">>> Accelerometer calibration in progress...")
    # Optimized acceleromater calibration parameters
    params_acc, (starts, ends) = calibrate_accel_from_data(t_init, t_wait,
                                                           raw_accel_data, fs,
                                                           theta_init_acc, g,
                                                           n_iteration)
    #params_acc = np.array([1.043860522409845042e-05,2.098452816658057195e-06,2.005985606930172646e-06,6.020083400010406817e-04,5.968449488725015338e-04,5.859241382355084718e-04,-7.115163422327075295e+02,3.585054836108664063e+02,1.840128451085784491e+03])
    #starts = np.array([150,4297,4934,5585,6205,7018,7570,8297,9094,9662])
    #ends = np.array([3613,4299,4946,5690,6368,7046,7765,8457,9158,10095])

    if spanish:
        print(">>> Calibración del acelerómetro finalizada")
    else:
        print(">>> Accelerometer calibration finished")

    # Compute calibrated static acceleration data
    static_accel_data = compute_accel_averages(starts, ends, raw_accel_data)
    calibrated_accel_avg_data = apply_accel_calibration(params_acc, static_accel_data)

    if spanish:
        print(">>> Calibración del giroscopio en progreso...")
    else:
        print(">>> Gyroscope calibration in progress...")
    # Optimized gyroscope calibration parameters
    params_gyro = calibrate_gyro_from_data(t_init, calibrated_accel_avg_data,
                                           raw_gyro_data, fs, starts,
                                           ends, theta_init_gyro)
    #params_gyro = np.array([-0.00996739,0.00918384,-0.0029122,0.00723488,-0.00984196,0.00579592,0.00013737,0.00013292,0.00013394,-427.46176147,147.94793701,-80.72266388])
    if spanish:
        print(">>> Calibración del giroscopio finalizada")
    else:
        print(">>> Gyroscope calibration finished")

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

        # Optimization parameters
        T_opt_gyro= params_gyro[:6]
        k_opt_gyro = params_gyro[6:9]
        b_opt_gyro = params_gyro[9:]

        if spanish:
            print(f">>> Bias sistemático del acelerómetro optimizado: {b_opt_acc}")
            print(f">>> Factores de escala del acelerómetro optimizados: {k_opt_acc}")
            print(f">>> Desalineamientos del acelerómetro optimizados: {T_opt_acc}")
            print(f">>> Bias sistemático del giroscopio optimizado: {b_opt_gyro}")
            print(f">>> Factores de escala del giroscopio optimizados: {k_opt_gyro}")
            print(f">>> Desalineamientos del giroscopio optimizados: {T_opt_gyro}")
        else:
            print(f">>> Accelerometer optimized sistematic bias: {b_opt_acc}")
            print(f">>> Accelerometer optimized scale factors: {k_opt_acc}")
            print(f">>> Accelerometer optimized missalignments: {T_opt_acc}")
            print(f">>> Gyroscope optimized sistematic bias: {b_opt_gyro}")
            print(f">>> Gyroscope optimized scale factors: {k_opt_gyro}")
            print(f">>> Gyroscope optimized missalignments: {T_opt_gyro}")

        # Show data plots
        show_data(data, fs, g/16384, 0.01745/131, params_acc, params_gyro, "Calibration", spanish=spanish)

    return params_acc, params_gyro

def show_data(
    data: np.ndarray,
    fs: Union[int,float],
    acc_scale: float, 
    gyro_scale: float,
    params_acc: np.ndarray,
    params_gyro: np.ndarray,
    title: str,
    spanish: bool = False
) -> None:
    """
    Show IMU data in different plots of raw and calibrated data, applying calibration models as
    presented in Tedaldi et al. (2014).

    Args:
        data: Input array of shape (N, 6) where N is number of samples.
            - data[:,:3]: acceleration data
            - data[:,3:]: angular velocity data
        fs: Sampling rate in Hz.
        acc_scale: Manufacturer accelerometer scale.
        gyro_scale: Manufacturer gyroscope scale.
        params_acc: Accelerometer calibration parameters (9,)
        params_gyro: Gyroscope calibration parameters (12,)
        title: Data title
        spanish: Spanish comments

    Returns:
        None
    """
    raw_accel_data = data[:,:3]
    raw_gyro_data = data[:,3:]

    cal_accel_data = apply_accel_calibration(params_acc, raw_accel_data)
    cal_gyro_data = apply_gyro_calibration(params_gyro[:9], raw_gyro_data - params_gyro[9:])

    if spanish:
        xlabel_plot = "Tiempo [s]"
        legend_plot_1 = ['Eje X','Eje Y','Eje Z','Magnitud']
        legend_plot_2 = ['Eje X','Eje Y','Eje Z']
        ylabel_plot_1 = "[m/s^2]"
        ylabel_plot_2 = "[rad/s]"
        title_plot_1 = "Medición del acelerómetro sin calibrar de los datos de " + title
        title_plot_2 = "Medición del acelerómetro calibrada de los datos de "+ title
        title_plot_3 = "Medición del giroscopio sin calibrar de los datos de " + title
        title_plot_4 = "Medición del giroscopio calibrada de los datos de " + title
    else:
        xlabel_plot = "Time [s]"
        legend_plot_1 = ['X axis','Y axis','Z axis','Magnitude']
        legend_plot_2 = ['X axis','Y axis','Z axis']
        ylabel_plot_1 = "[m/s^2]"
        ylabel_plot_2 = "[rad/s]"
        title_plot_1 = "Uncalibrated Accelerometer Measurement of " + title + " data"
        title_plot_2 = "Calibrated Accelerometer Measurement of " + title + " data"
        title_plot_3 = "Uncalibrated Gyroscope Measurement of " + title + " data"
        title_plot_4 = "Calibrated Gyroscope Measurement of " + title + " data"

    # Completed accelerometer uncalibrated data
    utils.show_time_data(np.hstack([acc_scale*raw_accel_data.reshape(-1,3),np.linalg.norm(acc_scale*raw_accel_data, axis=1).reshape(-1,1)]), 
                         fs, legend=legend_plot_1, xlabel=xlabel_plot, ylabel=ylabel_plot_1, title=title_plot_1)
    

    # Completed accelerometer calibrated data
    utils.show_time_data(np.hstack([cal_accel_data.reshape(-1,3),np.linalg.norm(cal_accel_data, axis=1).reshape(-1,1)]), 
                         fs, legend=legend_plot_1, xlabel=xlabel_plot, ylabel=ylabel_plot_1, title=title_plot_2)

    # Completed gyroscope uncalibrated data
    utils.show_time_data(gyro_scale*raw_gyro_data.reshape(-1,3), 
                         fs, legend=legend_plot_2, xlabel=xlabel_plot, ylabel=ylabel_plot_2, title=title_plot_3)

    # Completed gyroscope calibrated data
    utils.show_time_data(cal_gyro_data.reshape(-1,3), 
                         fs, legend=legend_plot_2, xlabel=xlabel_plot, ylabel=ylabel_plot_2, title=title_plot_4)

    plt.show()

