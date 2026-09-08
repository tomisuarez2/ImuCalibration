# IMU Calibration & Characterization Toolbox

**An experimental toolbox for IMU parameter identification, calibration, and stochastic sensor characterization.**

This project provides a collection of tools for working with experimental IMU data, using a low-cost **MPU-6050** as the main case study.

The toolbox was developed as a practical application of concepts from **estimation theory, numerical optimization, system identification, and stochastic process characterization**.

The main objective is not only to calibrate a sensor, but to explore how **experimental measurements can be used to identify the parameters of a sensor error model and characterize the stochastic behavior that remains after calibration**.

---

## Overview

Real IMUs exhibit both **systematic** and **stochastic** measurement errors.

A simplified view of the experimental process implemented in this project is:

```text
              Experimental measurements
                        │
                        ▼
                  Raw IMU data
                        │
             ┌──────────┴──────────┐
             │                     │
             ▼                     ▼
      Systematic errors      Stochastic errors
             │                     │
             ▼                     ▼
     Parameter estimation    Allan variance
             │                     │
             ▼                     ▼
       Optimization          Noise characterization
             │                     │
             └──────────┬──────────┘
                        ▼
             Calibrated & characterized
                        IMU
```

The toolbox therefore addresses two complementary problems:

### 1. Deterministic parameter identification

The calibration process estimates parameters associated with systematic errors, including:

- Sensor biases
- Scale factors
- Axis misalignment
- Calibration parameters for the accelerometer and gyroscope

The parameters are obtained by formulating the calibration problem as a nonlinear optimization problem and solving it using the **Levenberg–Marquardt algorithm**.

### 2. Stochastic sensor characterization

After accounting for systematic errors, the remaining measurement behavior can be studied statistically.

The toolbox includes **Allan variance / Allan deviation** analysis to characterize properties of sensor noise and bias instability over different averaging times.

This provides a way of connecting experimental sensor data with stochastic models used in estimation and navigation applications.

---

## Toolbox Contents

The repository is organized around the complete experimental workflow:

```text
Raw sensor data
      │
      ├── Data acquisition
      │
      ├── Static interval detection
      │
      ├── Calibration
      │      ├── Accelerometer
      │      └── Gyroscope
      │
      ├── Optimization
      │
      ├── Calibration validation
      │
      └── Stochastic characterization
             └── Allan variance / deviation
```

The repository contains:

- IMU calibration algorithms
- Automatic detection of static measurements
- Accelerometer calibration
- Gyroscope calibration
- Nonlinear parameter optimization
- Quaternion-based numerical integration
- Allan variance / Allan deviation analysis
- Experimental datasets
- Calibration and characterization results
- Visualization and analysis tools
- Arduino code for MPU-6050 data acquisition
- Supporting theoretical material

---

# Calibration

## Sensor Error Model

The calibration procedure is based on the error models described by Tedaldi, Pretto and Menegatti.

Real IMUs can exhibit effects such as:

- Non-zero biases
- Different scale factors between axes
- Non-orthogonality between sensing axes
- Measurement noise

The calibration procedure estimates the deterministic parameters of these models from experimental measurements.

For the accelerometer, the sensor is placed in different static orientations. Since the magnitude of the gravitational acceleration is known, the measurements provide constraints from which the calibration parameters can be identified.

For the gyroscope, the calibration depends on the calibrated accelerometer. Angular-rate measurements are integrated and compared through the corresponding changes in the measured gravity direction.

---

## Parameter Estimation

Rather than assigning calibration parameters manually, the toolbox formulates calibration as a parameter estimation problem.

In general terms:

```text
Experimental data
       │
       ▼
Sensor error model
       │
       ▼
Parameter vector
       │
       ▼
Cost function
       │
       ▼
Nonlinear optimization
       │
       ▼
Estimated calibration parameters
```

The calibration parameters are estimated by minimizing a nonlinear cost function using **Levenberg–Marquardt** optimization.

This makes the project a practical example of parameter identification from experimental data.

---

## Experimental Procedure

The calibration does not require an external calibration device.

The required data can be obtained by manually placing the IMU in different static orientations.

The general procedure is:

1. Acquire raw accelerometer and gyroscope measurements.
2. Detect static intervals in the data.
3. Estimate representative measurements for each static interval.
4. Estimate accelerometer calibration parameters.
5. Use the calibrated accelerometer to support gyroscope calibration.
6. Estimate gyroscope calibration parameters.
7. Validate the resulting calibration using experimental data.

The method requires sufficiently diverse orientations for the parameters to be observable. The reference paper recommends collecting approximately **36–50 different static attitudes**.

---

# Automatic Static Detection

Static intervals are required during the calibration procedure.

The toolbox includes an automatic static detector based on the statistical behavior of the measured signal.

This allows the calibration process to operate on experimental datasets without requiring every static interval to be manually selected.

---

# Quaternion Integration

Gyroscope calibration requires integrating angular velocity measurements.

The toolbox implements quaternion-based attitude propagation using a **fourth-order Runge–Kutta integration scheme**.

The quaternion is normalized during the integration process to maintain a valid attitude representation.

This provides an additional practical application of:

- Quaternion kinematics
- Numerical integration
- Rigid-body attitude representation

---

# Stochastic Characterization

Calibration removes or reduces deterministic errors, but it does not make the sensor perfect.

The remaining error contains stochastic components that are important when the sensor is later used in applications such as:

- State estimation
- Sensor fusion
- Inertial navigation
- Robotics
- Guidance, Navigation and Control

For this reason, the toolbox also includes tools for **stochastic characterization**.

## Allan Variance

The **Allan variance** is computed from sensor time-series data to study how the statistical behavior of the measurements changes with averaging time.

Its square root, the **Allan deviation**, provides a convenient representation of the same analysis.

The resulting curves can be used to identify different noise behaviors and to obtain parameters useful for stochastic sensor models.

Conceptually:

```text
Sensor time series
       │
       ▼
 Allan variance
       │
       ▼
Allan deviation
       │
       ▼
Noise characteristics
       │
       ▼
Stochastic sensor model
```

This part of the project complements calibration: while calibration focuses primarily on **deterministic parameter estimation**, Allan analysis focuses on the **statistical characterization of measurement errors**.

---

# Why This Project?

The project was developed as an experimental exercise in applying estimation and optimization concepts to a real physical system.

Instead of considering calibration only as a collection of correction equations, the approach is treated as an estimation problem:

> **Given experimental measurements and a model of the sensor, can the unknown parameters of that model be estimated from the data?**

This involves several important concepts:

| Concept                  | Application                          |
| ------------------------ | ------------------------------------ |
| System modeling          | IMU measurement error models         |
| Experimental design      | Selection of static orientations     |
| Parameter identification | Estimation of calibration parameters |
| Nonlinear optimization   | Levenberg–Marquardt                  |
| Numerical methods        | Quaternion integration               |
| Statistical analysis     | Sensor noise characterization        |
| Allan variance           | Stochastic error analysis            |
| Experimental validation  | Evaluation using real measurements   |

One of the main lessons from the project is that **optimization convergence alone does not guarantee a physically meaningful identification**. The quality and diversity of the experimental data, the chosen model, and the observability of its parameters are equally important.

---

# Data Acquisition

Experimental data can be acquired using the included Arduino code and an **MPU-6050**.

The acquisition setup provides:

- Accelerometer measurements
- Gyroscope measurements
- Approximately 100 Hz sampling frequency by default
- Data-ready synchronization
- CSV-compatible output

The recorded measurements follow the format:

```text
ax, ay, az, gx, gy, gz
```

The resulting datasets can then be processed by the Python toolbox.

---

# Repository Structure

```text
ImuCalibration/
│
├── CalibrationTests/
│   └── Calibration and validation experiments
│
├── ImuCalibrationModules/
│   └── Calibration and characterization toolbox
│
├── arduino code/
│   └── MPU-6050 data acquisition
│
├── calibration data/
│   └── Experimental calibration datasets
│
├── characterization data/
│   └── Sensor characterization datasets
│
├── characterization result data/
│   └── Characterization results
│
├── characterization result images/
│   └── Characterization plots
│
├── optimization result images/
│   └── Optimization results and plots
│
├── results test data/
│   └── Experimental validation data
│
└── theory/
    └── Supporting theoretical material
```

---

# Quick Start

Clone the repository and install the required Python packages:

```bash
git clone https://github.com/tomisuarez2/ImuCalibration.git
cd ImuCalibration

pip install -r requirements.txt
```

The complete calibration workflow can then be executed with:

```bash
python -m CalibrationTests.test_complete_imu_calibration
```

The repository also includes experimental datasets and generated results that can be used to explore the different stages of the toolbox.

---

## Example visualization

For example, by running the above example you can obtain the following IMU calibration parameters:

```bash
Accelerometer optimized bias: [-711.51634223  358.50548361 1840.12845109]
Accelerometer optimized scale factors: [0.00060201 0.00059684 0.00058592]
Accelerometer optimized missalignments: [1.04386052e-05 2.09845282e-06 2.00598561e-06]
Gyroscope optimized bias: [-427.46176147  147.94793701  -80.72266388]
Gyroscope optimized scale factors: [0.00013737 0.00013292 0.00013394]
Gyroscope optimized missalignments: [-0.00996739  0.00918384 -0.0029122  0.00723488 -0.00984196  0.00579592]
```
![Static interval detector](optimization%20result%20images/static_detector_test.png)

![Non calibrated vs calibrated accel meas](optimization%20result%20images/accel_magnitude_calibration.png)

![gyro pitch](optimization%20result%20images/gyro_pitch.png)

![gyro roll](optimization%20result%20images/gyro_roll.png)


- **Allan deviation curve** with fitted slopes
- Estimated noise parameters:

 ```bash
>>> Y axis accelerometer white measurement–noise variance [m^2/s^4]: 0.000927991675334452
>>> Y axis accelerometer bias random–walk intensity [m^2/s^5]: nan
>>> Y axis gyroscope white measurement–noise variance [rad^2/s^2]: 4.9340091913017585e-06
>>> Y axis gyroscope bias random–walk intensity [rad^2/s^3]: nan
 ```
- Visualization of white noise (−½ slope) and random walk (+½ slope) regions

![Allan Deviation Plot](characterization%20result%20images/allan_dev_plot_ay.png)

![Real vs Simulated data](characterization%20result%20images/real_vs_sim_ay.png)

![Allan Deviation Plot](characterization%20result%20images/allan_dev_plot_gy.png)

![Real vs Simulated data](characterization%20result%20images/real_vs_sim_gy.png)

It can be seen from above pictures that there is no apreciable random bias walk sensor noise. Practically all the sensor noise is dominated by measurement white gaussian noise.

---

# Validation

Calibration results can be evaluated by comparing the behavior of the raw and calibrated measurements.

The repository includes experimental results and visualization tools for examining:

- Calibration residuals
- Sensor measurements before and after calibration
- Optimization results
- Characterization results
- Allan deviation curves

The goal is to evaluate whether the identified parameters actually improve the physical consistency of the measurements rather than merely obtaining numerical convergence.

---

# Limitations

The calibration procedure is subject to the limitations of the underlying model and experimental setup.

In particular:

- The optimization problem is nonlinear and solved using a local optimization method.
- The quality of the solution depends on the initial parameter estimates.
- The experimental dataset must contain sufficiently diverse orientations.
- Static intervals must be correctly identified.
- Sensor temperature can affect the measurements.
- The calibration model does not necessarily capture every physical error present in a real IMU.

The calibration method described in the reference paper recommends at least **nine different orientations for observability**, with approximately **36–50 distinct attitudes** recommended for practical calibration.

---

# Theoretical Background

The project draws primarily from concepts in:

- Estimation theory
- Parameter identification
- Nonlinear least-squares optimization
- Sensor modeling
- Quaternion kinematics
- Numerical integration
- Stochastic process characterization
- Allan variance analysis

The main theoretical reference used during development was:

>**Optimal Estimation of Dynamic Systems**
> Jhon L. Crassidis
> Jhon L. Junkins

The calibration methodology is based primarily on:

> D. Tedaldi, A. Pretto, E. Menegatti,
> *A Robust and Easy to Implement Method for IMU Calibration without External Equipments*,
> IEEE International Conference on Robotics and Automation (ICRA), 2014.

---

# Reference

```text
Tedaldi, D., Pretto, A., Menegatti, E.
"A Robust and Easy to Implement Method for IMU Calibration
without External Equipments."
IEEE International Conference on Robotics and Automation (ICRA), 2014.
DOI: 10.1109/ICRA.2014.6907165
```

---

# Motivation

This project was developed to bridge the gap between theoretical estimation methods and real experimental data.

A real sensor provides an opportunity to study the complete chain:

```text
Physical system
      ↓
Experimental measurement
      ↓
Mathematical model
      ↓
Parameter estimation
      ↓
Optimization
      ↓
Model validation
      ↓
Stochastic characterization
```

The same ideas form the basis of more advanced problems in **robotics, navigation, autonomous systems and Guidance, Navigation and Control (GNC)**.

---

### Arduino Libraries Required

- [`I2Cdev`](https://github.com/ElectronicCats/mpu6050/tree/master)
- [`MPU6050`](https://github.com/ElectronicCats/mpu6050)

To install the required libraries, download or clone them from the [ElectronicCats GitHub repository](https://github.com/ElectronicCats/mpu6050) and place them in your Arduino `libraries/` folder.

---

### Acknowledgements

This Arduino sketch is based on the excellent open-source library provided by [**Electronic Cats**](https://github.com/ElectronicCats/mpu6050).
All rights and credits for the original `MPU6050` library belong to its authors.

---

## Contact

If you have questions or want to collaborate, feel free to reach out:
**Tomás Suárez**
Mechatronics Engineer
📧 [suareztomasm@gmail.com](mailto:suareztomasm@gmail.com)
