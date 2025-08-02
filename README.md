# 🧭 IMU Auto Calibration Module (MPU-6050)

This project implements an automatic calibration method for Inertial Measurement Units (IMUs), based on the paper:

> **A Robust and Easy to Implement Method for IMU Calibration without External Equipments**  
> Daniele Tedaldi, Andrea Pretto, Emanuele Menegatti – *IEEE ICRA 2014*

The method is designed for low-cost IMUs like the **MPU-6050**, and requires **no external equipment**. It detects static data segments automatically and estimates sensor biases and calibration matrices.

---

## 📌 Summary

- ✅ Fully automatic calibration (no user interaction)
- ✅ Detects static windows in raw IMU data
- ✅ Estimates:
  - Gyroscope biases, missalignment and scaling matrices
  - Accelerometer offsets, missalignment and scaling matrices
- ✅ Plots and tools to validate calibration
- ✅ Written in clean, modular Python

---

## 👨‍💻 Authors

**Tomás Suárez, Agustín Corazza, Rodrigo Pérez**  
Mechatronics Engineering Students 
Universidad Nacional de Cuyo  
📧 suareztomasm@gmail.com
📧 corazzaagustin@gmail.com
📧 rodrigoperez2110@gmail.com

---

## 📁 Project Structure

```text
ImuCalibration/
├── __init__.py
├── arduino code/
│ ├── connection.jpeg # Connection diagram between Arduino UNO and MPU6050 
│ ├── MPU-6000-Register.pdf # MPU6050 registers file
│ ├── MPU-6000.pdf # MPU6050 datasheet 
│ ├── MPU6050_raw.ino # Arduino UNO code for MPU6050 raw values UART communication based on the specified connection
├── ImuCalibrationModules/
│ ├── __init__.py
│ ├── imu_calibration.py # Main calibration logic
│ └── utils.py # Helpers and data loaders
├── calibration data/
│ └── example_data_calibration.csv # Example raw IMU data
│ └── example_data_tinit_calc.csv # Example static IMU raw data for initial static time calculation
├── optimization result images/
│ └── cal_accel.png # Calibrated acceleration data
│ └── cal_ang_vel.png # Calibrated angular velocity data
│ └── non_cal_accel.png # Non calibrated acceleration data
│ └── non_cal_ang_vel.png # Non alibrated angular velocity data
│ └── static_detector_test.png # Static interval detector test
├── optimization result data/
│ └── calibrated_accel_avg_data.csv # Example calibrated static acceleration data for gyroscope calibration
│ └── params_acc.csv # Optimization result parameters for accelerometer calibration
│ └── params_gyro.csv # Optimization result parameters for gyroscope calibration
│ └── static_intervals.csv # Static intervals start and end indices for gyroscope calibration
├── results test data/
├── optimization result data/
│ └── imu_90_x_rot.csv # Raw imu data from +90 degrees X rotation for optmization result validation
│ └── imu_static_test.csv # Raw imu data from static test for optimization result validation
├── CalibrationTests/
│ └── __init__.py
│ └── test_data_logging.py # Imu raw accelerometer and gyroscope data logging in a csv file
│ └── test_tinit_calculation.py # Initial static time calculation test example
│ └── test_static_detector.py # Static intervals detector test example
│ └── test_accel_calibration.py # Accelerometer calibration test example
│ └── test_gyro_calibration.py # Gyroscope calibration test example
│ └── test_calibration_result.py # Imu optimization results example test
│ └── test_complete_imu_calibration.py # Complete imu (accelerometer and gyroscope) calibration test example
├── README.md # This file
├── LICENSE # MIT License
└── requirements.txt # Python dependencies
```
--

## 🚀 Quick Start

### 1. 📥 Clone the Repository

```bash
git clone https://github.com/tomisuarez2/ImuCalibration
cd ImuCalibration
```

--

### 2. 📦 Install Requirements

```bash
pip install -r requirements.txt
```

--

### 3. ▶️ Run example tests (e.g. complete calibration test)

```bash
cd ImuCalibration
python -m CalibrationTests.test_complete_imu_calibration
```
--

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
![Non calibrated accelerometer data](optimization%20result%20images/non_cal_accel.png)

![Calibrated accelerometer data](optimization%20result%20images/cal_accel.png)

![Non calibrated gyroscope data](optimization%20result%20images/non_cal_ang_vel.png)

![Calibrated gyroscope data](optimization%20result%20images/cal_ang_vel.png)

![Static interval detector](optimization%20result%20images/static_detector_test.png)

--

## 📈 Input Data Format

The module expects a CSV file with the following columns:

```bash
ax, ay, az, gx, gy, gz
```

ax, ay, az: Accelerometer data (raw values)
gx, gy, gz: Gyroscope data (raw values)

Additional information as sampling frecuency, waiting time, number or moves and so on is also required.

--

## 📟 Arduino sketch for MPU6050 UART communication

To collect raw IMU data for calibration, this repository includes an Arduino sketch that interfaces with the **MPU-6050** sensor via I2C. The code configures the device to operate at a user-defined sampling frequency (default: 100 Hz), uses data-ready interrupts, and streams raw accelerometer and gyroscope readings over UART.

### ⚙️ Features

- Configurable sampling frequency via `SMPL_RT_FREQ` (default: 100 Hz)
- I2C communication using `Wire` interface
- Interrupt-based data acquisition (using pin D2)
- Outputs raw data in **readable CSV format** or **compact binary format**
- Blinks onboard LED as activity indicator

### 🧾 Output Format

By default, the output is a comma-separated string:

```bash
ax, ay, az, gx, gy, gz
```

ax, ay, az: Accelerometer data (raw values)
gx, gy, gz: Gyroscope data (raw values)

### 📦 Arduino Libraries Required

- [`I2Cdev`](https://github.com/ElectronicCats/mpu6050/tree/master)
- [`MPU6050`](https://github.com/ElectronicCats/mpu6050)

To install the required libraries, download or clone them from the [ElectronicCats GitHub repository](https://github.com/ElectronicCats/mpu6050) and place them in your Arduino `libraries/` folder.

### 👏 Acknowledgements

This Arduino sketch is based on the excellent open-source library provided by [**Electronic Cats**](https://github.com/ElectronicCats/mpu6050).  
All rights and credits for the original `MPU6050` library belong to its authors.

---

## 🧪 Validation

The module includes tools to validate calibration by:

Plotting acceleration norm (should converge to 9.81 m/s² in rest)

Comparing gyroscope biases before/after

Visualizing the static windows detected

Showing n-point cloud of raw vs. calibrated accelerometer data

All of this is included in the test examples.

--

## ⚠️ Limitations

Remember that **Levenberg-Marquardt** algorithm is a **local optimization** method, it converges to a local minimum, not necessarily the global one.

This calibration algorithm requires sufficiently rich and diverse motion to ensure parameter observability.

In particular:

- The IMU must be rotated between all static intervals, with significant angular displacement around all three axes (X, Y, and Z).

- Avoid repeating the same type of rotation (e.g., yaw-only). Instead, include:

- Tilts (pitch/roll),

- Twists (yaw),

- Compound 3D rotations (diagonal movements or figure-eight patterns).

At least 9 static orientations are required. Authors recommend 36–50 poses, each preceded by short, clean motions (1–4 seconds).

Without sufficient motion excitation:

- The gyroscope scale factors and misalignment parameters may become weakly observable.

- This can lead to incorrect calibration results, such as negative scale factors or parameter ambiguity.

- The optimizer may converge, but to a mathematically plausible yet physically incorrect solution.

For best results:

- Record data in a stable thermal environment.

. Ensure the sensor is completely still during static intervals.

- Visually inspect the gyroscope data to confirm motion diversity.

--

## 📚 Citation

If you use this module or code, please cite the original paper:

Tedaldi, D., Pretto, A., & Menegatti, E. (2014).
A Robust and Easy to Implement Method for IMU Calibration without External Equipments.
In Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), 3042–3049.
DOI: 10.1109/ICRA.2014.6907165

--

## 🤝 Contributing

Feel free to fork the repo and submit a pull request!
This is my first repo so any suggestions or improvements are welcome.

--

## 🛰️ Contact

If you have questions or want to collaborate, feel free to reach out:
📧 suareztomasm@gmail.com






