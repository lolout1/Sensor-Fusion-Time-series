```markdown
# Sensor-Fusion-Time-series

## Overview

This repository demonstrates an **offline sensor-fusion** pipeline that uses **accelerometer** and **gyroscope** data (from a smartwatch) to remove gravity, producing near-zero-mean linear acceleration signals for **fall detection**. These signals can then be passed into downstream classifiers (e.g., a Transformer) to distinguish falls from daily activities.

> **Note**: The actual **SmartFallMM** CSV files (watch data) are **not** included. You must supply your own CSV files under:
>
> ```
> data/
> └── smartfallmm/
>     └── young/
>         ├── accelerometer
>         │   └── watch
>         └── gyroscope
>             └── watch
> ```
>
> Each file is named `SxxAyyTzz.csv`, where `xx` = subject ID, `yy` = activity ID, and `zz` = trial ID.

---

## Main Script

- **File**: `Filtering.ipynb`
- **Purpose**:
  1. Matches accelerometer & gyroscope files (watch) by `(SxxAyyTzz)`.
  2. Applies four orientation filters (Complementary, Madgwick, Mahony, EKF) to subtract gravity.
  3. Saves “raw vs. filtered” plots to `visualizations/{subject}/{activity}/`, along with RMSE/correlation logs.

### How to Run

1. **Install** required Python packages:  
   `numpy`, `pandas`, `matplotlib`, `scipy`, `scikit-learn`, `torch`, etc.
2. **Add** your watch data in:
   - `data/smartfallmm/young/accelerometer/watch`
   - `data/smartfallmm/young/gyroscope/watch`
3. **Run**:
   ```bash
   python ExtendedKalmanFilter_FallDetection_YoungWatch_MultiFilters.py
   ```

In the **notebook** (and similarly in the script above), we:
- **Load** accelerometer (`ACC`) and gyroscope (`GYRO`) CSV files.
- **Merge** them by matching file IDs (`SxxAyyTzz`).
- **Apply** each filter in a time-stepped manner to estimate orientation and subtract gravity.
- **Plot** the original `ACC` vs. the gravity-free outputs.

---

## Mathematical Foundations of the Filters

All four filters estimate the watch’s **3D orientation** from accelerometer + gyroscope data. Once orientation is known, we compute the **gravity component** in the sensor frame and subtract it from the raw accelerometer readings, producing **linear acceleration**. This helps highlight abrupt changes crucial for **fall detection**.

---

## Filter Descriptions

Below are the four main orientation filters used in this repository, **all ignoring any magnetometer** (i.e., 6-DoF sensor fusion from accelerometer + gyroscope only).

---

### 1) Complementary Filter (Pitch/Roll Only)

#### Basic Idea

The Complementary Filter (as used here) estimates **pitch** and **roll** angles directly, ignoring yaw. It combines low-frequency orientation cues from the **accelerometer** (which senses gravity) with high-frequency cues from the **gyroscope** (which measures rotational velocity).

#### Accelerometer-Derived Angles

We calculate approximate angles:

```
θ_acc = arctan2(-a_x, √(a_y² + a_z²))
φ_acc = arctan2(a_y, a_z)
```
- θ_acc is an estimate of **pitch**  
- φ_acc is an estimate of **roll**

#### Gyro Integration (small-angle approximation)

If the gyroscope outputs angular rates (ω_x, ω_y, ω_z) in rad/s:

```
θ̇ = ω_y
φ̇ = ± ω_x
```
The sign for φ̇ depends on the sensor’s axis definitions.

#### Blending

A complementary filter parameter `α ∈ [0,1]` is used to weight each source:

```
θ ← α [ θ + ω_y * Δt ] + (1 - α) θ_acc
φ ← α [ φ + ω_x * Δt ] + (1 - α) φ_acc
```

Here, `α` near 1 emphasizes gyro integration (good for transient changes), whereas `α` near 0 would rely more on the accelerometer angles (good for drift correction).

#### Gravity Subtraction

Let `g = 9.81 m/s²`. Once pitch (θ) and roll (φ) are estimated:

```
g_x = -sin(θ) * g
g_y = sin(φ) * cos(θ) * g
g_z = cos(φ) * cos(θ) * g
```

Then:

```
a_lin = a_raw - g_sensor
```

In the **notebook**, we implement `complementary_fusion(...)` to read `ACC` + `GYRO` over time, update θ and φ each step, and subtract `g_sensor`.

---

### 2) Madgwick Filter (6‐DoF, No Magnetometer)

#### Basic Idea

Madgwick’s filter represents orientation via a **quaternion** `q = [q₀, q₁, q₂, q₃]`. It has two steps each time-step:

1. **Gyro Predict**: Integrate angular velocity to update the quaternion.
2. **Accelerometer Correction**: Use a gradient-descent step to force the accelerometer reading to align with gravity in the global frame.

This approach effectively fuses the gyroscope and accelerometer to estimate the sensor’s 3D orientation.

#### State

```
q = [q₀, q₁, q₂, q₃]
```

#### Gyro Predict

```
q̇_ω = (1/2) q ⨂ ω
```

where `⨂` is quaternion multiplication and `ω` is the gyroscope reading (in rad/s). The filter typically integrates `q += q̇_ω * Δt`, then normalizes `q`.

#### Accelerometer Correction

Madgwick’s key insight is a **gradient-descent** step that drives:

```
f(q) = 0
```

so that the accelerometer vector (normalized) points along the gravity axis in the **earth** frame. A small correction is computed and blended with the gyro integration to refine `q`.

#### Gravity Subtraction

Assuming gravity in the global frame is (0,0,9.81), we transform it into the **sensor** frame:

```
g_sensor = R(q)ᵀ [ 0, 0, 9.81 ]ᵀ
```

Then:

```
a_lin = a_raw - g_sensor
```

In the **notebook**, we implement `madgwick_fusion(...)`, stepping through each sample’s `ACC` + `GYRO`, updating `q`, and storing `a_lin`.

---

### 3) Mahony Filter (6‐DoF, No Magnetometer)

#### Basic Idea

Mahony’s filter also uses a quaternion `q`, but it introduces a **Proportional–Integral (PI) feedback** approach to handle the orientation error derived from the accelerometer. This can help suppress drift more robustly and handle steady-state errors.

#### Feedback Correction

- Normalize `a_raw` to get `a_norm`.
- Let `v = R(q)ᵀ (0, 0, 1)` be the estimated gravity direction in the sensor frame.
- Compute the orientation error:

  ```
  e = v × a_norm
  ```

- We accumulate an integral term over time if `K_i` > 0:

  ```
  e_int += e * Δt
  ```

- The corrected gyro rate:

  ```
  ω_corr = ω - K_p * e - K_i * e_int
  ```

#### Quaternion Integration

We then integrate `ω_corr` using:

```
q̇ = (1/2) q ⨂ ω_corr
```

Normalize `q` afterward to avoid numerical drift.

#### Gravity Subtraction

```
g_sensor = R(q)ᵀ [ 0, 0, 9.81 ]ᵀ
a_lin    = a_raw - g_sensor
```

In the **notebook**, we call `mahony_fusion(...)` with user-defined `K_p`, `K_i`, read each `(ACC, GYRO)` pair, update `e`, integrate `ω_corr`, and compute `a_lin`.

---

### 4) Extended Kalman Filter (6‐DoF, No Magnetometer)

#### Basic Idea

The EKF maintains a **joint state** of orientation (quaternion) and possibly **gyroscope biases**. Instead of a purely geometric or gradient-based correction, it uses **probabilistic sensor fusion**, combining process models (gyro integration) with measurement updates (accelerometer → gravity).

In this repository’s **notebook** and script, we define an EKF class, feed it each `(GYRO, ACC)` sample in time, predict the new orientation, then update based on how well the expected gravity matches the measured accelerometer.

#### State

```
x = [ q₀, q₁, q₂, q₃, b_ωx, b_ωy, b_ωz ]
```

- `q` is the quaternion representing orientation.
- `b_ω` is an estimate of the gyro bias in each axis.

#### Predict Step

1. **Gyro Bias Correction**:

   ```
   ω_corrected = ω_meas - b_ω
   ```

2. **Quaternion Integration**:

   ```
   q̇ = (1/2) q ⨂ ω_corrected
   q_new = q + q̇ * Δt
   ```

3. **Update Covariance** `P` with process noise (Q).

#### Update Step

We measure `a_raw`. In an ideal scenario, if `q` were exact, then:

```
g_sensor = R(q)ᵀ [ 0, 0, 9.81 ]ᵀ
```

Hence, the innovation is:

```
z = a_raw - g_sensor
```

We compute the **Jacobian** of this measurement wrt the state. The EKF uses that to form the **Kalman gain**, correct both `q` and `b_ω`, and then re-normalize `q`.

#### Orientation Estimation vs. Gravity-Free Acceleration

- **Primary EKF Goal**: Estimate orientation (and bias) from sensor fusion.  
- **Gravity Subtraction**: Once we have the best estimate of `q`, we get `g_sensor` in the sensor frame and compute:

  ```
  a_lin = a_raw - g_sensor
  ```

In the **notebook**, this logic appears as `ekf_fusion(...)`, where for each time step:
- We call `predict(...)` with `ω_meas`.
- Then `update(...)` with `a_raw`.
- Finally, we retrieve the updated `g_sensor` and subtract it from `a_raw`.

---

## Conclusion

All four filters—Complementary, Madgwick, Mahony, and Extended Kalman—**estimate orientation** using different mathematical approaches. They each enable subtracting gravity in the sensor frame to produce **linear acceleration** (near zero mean when the device is at rest). This **gravity-free signal** is essential for detecting high-impact or sudden changes, making it ideal for **fall detection** tasks.

**Happy Sensing!**
```
