
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

- **File**: `Filtering.pynb`
- **Purpose**:
  1. Matches accelerometer & gyroscope files (watch) by `(SxxAyyTzz)`.
  2. Applies four orientation filters (Complementary, Madgwick, Mahony, EKF) to subtract gravity.
  3. Saves “raw vs. filtered” plots to `visualizations/{subject}/{activity}/`, along with RMSE/correlation logs.

### How to Run

1. **Install** required Python packages: `numpy`, `pandas`, `matplotlib`, `scipy`, `scikit-learn`, `torch`, etc.
2. **Add** your watch data in:
   - `data/smartfallmm/young/accelerometer/watch`
   - `data/smartfallmm/young/gyroscope/watch`
3. **Run**:
   ```bash
   python ExtendedKalmanFilter_FallDetection_YoungWatch_MultiFilters.py
   ```

---

## Mathematical Foundations of the Filters

Each filter aims to reconstruct the watch's orientation using data from accelerometers and gyroscopes, then subtract the gravity vector in sensor coordinates to obtain linear acceleration. This linear acceleration is crucial for highlighting dynamic movements.

### Relevance for Fall Detection

Accurate removal of gravity from inertial data is essential to isolate purely human-induced accelerations. After subtracting gravity, the resulting linear acceleration signals emphasize abrupt changes (e.g., impacts or free-fall phases) which are strong indicators for detecting falls versus normal daily activities.

Once gravity-free signals are obtained, they can be segmented and fed into a classifier (e.g., a Transformer, logistic regression, or other deep neural networks). 

For instance, a simple logistic model might define:

$$
p(\text{fall}) = \frac{1}{1 + \exp(-\mathbf{w}^T \mathbf{x})},
$$

where \(\mathbf{x}\) is the feature vector from a window and \(\mathbf{w}\) are the model parameters. We label an event as a “fall” if \(p(\text{fall}) > 0.5\). Transformers and other architectures can similarly ingest these segmented windows for robust fall detection.

---

## Filter Descriptions

## 1) Complementary Filter (Pitch/Roll Only)

### Accelerometer-Derived Angles

$$
\theta_{\text{acc}} = \arctan2 \bigl(-a_x, \sqrt{a_y^2 + a_z^2}\bigr), 
\quad 
\phi_{\text{acc}} = \arctan2(a_y, a_z).
$$

### Gyro Integration (small-angle approximation)

$$
\dot{\theta} = \omega_y,
\quad 
\dot{\phi} = \pm \,\omega_x.
$$

(The sign for \(\dot{\phi}\) depends on the sensor axis convention.)

### Blending

$$
\theta \leftarrow \alpha \bigl[\theta + \omega_y \,\Delta t\bigr] 
               + (1 - \alpha)\,\theta_{\text{acc}},
$$

and similarly for \(\phi\). Here, \(\alpha \in [0,1]\) is a tunable parameter that balances gyro integration (high-frequency) with accelerometer estimates (low-frequency).

### Gravity Subtraction

Let \(g = 9.81\) m/s\(^2\). The gravity vector in sensor coordinates is computed as:

$$
\begin{aligned}
g_x &= -\sin(\theta)\,g, \\
g_y &= \sin(\phi)\cos(\theta)\,g, \\
g_z &= \cos(\phi)\cos(\theta)\,g.
\end{aligned}
$$

Then the linear acceleration is:

$$
\mathbf{a}_{\text{lin}} 
= \mathbf{a}_{\text{raw}} - \mathbf{g}_{\text{sensor}}.
$$

---

## 2) Madgwick Filter (6‐DoF, No Magnetometer)

### Basic Idea

Madgwick’s filter maintains a quaternion \(\mathbf{q}\) representing the sensor orientation. It uses a gradient-descent approach to align the measured accelerometer vector with the estimated gravity direction in the inertial frame.

### State

$$
\mathbf{q} = \bigl[q_0,\, q_1,\, q_2,\, q_3 \bigr].
$$

### Gyro Predict

$$
\dot{\mathbf{q}}_{\omega} 
= \tfrac{1}{2}\,\mathbf{q} \otimes \boldsymbol{\omega},
$$

where \(\otimes\) denotes quaternion multiplication and \(\boldsymbol{\omega}\) is the gyroscope reading (converted to the proper units).

### Accelerometer Correction

A gradient-descent step drives

$$
f(\mathbf{q}) = 0
$$

so that the measured acceleration \(\mathbf{a}_{\text{norm}}\approx \mathbf{q}\cdot(0,0,1)\).  
(In other words, the sensor’s \((0,0,1)\) axis should align with the true gravity direction.)

### Gravity Subtraction

To remove gravity, rotate \(\mathbf{g} = (0,0,9.81)\) from the global frame into the sensor frame using the orientation \(\mathbf{q}\):

$$
\mathbf{g}_{\text{sensor}} 
= R(\mathbf{q})^T 
  \begin{bmatrix} 0 \\ 0 \\ 9.81 \end{bmatrix}.
$$

Then,

$$
\mathbf{a}_{\text{lin}} 
= \mathbf{a}_{\text{raw}} - \mathbf{g}_{\text{sensor}}.
$$



