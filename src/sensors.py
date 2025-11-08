# src/sensors.py
import numpy as np
from src.quaternion_utils import quat_mult, quat_conj

def measure_gyro(true_w, bias, sigma):
    """
    Simulate a noisy gyroscope measurement.
    true_w: true angular rate [wx, wy, wz]
    bias: constant gyro bias
    sigma: standard deviation of noise
    """
    noise = np.random.randn(3) * sigma
    return true_w + bias + noise

def body_vector_from_inertial(q, v_inertial, sigma):
    """
    Simulate a body-frame measurement of an inertial vector (e.g. sun vector).
    q: current attitude quaternion
    v_inertial: known inertial vector (e.g., direction to sun)
    sigma: standard deviation of measurement noise
    """
    # Rotate inertial vector into body frame: v_body = q * [0, v] * q_conj
    v_body = quat_mult(quat_mult(q, np.concatenate(([0.0], v_inertial))), quat_conj(q))[1:]
    noise = np.random.randn(3) * sigma
    return v_body + noise
