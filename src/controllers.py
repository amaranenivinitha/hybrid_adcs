# src/controllers.py
import numpy as np
from src.quaternion_utils import quat_mult, quat_conj

def quaternion_error(q_des, q):
    """
    Compute error quaternion (current relative to desired).
    Returns q_err = q * q_des_conj (rotation needed to go from desired to current).
    """
    q_err = quat_mult(q, quat_conj(q_des))
    if q_err[0] < 0:
        q_err = -q_err
    return q_err

def pd_controller(q_des, q_est, w_est, Kp=3.0, Kd=1.5, tau_max=0.15):
    """
    Simple PD controller for attitude control.
    q_des: desired quaternion
    q_est: estimated quaternion
    w_est: angular velocity estimate
    Kp, Kd: gains
    tau_max: actuator torque limit
    """
    q_err = quaternion_error(q_des, q_est)
    e = q_err[1:4]
    tau = -Kp * e - Kd * w_est
    tau = np.clip(tau, -tau_max, tau_max)
    return tau
