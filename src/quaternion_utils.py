# src/quaternion_utils.py
import numpy as np

def quat_mult(q, r):
    """Multiply two quaternions."""
    w0, x0, y0, z0 = q
    w1, x1, y1, z1 = r
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1
    ])

def quat_conj(q):
    """Return quaternion conjugate."""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_normalize(q):
    """Normalize a quaternion."""
    return q / np.linalg.norm(q)

def omega_to_quat_dot(q, omega):
    """Convert angular velocity to quaternion derivative."""
    wx, wy, wz = omega
    Omega = np.array([
        [0.0, -wx, -wy, -wz],
        [wx,  0.0,  wz, -wy],
        [wy, -wz,  0.0,  wx],
        [wz,  wy, -wx,  0.0]
    ])
    return 0.5 * Omega.dot(q)

def quat_error_vector(q_des, q):
    """Compute the small-angle error between desired and current quaternion."""
    q_des = quat_normalize(q_des)
    q = quat_normalize(q)
    q_err = quat_mult(q_des, quat_conj(q))
    return q_err[1:4]
