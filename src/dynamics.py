# src/dynamics.py
import numpy as np
from src.quaternion_utils import omega_to_quat_dot, quat_normalize

def rigid_deriv(state, torque, disturbance, I):
    """
    Compute the derivative of the satellite state.
    state: [q0,q1,q2,q3, wx,wy,wz]
    torque: control torque from actuators (3x1)
    disturbance: external disturbance torque (3x1)
    I: inertia matrix (3x3)
    """
    q = state[0:4]
    w = state[4:7]

    # Quaternion derivative
    qdot = omega_to_quat_dot(q, w)

    # Angular acceleration
    w_cross = np.cross(w, I.dot(w))
    wdot = np.linalg.inv(I).dot(torque + disturbance - w_cross)

    return np.concatenate([qdot, wdot])

def step(state, torque_cmd, disturbance, I, dt):
    """
    Integrate one time step using RK4.
    """
    k1 = rigid_deriv(state, torque_cmd, disturbance, I)
    k2 = rigid_deriv(state + 0.5*dt*k1, torque_cmd, disturbance, I)
    k3 = rigid_deriv(state + 0.5*dt*k2, torque_cmd, disturbance, I)
    k4 = rigid_deriv(state + dt*k3, torque_cmd, disturbance, I)
    new_state = state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    new_state[0:4] = quat_normalize(new_state[0:4])
    return new_state
