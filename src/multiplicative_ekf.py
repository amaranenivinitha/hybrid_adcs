# src/multiplicative_ekf.py
import numpy as np
from src.quaternion_utils import quat_mult, quat_conj, quat_normalize

class MultiplicativeEKF:
    """
    Robust attitude estimator for spacecraft using multiplicative EKF formulation.
    State: quaternion (4x1), gyro bias (3x1)
    """
    def __init__(self, dt):
        self.dt = dt
        self.q = np.array([1., 0., 0., 0.])
        self.b = np.zeros(3)

        self.P = np.eye(6) * 1e-2
        self.Q = np.diag([1e-5]*3 + [1e-7]*3)
        self.R = np.eye(3) * 2e-4

    def predict(self, gyro_meas):
        w = gyro_meas - self.b
        Omega = np.array([
            [0., -w[0], -w[1], -w[2]],
            [w[0], 0., w[2], -w[1]],
            [w[1], -w[2], 0., w[0]],
            [w[2], w[1], -w[0], 0.]
        ])
        self.q = quat_normalize(self.q + 0.5 * self.dt * Omega.dot(self.q))

        F = np.eye(6)
        F[0:3, 0:3] -= self.dt * self.skew(w)
        F[0:3, 3:6] = -self.dt * np.eye(3)

        G = np.zeros((6, 6))
        G[0:3, 0:3] = -np.eye(3)
        G[3:6, 3:6] = np.eye(3)

        self.P = F @ self.P @ F.T + G @ self.Q @ G.T

    def update_vector(self, v_body, v_inertial):
        # predicted measurement from current attitude
        v_pred = self.rotate_vector(self.q, v_inertial)
        H = np.zeros((3, 6))
        H[0:3, 0:3] = -self.skew(v_pred)

        y = v_body - v_pred
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        dx = K @ y
        dq = np.concatenate(([1.0], 0.5 * dx[0:3]))
        dq = quat_normalize(dq)
        self.q = quat_normalize(quat_mult(self.q, dq))
        self.b += dx[3:6]
        I = np.eye(6)
        self.P = (I - K @ H) @ self.P

    def get_state(self):
        return self.q, self.b, self.P

    @staticmethod
    def skew(v):
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    @staticmethod
    def rotate_vector(q, v):
        qv = np.concatenate(([0.0], v))
        return quat_mult(quat_mult(q, qv), quat_conj(q))[1:]
