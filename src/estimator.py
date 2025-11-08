# src/estimator.py
import numpy as np
from src.quaternion_utils import omega_to_quat_dot, quat_mult, quat_conj, quat_normalize

class SimpleEKF:
    """
    Simple quaternion-aware EKF-like estimator.
    State tracked outside P: quaternion q_est (4) and bias b_est (3).
    Covariance P corresponds to small-angle (3) and bias (3): P is 6x6.
    """

    def __init__(self, dt):
        self.dt = dt
        # attitude estimate (quaternion) and gyro bias estimate
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.bias = np.zeros(3)

        # Covariance for [delta_theta(3); bias(3)]
        self.P = np.diag(np.concatenate([np.ones(3) * 1e-3, np.ones(3) * 1e-6]))

        # Process and measurement noise (tunable)
        self.Q = np.diag(np.concatenate([np.ones(3)*1e-5, np.ones(3)*1e-7]))
        self.R = np.eye(3) * 1e-3

    def predict(self, gyro_meas):
        """
        Predict step: integrate quaternion using gyro measurement minus estimated bias.
        gyro_meas: measured angular rate (3,)
        """
        omega = gyro_meas - self.bias
        qdot = omega_to_quat_dot(self.q, omega)
        self.q = self.q + qdot * self.dt
        self.q = quat_normalize(self.q)

        # simple covariance propagation (approx)
        self.P = self.P + self.Q * self.dt

    def update_vector(self, v_body_meas, v_inertial):
        """
        Update step using one vector measurement: measured v in body-frame and known inertial vector.
        Uses numeric Jacobian approximation for H (3x6).
        """
        # Predicted body vector from current attitude
        v_pred = quat_mult(quat_mult(self.q, np.concatenate(([0.0], v_inertial))), quat_conj(self.q))[1:]
        y = v_body_meas - v_pred  # innovation (3,)

        # Numeric Jacobian for small-angle part (3x3)
        H = np.zeros((3, 6))
        eps = 1e-7
        for i in range(3):
            dq = np.zeros(4); dq[i+1] = eps
            qp = self.q + dq
            qp = qp / np.linalg.norm(qp)
            v_plus = quat_mult(quat_mult(qp, np.concatenate(([0.0], v_inertial))), quat_conj(qp))[1:]
            H[:, i] = (v_plus - v_pred) / eps

        # bias part has no direct effect on instantaneous vector measurement -> zeros in H[:,3:6]

        S = H.dot(self.P).dot(H.T) + self.R
        K = self.P.dot(H.T).dot(np.linalg.inv(S + 1e-12 * np.eye(3)))  # Kalman gain (6x3)

        dx = K.dot(y)  # (6,)
        dtheta = dx[0:3]
        db = dx[3:6]

        # Apply small-angle correction to quaternion multiplicatively
        angle = np.linalg.norm(dtheta)
        if angle > 1e-12:
            dq = np.concatenate(([np.cos(angle/2.0)], np.sin(angle/2.0) * (dtheta / angle)))
        else:
            dq = np.concatenate(([1.0], dtheta / 2.0))
        self.q = quat_mult(dq, self.q)
        self.q = quat_normalize(self.q)

        # Update bias estimate
        self.bias = self.bias + db

        # Covariance update
        I6 = np.eye(6)
        self.P = (I6 - K.dot(H)).dot(self.P)

    def get_state(self):
        """
        Return tuple: (q_est (4,), bias_est (3,), P (6x6))
        """
        return self.q.copy(), self.bias.copy(), self.P.copy()
