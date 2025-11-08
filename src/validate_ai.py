# src/validate_ai.py
import numpy as np
from src.sim import run_single  # assumes your sim has run_single()
from src.ai_controller import AIController
from src.controllers import pd_controller
from src.quaternion_utils import quat_mult, quat_conj

def run_and_measure(use_ai=False):
    from src.multiplicative_ekf import MultiplicativeEKF
    from src.dynamics import step
    from src.sensors import measure_gyro, body_vector_from_inertial

    I = np.diag([0.02, 0.025, 0.03])
    dt = 0.02
    t_final = 40.0
    tvec = np.arange(0, t_final, dt)
    q_des = np.array([1., 0., 0., 0.])

    axis = np.array([0.3, -0.4, 0.2]); axis /= np.linalg.norm(axis)
    angle = 0.2
    q0 = np.concatenate(([np.cos(angle/2)], np.sin(angle/2)*axis))
    w0 = np.array([0.02, -0.01, 0.015])
    state = np.concatenate([q0, w0])
    ekf = MultiplicativeEKF(dt)
    ai = AIController(model_path="results/ai_model.pt") if use_ai else None
    v_inertial = np.array([1., 0.1, -0.05]); v_inertial /= np.linalg.norm(v_inertial)
    bias_true = np.array([1e-4, 0., -5e-5])

    err_log, tau_log = [], []
    for t in tvec:
        q_true, w_true = state[:4], state[4:]
        gyro = measure_gyro(w_true, bias_true, 5e-4)
        v_body = body_vector_from_inertial(q_true, v_inertial, 1e-3)
        ekf.predict(gyro)
        ekf.update_vector(v_body, v_inertial)
        q_est, b_est, _ = ekf.get_state()
        w_est = gyro - b_est
        q_err = quat_mult(q_est, quat_conj(q_des))[1:4]
        tau_pd = pd_controller(q_des, q_est, w_est)
        tau_ai = ai.compute(q_err, w_est) if use_ai else np.zeros(3)
        tau = tau_pd + tau_ai
        if 12.0 <= t < 12.4:
            disturbance = np.array([5e-3, -4e-3, 3e-3])
        else:
            disturbance = np.zeros(3)
        state = step(state, tau, disturbance, I, dt)
        q_err_full = quat_mult(q_des, quat_conj(state[:4]))
        ang_err = 2*np.degrees(np.arccos(np.clip(q_err_full[0], -1, 1)))
        err_log.append(ang_err)
        tau_log.append(np.linalg.norm(tau))

    err_log, tau_log = np.array(err_log), np.array(tau_log)
    rms_err = np.sqrt(np.mean(err_log**2))
    max_err = np.max(err_log)
    avg_torque = np.mean(tau_log)
    return rms_err, max_err, avg_torque

if __name__ == "__main__":
    rms_pd, max_pd, torque_pd = run_and_measure(use_ai=False)
    rms_ai, max_ai, torque_ai = run_and_measure(use_ai=True)
    print("\n=== FINAL PERFORMANCE COMPARISON ===")
    print(f"PD-only RMS error: {rms_pd:.3f}°, max: {max_pd:.2f}°, avg torque: {torque_pd:.5f}")
    print(f"Hybrid AI RMS error: {rms_ai:.3f}°, max: {max_ai:.2f}°, avg torque: {torque_ai:.5f}")
    print(f"Improvement: {(1 - rms_ai/rms_pd)*100:.1f}%")
