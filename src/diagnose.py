# src/diagnose.py
import numpy as np
import matplotlib.pyplot as plt
from src.dynamics import step
from src.sensors import measure_gyro, body_vector_from_inertial
from src.multiplicative_ekf import MultiplicativeEKF as SimpleEKF
from src.controllers import pd_controller
from src.quaternion_utils import quat_mult, quat_conj

def run_and_report():
    I = np.diag([0.02, 0.025, 0.03])
    dt = 0.02
    t_final = 40.0
    tvec = np.arange(0, t_final, dt)

    axis = np.array([0.3, -0.4, 0.2]); axis /= np.linalg.norm(axis)
    angle = 0.2
    q0 = np.concatenate(([np.cos(angle/2)], np.sin(angle/2)*axis))
    state = np.concatenate([q0, np.array([0.02, -0.01, 0.015])])

    ekf = SimpleEKF(dt)
    v_inertial = np.array([1.0, 0.1, -0.05]); v_inertial /= np.linalg.norm(v_inertial)
    bias_true = np.array([1e-4, 0.0, -5e-5])
    gyro_sigma = 5e-4; vec_sigma = 1e-3
    q_des = np.array([1.0,0.0,0.0,0.0])

    err_deg = []; tau_log = []; tau_norm = []; est_err = []; P_trace = []

    for t in tvec:
        q_true = state[0:4]; w_true = state[4:7]
        gyro = measure_gyro(w_true, bias_true, gyro_sigma)
        v_body = body_vector_from_inertial(q_true, v_inertial, vec_sigma)

        ekf.predict(gyro)
        ekf.update_vector(v_body, v_inertial)
        q_est, bias_est, P = ekf.get_state()

        # use estimated angular rate (gyro - bias_est)
        w_est = gyro - bias_est
        tau = pd_controller(q_des, q_est, w_est)

        # disturbance
        if 12.0 <= t < 12.4:
            disturbance = np.array([5e-3, -4e-3, 3e-3])
        else:
            disturbance = np.zeros(3)

        state = step(state, tau, disturbance, I, dt)

        err_q = quat_mult(q_des, quat_conj(state[0:4]))
        angle_deg = 2 * np.degrees(np.arccos(np.clip(err_q[0], -1.0, 1.0)))
        err_deg.append(angle_deg)
        tau_log.append(tau.copy())
        tau_norm.append(np.linalg.norm(tau))
        P_trace.append(np.trace(P[0:3,0:3]))
        q_err_est = quat_mult(q_est, quat_conj(state[0:4]))
        est_err.append(2*np.degrees(np.arccos(np.clip(q_err_est[0], -1.0, 1.0))))

    err_deg = np.array(err_deg); tau_log = np.array(tau_log); tau_norm = np.array(tau_norm)
    est_err = np.array(est_err); P_trace = np.array(P_trace)

    print("=== Diagnostic single-run summary ===")
    print("RMS pointing error (deg):", np.sqrt(np.mean(err_deg**2)))
    print("Max pointing error (deg):", np.max(err_deg))
    print("Mean torque norm (Nm):", np.mean(tau_norm))
    print("Max torque magnitude (Nm):", np.max(tau_norm))
    print("Fraction timesteps with torque at limit (per-axis):")
    tau_max = 5e-2  # match controller default
    per_axis_sat = np.mean(np.isclose(tau_log, tau_max, atol=1e-6) | np.isclose(tau_log, -tau_max, atol=1e-6), axis=0)
    print(per_axis_sat)
    print("Estimator RMS error (deg):", np.sqrt(np.mean(est_err**2)))
    print("Estimator mean trace(P_quat_sub):", np.mean(P_trace))

    # save plots
    import os
    os.makedirs('results', exist_ok=True)
    t = tvec
    plt.figure()
    plt.plot(t, err_deg); plt.xlabel('t (s)'); plt.ylabel('pointing error (deg)')
    plt.savefig('results/diag_pointing_error.png', dpi=150)
    plt.figure()
    plt.plot(t, tau_norm); plt.xlabel('t (s)'); plt.ylabel('torque norm (Nm)')
    plt.savefig('results/diag_tau_norm.png', dpi=150)
    plt.figure()
    plt.plot(t, P_trace); plt.yscale('log'); plt.xlabel('t (s)'); plt.ylabel('trace P')
    plt.savefig('results/diag_P_trace.png', dpi=150)
    print("Saved diagnostic plots to results/ (diag_*.png)")

if __name__ == "__main__":
    run_and_report()
