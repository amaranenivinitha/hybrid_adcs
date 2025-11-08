# src/sim.py
import numpy as np
import matplotlib.pyplot as plt
from src.dynamics import step
from src.sensors import measure_gyro, body_vector_from_inertial
from src.multiplicative_ekf import MultiplicativeEKF as SimpleEKF
from src.controllers import pd_controller
from src.quaternion_utils import quat_mult, quat_conj, quat_normalize
from src.ai_controller import AIController

def run_single():
    I = np.diag([0.02, 0.025, 0.03])
    dt = 0.02
    t_final = 40.0
    tvec = np.arange(0, t_final, dt)

    # --- Initial true state (some rotation and angular velocity)
    axis = np.array([0.3, -0.4, 0.2])
    axis = axis / np.linalg.norm(axis)
    angle = 1.0
    q0 = np.concatenate(([np.cos(angle/2)], np.sin(angle/2)*axis))
    state = np.concatenate([q0, np.array([0.02, -0.01, 0.015])])

    # --- Estimator and constants
    ekf = SimpleEKF(dt)
    v_inertial = np.array([1.0, 0.1, -0.05])
    v_inertial /= np.linalg.norm(v_inertial)
    bias_true = np.array([1e-4, 0.0, -5e-5])
    gyro_sigma = 5e-4
    vec_sigma = 1e-3

    q_des = np.array([1.0, 0.0, 0.0, 0.0])
    err_list, tau_list = [], []

    ai = AIController(model_path="results/ai_model.pt")

    # --- Simulation loop
    for t in tvec:
        true_q = state[0:4]
        true_w = state[4:7]

        # Sensor readings
        gyro = measure_gyro(true_w, bias_true, gyro_sigma)
        v_body = body_vector_from_inertial(true_q, v_inertial, vec_sigma)

        # Estimator update
        ekf.predict(gyro)
        ekf.update_vector(v_body, v_inertial)
        q_est, bias_est, P = ekf.get_state()

        # estimated angular rate from gyro measurement minus estimated bias
        w_est = gyro - bias_est
        # --- Hybrid control (PD + AI correction) ---
        tau_pd = pd_controller(q_des, q_est, w_est)

        # get quaternion error vector (imaginary part of quaternion)
        q_err = quat_mult(q_est, quat_conj(q_des))[1:4]

        # AI compensator adds a small correction torque
        tau_ai = ai.compute(q_err, w_est) * 0.001  # scale factor (adjust later)

        # total torque command
        tau = tau_pd + tau_ai

        # Inject a disturbance impulse between 12s–12.4s
        if 12.0 <= t < 12.4:
            disturbance = np.array([5e-3, -4e-3, 3e-3])
        else:
            disturbance = np.zeros(3)

        # Dynamics step
        state = step(state, tau, disturbance, I, dt)

        # Pointing error (deg)
        err_q = quat_mult(q_des, quat_conj(state[0:4]))
        err_deg = 2 * np.degrees(np.arccos(np.clip(err_q[0], -1.0, 1.0)))
        err_list.append(err_deg)
        tau_list.append(np.linalg.norm(tau))

    # --- Plot results
    plt.figure(figsize=(8, 4))
    plt.plot(tvec, err_list)
    plt.xlabel("Time (s)")
    plt.ylabel("Pointing error (deg)")
    plt.title("Satellite Attitude Control (PD + EKF)")
    plt.grid(True)
    plt.savefig("results/attitude_error_plot.png", dpi=300)

    plt.figure(figsize=(8, 3))
    plt.plot(tvec, tau_list)
    plt.xlabel("Time (s)")
    plt.ylabel("Torque magnitude (Nm)")
    plt.title("Control Effort")
    plt.grid(True)
    plt.savefig("results/torque_plot.png", dpi=300)
    print("✅ Simulation complete — plots saved in 'results' folder.")

if __name__ == "__main__":
    run_single()
