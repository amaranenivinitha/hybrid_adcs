# src/validate.py
import os, sys
import numpy as np
import matplotlib.pyplot as plt

# ensure imports work when running as a module
from src.dynamics import step
from src.sensors import measure_gyro, body_vector_from_inertial
from src.estimator import SimpleEKF
from src.controllers import pd_controller
from src.quaternion_utils import quat_mult, quat_conj, quat_normalize

# ---------- Simulation function (returns time series) ----------
def run_one_trial(params):
    I = params['I']
    dt = params['dt']
    t_final = params['t_final']
    tvec = np.arange(0.0, t_final, dt)

    # initial true state (randomized if requested)
    axis = np.array([0.3, -0.4, 0.2])
    axis = axis / np.linalg.norm(axis)
    angle = params.get('init_angle', 1.0)
    q0 = np.concatenate(([np.cos(angle/2)], np.sin(angle/2) * axis))
    state = np.concatenate([q0, params.get('init_w', np.array([0.02, -0.01, 0.015]))])

    ekf = SimpleEKF(dt)
    v_inertial = np.array([1.0, 0.1, -0.05]); v_inertial /= np.linalg.norm(v_inertial)
    bias_true = params.get('bias_true', np.array([1e-4, 0.0, -5e-5]))
    gyro_sigma = params.get('gyro_sigma', 5e-4)
    vec_sigma = params.get('vec_sigma', 1e-3)
    q_des = np.array([1.0,0.0,0.0,0.0])
    tau_max = params.get('tau_max', 1e-2)

    # logs
    err_deg = []
    tau_cmds = []
    tau_norm = []
    q_true_hist = []
    q_est_hist = []
    P_trace_hist = []
    est_err_deg = []

    for t in tvec:
        q_true = state[0:4]; w_true = state[4:7]

        gyro = measure_gyro(w_true, bias_true, gyro_sigma)
        v_body = body_vector_from_inertial(q_true, v_inertial, vec_sigma)

        ekf.predict(gyro)
        ekf.update_vector(v_body, v_inertial)
        q_est, bias_est, P = ekf.get_state()

        # choose controller (PD for baseline)
        tau = pd_controller(q_des, q_est, w_true, Kp=params['Kp'], Kd=params['Kd'], tau_max=tau_max)

        # disturbance impulse injection
        if params.get('disturbance_time') is not None and params.get('disturbance_duration') is not None:
            if params['disturbance_time'] <= t < params['disturbance_time'] + params['disturbance_duration']:
                disturbance = params.get('disturbance_vector', np.array([5e-3, -4e-3, 3e-3]))
            else:
                disturbance = np.zeros(3)
        else:
            disturbance = np.zeros(3)

        # step dynamics
        state = step(state, tau, disturbance, I, dt)

        # logs
        err_q = quat_mult(q_des, quat_conj(state[0:4]))
        angle = 2 * np.degrees(np.arccos(np.clip(err_q[0], -1.0, 1.0)))
        err_deg.append(angle)
        tau_cmds.append(tau.copy())
        tau_norm.append(np.linalg.norm(tau))
        q_true_hist.append(state[0:4].copy())
        q_est_hist.append(q_est.copy())
        P_trace_hist.append(np.trace(P[0:3,0:3]))
        # estimator attitude error (angle between q_est and q_true)
        q_err_est = quat_mult(q_est, quat_conj(state[0:4]))
        est_angle = 2 * np.degrees(np.arccos(np.clip(q_err_est[0], -1.0, 1.0)))
        est_err_deg.append(est_angle)

    # convert to arrays
    return {
        'tvec': tvec,
        'err_deg': np.array(err_deg),
        'tau_cmds': np.array(tau_cmds),
        'tau_norm': np.array(tau_norm),
        'q_true': np.array(q_true_hist),
        'q_est': np.array(q_est_hist),
        'P_trace': np.array(P_trace_hist),
        'est_err_deg': np.array(est_err_deg)
    }

# ---------- Metrics calculators ----------
def compute_metrics(run_data, params):
    t = run_data['tvec']
    err = run_data['err_deg']
    tau_norm = run_data['tau_norm']
    est_err = run_data['est_err_deg']

    # RMS pointing error (whole run)
    rms = np.sqrt(np.mean(err**2))
    maxerr = np.max(err)
    # average control effort (RMS of torque norm)
    effort = np.sqrt(np.mean(tau_norm**2))

    # actuator saturation events: where any axis equals tau_max (approx)
    tau_cmds = run_data['tau_cmds']
    sat_mask = np.any(np.isclose(tau_cmds, params['tau_max'], atol=1e-8) | np.isclose(tau_cmds, -params['tau_max'], atol=1e-8), axis=1)
    sat_fraction = np.sum(sat_mask) / len(tau_cmds)

    # settling time after disturbance (if disturbance configured)
    settling_time = None
    if params.get('disturbance_time') is not None:
        # Define settling: time after disturbance end when error stays below threshold_deg
        thr = params.get('settle_threshold_deg', 1.0)
        disturb_end = params['disturbance_time'] + params['disturbance_duration']
        # find index after disturbance end
        idx0 = np.searchsorted(t, disturb_end)
        # search for first time after idx0 when error < thr and remains below for the remainder of run
        settling_time = np.nan
        for i in range(idx0, len(t)):
            if np.all(err[i:] < thr):
                settling_time = t[i] - disturb_end
                break

    # estimation RMS (deg)
    est_rms = np.sqrt(np.mean(est_err**2))

    return {
        'rms_deg': rms,
        'max_deg': maxerr,
        'effort': effort,
        'sat_fraction': sat_fraction,
        'settling_time_after_disturb_s': settling_time,
        'estimator_rms_deg': est_rms
    }

# ---------- Monte Carlo runner ----------
def monte_carlo(params, n_runs=30, randomize=True, save_plots=True):
    all_metrics = []
    # collect representative time series for one run (first run)
    rep = None
    for k in range(n_runs):
        # optionally randomize initial conditions & disturbances
        if randomize:
            params_run = params.copy()
            # random initial angle 0.2 to 1.5 rad (approx 11° to 86°)
            params_run['init_angle'] = np.random.uniform(0.2, 1.5)
            params_run['init_w'] = np.random.randn(3) * 0.02
            # small random bias
            params_run['bias_true'] = np.random.randn(3) * 1e-4
        else:
            params_run = params.copy()

        run_data = run_one_trial(params_run)
        metrics = compute_metrics(run_data, params_run)
        all_metrics.append(metrics)
        if rep is None:
            rep = run_data

    # aggregate metrics
    keys = list(all_metrics[0].keys())
    agg = {}
    for k in keys:
        vals = np.array([m[k] if m[k] is not None else np.nan for m in all_metrics])
        agg[k+'_median'] = np.nanmedian(vals)
        agg[k+'_mean'] = np.nanmean(vals)
        agg[k+'_95pct'] = np.nanpercentile(vals, 95)

    # Save CSV summary
    import csv
    os.makedirs('results', exist_ok=True)
    csvfile = os.path.join('results', 'metrics_summary.csv')
    with open(csvfile, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric','median','mean','95pct'])
        for k,v in agg.items():
            w.writerow([k, v, agg[k.replace('_median','_mean')] if '_median' in k else v, agg[k.replace('_median','_95pct')] if '_median' in k else v])
    print("Saved metrics summary to", csvfile)

    # Save representative time-series plots
    if save_plots and rep is not None:
        t = rep['tvec']
        plt.figure(figsize=(8,4))
        plt.plot(t, rep['err_deg'], label='pointing err (deg)')
        plt.axvline(params['disturbance_time'], color='r', linestyle='--', alpha=0.5, label='disturbance start')
        plt.xlabel('Time (s)'); plt.ylabel('Pointing error (deg)')
        plt.legend(); plt.grid(True)
        plt.savefig('results/rep_pointing_error.png', dpi=200)

        plt.figure(figsize=(8,3))
        plt.plot(t, rep['tau_norm'])
        plt.xlabel('Time (s)'); plt.ylabel('Torque norm (Nm)')
        plt.grid(True)
        plt.savefig('results/rep_torque_norm.png', dpi=200)

        plt.figure(figsize=(8,3))
        plt.plot(t, rep['P_trace'])
        plt.xlabel('Time (s)'); plt.ylabel('trace(P_quat_sub)')
        plt.yscale('log'); plt.grid(True)
        plt.savefig('results/rep_P_trace.png', dpi=200)
        print("Saved representative plots to results/")

    return agg, all_metrics

# ---------- Main ----------
if __name__ == "__main__":
    # default params (tune as needed)
    params = {
        'I': np.diag([0.02, 0.025, 0.03]),
        'dt': 0.02,
        't_final': 40.0,
        'Kp': 12.0,
        'Kd': 6.0,
        'tau_max': 1e-2,
        'gyro_sigma': 5e-4,
        'vec_sigma': 1e-3,
        'disturbance_time': 12.0,
        'disturbance_duration': 0.4,
        'disturbance_vector': np.array([5e-3, -4e-3, 3e-3]),
        'settle_threshold_deg': 1.0
    }

    print("Running Monte Carlo validation (30 runs)...")
    agg, all_metrics = monte_carlo(params, n_runs=30, randomize=True, save_plots=True)
    print("Aggregated results (median/mean/95pct):")
    for k,v in agg.items():
        print(k, v)

    # Basic pass/fail checks (your success criteria defaults)
    print("\nPass/fail checks (defaults):")
    checks = {
        'pointing_rms_median_leq_0.5deg': agg['rms_deg_median'] <= 0.5,
        'estimator_rms_median_leq_3deg': agg['estimator_rms_deg_median'] <= 3.0,
        'settling_median_leq_10s': (agg['settling_time_after_disturb_s_median'] is not None and agg['settling_time_after_disturb_s_median'] <= 10.0),
        'saturation_95pct_leq_0.05': agg['sat_fraction_95pct'] <= 0.05
    }
    for k,v in checks.items():
        print(f"{k}: {'PASS' if v else 'FAIL'}")
    print("\nDetailed metrics saved to results/metrics_summary.csv and representative plots in results/")
