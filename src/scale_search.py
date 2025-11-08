# src/scale_search.py
import numpy as np
import importlib
from importlib import reload
from src.validate_ai import run_and_measure  # we will modify below to import this function
# If your validate_ai uses different structure, we will call its internals inline.

# We'll import your validate function by reusing run_and_measure in validate_ai.
# If validate_ai is a module, ensure it's importable.
import src.validate_ai as vmod
reload(vmod)

def test_scale(scale):
    # monkey-patch AI compute scaling by wrapping AIController.compute if needed
    # In our sim, AIController is in src.ai_controller — easiest is to set a global scale in that module
    import src.ai_controller as aic
    aic.SCALE = scale  # script expects ai_controller to use aic.SCALE if present

    # run one PD-only and one hybrid run using validate_ai internals
    rms_pd, max_pd, torque_pd = vmod.run_and_measure(use_ai=False)
    rms_ai, max_ai, torque_ai = vmod.run_and_measure(use_ai=True)
    return {
        'scale': scale,
        'rms_pd': rms_pd, 'max_pd': max_pd, 'torque_pd': torque_pd,
        'rms_ai': rms_ai, 'max_ai': max_ai, 'torque_ai': torque_ai
    }

if __name__ == "__main__":
    scales = list(np.concatenate((np.arange(-0.01, -0.06, -0.01), np.arange(-0.02, -0.11, -0.02))))
    # remove duplicates and sort
    scales = sorted(set(scales))
    print("Testing scales:", scales)
    results = []
    for s in scales:
        print(f"Testing scale {s:.4f} ...", flush=True)
        res = test_scale(s)
        results.append(res)
        print(f" scale {s:.4f} => RMS PD {res['rms_pd']:.3f}, RMS AI {res['rms_ai']:.3f}")
    best = min(results, key=lambda r: r['rms_ai'])
    print("\nBEST SCALE:", best['scale'], "RMS_AI:", best['rms_ai'])
    # save summary
    import csv, os
    os.makedirs('results', exist_ok=True)
    with open('results/scale_search.csv','w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in results: w.writerow(r)
    print("Saved results/scale_search.csv")
