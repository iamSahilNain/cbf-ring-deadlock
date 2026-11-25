"""Baseline symmetric deadlock is a saddle: stall time should grow like ln(1/eps)/lambda,
lambda = rho/((rho + r*) r*), when the ring's starting phase is perturbed by eps (all robots
rotated by eps, goals fixed). Same model as
check_lemmas.py (disc-saturated nominal, no input bounds, centralized). Circulation shown
for contrast. Run: python3 check_stall.py
"""
import json
from pathlib import Path
import numpy as np
from check_lemmas import antipodal, rot, run, TOL, D

rows = []
for N in (3, 5):
    s, g = antipodal(N)
    rs = D / (2*np.sin(np.pi/N))
    lam = 2. / ((2. + rs) * rs)
    for circulation in (False, True):
        times = []
        for eps in (1e-2, 1e-4, 1e-6, 1e-8):
            traj, _, _ = run(s @ rot(-eps).T, g, True, circulation, horizon=60.)
            done = bool(np.max(np.linalg.norm(g - traj[-1], axis=1)) <= TOL)
            times.append((eps, (len(traj)-1)*0.02, done))
        logs = np.log(1/np.array([t[0] for t in times]))
        slope = float(np.polyfit(logs, [t[1] for t in times], 1)[0])
        rows.append(dict(N=N, circulation=circulation, predicted_slope=None if circulation else 1/lam,
                         fitted_slope=slope, runs=times))
        print(rows[-1], flush=True)
(Path(__file__).parent / 'stall.json').write_text(json.dumps(rows, indent=2))
