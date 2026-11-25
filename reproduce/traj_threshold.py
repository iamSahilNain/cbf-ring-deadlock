"""Gain at which a run from the exact ring first gets stuck (separate filters), by bisection.

Report: second row of Table 7.2. Run: python3 traj_threshold.py  (writes results/traj_threshold.json)
"""
import json
from pathlib import Path
import numpy as np
import check_lemmas as cl

out = {}
for N, lo, hi in ((7, 1.2, 1.394), (8, 1.2, 1.751)):
    s, g = cl.antipodal(N)

    def ok(k):
        cl.KC = k
        traj, _, _ = cl.run(s, g, False, horizon=60.)
        return bool(np.max(np.linalg.norm(g-traj[-1], axis=1)) <= cl.TOL)

    assert not ok(lo) and ok(hi)
    for _ in range(7):
        mid = (lo+hi)/2
        lo, hi = (mid, hi) if not ok(mid) else (lo, mid)
    out[N] = dict(trajectory_threshold_between=[lo, hi])
    print(N, out[N], flush=True)
(Path(__file__).parent/'results'/'traj_threshold.json').write_text(json.dumps(out, indent=2))
