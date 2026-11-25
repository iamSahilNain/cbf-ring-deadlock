"""Nearly symmetric rings, separate filters, gain 1.2: does a tiny perturbation free them?

Report: Section 7.1 (text after Table 7.1). Run: python3 check_perturbed.py > results/check_perturbed.log
"""
import sys, numpy as np
from multiprocessing import Pool
import check_lemmas as cl
def job(a):
    N, eps, seed = a
    cl.KC = 1.2
    s, g = cl.antipodal(N)
    rng = np.random.default_rng(seed)
    traj, _, _ = cl.run(s + rng.uniform(-eps, eps, s.shape), g, False, horizon=60.)
    done = bool(np.max(np.linalg.norm(g-traj[-1], axis=1)) <= cl.TOL)
    return N, eps, seed, done, round((len(traj)-1)*cl.DT, 1), round(float(np.linalg.norm(traj[-1,0])), 3)
if __name__ == '__main__':
    jobs = [(N, e, s) for N in (7, 8) for e in (1e-6, 1e-2) for s in range(3)]
    with Pool(8) as p:
        for r in p.imap_unordered(job, jobs): print(r, flush=True)
