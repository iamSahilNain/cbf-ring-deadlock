"""Decentralized ring: exact contact-deadlock set E_N(k) over all phases, threshold k*(N).

E_N(k) = { phi in [-pi, 0] : at r = r*, robot 0's reference r satisfies r_r < 0 and
           |r_t| <= cot(pi/N) * (-r_r) }  (normal cone of the two-contact wedge, N >= 3).
k0(N): closed form at phi = 0.  k*(N) = inf{k : E_N(k) empty}, by bisection on a fine phi grid.
Then simulate the decentralized symmetric swap just above k* and just below k0.
Run: python3 check_threshold.py   (writes results/threshold.json)
"""
import json
from pathlib import Path
import numpy as np
import check_lemmas as cl

RHO = 2.0
PHI = np.linspace(-np.pi, 0., 6001)


def ring(N, r, phi):
    th = 2*np.pi*np.arange(N)/N
    p = r*np.column_stack((np.cos(th+phi), np.sin(th+phi)))
    g = -RHO*np.column_stack((np.cos(th), np.sin(th)))
    return p, g


def deadlock_phases(N, k):
    cl.KC = k
    rs = cl.D/(2*np.sin(np.pi/N))
    cot = np.cos(np.pi/N)/np.sin(np.pi/N)
    out = []
    for phi in PHI:
        p, g = ring(N, rs, phi)
        ref = cl.reference(p, g)[0]
        rhat = p[0]/np.linalg.norm(p[0])
        rr, rt = ref @ rhat, ref @ (cl.J @ rhat)
        if rr < 0 and abs(rt) <= cot*(-rr):
            out.append(phi)
    return out


def k0(N):
    m = np.arange(1, N)
    d = cl.D*np.sin(np.pi*m/N)/np.sin(np.pi/N)
    return (np.cos(np.pi/N)/np.sin(np.pi/N))/np.sum(np.clip(1-d/2, 0, None)*np.sin(np.pi*m/N)**2)


def kstar(N, hi=60.):
    lo = 0.
    assert not deadlock_phases(N, hi)
    for _ in range(30):
        mid = (lo+hi)/2
        lo, hi = (lo, mid) if not deadlock_phases(N, mid) else (mid, hi)
    return hi


def main():
    rows = []
    for N in range(3, 11):
        ks, kz = kstar(N), k0(N)
        # monotonicity spot check: E empty for all sampled k above k*?
        mono = all(not deadlock_phases(N, ks*f) for f in (1.01, 1.2, 1.5, 2., 3.))
        E = deadlock_phases(N, 1.2)
        rows.append(dict(N=N, k0=kz, kstar=ks, monotone_above=mono,
                         E_at_gain_1p2=None if not E else [min(E), max(E)]))
        print(rows[-1], flush=True)
    sims = []
    for N in (5, 7, 8, 9):
        r = next(x for x in rows if x['N'] == N)
        for label, k in (('above_kstar', 1.03*r['kstar']), ('below_k0', 0.97*r['k0'])):
            cl.KC = k
            s, g = cl.antipodal(N)
            traj, _, _ = cl.run(s, g, False, horizon=60.)
            done = bool(np.max(np.linalg.norm(g-traj[-1], axis=1)) <= cl.TOL)
            sims.append(dict(N=N, case=label, gain=k, completed=done, time_s=round((len(traj)-1)*cl.DT, 2),
                             final_phi=float(np.unwrap(np.arctan2(traj[:, 0, 1], traj[:, 0, 0]))[-1]),
                             final_radius=float(np.linalg.norm(traj[-1, 0]))))
            print(sims[-1], flush=True)
    (Path(__file__).parent/'results'/'threshold.json').write_text(json.dumps(dict(thresholds=rows, sims=sims), indent=2))


if __name__ == '__main__':
    main()
