"""Refined chiral law: side gate only for neighbours AHEAD.

beta_ij = k * gamma(d) * [ (eh.e)_+  +  (eh x e)_+ * min(1, (eh.e)_+ / delta) ] * tau
Every active nudge has eh.e > 0, so Lemma 1 (no free-space deadlock) applies unchanged.
Theorem 6 claim: contact deadlock phases E_N = empty whenever k*gamma(D)*min(1, sin(pi/N)/delta) > 1.
Checks: contact sweep, Lemma-1 style free-space search, rings N=3..20, perturbed rings, parked, head-on.
Run: python3 check_chiral2.py  (writes results/chiral2.json)
"""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import check_lemmas as cl
from check_chiral import contact_deadlock_phases, ring, done

DELTA = 0.02
orig = cl.reference


def chiral2(p, g, circulation=True):
    err = g - p
    ref = np.array([cl.sat(x) for x in err])
    if not circulation:
        return ref
    for i in range(len(p)):
        gd = np.linalg.norm(err[i])
        if gd < 1e-12:
            continue
        eh = err[i]/gd
        for j in range(len(p)):
            if i == j:
                continue
            e = p[j]-p[i]; d = np.linalg.norm(e); e = e/d
            a = max(0., eh @ e)
            gate = a + max(0., eh[0]*e[1]-eh[1]*e[0])*min(1., a/DELTA)
            ref[i] = ref[i] + cl.KC*max(0., 1-d/2)*gate*min(1., gd/.3)*(cl.J @ e)
    return ref


def free_space_min_ratio(rng, trials=300):
    best = np.inf
    for _ in range(trials):
        k = int(rng.integers(1, 5)); goal = rng.uniform(-3, 3, 2)
        def f(x):
            q = x.reshape(k, 2)
            if np.min(np.linalg.norm(q, axis=1)) < cl.D:
                return 10.
            p = np.vstack(([0., 0.], q)); g = np.vstack((goal, q))
            return np.linalg.norm(cl.reference(p, g)[0])/np.linalg.norm(cl.sat(goal))
        r = minimize(f, rng.uniform(-1.6, 1.6, 2*k), method='Nelder-Mead', options={'maxiter': 3000})
        best = min(best, float(r.fun))
    return best


def main():
    rng = np.random.default_rng(5)
    cl.reference = chiral2
    res = dict(delta=DELTA, sweep=[], free_space=None, rings=[], perturbed=[], parked=[], headon=[])
    for k in (1.2, 1.6, 2.0, 3.0):
        cl.KC = k
        for N in (3, 6, 8, 10, 12, 14, 20):
            s, g = ring(N)
            E = contact_deadlock_phases(N, float(np.linalg.norm(s[0])))
            pred_free = k*(1-cl.D/2)*min(1., np.sin(np.pi/N)/DELTA) > 1
            res['sweep'].append(dict(k=k, N=N, deadlock_phases=len(E), theorem6_predicts_none=bool(pred_free)))
            print('sweep', res['sweep'][-1], flush=True)
    cl.KC = 2.0
    res['free_space'] = free_space_min_ratio(rng)
    print('free-space min ||r||/||n||', res['free_space'], flush=True)
    for N in (3, 6, 10, 14, 20):
        s, g = ring(N)
        for central in (False, True):
            traj, _, _ = cl.run(s, g, central, horizon=90.)
            res['rings'].append(dict(N=N, central=central, completed=done(traj, g), time_s=round((len(traj)-1)*cl.DT, 2)))
            print('ring', res['rings'][-1], flush=True)
    for N in (6, 10, 14):
        s, g = ring(N); ok = 0
        for _ in range(8):
            traj, _, _ = cl.run(s + rng.uniform(-.05, .05, s.shape), g, False, horizon=90.)
            ok += done(traj, g)
        res['perturbed'].append(dict(N=N, decentralized_completed=f'{ok}/8'))
        print('perturbed', res['perturbed'][-1], flush=True)
    for y0 in (-1.0, -0.65, 0.65, 1.0):
        s = np.array([[-3., 0.], [0., y0]]); g = np.array([[3., 0.], [0., y0]])
        traj, _, _ = cl.run(s, g, False, horizon=60.)
        res['parked'].append(dict(y0=y0, completed=done(traj, g), time_s=round((len(traj)-1)*cl.DT, 2)))
        print('parked', res['parked'][-1], flush=True)
    s = np.array([[-2., 0.], [2., 0.]])
    traj, _, _ = cl.run(s, -s, False, horizon=60.)
    res['headon'].append(dict(completed=done(traj, -s), time_s=round((len(traj)-1)*cl.DT, 2)))
    print('headon', res['headon'][-1], flush=True)
    cl.reference = orig
    (Path(__file__).parent/'results'/'chiral2.json').write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
