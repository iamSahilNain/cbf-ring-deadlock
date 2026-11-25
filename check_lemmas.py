"""Numerical sanity checks for the multi-robot circulation results in the report.

Theory model (differs from the September sims on purpose):
  - nominal n_i = sat_disc(g_i - p_i), unit disc saturation (rotation-equivariant;
    the sims used component clipping, which breaks C_N symmetry),
  - circulation c_i exactly as in multi_robot.py,
  - CBF-QP with NO input bounds (centralized: joint; decentralized: halved split).
Checks:
  A  centralized QP preserves total reference force and torque (Lemma 2)
  B  no free-space equilibria: min ||n_i + c_i|| / ||n_i|| over sampled neighbour sets (Lemma 1)
  C  symmetric antipodal swap, N=2..8: completion, phi monotone, symmetry, tangential preserved
  D  baseline on the same family: phi stays 0, converges to the N-gon deadlock
  E  robustness: random 0.05 m perturbations of the symmetric starts
  F  decentralized wedge margin along the symmetric circulation trajectory
Run: python3 check_lemmas.py   (writes results.json next to this file)
"""
from itertools import combinations
from pathlib import Path
import json
import numpy as np
from scipy.optimize import minimize, LinearConstraint

D, ALPHA, DT, TOL = 0.6, 2.0, 0.02, 0.05
KC = 1.2  # circulation gain; check_threshold.py varies it
J = np.array([[0., -1.], [1., 0.]])
OUT = Path(__file__).parent


def sat(v, vmax=1.0):
    n = np.linalg.norm(v)
    return v if n <= vmax else v * vmax / n


def reference(p, g, circulation=True):
    err = g - p
    ref = np.array([sat(e) for e in err])
    if not circulation:
        return ref
    for i in range(len(p)):
        gd = np.linalg.norm(err[i])
        for j in range(len(p)):
            if i == j:
                continue
            e = p[j] - p[i]
            d = np.linalg.norm(e)
            e = e / d
            align = max(0., err[i] @ e / max(gd, 1e-12))
            beta = KC * max(0., 1 - d / 2) * align * min(1., gd / .3)
            ref[i] = ref[i] + beta * (J @ e)
    return ref


def constraints(p):
    pairs = list(combinations(range(len(p)), 2))
    A = np.zeros((len(pairs), 2 * len(p)))
    b = np.empty(len(pairs))
    for r, (i, j) in enumerate(pairs):
        rel = p[i] - p[j]
        A[r, 2*i:2*i+2], A[r, 2*j:2*j+2] = 2*rel, -2*rel
        b[r] = -ALPHA * (rel @ rel - D**2)
    return pairs, A, b


def project(ref, A, b):
    if np.min(A @ ref - b) >= 0:
        return ref.copy()
    res = minimize(lambda u: .5*np.sum((u-ref)**2), np.zeros_like(ref), jac=lambda u: u-ref,
                   method='SLSQP', constraints=[LinearConstraint(A, b, np.inf)],
                   options={'ftol': 1e-13, 'maxiter': 300})
    if not res.success or np.min(A @ res.x - b) < -1e-7:
        raise RuntimeError(res.message)
    return res.x


def control(p, g, central, circulation=True):
    pairs, A, b = constraints(p)
    ref = reference(p, g, circulation)
    if central:
        return project(ref.ravel(), A, b).reshape(-1, 2), ref
    u = np.empty_like(p)
    for i in range(len(p)):
        rows = [r for r, pr in enumerate(pairs) if i in pr]
        u[i] = project(ref[i], A[rows, 2*i:2*i+2], b[rows] / 2)
    return u, ref


def run(p0, g, central, circulation=True, horizon=40.):
    p = np.array(p0, float)
    traj, refs, us = [p.copy()], [], []
    for _ in range(int(horizon / DT)):
        if np.max(np.linalg.norm(g - p, axis=1)) <= TOL:
            break
        u, ref = control(p, g, central, circulation)
        p = p + DT * u
        traj.append(p.copy()); refs.append(ref); us.append(u)
    return np.array(traj), np.array(refs), np.array(us)


def antipodal(N, R=2.):
    th = 2*np.pi*np.arange(N)/N
    s = R*np.column_stack((np.cos(th), np.sin(th)))
    return s, -s


def rot(t):
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def check_A(rng, trials=300):
    worst_f = worst_t = 0.
    for _ in range(trials):
        N = rng.integers(2, 7)
        p = rng.uniform(-1.2, 1.2, (N, 2))
        if min(np.linalg.norm(p[i]-p[j]) for i, j in combinations(range(N), 2)) < D:
            continue
        g = rng.uniform(-2, 2, (N, 2))
        u, ref = control(p, g, True)
        w = u - ref
        worst_f = max(worst_f, float(np.linalg.norm(w.sum(0))))
        worst_t = max(worst_t, float(abs(np.sum(p[:, 0]*w[:, 1] - p[:, 1]*w[:, 0]))))
    return dict(max_net_force=worst_f, max_net_torque=worst_t)


def check_B(rng, trials=400):
    """Minimise ||n+c||/||n|| over neighbour sets with every neighbour farther than D."""
    def ratio(x, k):
        q = x.reshape(k, 2)
        if np.min(np.linalg.norm(q, axis=1)) < D:          # no contact allowed
            return 10.
        p = np.vstack(([0., 0.], q))
        g = np.vstack((x_goal, q))                       # neighbours sit at their goals
        ref = reference(p, g)
        return np.linalg.norm(ref[0]) / np.linalg.norm(sat(x_goal))
    best = np.inf
    for _ in range(trials):
        k = rng.integers(1, 5)
        x_goal = rng.uniform(-3, 3, 2)
        x0 = rng.uniform(-1.6, 1.6, 2*k)
        r = minimize(ratio, x0, args=(k,), method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-9})
        best = min(best, float(r.fun))
    return dict(min_ratio_ref_over_nominal=best)


def symmetric_report(N, central, circulation=True):
    s, g = antipodal(N)
    traj, refs, us = run(s, g, central, circulation)
    phi = np.unwrap(np.arctan2(traj[:, 0, 1], traj[:, 0, 0]))
    dphi = np.diff(phi)
    asym = max(float(np.max(np.linalg.norm(traj[:, k] - traj[:, 0] @ rot(2*np.pi*k/N).T, axis=1)))
               for k in range(1, N))
    tang_err = 0.
    if central and len(us):
        t = np.array([J @ (x/np.linalg.norm(x)) for x in traj[:-1, 0]])
        tang_err = float(np.max(np.abs(np.sum((us[:, 0]-refs[:, 0])*t, axis=1))))
    sep = min(float(np.min(np.linalg.norm(traj[:, i]-traj[:, j], axis=1))) for i, j in combinations(range(N), 2))
    done = bool(np.max(np.linalg.norm(g - traj[-1], axis=1)) <= TOL)
    return dict(N=N, central=central, circulation=circulation, completed=done,
                time_s=round((len(traj)-1)*DT, 2), final_phi=float(phi[-1]),
                max_positive_dphi=float(dphi.max(initial=0.)), max_asymmetry_m=asym,
                max_tangential_change=tang_err, min_separation=sep,
                final_radius=float(np.linalg.norm(traj[-1, 0])),
                deadlock_radius=D/(2*np.sin(np.pi/N)))


def check_E(rng, Ns=(3, 4, 5, 6), trials=15, eps=.05):
    out = []
    for N in Ns:
        s, g = antipodal(N)
        for central in (True, False):
            ok = 0
            for _ in range(trials):
                traj, _, _ = run(s + rng.uniform(-eps, eps, s.shape), g, central)
                ok += bool(np.max(np.linalg.norm(g - traj[-1], axis=1)) <= TOL)
            out.append(dict(N=N, central=central, completed=f'{ok}/{trials}'))
    return out


def check_F(N):
    """Decentralized symmetric deadlock iff ref lies in the normal cone of the contact wedge:
    |ref_t| <= cot(pi/N) * max(0, -ref_r). Report the smallest |ref_t| / (cot * inward) at contact."""
    s, g = antipodal(N)
    traj, refs, _ = run(s, g, False)
    worst = np.inf
    for p, ref in zip(traj[:-1], refs):
        if np.min(np.linalg.norm(p[1:] - p[0], axis=1)) > D + 1e-3:
            continue
        rhat = p[0] / np.linalg.norm(p[0])
        inward = max(0., -ref[0] @ rhat)
        if inward > 0:
            worst = min(worst, abs(ref[0] @ (J @ rhat)) / (inward / np.tan(np.pi/N)))
    return dict(N=N, min_wedge_margin=None if worst == np.inf else float(worst))


def main():
    rng = np.random.default_rng(7)
    res = dict(A=check_A(rng), B=check_B(rng))
    print('A', res['A'], '\nB', res['B'], flush=True)
    res['C'] = [symmetric_report(N, c) for N in range(2, 9) for c in (True, False)]
    for r in res['C']:
        print('C', r, flush=True)
    res['D'] = [symmetric_report(N, True, False) for N in (3, 5, 8)]
    for r in res['D']:
        print('D', r, flush=True)
    res['F'] = [check_F(N) for N in range(2, 9)]
    print('F', res['F'], flush=True)
    res['E'] = check_E(rng)
    print('E', res['E'], flush=True)
    (OUT / 'results.json').write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
