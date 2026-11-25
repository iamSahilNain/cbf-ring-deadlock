"""Clutter stress test: does the chiral law (Theorem 6) cause stalls outside the ring?

Decentralized CBF-QP only (the architecture Theorem 6 is about). Movers have goals; obstacles
are static points of the same radius (movers take full responsibility for them). Circulation
nudges come from movers and obstacles alike.

Laws: none (baseline), orig@1.2, orig@2.0, chiral@2.0 (delta = 0.02).
Scenarios (20 seeds each):
  A  random goal permutation in a crowded box, N in {6, 10, 16}
  B  6 movers crossing a field of 10 static obstacles
  C  corridor between two obstacle walls, 2 and 4 movers swapping
  D  jittered antipodal ring, N in {8, 12}
Metrics: completion, time, stall time (speed < 0.02 m/s while > 0.1 m from goal, summed
over movers), min separation (sampled plus exact segment check), max speed.
Run: python3 check_clutter.py   (writes results/clutter.json; uses all cores)
"""
import json, os, sys, time
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path
import numpy as np

D, ALPHA, DT, TOL, RC, ELL, DELTA = 0.6, 2.0, 0.02, 0.05, 2.0, 0.3, 0.02
EPS_SIDE = 0.15   # range of the contact-only side nudge (law 'chiralc')
J = np.array([[0., -1.], [1., 0.]])


def project2d(r, A, b):
    """Exact Euclidean projection of r onto {u in R^2 : A u >= b}."""
    if A.shape[0] == 0 or np.all(A @ r - b >= -1e-12):
        return r
    best, bestd = None, np.inf
    def ok(u):
        return np.all(A @ u - b >= -1e-9)
    nn = np.einsum('ij,ij->i', A, A)
    for k in range(len(b)):
        u = r + (b[k] - A[k] @ r)/nn[k]*A[k]
        if ok(u):
            d = np.sum((u-r)**2)
            if d < bestd:
                best, bestd = u, d
    for k, l in combinations(range(len(b)), 2):
        M = A[[k, l]]
        det = M[0, 0]*M[1, 1]-M[0, 1]*M[1, 0]
        if abs(det) < 1e-14:
            continue
        u = np.linalg.solve(M, b[[k, l]])
        if ok(u):
            d = np.sum((u-r)**2)
            if d < bestd:
                best, bestd = u, d
    if best is None:
        raise RuntimeError('infeasible projection')
    return best


def wishes(p, g, obs, law, k):
    err = g - p
    nrm = np.linalg.norm(err, axis=1)
    ref = err/np.maximum(1., nrm)[:, None]
    if law == 'none':
        return ref
    allpts = np.vstack((p, obs)) if len(obs) else p
    for i in range(len(p)):
        if nrm[i] < 1e-12:
            continue
        eh = err[i]/nrm[i]
        v = allpts - p[i]
        d = np.linalg.norm(v, axis=1)
        mask = (d < RC) & (d > 1e-9)
        if not mask.any():
            continue
        e = v[mask]/d[mask][:, None]
        a = np.clip(e @ eh, 0, None)
        g_far = 1-d[mask]/RC
        side = np.clip(eh[0]*e[:, 1]-eh[1]*e[:, 0], 0, None)*np.minimum(1., a/DELTA)
        if law == 'orig':
            w = g_far*a
        elif law == 'chiral':                      # side nudge with the full 2 m range
            w = g_far*(a + side)
        else:                                      # 'chiralc': side nudge only near contact
            w = g_far*a + np.clip(1-(d[mask]-D)/EPS_SIDE, 0, 1)*side
        beta = k*w*min(1., nrm[i]/ELL)
        ref[i] = ref[i] + (beta[:, None]*(e @ J.T)).sum(0)
    return ref


def control(p, g, obs, law, k):
    ref = wishes(p, g, obs, law, k)
    u = np.empty_like(p)
    for i in range(len(p)):
        rows, rhs = [], []
        # rhs capped at 0: rounding can leave h at -1e-12, and a caged robot must still be allowed u = 0
        for j in range(len(p)):
            if j != i:
                rel = p[i]-p[j]; rows.append(2*rel); rhs.append(min(0., -ALPHA*(rel@rel-D**2)/2))
        for o in obs:
            rel = p[i]-o; rows.append(2*rel); rhs.append(min(0., -ALPHA*(rel@rel-D**2)))
        u[i] = project2d(ref[i], np.array(rows).reshape(-1, 2), np.array(rhs))
    return u


def simulate(job):
    try:
        return _simulate(job)
    except Exception as exc:          # record, don't kill the batch
        name, seed, law, k = job[:4]
        return dict(scenario=name, seed=seed, law=law, k=k, completed=False, time_s=None, stall_s=None,
                    movers_at_goal=None, movers=len(job[4]), min_sep=None, max_speed=None, error=repr(exc))


def _simulate(job):
    name, seed, law, k, starts, goals, obs, horizon = job
    p = np.array(starts, float); g = np.array(goals, float); obs = np.array(obs, float).reshape(-1, 2)
    minsep, stall, vmax = np.inf, 0., 0.
    steps = int(horizon/DT)
    for t in range(steps):
        dist = np.linalg.norm(g-p, axis=1)
        if dist.max() <= TOL:
            break
        u = control(p, g, obs, law, k)
        sp = np.linalg.norm(u, axis=1); vmax = max(vmax, float(sp.max()))
        stall += DT*float(np.sum((sp < .02) & (dist > .1)))
        q = p + DT*u
        pts0 = np.vstack((p, obs)); pts1 = np.vstack((q, obs))
        for a_, b_ in combinations(range(len(pts0)), 2):
            if a_ >= len(p) and b_ >= len(p):
                continue
            rel = pts0[a_]-pts0[b_]; ch = (pts1[a_]-pts1[b_]) - rel
            f = np.clip(-rel@ch/max(ch@ch, 1e-30), 0, 1)
            minsep = min(minsep, float(np.linalg.norm(rel+f*ch)))
        p = q
    dist = np.linalg.norm(g-p, axis=1)
    return dict(scenario=name, seed=seed, law=law, k=k, completed=bool(dist.max() <= TOL),
                time_s=round((t+1)*DT if dist.max() <= TOL else horizon, 2), stall_s=round(stall, 2),
                movers_at_goal=int((dist <= TOL).sum()), movers=len(p), min_sep=minsep, max_speed=vmax)


def sample_points(rng, n, lo, hi, sep, avoid=()):
    pts = []
    tries = 0
    while len(pts) < n:
        tries += 1
        if tries > 200000:
            raise RuntimeError('sampling failed')
        x = rng.uniform(lo, hi, 2)
        if all(np.linalg.norm(x-q) >= sep for q in list(pts)+list(avoid)):
            pts.append(x)
    return np.array(pts)


def scenarios():
    jobs_geo = []
    for seed in range(20):
        rng = np.random.default_rng(1000+seed)
        for N in (6, 10, 16):
            L = 1.05*np.sqrt(N)
            s = sample_points(rng, N, -L, L, .8); g = sample_points(rng, N, -L, L, .8)
            jobs_geo.append((f'A_random_N{N}', seed, s, g, np.zeros((0, 2))))
        obs = sample_points(rng, 10, -2., 2., .9)
        def column(x0, n):           # n points on x = x0, |y| <= 2, spaced >= 0.8
            ys = sample_points(rng, n, -2., 2., .8)[:, 1] if False else np.sort(rng.choice(np.linspace(-2, 2, 6), n, replace=False))
            return np.column_stack((np.full(n, x0), ys + rng.uniform(-.05, .05, n)))
        s = np.vstack((column(-4., 3), column(4., 3))); g = np.vstack((column(4.5, 3), column(-4.5, 3)))
        jobs_geo.append(('B_obstacle_field', seed, s, g, obs))
        wx = np.arange(-2.4, 2.41, .62)
        walls = np.vstack([np.column_stack((wx, np.full_like(wx, y))) for y in (-1.25, 1.25)])
        jit = rng.uniform(-.15, .15, 4)
        s2 = np.array([[-4., jit[0]], [4., jit[1]]]); g2 = np.array([[4., jit[2]], [-4., jit[3]]])
        jobs_geo.append(('C_corridor_2', seed, s2, g2, walls))
        s4 = np.array([[-4., .3+jit[0]], [-4.7, -.3+jit[1]], [4., -.3+jit[2]], [4.7, .3+jit[3]]])
        g4 = np.array([[4.2, .3], [4.9, -.3], [-4.2, -.3], [-4.9, .3]])
        jobs_geo.append(('C_corridor_4', seed, s4, g4, walls))
        if seed == 0:                               # exact symmetric rings: the Theorem 6 setting
            for N in (8, 12, 16, 20):
                rho = max(2., 1/np.sin(np.pi/N)); th = 2*np.pi*np.arange(N)/N
                s = rho*np.column_stack((np.cos(th), np.sin(th)))
                jobs_geo.append((f'E_exact_ring_N{N}', 0, s, -s, np.zeros((0, 2))))
        for N in (8, 12):
            rho = max(2., 1/np.sin(np.pi/N))
            th = 2*np.pi*np.arange(N)/N + rng.uniform(-.17, .17, N)
            rr = rho + rng.uniform(-.3, .3, N)
            s = np.column_stack((rr*np.cos(th), rr*np.sin(th)))
            if min(np.linalg.norm(a-b) for a, b in combinations(s, 2)) < D:
                continue
            jobs_geo.append((f'D_jittered_ring_N{N}', seed, s, -s, np.zeros((0, 2))))
    return jobs_geo


LAWS = [('none', 0.), ('orig', 1.2), ('orig', 2.0), ('chiral', 2.0)]


def main():
    global LAWS
    if len(sys.argv) > 1 and sys.argv[1] == 'contact':
        LAWS = [('chiralc', 1.5), ('chiralc', 2.0), ('orig', 2.0)]
    geo = scenarios()
    jobs = [(name, seed, law, k, s.tolist(), g.tolist(), np.asarray(o).tolist(), 90.)
            for (name, seed, s, g, o) in geo for (law, k) in LAWS]
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        jobs = jobs[:8]
    t0 = time.time()
    with Pool(max(1, (os.cpu_count() or 2)-1)) as pool:
        res = []
        for i, r in enumerate(pool.imap_unordered(simulate, jobs, chunksize=1)):
            res.append(r)
            if (i+1) % 50 == 0:
                print(f'{i+1}/{len(jobs)} done, {time.time()-t0:.0f}s', flush=True)
    (Path(__file__).parent/'results'/('clutter_contact.json' if LAWS[0][0] == 'chiralc' else 'clutter.json')).write_text(json.dumps(res, indent=1))
    # summary table
    import collections
    agg = collections.defaultdict(list)
    for r in res:
        agg[(r['scenario'], f"{r['law']}@{r['k']}")].append(r)
    print(f"{'scenario':24s} {'law':11s} {'done':>7s} {'med t':>6s} {'stall':>6s} {'minsep':>7s} {'vmax':>5s}")
    errs = [r for r in res if r.get('error')]
    print('errors:', len(errs), errs[:3])
    for (sc, law), rs in sorted(agg.items()):
        ok = [r for r in rs if not r.get('error')]
        dn = sum(r['completed'] for r in ok); ts = [r['time_s'] for r in ok if r['completed']]
        print(f"{sc:24s} {law:11s} {dn:3d}/{len(rs):<3d} {np.median(ts) if ts else float('nan'):6.1f} "
              f"{np.mean([r['stall_s'] for r in ok]) if ok else float('nan'):6.2f} "
              f"{min(r['min_sep'] for r in ok) if ok else float('nan'):7.3f} {max(r['max_speed'] for r in ok) if ok else float('nan'):5.2f}")


if __name__ == '__main__':
    main()
