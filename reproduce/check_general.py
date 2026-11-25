"""Theorem 5 (class-level jam conditions for the decentralized ring) checked against the actual QP.

At the symmetric contact state (r = r*, phi = 0), robot 0's wish is r0 = mu*n_hat + c with
n_hat = -p_hat (toward the centre). Claims:
 (i)   any c with ||c|| < mu cos(pi/N)                      -> u0 = 0 (jam)
 (ii)  nudge laws c = sum_m beta_m J e_m, beta >= 0:  jam  iff  sum beta_m w_m <= mu cos(pi/N),
       w_m = -cos(pi (m+1)/N)
 (iii) mirror-symmetric strengths (beta_m = beta_{N-m}): jam iff sum beta_m sin(pi m/N) <= mu cot(pi/N)
Jam is decided by solving robot 0's decentralized QP (all N-1 neighbour rows) and testing ||u0|| < 1e-7.
Run: python3 check_general.py
"""
import numpy as np
import check_lemmas as cl

rng = np.random.default_rng(11)
bad = {'i': 0, 'ii': 0, 'iii': 0}
tested = {'i': 0, 'ii': 0, 'iii': 0}
for trial in range(3000):
    N = int(rng.integers(3, 16))
    rs = cl.D/(2*np.sin(np.pi/N))
    th = 2*np.pi*np.arange(N)/N
    p = rs*np.column_stack((np.cos(th), np.sin(th)))
    pairs, A, b = cl.constraints(p)
    rows = [r for r, pr in enumerate(pairs) if 0 in pr]
    A0, b0 = A[rows, 0:2], b[rows]/2
    phat = np.array([1., 0.])
    mu = float(rng.uniform(.2, 1.))
    e = (p[1:]-p[0])/np.linalg.norm(p[1:]-p[0], axis=1)[:, None]
    Je = np.column_stack((-e[:, 1], e[:, 0]))
    m = np.arange(1, N)
    kind = ('i', 'ii', 'iii')[trial % 3]
    if kind == 'i':
        ang = rng.uniform(0, 2*np.pi)
        c = rng.uniform(0, .999)*mu*np.cos(np.pi/N)*np.array([np.cos(ang), np.sin(ang)])
        predicted = True
    else:
        beta = rng.exponential(1., N-1)*rng.uniform(0, 3)/N*rng.integers(0, 2, N-1)
        if kind == 'iii':
            beta = (beta + beta[::-1])/2
        c = beta @ Je
        if kind == 'ii':
            lhs, rhs = np.sum(beta*(-np.cos(np.pi*(m+1)/N))), mu*np.cos(np.pi/N)
        else:
            lhs, rhs = np.sum(beta*np.sin(np.pi*m/N)), mu*np.cos(np.pi/N)/np.sin(np.pi/N)
        if abs(lhs-rhs) < 1e-6:
            continue
        predicted = lhs <= rhs
    ref = -mu*phat + c
    u = cl.project(ref, A0, b0)
    jam = np.linalg.norm(u) < 1e-7
    tested[kind] += 1
    bad[kind] += int(jam != predicted)
print('tested', tested, 'disagreements', bad)
