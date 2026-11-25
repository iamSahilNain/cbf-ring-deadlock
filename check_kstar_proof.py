"""Numerical audit of the proof that k*(N) = k0(N) (Lemma 6 = the "worst moment" step of report claim C2).

At contact, in the frame where p_0 = (r*, 0):
  r_0 = -V,  V = mu (cos chi, sin chi) + k tau sum_m gamma_m (sin(th_m - chi))_+ (cos th_m, sin th_m),
  th_m = pi m / N,  chi = arg(r* + rho e^{i psi}),  psi = -phi.
Deadlock  <=>  G(chi) := u_beta x V <= 0,  beta = pi/2 - pi/N.
Claims checked: (1) the closed form matches the full reference code;
(2) the cone test via G matches the direct cone test; (3) each pair ratio
R_m(chi) = P_m(chi)/sin(beta-chi) is nondecreasing on [0, beta);
(4) G(chi) >= mu sin(beta-chi) G(0)/sin(beta) on [0, beta); (5) E empty iff k > k0,
including unsaturated goals (rho close to r*, rho < r*).
Run: python3 check_kstar_proof.py
"""
import numpy as np
import check_lemmas as cl

rng = np.random.default_rng(3)
worst_formula = 0.
mismatch = bound_viol = mono_viol = iff_viol = 0
for trial in range(400):
    N = int(rng.integers(2, 13))
    rs = cl.D/(2*np.sin(np.pi/N))
    rho = float(rng.choice([2., rs + .5, rs + .05, max(.2, rs - .2), 5.])) if rs + .05 > 0 else 2.
    if rho + rs < 1:          # theorem assumes rho + r* >= 1
        continue
    m = np.arange(1, N); th = np.pi*m/N
    gam = np.clip(1 - cl.D*np.sin(th)/np.sin(np.pi/N)/2, 0, None)
    k0 = (np.cos(np.pi/N)/np.sin(np.pi/N))/np.sum(gam*np.sin(th)**2) if N > 2 else 0.
    k = float(rng.uniform(.3, 2.)*max(k0, .2))
    cl.KC = k
    beta = np.pi/2 - np.pi/N
    ub = np.array([np.cos(beta), np.sin(beta)])
    anyE = False
    for psi in np.linspace(0, np.pi, 721):
        p = rs*np.column_stack((np.cos(2*np.pi*np.arange(N)/N), np.sin(2*np.pi*np.arange(N)/N)))
        g0 = -rho*np.array([np.cos(psi), np.sin(psi)])              # goal of robot 0 in this frame
        g = np.vstack((g0, p[1:]))                                    # others' goals irrelevant for robot 0
        ref = cl.reference(p, g)[0]
        w = g0 - p[0]; nw = np.linalg.norm(w)
        chi = np.arctan2(rho*np.sin(psi), rho*np.cos(psi) + rs)
        mu, tau = min(1, nw), min(1, nw/.3)
        V = mu*np.array([np.cos(chi), np.sin(chi)]) + k*tau*np.sum(
            (gam*np.clip(np.sin(th-chi), 0, None))[:, None]*np.column_stack((np.cos(th), np.sin(th))), axis=0)
        worst_formula = max(worst_formula, float(np.abs(ref + V).max()))
        rr, rt = ref[0], ref[1]
        direct = rr < 0 and abs(rt) <= (np.cos(np.pi/N)/np.sin(np.pi/N))*(-rr) + 1e-12 if N > 2 else (rr < 0 and abs(rt) < 1e-12)
        G = ub[0]*V[1] - ub[1]*V[0]
        if direct != (G <= 1e-12):
            mismatch += 1
        anyE |= direct
        if chi < beta - 1e-9:
            G0 = -np.sin(beta) + k*np.sum(gam*np.sin(th)*np.sin(th-beta))
            if G < mu*np.sin(beta-chi)*G0/np.sin(beta) - 1e-10:
                bound_viol += 1
    # pair-ratio monotonicity on a fine chi grid
    chis = np.linspace(0, beta, 2001, endpoint=False)
    for mm, t in zip(m, th):
        if t > np.pi/2 + 1e-12:
            continue
        tb = np.pi - t
        P = gam[mm-1]*(np.clip(np.sin(t-chis), 0, None)*np.sin(t-beta) +
                       (np.clip(np.sin(tb-chis), 0, None)*np.sin(tb-beta) if abs(t-np.pi/2) > 1e-12 else 0))
        R = P/np.sin(beta-chis)
        mono_viol += int(np.sum(np.diff(R) < -1e-12))
    if anyE != (k <= k0 + 1e-12):
        iff_viol += 1
print(dict(max_formula_error=worst_formula, cone_test_mismatches=mismatch,
           bound_violations=bound_viol, monotonicity_violations=mono_viol, iff_violations=iff_viol))
