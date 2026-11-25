"""Chiral fix experiment.

Law: beta_ij = k * gamma(d_ij) * [ (e_hat_i . e_ij)_+  +  (e_hat_i x e_ij)_+ ] * tau_i
  - first term: the original alignment gate
  - second term: "neighbour on my left" gate (chirality in the *strength*)
Claim to test (Theorem 6 = report claim C5): for k > 1/gamma(D) the decentralized
ring has no contact deadlock phase for ANY N.
Risk to test: backing off from a side neighbour may create a new stall when passing a parked robot.
Run: python3 check_chiral.py   (writes chiral.json)
"""
import json
from pathlib import Path
import numpy as np
import check_lemmas as cl

original_reference = cl.reference


def chiral_reference(p, g, circulation=True):
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
            gate = max(0., eh @ e) + max(0., eh[0]*e[1]-eh[1]*e[0])
            beta = cl.KC*max(0., 1-d/2)*gate*min(1., gd/.3)
            ref[i] = ref[i] + beta*(cl.J @ e)
    return ref


def done(traj, g):
    return bool(np.max(np.linalg.norm(g-traj[-1], axis=1)) <= cl.TOL)


def ring(N):
    rho = max(2., 1/np.sin(np.pi/N))            # assumption (A): rho sin(pi/N) >= 1
    s, g = cl.antipodal(N, rho)
    return s, g


def contact_deadlock_phases(N, rho):
    rs = cl.D/(2*np.sin(np.pi/N)); cot = np.cos(np.pi/N)/np.sin(np.pi/N)
    th = 2*np.pi*np.arange(N)/N; out = []
    for phi in np.linspace(-np.pi, 0, 3001):
        p = rs*np.column_stack((np.cos(th+phi), np.sin(th+phi)))
        g = -rho*np.column_stack((np.cos(th), np.sin(th)))
        r0 = cl.reference(p, g)[0]
        rh = p[0]/np.linalg.norm(p[0]); rr, rt = r0 @ rh, r0 @ (cl.J @ rh)
        if rr < 0 and abs(rt) <= cot*(-rr):
            out.append(phi)
    return out


def main():
    res = dict(ring_phase_sweep=[], rings=[], parked=[], headon=[])
    for law_name, law in (('original', original_reference), ('chiral', chiral_reference)):
        cl.reference = law
        for k in (1.2, 1.6, 2.0):
            cl.KC = k
            for N in (3, 6, 8, 10, 12, 14):
                s, g = ring(N)
                E = contact_deadlock_phases(N, float(np.linalg.norm(s[0])))
                res['ring_phase_sweep'].append(dict(law=law_name, k=k, N=N, deadlock_phases=len(E)))
                print('sweep', res['ring_phase_sweep'][-1], flush=True)
        cl.KC = 2.0
        for N in (3, 6, 8, 10, 12, 14):
            s, g = ring(N)
            for central in (False, True):
                traj, _, _ = cl.run(s, g, central, horizon=80.)
                res['rings'].append(dict(law=law_name, k=2.0, N=N, central=central, completed=done(traj, g),
                                         time_s=round((len(traj)-1)*cl.DT, 2)))
                print('ring', res['rings'][-1], flush=True)
        # robot 0 drives (-3,0)->(3,0) past robot 1 parked at (0, y0) (its goal = its position)
        for y0 in (-1.3, -1.0, -0.8, -0.65, 0.65, 0.8, 1.0, 1.3):
            s = np.array([[-3., 0.], [0., y0]]); g = np.array([[3., 0.], [0., y0]])
            traj, _, us = cl.run(s, g, False, horizon=60.)
            speeds = np.linalg.norm(us[:, 0], axis=1) if len(us) else np.array([0.])
            res['parked'].append(dict(law=law_name, y0=y0, completed=done(traj, g), time_s=round((len(traj)-1)*cl.DT, 2),
                                      final_x=float(traj[-1, 0, 0]), min_speed_after_1s=float(speeds[50:].min()) if len(speeds) > 50 else None))
            print('parked', res['parked'][-1], flush=True)
        s = np.array([[-2., 0.], [2., 0.]]); g = -s
        for central in (False, True):
            traj, _, _ = cl.run(s, g, central, horizon=60.)
            res['headon'].append(dict(law=law_name, central=central, completed=done(traj, g), time_s=round((len(traj)-1)*cl.DT, 2)))
            print('headon', res['headon'][-1], flush=True)
    cl.reference = original_reference
    (Path(__file__).parent/'chiral.json').write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
