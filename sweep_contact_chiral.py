"""Search every touching state of the ring for stuck states under the keep-left rule.

Report: claim C5, first check. Run: python3 sweep_contact_chiral.py  (prints; we saved it as
chiralc_sweep.log)
"""
import numpy as np
import check_lemmas as cl
from check_chiral import contact_deadlock_phases, ring

DELTA, EPS = 0.02, 0.15


def chiralc(p, g, circulation=True):
    err = g - p
    ref = np.array([cl.sat(x) for x in err])
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
            s = max(0., eh[0]*e[1]-eh[1]*e[0])*min(1., a/DELTA)
            w = max(0., 1-d/2)*a + min(1., max(0., 1-(d-cl.D)/EPS))*s
            ref[i] = ref[i] + cl.KC*w*min(1., gd/.3)*(cl.J @ e)
    return ref


cl.reference = chiralc
for k in (0.8, 1.05, 1.5, 2.0):
    cl.KC = k
    out = []
    for N in (3, 6, 8, 12, 16, 20, 30):
        s, g = ring(N)
        out.append((N, len(contact_deadlock_phases(N, float(np.linalg.norm(s[0]))))))
    print(f'k={k}: theorem predicts none iff k*min(1,sin(pi/N)/delta) > 1 (k>1 for N<=157);  '
          'deadlock phases per N:', out, flush=True)
