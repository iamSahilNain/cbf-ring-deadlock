"""Symbolic checks (sympy) of every trig identity and derivative used in the report proofs.
Run: python3 check_identities.py   -> prints each identity name and True.
"""
import sympy as sp

th, chi, beta, x, r, phi, rho, m, N = sp.symbols('theta chi beta x r phi rho m N', real=True)
J = sp.Matrix([[0, -1], [1, 0]])
checks = {}

# Bearing from p0=(r,0) to p_m = r(cos 2th, sin 2th): direction (-sin th, cos th)
pm = r*sp.Matrix([sp.cos(2*th), sp.sin(2*th)]) - sp.Matrix([r, 0])
dm = 2*r*sp.sin(th)
checks['bearing e_m'] = sp.simplify(pm/dm - sp.Matrix([-sp.sin(th), sp.cos(th)])) == sp.zeros(2, 1)
e = sp.Matrix([-sp.sin(th), sp.cos(th)])
checks['J e_m = -(cos th, sin th)'] = sp.simplify(J*e + sp.Matrix([sp.cos(th), sp.sin(th)])) == sp.zeros(2, 1)
w_hat = -sp.Matrix([sp.cos(chi), sp.sin(chi)])
checks['alignment = sin(th - chi)'] = sp.simplify((w_hat.T*e)[0] - sp.sin(th - chi)) == 0

# Pair formula (active branch): sin(th-chi)sin(th-beta) + sin(pi-th-chi)sin(pi-th-beta)
pair = sp.sin(th-chi)*sp.sin(th-beta) + sp.sin(sp.pi-th-chi)*sp.sin(sp.pi-th-beta)
checks['pair active formula'] = sp.simplify(sp.expand_trig(pair - (sp.cos(beta-chi) - sp.cos(2*th)*sp.cos(beta+chi)))) == 0

# d/dx of [cos x - c cos(2beta - x)]/sin x  ==  (-1 + c cos 2beta)/sin^2 x
c = sp.symbols('c', real=True)
Q = (sp.cos(x) - c*sp.cos(2*beta - x))/sp.sin(x)
checks['dQ/dx'] = sp.simplify(sp.diff(Q, x) - (-1 + c*sp.cos(2*beta))/sp.sin(x)**2) == 0

# inactive branch: d/dchi [sin(th+chi)/sin(beta-chi)] == sin(th+beta)/sin^2(beta-chi)
Rin = sp.sin(th+chi)/sp.sin(beta-chi)
checks['inactive derivative'] = sp.simplify(sp.diff(Rin, chi) - sp.sin(th+beta)/sp.sin(beta-chi)**2) == 0

# continuity at chi = th: active formula equals inactive branch
checks['continuity at chi=theta'] = sp.simplify(sp.expand_trig(
    (sp.cos(beta-th) - sp.cos(2*th)*sp.cos(beta+th)) - sp.sin(2*th)*sp.sin(th+beta))) == 0

# R_m(0): pair at chi=0 equals 2 sin^2 th cos beta
checks['pair at chi=0'] = sp.simplify(sp.expand_trig(pair.subs(chi, 0) - 2*sp.sin(th)**2*sp.cos(beta))) == 0

# tan(pi/2 - pi/N) == cot(pi/N)
checks['tan beta = cot(pi/N)'] = sp.simplify(sp.tan(sp.pi/2 - sp.pi/N) - sp.cot(sp.pi/N)) == 0

# ring constraint reduces to radial: (I - R)^T (I - R) = 2(1-cos a) I
a = sp.symbols('a', real=True)
R = sp.Matrix([[sp.cos(a), -sp.sin(a)], [sp.sin(a), sp.cos(a)]])
checks['(I-R)^T(I-R) = 2(1-cos a) I'] = sp.simplify((sp.eye(2)-R).T*(sp.eye(2)-R) - 2*(1-sp.cos(a))*sp.eye(2)) == sp.zeros(2, 2)

# saddle eigenvalue: phi' = rho sin(phi) / (r * |g - p|),  |g-p|^2 = rho^2 + r^2 + 2 rho r cos phi
rs, al, rhop = sp.symbols('r_star alpha rho_p', positive=True)
phidot = rhop*sp.sin(phi)/(r*sp.sqrt(rhop**2 + r**2 + 2*rhop*r*sp.cos(phi)))
lam = sp.diff(phidot, phi).subs({phi: 0, r: rs})
lam = sp.simplify(lam.subs(sp.sqrt(rs**2 + 2*rs*rhop + rhop**2), rs + rhop))  # sqrt((rho+r*)^2) = rho+r*
checks['lambda = rho/((rho+r*) r*)'] = sp.simplify(lam - rhop/((rhop+rs)*rs)) == 0
b1 = -al*(r**2 - rs**2)/(2*r)
checks["b1'(r*) = -alpha"] = sp.simplify(sp.diff(b1, r).subs(r, rs) + al) == 0

# tangential part of goal pull: g=(-rho,0), p = r(cos phi, sin phi), t = (-sin phi, cos phi)
g = sp.Matrix([-rho, 0]); p = r*sp.Matrix([sp.cos(phi), sp.sin(phi)]); t = sp.Matrix([-sp.sin(phi), sp.cos(phi)])
checks['(g-p).t = rho sin phi'] = sp.simplify(((g-p).T*t)[0] - rho*sp.sin(phi)) == 0

# chiral law (Theorem 6): side gate  w_hat x e_m = -cos(th - chi);  trailing neighbour term
cross = w_hat[0]*e[1] - w_hat[1]*e[0]
checks['w_hat x e_m = -cos(th - chi)'] = sp.simplify(cross + sp.cos(th - chi)) == 0
trail = (-sp.cos(th - chi)).subs(th, sp.pi - sp.pi/N)
checks['trailing side gate = sin(beta-chi)'] = sp.simplify(sp.expand_trig(trail - sp.sin(sp.pi/2 - sp.pi/N - chi))) == 0
checks['sin(th_{N-1} - beta) = 1'] = sp.simplify(sp.sin((sp.pi - sp.pi/N) - (sp.pi/2 - sp.pi/N)) - 1) == 0
# Theorem 5: sin(th_m - beta) = -cos(pi(m+1)/N), and pair sum
checks['w_m = -cos(pi(m+1)/N)'] = sp.simplify(sp.sin(sp.pi*m/N - (sp.pi/2 - sp.pi/N)) + sp.cos(sp.pi*(m+1)/N)) == 0
checks['w_m + w_{N-m} = 2 sin(pi/N) sin th_m'] = sp.simplify(sp.expand_trig(
    -sp.cos(sp.pi*(m+1)/N) - sp.cos(sp.pi*(N-m+1)/N) - 2*sp.sin(sp.pi/N)*sp.sin(sp.pi*m/N))) == 0

for k, v in checks.items():
    print(f'{k:34s} {bool(v)}')
assert all(checks.values())
