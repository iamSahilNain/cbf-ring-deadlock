"""Make every figure in the report from the saved results and the proof-folder model.

Run from anywhere:  python3 make_figures.py
Reads  ../{threshold.json, traj_threshold.json, stall.json}
Writes fig_*.pdf next to this file.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge

HERE = Path(__file__).resolve().parent
PROOF = HERE.parent
sys.path.insert(0, str(PROOF))
import check_lemmas as cl  # noqa: E402  (same model as every result in the report)

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#d9d8d4'
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 0.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'lines.linewidth': 1.6, 'savefig.bbox': 'tight', 'savefig.dpi': 300,
})


def save(fig, name):
    fig.savefig(HERE / name)
    plt.close(fig)
    print('wrote', name)


# ---------------------------------------------------------------- one robot, one obstacle
def single_obstacle():
    """Goal at the origin, round obstacle centred at c = (3, 0), radius 1, k = gamma = 1."""
    c, r, k, gam = np.array([3., 0.]), 1., 1., 1.

    def u(x):
        a = x - c
        h = 0.5*(a @ a - r*r)
        u0 = -k*x
        s = a @ u0 + gam*h
        return u0 if s >= 0 else u0 + (-s/(a @ a))*a

    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax.add_patch(Circle(c, r, facecolor='#ecebe7', edgecolor=MUTED, lw=1))
    for y0, col in ((0.0, ORANGE), (0.15, BLUE), (-0.15, BLUE), (0.6, BLUE), (-0.6, BLUE)):
        x = np.array([6., y0]); path = [x.copy()]
        for _ in range(3000):
            x = x + 0.01*u(x); path.append(x.copy())
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], color=col, lw=1.4)
        ax.plot(*path[0], 'o', color=col, ms=4)
    ax.plot(0, 0, marker='*', color=INK, ms=10)
    ax.annotate('goal', (0, 0), (0.15, -0.55), color=INK)
    ax.plot(4, 0, 'o', mfc='white', mec=ORANGE, mew=1.5, ms=7)
    ax.annotate('stuck point\n(far side)', (4, 0), (4.25, 0.55), color=INK,
                arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.7))
    ax.plot(2, 0, 'x', color=MUTED, ms=6)
    ax.annotate('April report said\nthe robot stops here', (2, 0), (0.6, 0.75), color=MUTED,
                arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.7))
    ax.text(5.2, -1.15, 'start exactly on the line: stops', color=ORANGE, fontsize=8)
    ax.text(5.2, -1.45, 'start slightly off the line: escapes', color=BLUE, fontsize=8)
    ax.set_aspect('equal'); ax.set_xlim(-0.5, 8.6); ax.set_ylim(-1.7, 1.6)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    save(fig, 'fig_single_obstacle.pdf')


# ---------------------------------------------------------------- ring snapshots
def ring_panel(ax, N, central, circulation, gain, title):
    cl.KC = gain
    s, g = cl.antipodal(N)
    traj, _, _ = cl.run(s, g, central, circulation, horizon=40.)
    done = bool(np.max(np.linalg.norm(g - traj[-1], axis=1)) <= cl.TOL)
    for i in range(N):
        ax.plot(traj[:, i, 0], traj[:, i, 1], color=BLUE if done else ORANGE, lw=1.0, alpha=0.9)
        ax.plot(*s[i], 'o', color=MUTED, ms=3)
        ax.add_patch(Circle(traj[-1, i], cl.D/2, facecolor='none',
                            edgecolor=BLUE if done else ORANGE, lw=1.1))
    ax.set_aspect('equal'); ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.6, 2.6)
    ax.set_xticks([-2, 0, 2]); ax.set_yticks([-2, 0, 2])
    t = (len(traj)-1)*cl.DT
    verdict = f'all arrive ({t:.0f} s)' if done else 'stuck (40 s)'
    ax.set_title(f'{title}\n{verdict}', fontsize=8.5, color=INK)
    return done


def rings():
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.3))
    ring_panel(axs[0], 2, False, False, 0.0, '2 robots, no circulation')
    ring_panel(axs[1], 8, True, True, 1.2, '8 robots, joint, gain 1.2')
    ring_panel(axs[2], 8, False, True, 1.2, '8 robots, separate, gain 1.2')
    ring_panel(axs[3], 8, False, True, 2.0, '8 robots, separate, gain 2.0')
    cl.KC = 1.2
    fig.tight_layout()
    save(fig, 'fig_rings.pdf')


# ---------------------------------------------------------------- threshold k0(N)
def threshold():
    rows = json.loads((PROOF/'threshold.json').read_text())['thresholds']
    traj = json.loads((PROOF/'traj_threshold.json').read_text())
    N = np.array([r['N'] for r in rows]); k0 = np.array([r['k0'] for r in rows])
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    ax.plot(N, k0, '-o', color=BLUE, ms=5, label='formula $k_0(N)$ (equals bisection to 7 digits)')
    tn = [int(n) for n in traj]; tk = [np.mean(traj[n]['trajectory_threshold_between']) for n in traj]
    ax.plot(tn, tk, 's', color=ORANGE, ms=6, label='gain where a real run starts to get stuck')
    ax.axhline(1.2, color=MUTED, lw=1, ls='--')
    ax.text(10.0, 0.88, 'gain 1.2 used in our first simulations', color=MUTED, fontsize=8, ha='right')
    ax.set_xlabel('number of robots N'); ax.set_ylabel('circulation gain')
    ax.set_xticks(N); ax.set_ylim(0, 3.8)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    save(fig, 'fig_threshold.pdf')


# ---------------------------------------------------------------- stall time
def stall():
    rows = json.loads((PROOF/'stall.json').read_text())
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    for r in rows:
        eps = np.array([x[0] for x in r['runs']]); t = np.array([x[1] for x in r['runs']])
        L = np.log(1/eps)
        if r['circulation']:
            ax.plot(L, t, '-^', color=AQUA, ms=6)
            continue
        col = BLUE if r['N'] == 3 else ORANGE
        ax.plot(L, t, 'o', color=col, ms=6)
        ax.plot(L, t[0] + r['predicted_slope']*(L - L[0]), '-', color=col, lw=1.2)
        xy = (4.6, 15.6) if r['N'] == 5 else (11.2, 8.0)
        ax.text(*xy, f"N = {r['N']}, no circulation\nslope {r['fitted_slope']:.3f} "
                f"(predicted {r['predicted_slope']:.3f})", color=INK, fontsize=8)
    ax.text(4.6, 6.5, 'N = 3 and 5 with circulation: about 6 s, whatever the start', color=INK, fontsize=8)
    ax.set_xlabel(r'$\ln(1/\varepsilon)$     ($\varepsilon$ = distance of the start from perfect symmetry)')
    ax.set_ylim(5, 19)
    ax.set_ylabel('time to finish (s)')
    save(fig, 'fig_stall.pdf')


# ---------------------------------------------------------------- the stuck cone
def cone():
    """Robot 0 touching its two ring neighbours; the wishes that leave it stuck."""
    N = 8
    rs = cl.D/(2*np.sin(np.pi/N))
    th = 2*np.pi*np.arange(N)/N
    P = rs*np.column_stack((np.cos(th), np.sin(th)))
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    for i, p in enumerate(P):
        ax.add_patch(Circle(p, cl.D/2, facecolor='#ecebe7' if i else '#ffffff',
                            edgecolor=INK if i == 0 else MUTED, lw=1.2 if i == 0 else 0.8))
    p0 = P[0]
    beta = np.pi/2 - np.pi/N
    ang_in = 180.0          # direction from robot 0 to the centre
    ax.add_patch(Wedge(p0, 0.75, ang_in - np.degrees(beta), ang_in + np.degrees(beta),
                       facecolor=ORANGE, alpha=0.25, edgecolor=ORANGE, lw=1))
    for j in (1, N-1):
        d = P[j]-p0
        ax.annotate('', p0 + 0.75*d/np.linalg.norm(d), p0,
                    arrowprops=dict(arrowstyle='->', color=MUTED, lw=1))
    ax.annotate('', p0 + np.array([-0.75, 0.]), p0, arrowprops=dict(arrowstyle='->', color=INK, lw=1.4))
    ax.text(p0[0]-0.72, p0[1]-0.17, 'goal pull', fontsize=8, color=INK)
    ax.text(p0[0]-0.74, p0[1]+0.4, 'stuck cone', fontsize=7, color=INK)
    ax.text(p0[0]+0.36, p0[1]-0.05, 'robot 0', fontsize=8, color=INK)
    ax.set_aspect('equal'); ax.set_xlim(-1.25, 1.55); ax.set_ylim(-1.25, 1.25)
    ax.axis('off')
    save(fig, 'fig_cone.pdf')


if __name__ == '__main__':
    which = sys.argv[1:] or ['single', 'cone', 'threshold', 'stall', 'rings']
    for w in which:
        dict(single=single_obstacle, cone=cone, threshold=threshold, stall=stall, rings=rings)[w]()
