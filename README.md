# Deadlocks in CBF safety filters for many robots: reproduction code

Code and saved results for the B.Tech. project report *When Do Robots Get Stuck? Deadlocks in
Safety Filters for Many Robots* (IIIT Delhi, 2026).

Model: single-integrator disc robots in the plane, CBF safety filter (joint QP or one QP per
robot), goal pull plus a circulation term. Parameters: safe distance D = 0.6 m, alpha = 2, push
range 2 m, time step 0.02 s. Every script imports the model from `check_lemmas.py`.

## Setup

```
pip install -r requirements.txt
```

The saved outputs from our runs are committed (some scripts only print; their output is shown in the table),
so you can compare them without rerunning anything.

## What reproduces what

| Report item | Script | Saved output |
|---|---|---|
| R1, R3 numerical checks; Table 7.1 (ring runs); perturbed rings | `check_lemmas.py` | `results.json`, `run.log` |
| Nearly symmetric rings, N = 7, 8 (text after Table 7.1) | `check_perturbed.py` | `check_perturbed.log` |
| Table 7.2, Figure 7.2 ($k_0(N)$ by formula and bisection) | `check_threshold.py` | `threshold.json`, `threshold.log` |
| Table 7.2, second row (first stuck run) | `traj_threshold.py` | `traj_threshold.json`, `traj_threshold.log` |
| C2, "worst moment" step (400 random cases) | `check_kstar_proof.py` | printed |
| Figure 7.3 (waiting time against epsilon) | `check_stall.py` | `stall.json` |
| C4 against the actual QP (3000 random cases) | `check_general.py` | printed |
| Identities used in the proofs (SymPy) | `check_identities.py` | printed |
| C5, first yield-to-the-left versions | `check_chiral.py`, `check_chiral2.py` | `chiral*.json`, `chiral*.log` |
| C5, contact sweep | `sweep_contact_chiral.py` | `chiralc_sweep.log` |
| Table 7.3 (clutter) and the long-range yield-to-the-left run | `check_clutter.py` | `clutter.json`, `clutter.log` |
| C5 clutter results | `check_clutter.py contact` | `clutter_contact.json`, `clutter_contact.log` |
| All figures | `figures/make_figures.py` | `figures/fig_*.pdf` |

The clutter benchmark uses every core but one, and takes about ten minutes per run on a laptop.
The other scripts take seconds to a few minutes.

## Names in the code and in the report

The scripts were written before the report, so they use older names:

| Code | Report |
|---|---|
| Lemma 1 | R1 (no stuck state away from contact) |
| Lemma 2 | R3 (equal and opposite pushes) |
| Theorem 1 | C1 (joint filter frees the ring) |
| Theorem 2(a) | R4 (the stuck cone) |
| Theorem 2(b-e), Lemma 6, k* | C2 (gain threshold) |
| Theorem 3 | C3 (saddle and waiting time) |
| Theorem 5 | C4 (what any rule needs) |
| Theorem 6, `chiral`, `chiralc` | C5 (yield-to-the-left rule) |
| `central=True / False` | joint / separate filters |
| `KC`, `k` | circulation gain |
