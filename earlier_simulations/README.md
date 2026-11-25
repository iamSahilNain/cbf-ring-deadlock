# Earlier simulations (September 2026)

The first simulations of the project, written on 9 and 15 September 2026 (see the commit
history). They explored two-robot and small multi-robot encounters and found the questions that
the report later answers. **No number in the report comes from this folder**: the report uses
the model in [`../reproduce/`](../reproduce).

## How this model differs from the report's

- Speeds are bounded component-wise, |u_x|, |u_y| <= 1/sqrt(2) m/s (a square, not a disc).
- The goal pull is clipped coordinate by coordinate.

Both break the rotational symmetry of the ring, which is why the report switched to a
disc-saturated goal pull with no speed bound. The circulation rule and the CBF filter are the
same: safe distance 0.6 m, alpha = 2, time step 0.02 s, circulation gain 1.2.

## Files

| File | What it does | Date |
|---|---|---|
| `simulate.py` | Two robots, joint or separate CBF filter, with or without circulation: head-on, crossing and offset cases. Writes `results/*.npz`, `results/metrics.csv`, plots and a head-on animation | 9 Sep |
| `verify.py` | Independent check of the QP solver and controller. Writes `results/verification.json` | 9 Sep |
| `oblique.py` | Oblique encounters at 30, 120 and 150 degrees. Writes `results/oblique_*` | 9 Sep |
| `multi_robot.py` | 3, 4 and 5 robots on a circle swapping to the opposite point, four controllers each. Writes `results/multi_robot/` | 9 Sep |
| `verify_multi.py` | Independent QP and trajectory checks for `multi_robot.py` | 9 Sep |
| `RELATED_WORK.md` | Reading notes at the time | 9 Sep |
| `paper_experiments.py` | Collinear two-robot deadlocks in the setting of Grover et al. (IFAC 2020), and an exact equilibrium of the circulation rule. Writes `results/paper_cases/` | 15 Sep |
| `PAPER_EXPERIMENTS.md` | What those experiments showed, including the exact equilibrium (b = 0.5596...) | 15 Sep |
| `package_simulations.py` | Packs every saved trajectory into `simulations.html`, an offline replay page | 15 Sep |
| `simulations.html` | Open in a browser to replay all runs | 15 Sep |

Run each script from inside this folder, for example `python3 simulate.py`.
