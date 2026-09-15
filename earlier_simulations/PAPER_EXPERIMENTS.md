# Two-robot paper conditions and a circulation counterexample

Run date: 2026-09-15. See simulations.html for all replays; the four new scenario choices appear at the end of the selector.

## What was reproduced

Grover et al., Why Does Symmetry Cause Deadlocks?, Section 4, equations (17)–(18), Theorem 1: collinear starts separated by D_init, opposite goal directions, both initial goal travel distances strictly greater than D_init, positive proportional gains. The paper uses decentralized equal-responsibility CBF QPs with no input bounds. Our previous exact swap had goal travel distances equal to D_init, so it was not literally this strict overshoot setup.

New scenarios use initial separation 4 m: (1) starts (-2,0),(2,0), goals (3,0),(-3,0), gains (1,1); (2) same starts, goals (3,0),(-4,0), gains (0.15,0.08); (3) scenario 2 rotated by 0.63 radians. D_s=0.6, alpha=2, dt=0.02, maximum horizon 60 s, goal tolerance 0.05 m.

Each has four bounded-controller comparisons and two paper-QP comparisons. “Paper QP · Baseline” implements the paper's unclipped proportional nominal velocity and unconstrained-speed decentralized projection. “Paper QP · Our circulation” adds our heuristic; it is not the paper's resolution algorithm. These numerically integrate the model; they do not reproduce a published figure's exact parameters or prove the theorem.

All nine new baseline traces are unfinished at 60 s. All nine corresponding circulation traces finish. Some baseline traces leave their near-collinear stall because of numerical perturbations; in particular the rotated centralized baseline has no interval meeting the system-stall threshold. Do not label every finite numerical trace permanently deadlocked.

## Exact equilibrium of our circulation law

Take positions p1=(-0.3,0), p2=(0.3,0). Let a=0.5, c=1.2(1-0.6/2)=0.84 and

b = sqrt((-a² + sqrt(a⁴+4c²a²))/2) = 0.5596486150184862.

Set goals g1=(0.2,-b), g2=(-0.2,b). The goals are 1.188623695 m apart, so simultaneous goal occupancy is safe. Goal errors exceed the 0.3 m taper distance; all nominal components are within the component bounds.

Robot 1 has nominal (a,-b). Its circulation term is (0,c*a/sqrt(a²+b²)). The chosen b satisfies b=c*a/sqrt(a²+b²), so its reference is exactly (a,0). Robot 2 has the opposite reference. At separation D_s, the decentralized constraints require u1x<=0 and u2x>=0; projecting those references gives both velocities zero. The centralized joint projection also gives zero. Both are away from their goals. This proves an undesired equilibrium exists for the exact mathematical circulation controller.

This is an equilibrium construction on the safe-set boundary, not a proof of convergence into it from an open neighborhood, and not a new-literature novelty claim. It is NOT the ordinary baseline's collinear-goal condition: circulation changes the force balance and thus changes the equilibrium geometry.

Numerics: computed initial max absolute velocity 1.44e-15 m/s. The bounded baseline finishes in 2.78 s. Circulation stalls for 13.76 s and finishes in 19.08 s at dt=0.02; at dt=0.01 it stalls 14.02 s and finishes in 19.32 s. Perturbing the two initial y positions oppositely by ±0.0001 m gives completion in 6.02 or 6.92 s (6.00 or 6.88 s at dt=0.01). These checks show numerical escape and sensitivity; no formal stability/basin classification is asserted.

## Verification

22 saved traces independently checked for discrete state reconstruction, exact minimum separation along each held-velocity segment, joint CBF residual, applicable input bounds, and completion flags. Minimum separation was 0.6 m. Random safe configurations generated from 100 draws give zero disagreement between the new gain-1 bounded implementation and the original controller, across both architectures and remedies. Six additional sensitivity/refinement runs are recorded in results/paper_cases/verification.json. HTML checked for all four new scenario selectors, 67 rows, and no JavaScript errors.

## Published circulation methods

- Keyumarsi, Atman and Gusrialdi (2025), Circulation-embedded Control Barrier Function for safe navigation: A solution to avoid undesired equilibria and dysfunctional circulation. https://doi.org/10.1016/j.robot.2025.105132 . Adds a circulation inequality inside the QP, with input bounds. Author implementation https://github.com/sh-keyumarsi/Circulation-embedded-CBF explicitly constrains the circulation-direction projection of control and uses activation memory. This differs from merely biasing our reference velocity. Full theorem assumptions were not verified because direct full-PDF access was blocked.
- Lee et al. (2025), Merry-Go-Round: Safe Control of Decentralized Multi-Robot Systems with Deadlock Prevention. https://arxiv.org/html/2503.05848v1 . Section IV, Eq. (11), uses tangential plus radial tracking of a temporary counterclockwise roundabout, with goal/roundabout switching and escape conditions. The reviewed model assumes homogeneous unicycles, perfect local communication, known map, and separated goals. Do not transfer its claims to our controller.

## Recorded outcomes

| Scenario | Controller | Complete | End (s) | Longest stall (s) |
|---|---|---|---:|---:|
| paper_symmetric | centralized_baseline | False | 60.00 | 54.60 |
| paper_symmetric | centralized_circulation | True | 8.82 | 0.00 |
| paper_symmetric | decentralized_baseline | False | 60.00 | 54.60 |
| paper_symmetric | decentralized_circulation | True | 8.82 | 0.00 |
| paper_symmetric | paper_decentralized_baseline | False | 60.00 | 55.32 |
| paper_symmetric | paper_decentralized_circulation | True | 5.62 | 0.00 |
| paper_heterogeneous | centralized_baseline | False | 60.00 | 23.98 |
| paper_heterogeneous | centralized_circulation | True | 59.16 | 0.00 |
| paper_heterogeneous | decentralized_baseline | False | 60.00 | 53.98 |
| paper_heterogeneous | decentralized_circulation | True | 59.16 | 0.00 |
| paper_heterogeneous | paper_decentralized_baseline | False | 60.00 | 53.98 |
| paper_heterogeneous | paper_decentralized_circulation | True | 59.18 | 0.00 |
| paper_rotated | centralized_baseline | False | 60.00 | 0.00 |
| paper_rotated | centralized_circulation | True | 59.20 | 0.00 |
| paper_rotated | decentralized_baseline | False | 60.00 | 20.56 |
| paper_rotated | decentralized_circulation | True | 59.16 | 0.00 |
| paper_rotated | paper_decentralized_baseline | False | 60.00 | 20.32 |
| paper_rotated | paper_decentralized_circulation | True | 59.18 | 0.00 |
| circulation_equilibrium | centralized_baseline | True | 2.78 | 0.00 |
| circulation_equilibrium | centralized_circulation | True | 19.08 | 13.76 |
| circulation_equilibrium | decentralized_baseline | True | 2.78 | 0.00 |
| circulation_equilibrium | decentralized_circulation | True | 19.08 | 13.76 |
