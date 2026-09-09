# Circulation and CBF deadlock: short prior-art check

Checked 9 September 2026. This is a bounded primary-source check, not an exhaustive novelty search.

The broad idea of using circulation or coordinated rotation to escape CBF-related multi-robot deadlock already appears in published work. Our current implementation is a heuristic nominal-velocity modification, not a reproduction of the algorithms below and not an established new method.

## Closest sources

1. **Shaghayegh Keyumarsi, Made Widhi Surya Atman, and Azwirman Gusrialdi. “Circulation-embedded Control Barrier Function for safe navigation: A solution to avoid undesired equilibria and dysfunctional circulation.”** Robotics and Autonomous Systems 194 (2025), 105132. Available online 8 August 2025; accepted 16 July 2025. [DOI](https://doi.org/10.1016/j.robot.2025.105132), [university PDF](https://researchportal.tuni.fi/files/153395188/Circulation-embedded_Control.pdf), [authors' code](https://github.com/sh-keyumarsi/Circulation-embedded-CBF).

   The abstract explicitly adds a circulation **linear inequality constraint inside the CBF-QP**, with input bounds and analysis of feasibility, continuity, equilibria, and convergence. It includes multi-robot experiments without communication. Our implementation instead adds a tangential term to the QP's reference velocity and keeps ordinary pairwise safety constraints. Therefore our circulation is a preference the safety filter can change, not a separately enforced inequality. The indexed university PDF first page supplied these details and dates; direct PDF retrieval was blocked during this check, so the complete theorem assumptions were not reviewed.

2. **Wonjong Lee et al. “Merry-Go-Round: Safe Control of Decentralized Multi-Robot Systems with Deadlock Prevention.”** Preprint submitted 7 March 2025; arXiv records publication in IROS 2025, pp. 4589–4595. [arXiv and publication metadata](https://arxiv.org/abs/2503.05848), [author-hosted full paper](https://wjlee.kr/MGR/static/pdfs/MGR.pdf).

   This is particularly close to the traffic/roundabout interpretation. Robots create or join temporary circular reference paths through local communication, travel in a consistent direction, and leave when an escape condition holds. Section IV-A and equation (9) construct desired velocity from tangential and radial components; the controller maintains safety through the QP formulation. Our pairwise tangential bias has no shared roundabout center, join/leave state machine, or radius-tracking term. Thus the conceptual overlap is strong, but the controller is different. The paper's guarantees cannot be transferred to our simulation.

3. **Jaskaran Grover, Changliu Liu, and Katia Sycara. “Deadlock Analysis and Resolution in Multi-Robot Systems (Extended Version).”** Preprint submitted 20 November 2019, revised 13 July 2020. [Primary arXiv record](https://arxiv.org/abs/1911.09146).

   Earlier theoretical background: characterizes CBF deadlock using optimization duality and force equilibrium, then proposes a decentralized resolution algorithm, with two- and three-robot simulations and physical experiments. This source establishes that both CBF deadlock characterization and resolution are longstanding research topics. Its abstract was checked here; this note does not claim that its exact resolution law equals our tangential bias.

## What we can presently claim

We implemented and compared a simple circulation-biased CBF simulation. Successful runs show behavior for those initial conditions and parameters; they do not establish deadlock freedom or research novelty. A defensible research contribution would need a specific difference, appropriate comparisons with existing methods, and evidence or analysis supporting that difference.
