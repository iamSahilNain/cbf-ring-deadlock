"""Independent active-face enumeration checks for the CBF velocity QPs.

Run this support-only verifier; it writes results/verification.json.
"""
import itertools
import json
from pathlib import Path

import numpy as np


def reference_projection(nominal, normal, rhs, bound):
    """Enumerate every box face, with the halfspace inactive or active."""
    dimension = len(nominal)
    best = None
    best_cost = float("inf")
    for status in itertools.product((-1, 0, 1), repeat=dimension):
        status = np.asarray(status)
        free = status == 0
        fixed = ~free
        candidate = nominal.copy()
        candidate[fixed] = bound * status[fixed]
        for active in (False, True):
            trial = candidate.copy()
            if active:
                denominator = normal[free] @ normal[free]
                residual = rhs - normal @ trial
                if denominator < 1e-24:
                    if abs(residual) > 1e-10:
                        continue
                else:
                    trial[free] += residual / denominator * normal[free]
            if np.any(np.abs(trial) > bound + 1e-10):
                continue
            if normal @ trial < rhs - 1e-10:
                continue
            cost = np.sum((trial - nominal) ** 2)
            if cost < best_cost:
                best, best_cost = trial, cost
    if best is None:
        raise ValueError("No feasible reference solution")
    return best


def check_projection(project):
    rng = np.random.default_rng(20260909)
    bound = 1 / np.sqrt(2)
    maximum_error = 0.0
    maximum_constraint_violation = 0.0
    for sample in range(500):
        dimension = 2 if sample % 2 == 0 else 4
        nominal = rng.normal(0, 1.5, dimension)
        normal = rng.normal(0, 2, dimension)
        capacity = bound * np.sum(np.abs(normal))
        rhs = rng.uniform(-1.3 * capacity, capacity)
        expected = reference_projection(nominal, normal, rhs, bound)
        actual = project(nominal, normal, rhs, bound)
        maximum_error = max(maximum_error, float(np.max(np.abs(actual - expected))))
        maximum_constraint_violation = max(maximum_constraint_violation, float(rhs - normal @ actual))
        assert np.all(np.abs(actual) <= bound + 1e-9)
        assert np.max(np.abs(actual - expected)) < 1e-8, (sample, actual, expected)
    return {"random_qp_cases": 500, "seed": 20260909,
            "maximum_coordinate_error": maximum_error,
            "maximum_halfspace_violation": maximum_constraint_violation}


def check_baseline_equilibrium(project):
    bound = 1 / np.sqrt(2)
    nominal = np.array([1., 0., -1., 0.])
    normal = np.array([-1.2, 0., 1.2, 0.])
    joint = project(nominal, normal, 0., bound)
    separate = np.concatenate([
        project(nominal[:2], normal[:2], 0., bound),
        project(nominal[2:], normal[2:], 0., bound)])
    assert np.max(np.abs(joint)) < 1e-10
    assert np.max(np.abs(separate)) < 1e-10
    return {"head_on_boundary_centralized_max_speed": float(np.max(np.abs(joint))),
            "head_on_boundary_decentralized_max_speed": float(np.max(np.abs(separate)))}


def main():
    import simulate
    results = check_projection(simulate.project)
    results.update(check_baseline_equilibrium(simulate.project))
    settled = np.array([[-2., 0.], [2., 0.]])
    positions, velocities, metrics = simulate.run(settled, settled, True, True)
    assert metrics['completed'] and metrics['elapsed_s'] == 0 and velocities.shape == (0, 2, 2)
    results['already_at_goal_check'] = True
    position = np.array([[-.5, 0.], [.5, 0.]])
    goals = np.array([[2., 0.], [-2., 0.]])
    for centralized in (True, False):
        velocity, _ = simulate.control(position, goals, centralized, True)
        assert velocity[0, 1] > 0 and velocity[1, 1] < 0
        results[f"left_sign_{'centralized' if centralized else 'decentralized'}"] = velocity.tolist()
    rng = np.random.default_rng(314159)
    records = []
    for scenario in range(20):
        while True:
            starts = rng.uniform(-3., 3., (2, 2))
            goals = rng.uniform(-3., 3., (2, 2))
            if np.linalg.norm(starts[0] - starts[1]) >= .6 and np.linalg.norm(goals[0] - goals[1]) >= .6:
                break
        for centralized, circulation in itertools.product((False, True), repeat=2):
            positions, velocities, metrics = simulate.run(starts, goals, centralized, circulation, dt=.04)
            assert np.isfinite(positions).all() and np.isfinite(velocities).all()
            # Independently sample within each held-control segment; exact segment
            # safety is also asserted by run(), and follows from the CBF inequality.
            for fraction in (.25, .5, .75):
                interpolated = positions[:-1] + fraction * .04 * velocities
                assert np.min(np.linalg.norm(interpolated[:, 0] - interpolated[:, 1], axis=1)) >= .6 - 1e-9
            records.append(dict(scenario=scenario, centralized=centralized, circulation=circulation,
                                starts=starts.tolist(), goals=goals.tolist(), **metrics))
    results.update(random_trajectory_seed=314159, random_trajectory_runs=len(records),
                   random_minimum_separation_m=min(item['min_separation_m'] for item in records),
                   random_maximum_speed_mps=max(item['max_speed_mps'] for item in records),
                   random_minimum_cbf_residual=min(item['min_cbf_residual'] for item in records),
                   random_completed=sum(item['completed'] for item in records))
    results['random_trajectory_records'] = records
    output = Path(__file__).parent / 'results' / 'verification.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(json.dumps({key: value for key, value in results.items() if key != 'random_trajectory_records'}, indent=2))


if __name__ == '__main__':
    main()
