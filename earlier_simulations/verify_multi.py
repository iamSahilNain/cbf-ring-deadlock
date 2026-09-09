"""Support-only independent QP and saved-trajectory checks; run this file directly."""
from itertools import combinations, product
from pathlib import Path
import csv
import json
import numpy as np
from scipy.optimize import nnls
import multi_robot
import simulate


def with_box(matrix, lower, dimension):
    return (np.vstack((matrix, np.eye(dimension), -np.eye(dimension))),
            np.concatenate((lower, np.full(2*dimension, -simulate.BOUND))))


def independent_2d(nominal, matrix, lower):
    matrix, lower = with_box(matrix, lower, 2)
    candidates = [nominal]
    for row, rhs in zip(matrix, lower):
        candidates.append(nominal + row*(rhs-row@nominal)/(row@row))
    for first, second in combinations(range(len(lower)), 2):
        face = matrix[[first, second]]
        if abs(np.linalg.det(face)) > 1e-12:
            candidates.append(np.linalg.solve(face, lower[[first, second]]))
    feasible = [point for point in candidates if np.min(matrix@point-lower) >= -1e-9]
    return min(feasible, key=lambda point: np.sum((point-nominal)**2))


def random_safe_positions(rng, count):
    while True:
        position = rng.uniform(-1.5, 1.5, (count, 2))
        if min(np.linalg.norm(position[i]-position[j]) for i,j in combinations(range(count),2)) >= .6:
            return position


def geometry_constraints(position):
    rows, lower = [], []
    for first, second in combinations(range(len(position)), 2):
        relative = position[first]-position[second]
        row = np.zeros(position.size)
        row[2*first:2*first+2] = 2*relative
        row[2*second:2*second+2] = -2*relative
        rows.append(row)
        lower.append(-2*(relative@relative-.36))
    return np.array(rows), np.array(lower)


def main():
    rng = np.random.default_rng(20260910)
    worst_2d = worst_stationarity = worst_feasibility = worst_complementarity = worst_reduction = 0.
    for _ in range(100):
        matrix = rng.normal(size=(rng.integers(2, 8), 2))
        lower = -rng.uniform(0., 2., len(matrix))
        nominal = rng.normal(size=2)*2
        actual = multi_robot.project_many(nominal, matrix, lower)
        expected = independent_2d(nominal, matrix, lower)
        worst_2d = max(worst_2d, float(np.max(np.abs(actual-expected))))
        assert np.max(np.abs(actual-expected)) < 1e-7
    for trial in range(100):
        position = random_safe_positions(rng, 3+trial%3)
        matrix, lower = geometry_constraints(position)
        nominal = rng.normal(size=position.size)*2
        actual = multi_robot.project_many(nominal, matrix, lower)
        matrix, lower = with_box(matrix, lower, position.size)
        slack = matrix@actual-lower
        active = slack < 1e-7
        if active.any():
            multipliers, _ = nnls(matrix[active].T, actual-nominal)
            stationarity = np.max(np.abs(actual-nominal-matrix[active].T@multipliers))
            complementarity = np.max(np.abs(multipliers*slack[active]))
        else:
            stationarity = np.max(np.abs(actual-nominal))
            complementarity = 0.
        worst_stationarity = max(worst_stationarity, float(stationarity))
        worst_feasibility = max(worst_feasibility, float(-slack.min()))
        worst_complementarity = max(worst_complementarity, float(complementarity))
        assert stationarity < 1e-7 and slack.min() >= -1e-8 and complementarity < 1e-7
    for _ in range(30):
        position = random_safe_positions(rng, 2)
        goals = rng.uniform(-3.,3.,(2,2))
        for centralized, circulation in product((False, True), repeat=2):
            old, _ = simulate.control(position, goals, centralized, circulation)
            new, _ = multi_robot.control(position, goals, centralized, circulation)
            worst_reduction = max(worst_reduction, float(np.max(np.abs(old-new))))
            assert np.max(np.abs(old-new)) < 1e-7
    results = dict(seed=20260910, independent_2d_qps=100, max_2d_coordinate_error=worst_2d,
                   geometry_kkt_qps=100, max_stationarity_residual=worst_stationarity,
                   max_feasibility_violation=worst_feasibility, max_complementarity_residual=worst_complementarity,
                   two_robot_reduction_cases=120, max_reduction_coordinate_error=worst_reduction)
    output = multi_robot.OUT
    if (output/'metrics.csv').exists():
        rows = list(csv.DictReader((output/'metrics.csv').open()))
        audits = []
        for row in rows:
            path = output/f"{row['robots']}_{row['controller']}.npz"
            trace = np.load(path)
            position, velocity, dt = trace['positions'], trace['velocities'], float(trace['dt'])
            minimum = float('inf')
            for first,second in combinations(range(int(row['robots'])),2):
                relative = position[:-1,first]-position[:-1,second]
                change = position[1:,first]-position[1:,second]-relative
                fraction = np.clip(-np.einsum('ij,ij->i',relative,change)/np.maximum(np.einsum('ij,ij->i',change,change),1e-30),0.,1.)
                minimum = min(minimum, float(np.linalg.norm(relative+fraction[:,None]*change,axis=1).min()))
            assert minimum >= .6-1e-8
            assert abs(minimum-float(row['min_separation_m'])) < 1e-8
            assert np.max(np.abs(np.diff(position,axis=0)-dt*velocity)) < 1e-12
            completed = np.linalg.norm(position[-1]-trace['goals'],axis=1).max() <= simulate.GOAL_TOL
            assert completed == (row['completed']=='True')
            audits.append(dict(robots=int(row['robots']), controller=row['controller'], independent_min_separation=minimum, completed=bool(completed)))
        assert len(audits)==12
        results['saved_trajectory_audits'] = audits
    else:
        results['saved_trajectory_audits'] = 'Pending: metrics.csv not available yet'
    output.mkdir(parents=True,exist_ok=True)
    (output/'verification.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))


if __name__ == '__main__':
    main()
