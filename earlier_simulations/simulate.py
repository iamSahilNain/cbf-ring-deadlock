"""Two single-integrator robots: joint / equal-responsibility CBF, with / without circulation."""
from pathlib import Path
import csv
import json
import numpy as np

SAFE_DISTANCE = 0.6
ALPHA = 2.0
BOUND = 1 / np.sqrt(2)  # Inscribed square: Euclidean speed never exceeds 1 m/s.
DT = 0.02
HORIZON = 30.0
GOAL_TOL = 0.05
OUT = Path(__file__).parent / 'results'


def project(nominal, normal, lower, bound=BOUND):
    """Euclidean projection onto a box intersected with normal @ u >= lower."""
    candidate = np.clip(nominal, -bound, bound)
    if np.dot(normal, candidate) >= lower:
        return candidate
    if bound * np.abs(normal).sum() < lower - 1e-10:
        raise ValueError('Infeasible control constraints')
    low, high = 0.0, 1.0
    while np.dot(normal, np.clip(nominal + high * normal, -bound, bound)) < lower:
        high *= 2
        if high > 1e14:
            raise RuntimeError('Projection failed to bracket')
    for _ in range(48):
        middle = (low + high) / 2
        if np.dot(normal, np.clip(nominal + middle * normal, -bound, bound)) < lower:
            low = middle
        else:
            high = middle
    return np.clip(nominal + high * normal, -bound, bound)


def control(position, goals, centralized, circulation):
    error = goals - position
    nominal = np.clip(error, -BOUND, BOUND)
    relative = position[0] - position[1]
    separation = np.linalg.norm(relative)
    barrier = separation**2 - SAFE_DISTANCE**2
    if circulation:
        for robot in range(2):
            toward_other = (position[1-robot] - position[robot]) / separation
            goal_distance = np.linalg.norm(error[robot])
            alignment = max(0.0, np.dot(error[robot], toward_other) / max(goal_distance, 1e-12))
            tangent = np.array([-toward_other[1], toward_other[0]])
            strength = 1.2 * max(0.0, 1 - separation / 2.0) * alignment * min(1.0, goal_distance / 0.3)
            nominal[robot] += strength * tangent
    normal = np.array([2 * relative, -2 * relative])
    if centralized:
        velocity = project(nominal.ravel(), normal.ravel(), -ALPHA * barrier).reshape(2, 2)
    else:
        velocity = np.array([project(nominal[robot], normal[robot], -ALPHA * barrier / 2) for robot in range(2)])
    residual = np.sum(normal * velocity) + ALPHA * barrier
    return velocity, residual


def run(starts, goals, centralized, circulation, dt=DT):
    position = np.array(starts, dtype=float)
    goals = np.array(goals, dtype=float)
    initial_separation = float(np.linalg.norm(position[0] - position[1]))
    if initial_separation < SAFE_DISTANCE - 1e-10:
        raise ValueError('Initial positions violate the safety distance')
    if np.max(np.linalg.norm(goals-position, axis=1)) <= GOAL_TOL:
        return position[None, ...], np.empty((0, 2, 2)), dict(
            completed=True, elapsed_s=0.0, min_separation_m=initial_separation,
            final_max_goal_error_m=float(np.linalg.norm(goals-position, axis=1).max()),
            longest_stall_s=0.0, stalled_3s=False, total_path_m=0.0,
            max_speed_mps=0.0, min_cbf_residual=None)
    positions, velocities, residuals = [position.copy()], [], []
    for _ in range(round(HORIZON / dt)):
        if np.max(np.linalg.norm(goals - position, axis=1)) <= GOAL_TOL:
            break
        velocity, residual = control(position, goals, centralized, circulation)
        position = position + dt * velocity  # Synchronous zero-order-held velocities.
        positions.append(position.copy())
        velocities.append(velocity)
        residuals.append(residual)
    positions, velocities = np.array(positions), np.array(velocities)
    errors = np.linalg.norm(positions - goals, axis=2)
    speeds = np.linalg.norm(velocities, axis=2)
    # Exact minimum separation along each straight segment, including between samples.
    relative = positions[:-1, 0] - positions[:-1, 1]
    change = dt * (velocities[:, 0] - velocities[:, 1])
    fraction = np.clip(-np.sum(relative * change, axis=1) / np.maximum(np.sum(change**2, axis=1), 1e-30), 0, 1)
    minimum = np.linalg.norm(relative + fraction[:, None] * change, axis=1).min()
    stalled = (speeds.max(axis=1) < 0.001) & (errors[:-1].max(axis=1) > GOAL_TOL)
    longest = streak = 0
    for stopped in stalled:
        streak = streak + 1 if stopped else 0
        longest = max(longest, streak)
    metrics = dict(completed=bool(errors[-1].max() <= GOAL_TOL), elapsed_s=len(velocities)*dt,
                   min_separation_m=float(minimum), final_max_goal_error_m=float(errors[-1].max()),
                   longest_stall_s=longest*dt, stalled_3s=bool(longest*dt >= 3),
                   total_path_m=float(np.linalg.norm(np.diff(positions, axis=0), axis=2).sum()),
                   max_speed_mps=float(speeds.max()), min_cbf_residual=float(min(residuals)))
    assert minimum >= SAFE_DISTANCE - 1e-9, metrics
    assert metrics['max_speed_mps'] <= 1 + 1e-9, metrics
    assert metrics['min_cbf_residual'] >= -1e-9, metrics
    return positions, velocities, metrics


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    OUT.mkdir(exist_ok=True)
    scenarios = {
        'head_on': ([[-2, 0], [2, 0]], [[2, 0], [-2, 0]]),
        'offset': ([[-2, 0.05], [2, -0.05]], [[2, 0.05], [-2, -0.05]]),
        'crossing_90': ([[-2, 0], [0, -2]], [[2, 0], [0, 2]]),
        'crossing_delayed': ([[-2, 0], [0, -3]], [[2, 0], [0, 2]]),
        'crossing_45': ([[-2, 0], [-1.4142, -1.4142]], [[2, 0], [1.4142, 1.4142]]),
    }
    configurations = [(True, False), (True, True), (False, False), (False, True)]
    records, head_runs = [], []
    for name, (starts, goals) in scenarios.items():
        for centralized, circulation in configurations:
            label = ('centralized' if centralized else 'decentralized') + ('_circulation' if circulation else '_baseline')
            positions, velocities, metrics = run(starts, goals, centralized, circulation)
            records.append(dict(scenario=name, controller=label, **metrics))
            np.savez_compressed(OUT / f'{name}_{label}.npz', positions=positions, velocities=velocities, goals=goals, dt=DT)
            if name == 'head_on':
                head_runs.append((label, positions, velocities))
    with (OUT / 'metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for axis, (label, positions, velocities) in zip(axes.flat, head_runs):
        for robot, color in enumerate(['#1675be', '#e47720']):
            axis.plot(positions[:, robot, 0], positions[:, robot, 1], color=color, label=f'Robot {robot+1}')
            axis.scatter(*positions[0, robot], color=color, marker='o')
            axis.scatter(*scenarios['head_on'][1][robot], color=color, marker='*', s=160)
            axis.add_patch(plt.Circle(positions[-1, robot], 0.25, color=color, alpha=.25))
        axis.set(title=label.replace('_', ' '), xlabel='x (m)', ylabel='y (m)', xlim=(-2.5,2.5), ylim=(-1,1), aspect='equal')
        axis.grid(alpha=.2)
    fig.suptitle('Head-on swap: circles = starts, stars = goals; shaded disks = final robots')
    fig.savefig(OUT / 'head_on_paths.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    for label, positions, velocities in head_runs:
        times = np.arange(len(positions))*DT
        axes[0].plot(times, np.linalg.norm(positions[:,0]-positions[:,1], axis=1), label=label)
        axes[1].plot(times, np.linalg.norm(positions-np.array(scenarios['head_on'][1]),axis=2).max(axis=1), label=label)
    axes[0].axhline(SAFE_DISTANCE, color='black', linestyle=':', label='Safety distance 0.6 m')
    axes[0].set(ylabel='Separation (m)')
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Time (s)', ylabel='Maximum distance to goal (m)')
    fig.savefig(OUT / 'head_on_metrics.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)
    disks, trails = [], []
    for axis, (label, positions, _) in zip(axes.flat, head_runs):
        axis.set(title=label.replace('_', ' '), xlim=(-2.5,2.5), ylim=(-1,1), aspect='equal')
        for robot, color in enumerate(['#1675be', '#e47720']):
            axis.scatter(*scenarios['head_on'][1][robot], marker='*', color=color, s=90)
            disk = plt.Circle(positions[0,robot], .25, color=color, alpha=.8)
            axis.add_patch(disk)
            disks.append(disk)
            trails.append(axis.plot([], [], color=color, alpha=.4)[0])
    def animate(frame):
        for panel, (_, positions, _) in enumerate(head_runs):
            step = min(frame*10, len(positions)-1)
            for robot in range(2):
                disks[panel*2+robot].center = positions[step,robot]
                trails[panel*2+robot].set_data(positions[:step+1,robot,0], positions[:step+1,robot,1])
        fig.suptitle(f'Head-on swap | t = {frame*10*DT:.1f} s | safety distance = 0.6 m')
        return disks + trails
    animation = FuncAnimation(fig, animate, frames=101, interval=80)
    animation.save(OUT/'head_on.gif', writer=PillowWriter(fps=12))
    plt.close(fig)
    # Step-size sensitivity for the four head-on runs.
    refinement = []
    for centralized, circulation in configurations:
        _, _, metrics = run(*scenarios['head_on'], centralized, circulation, dt=DT/2)
        refinement.append(dict(centralized=centralized, circulation=circulation, **metrics))
    (OUT/'refinement.json').write_text(json.dumps(refinement, indent=2))
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
