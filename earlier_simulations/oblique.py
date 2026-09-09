"""Run three oblique encounters through the existing, unchanged controllers.

Produce 12 trajectory archives, a CSV and a comparison figure. Angles refer to
goal-path directions, not necessarily instantaneous clipped velocities.
Edit main() here to change scenarios; simulate.py provides controller support.
Example: 120 degrees means A travels east and B travels northwest across A's path.
"""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulate import run, OUT, DT


def main():
    configurations = [(True, False), (True, True), (False, False), (False, True)]
    records = []
    fig, axes = plt.subplots(3, 4, figsize=(15, 11), constrained_layout=True)
    for row, angle in enumerate((30, 120, 150)):
        heading = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
        starts = np.array([[-2., 0.], -2*heading])
        goals = -starts
        for column, (centralized, circulation) in enumerate(configurations):
            label = ('centralized' if centralized else 'decentralized') + ('_circulation' if circulation else '_baseline')
            positions, velocities, metrics = run(starts, goals, centralized, circulation)
            records.append(dict(angle_deg=angle, controller=label, **metrics))
            np.savez_compressed(OUT/f'oblique_{angle}_{label}.npz', positions=positions, velocities=velocities, starts=starts, goals=goals, dt=DT)
            axis = axes[row, column]
            for robot, color in enumerate(('#1675be', '#e47720')):
                axis.plot(positions[:,robot,0], positions[:,robot,1], color=color)
                axis.scatter(*starts[robot], color=color, s=25)
                axis.scatter(*goals[robot], color=color, marker='*', s=90)
            axis.set(xlim=(-2.5,2.5), ylim=(-2.5,2.5), aspect='equal', xlabel='x (m)', ylabel='y (m)')
            outcome = f"{metrics['elapsed_s']:.2f} s" if metrics['completed'] else 'unfinished'
            axis.set_title(f"{angle} deg | {label.replace('_', ' ')}\n{outcome}; min gap {metrics['min_separation_m']:.3f} m", fontsize=9)
            axis.grid(alpha=.2)
    fig.suptitle('Oblique encounters | circles: starts, stars: goals | center separation >= 0.600 m')
    fig.savefig(OUT/'oblique_paths.png', dpi=160)
    plt.close(fig)
    with (OUT/'oblique_metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    for record in records:
        print(record)


if __name__ == '__main__':
    main()
