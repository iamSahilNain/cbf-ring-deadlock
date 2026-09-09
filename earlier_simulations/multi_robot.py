"""Run 3/4/5-robot antipodal goal exchanges, each with four controllers.

Contract: produce trajectories, pairwise-safety metrics and a comparison plot.
This extends the earlier heuristic; it does not reproduce a published controller
or claim general deadlock freedom. Example: four robots on a radius-2 circle
each aim at the point opposite their start, creating a shared central conflict.
Edit scenarios in main(), and the multi-robot formulation in reference/control.
simulate.py supplies unchanged constants; verify_multi.py is support-only.
"""
from itertools import combinations
from pathlib import Path
import csv
import json
from datetime import datetime
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
from simulate import SAFE_DISTANCE, ALPHA, BOUND, DT, HORIZON, GOAL_TOL

OUT = Path(__file__).parent / 'results' / 'multi_robot'


def project_many(nominal, matrix, lower):
    """One joint convex QP; never apply independent sequential safety clips."""
    clipped = np.clip(nominal, -BOUND, BOUND)
    if np.min(matrix @ clipped - lower) >= 0:
        return clipped
    result = minimize(lambda velocity: .5*np.sum((velocity-nominal)**2),
                      np.zeros_like(nominal), jac=lambda velocity: velocity-nominal,
                      method='SLSQP', bounds=Bounds(-BOUND, BOUND),
                      constraints=[LinearConstraint(matrix, lower, np.inf)],
                      options={'ftol': 1e-11, 'maxiter': 100})
    if not result.success:
        raise RuntimeError(f'QP solver failed: {result.message}')
    if np.min(matrix @ result.x-lower) < -1e-8 or np.max(np.abs(result.x)) > BOUND+1e-8:
        raise RuntimeError('QP returned constraint violation')
    return result.x


def reference(position, goals, circulation):
    error = goals-position
    nominal = np.clip(error, -BOUND, BOUND)
    if circulation:
        for robot in range(len(position)):
            goal_distance = np.linalg.norm(error[robot])
            for neighbor in range(len(position)):
                if robot == neighbor:
                    continue
                toward_other = position[neighbor]-position[robot]
                distance = np.linalg.norm(toward_other)
                toward_other = toward_other/distance
                alignment = max(0., np.dot(error[robot], toward_other)/max(goal_distance, 1e-12))
                strength = 1.2*max(0., 1-distance/2)*alignment*min(1., goal_distance/.3)
                nominal[robot] += strength*np.array([-toward_other[1], toward_other[0]])
    return nominal


def control(position, goals, centralized, circulation):
    count = len(position)
    pairs = list(combinations(range(count), 2))
    matrix = np.zeros((len(pairs), 2*count))
    lower = np.empty(len(pairs))
    for row, (first, second) in enumerate(pairs):
        relative = position[first]-position[second]
        matrix[row,2*first:2*first+2] = 2*relative
        matrix[row,2*second:2*second+2] = -2*relative
        lower[row] = -ALPHA*(relative@relative-SAFE_DISTANCE**2)
    nominal = reference(position, goals, circulation)
    if centralized:
        velocity = project_many(nominal.ravel(), matrix, lower).reshape(count,2)
    else:
        velocity = np.empty_like(position)
        for robot in range(count):
            incident = [row for row, pair in enumerate(pairs) if robot in pair]
            local_matrix = matrix[incident,2*robot:2*robot+2]
            velocity[robot] = project_many(nominal[robot], local_matrix, lower[incident]/2)
    return velocity, float(np.min(matrix@velocity.ravel()-lower))


def run(starts, goals, centralized, circulation, dt=DT, horizon=HORIZON):
    position = np.array(starts, dtype=float)
    goals = np.array(goals, dtype=float)
    count = len(position)
    pairs = list(combinations(range(count),2))
    min_separation = min(np.linalg.norm(position[first]-position[second]) for first,second in pairs)
    if min_separation < SAFE_DISTANCE-1e-9:
        raise ValueError('Unsafe initial positions')
    positions, velocities = [position.copy()], []
    min_residual, max_speed = float('inf'), 0.
    streak, longest = 0, 0
    for _ in range(round(horizon/dt)):
        if np.max(np.linalg.norm(goals-position,axis=1)) <= GOAL_TOL:
            break
        velocity, residual = control(position, goals, centralized, circulation)
        min_residual = min(min_residual,residual)
        speed = float(np.linalg.norm(velocity,axis=1).max())
        max_speed = max(max_speed,speed)
        streak = streak+1 if speed < .001 else 0
        longest = max(longest,streak)
        for first,second in pairs:
            relative = position[first]-position[second]
            change = dt*(velocity[first]-velocity[second])
            fraction = np.clip(-relative@change/max(change@change,1e-30),0,1)
            min_separation = min(min_separation,float(np.linalg.norm(relative+fraction*change)))
        position = position+dt*velocity
        positions.append(position.copy())
        velocities.append(velocity.copy())
    positions = np.array(positions)
    velocities = np.array(velocities).reshape(-1,count,2)
    errors = np.linalg.norm(goals-position,axis=1)
    metrics = dict(completed=bool(errors.max()<=GOAL_TOL), robots_at_goal=int((errors<=GOAL_TOL).sum()),
                   elapsed_s=len(velocities)*dt, min_separation_m=float(min_separation),
                   final_max_goal_error_m=float(errors.max()), longest_system_stall_s=longest*dt,
                   stalled_3s=bool(longest*dt>=3), max_speed_mps=max_speed,
                   min_cbf_residual=min_residual if len(velocities) else None,
                   total_path_m=float(np.linalg.norm(np.diff(positions,axis=0),axis=2).sum()))
    assert min_separation >= SAFE_DISTANCE-1e-8, metrics
    assert max_speed <= 1+1e-8, metrics
    assert min_residual >= -1e-8, metrics
    return positions, velocities, metrics


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    OUT.mkdir(exist_ok=True)
    records, audits = [], []
    fig, axes = plt.subplots(3,4,figsize=(15,11),constrained_layout=True)
    configurations = [(True,False),(True,True),(False,False),(False,True)]
    for row,count in enumerate((3,4,5)):
        angles = np.arange(count)*2*np.pi/count
        starts = 2*np.column_stack((np.cos(angles),np.sin(angles)))
        goals = -starts
        for column,(centralized,circulation) in enumerate(configurations):
            label = ('centralized' if centralized else 'decentralized')+('_circulation' if circulation else '_baseline')
            started_at = datetime.now().astimezone().isoformat()
            positions,velocities,metrics = run(starts,goals,centralized,circulation)
            records.append(dict(robots=count,controller=label,**metrics))
            audits.append(dict(robots=count,controller=label,started_at=started_at,finished_at=datetime.now().astimezone().isoformat()))
            np.savez_compressed(OUT/f'{count}_{label}.npz',positions=positions,velocities=velocities,starts=starts,goals=goals,dt=DT)
            axis = axes[row,column]
            for robot in range(count):
                color = f'C{robot}'
                axis.plot(positions[:,robot,0],positions[:,robot,1],color=color,lw=1.5)
                axis.scatter(*starts[robot],color=color,s=20)
                axis.scatter(*goals[robot],color=color,marker='*',s=75)
                axis.add_patch(plt.Circle(positions[-1,robot],.25,color=color,alpha=.18))
            outcome = f"{metrics['elapsed_s']:.2f} s" if metrics['completed'] else f"unfinished ({metrics['robots_at_goal']}/{count} at goal)"
            axis.set(title=f"{count} robots | {label.replace('_',' ')}\n{outcome}; min separation {metrics['min_separation_m']:.3f} m",xlim=(-2.6,2.6),ylim=(-2.6,2.6),aspect='equal')
            axis.title.set_fontsize(9)
            axis.grid(alpha=.2)
            print(json.dumps(records[-1]),flush=True)
    fig.suptitle('Antipodal goal exchanges | starts: dots; goals: stars; final robots: shaded disks')
    fig.savefig(OUT/'paths.png',dpi=160)
    plt.close(fig)
    with (OUT/'metrics.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    (OUT/'run_events.json').write_text(json.dumps(audits,indent=2))


if __name__=='__main__':
    main()
