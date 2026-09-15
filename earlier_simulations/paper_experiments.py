"""Reproduce collinear two-robot deadlocks and test a circulation equilibrium.

Produces saved trajectories and a manifest in results/paper_cases. Edit CASES for
geometry; existing simulation outputs remain unchanged. Example: overshoot goals
lie beyond the other robot's initial position, satisfying D_Gi > D_init.
"""
from pathlib import Path
import json
import numpy as np
from simulate import project, BOUND, SAFE_DISTANCE, ALPHA, GOAL_TOL, control as original

OUT = Path(__file__).parent / 'results/paper_cases'
SIDE = float(np.sqrt((-.5**2 + np.sqrt(.5**4 + 4*.84**2*.5**2))/2))
rotation = np.array([[np.cos(.63), -np.sin(.63)], [np.sin(.63), np.cos(.63)]])
CASES = [
    ('paper_symmetric', 'Paper · Collinear overshoot', [[-2,0],[2,0]], [[3,0],[-3,0]], [1,1]),
    ('paper_heterogeneous', 'Paper · Unequal gains and goals', [[-2,0],[2,0]], [[3,0],[-4,0]], [.15,.08]),
    ('paper_rotated', 'Paper · Rotated collinear encounter', np.array([[-2,0],[2,0]])@rotation.T,
     np.array([[3,0],[-4,0]])@rotation.T, [.15,.08]),
    ('circulation_equilibrium', 'Circulation · Constructed equilibrium', [[-.3,0],[.3,0]],
     [[.2,-SIDE],[-.2,SIDE]], [1,1]),
]


def controller(p, goals, gains, central, circulation, bounded):
    error = goals-p
    reference = gains[:,None]*error
    if bounded:
        reference = np.clip(reference,-BOUND,BOUND)
    relative=p[0]-p[1]
    distance=np.linalg.norm(relative)
    if circulation:
        for robot in range(2):
            toward=(p[1-robot]-p[robot])/distance
            goal_distance=np.linalg.norm(error[robot])
            strength=1.2*max(0.,1-distance/2)*max(0.,error[robot]@toward/max(goal_distance,1e-12))*min(1.,goal_distance/.3)
            reference[robot]+=strength*np.array([-toward[1],toward[0]])
    normals=np.array([2*relative,-2*relative])
    lower=-ALPHA*(distance**2-SAFE_DISTANCE**2)
    def projection(ref,normal,rhs):
        if bounded:
            return project(ref,normal,rhs)
        return ref+max(0.,(rhs-normal@ref)/(normal@normal))*normal
    u=projection(reference.ravel(),normals.ravel(),lower).reshape(2,2) if central else np.array([
        projection(reference[i],normals[i],lower/2) for i in range(2)])
    return u,float(np.sum(normals*u)-lower)


def run(starts,goals,gains,central,circulation,bounded,dt=.02,horizon=60):
    p=np.array(starts,dtype=float);goals=np.array(goals,dtype=float);gains=np.array(gains)
    positions=[p.copy()];velocities=[];residuals=[]
    for _ in range(round(horizon/dt)):
        if np.linalg.norm(goals-p,axis=1).max()<=GOAL_TOL:break
        u,residual=controller(p,goals,gains,central,circulation,bounded)
        p=p+dt*u;positions.append(p.copy());velocities.append(u);residuals.append(residual)
    positions=np.array(positions);velocities=np.array(velocities)
    relative=positions[:-1,0]-positions[:-1,1];change=dt*(velocities[:,0]-velocities[:,1])
    fraction=np.clip(-np.sum(relative*change,axis=1)/np.maximum(np.sum(change**2,axis=1),1e-30),0,1)
    minimum=float(np.linalg.norm(relative+fraction[:,None]*change,axis=1).min())
    errors=np.linalg.norm(positions-goals,axis=2);speeds=np.linalg.norm(velocities,axis=2)
    streak=longest=0
    for stopped in (speeds.max(axis=1)<.001)&(errors[:-1].max(axis=1)>GOAL_TOL):
        streak=streak+1 if stopped else 0;longest=max(longest,streak)
    m=dict(completed=bool(errors[-1].max()<=GOAL_TOL),elapsed_s=len(velocities)*dt,
           min_separation_m=minimum,final_max_goal_error_m=float(errors[-1].max()),
           longest_stall_s=longest*dt,max_speed_mps=float(speeds.max()),min_cbf_residual=min(residuals),
           total_path_m=float(np.linalg.norm(np.diff(positions,axis=0),axis=2).sum()))
    assert minimum>=SAFE_DISTANCE-1e-9 and min(residuals)>=-1e-9
    if bounded:assert np.abs(velocities).max()<=BOUND+1e-9
    return positions,velocities,m


def main():
    OUT.mkdir(exist_ok=True)
    # Check that the bounded gain-1 implementation matches the original controller.
    rng=np.random.default_rng(19);maximum=0.
    for _ in range(100):
        p=rng.uniform(-2,2,(2,2));goals=rng.uniform(-2,2,(2,2))
        if np.linalg.norm(p[0]-p[1])<.6:continue
        for central in (False,True):
            for circulation in (False,True):
                maximum=max(maximum,float(np.abs(controller(p,goals,np.ones(2),central,circulation,True)[0]-original(p,goals,central,circulation)[0]).max()))
    assert maximum<1e-12
    manifest=[]
    for scenario,title,starts,goals,gains in CASES:
        variants=[(c,r,True) for c in (True,False) for r in (False,True)]
        if scenario.startswith('paper_'):variants += [(False,r,False) for r in (False,True)]
        for central,circulation,bounded in variants:
            label=('centralized' if central else 'decentralized')+'_'+('circulation' if circulation else 'baseline')
            if not bounded:label='paper_'+label
            p,u,m=run(starts,goals,gains,central,circulation,bounded)
            filename=f'{scenario}_{label}.npz'
            np.savez_compressed(OUT/filename,positions=p,velocities=u,goals=goals,dt=.02)
            note=f'60 s maximum horizon; goal gains {gains}. '
            note+=('Collinear starts; initial separation 4 m; both goal travel distances strictly exceed 4 m. Paper variants use unclipped proportional input and no speed bounds; other variants retain our square speed bounds.' if scenario.startswith('paper_') else 'Analytically constructed equilibrium of our circulation controller. Floating-point perturbations can cause eventual escape; a completed numerical trace does not remove the exact equilibrium.')
            manifest.append(dict(file=filename,scenario=scenario,title=title,controller=label,note=note,metrics=m))
    perturb=[]
    _,_,starts,goals,gains=CASES[-1]
    for offset in [-1e-4,0,1e-4]:
        p=np.array(starts,dtype=float);p[:,1]+=[offset,-offset]
        for dt in [.02,.01]:
            _,_,m=run(p,goals,gains,False,True,True,dt=dt)
            perturb.append(dict(y_offset=offset,dt=dt,**m))
    equilibrium_u=controller(np.array(starts),np.array(goals),np.ones(2),False,True,True)[0]
    audit=dict(original_controller_max_difference=maximum,equilibrium_side=SIDE,
               equilibrium_velocity_max_abs=float(np.abs(equilibrium_u).max()),
               goal_separation=float(np.linalg.norm(np.array(goals)[0]-goals[1])),perturbations=perturb)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (OUT/'verification.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(dict(runs=len(manifest),audit=audit,results=[dict(scenario=r['scenario'],controller=r['controller'],**r['metrics']) for r in manifest]),indent=2))


if __name__=='__main__':main()
