"""Package every saved trajectory into an offline HTML replay and comparison page.

Run this file to produce simulations.html beside it. Existing simulation data are
support inputs; they are not modified. Edit PAGE below to change the presentation.
"""
from pathlib import Path
import base64
import csv
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'


def build():
    metrics = {}
    for filename, key, prefix in [('metrics.csv', 'scenario', ''),
                                  ('oblique_metrics.csv', 'angle_deg', 'oblique_'),
                                  ('multi_robot/metrics.csv', 'robots', 'multi_')]:
        for row in csv.DictReader((RESULTS / filename).open()):
            scenario = prefix + row.pop(key)
            controller = row.pop('controller')
            metrics[scenario, controller] = {
                field: value == 'True' if value in ('True', 'False') else float(value)
                for field, value in row.items()}
    runs = []
    manifest_path = RESULTS / 'paper_cases/manifest.json'
    extras = {row['file']: row for row in json.loads(manifest_path.read_text())} if manifest_path.exists() else {}
    for path in sorted(RESULTS.rglob('*.npz')):
        stem = path.stem
        extended = stem.endswith('_extended')
        if extended:
            stem = stem.removesuffix('_extended')
        scenario, architecture, remedy = stem.rsplit('_', 2)
        if path.parent.name == 'multi_robot':
            scenario = 'multi_' + scenario
        controller = architecture + '_' + remedy
        extra = extras.get(path.name) if path.parent.name == 'paper_cases' else None
        if extra:
            scenario, controller, record = extra['scenario'], extra['controller'], extra['metrics']
        else:
            record = metrics[scenario, controller] if not extended else json.loads(
                (RESULTS / 'multi_robot/extension.json').read_text())
        if controller.endswith('decentralized_baseline'):
            continue
        with np.load(path) as archive:
            positions, velocities = archive['positions'], archive['velocities']
            assert positions.shape[0] == velocities.shape[0] + 1
            assert np.isfinite(positions).all() and np.isfinite(velocities).all()
            dt = float(archive['dt'])
            assert abs((len(positions) - 1) * dt - record['elapsed_s']) < 1e-8
            encode = lambda array: base64.b64encode(np.asarray(array, dtype='<f8').tobytes()).decode()
            runs.append(dict(id=path.stem, scenario=scenario + ('_extended' if extended else ''),
                             controller=controller, n=positions.shape[1], frames=len(positions),
                             dt=dt, positions=encode(positions), velocities=encode(velocities),
                             goals=archive['goals'].tolist(), metrics=record,
                             title=extra['title'] if extra else None, note=extra['note'] if extra else None))
    payload = json.dumps(runs, separators=(',', ':'), allow_nan=False)
    output = ROOT / 'simulations.html'
    output.write_text(PAGE.replace('__RUN_DATA__', payload).replace('45 runs ·', f'{len(runs)} runs ·'))
    print(f'{output}: {len(runs)} runs, {output.stat().st_size:,} bytes')


PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CBF robot simulations · BTP</title>
<style>
:root{color-scheme:light;--ink:#17332f;--muted:#60736f;--green:#087c68;--line:#d9e3de}*{box-sizing:border-box}body{margin:0;background:#f3f5f0;color:var(--ink);font:15px/1.5 system-ui,sans-serif}main{max-width:1440px;margin:auto;padding:34px 28px}header{display:flex;justify-content:space-between;align-items:start;gap:24px}h1{font-size:clamp(26px,4vw,42px);letter-spacing:-1.5px;margin:3px 0 8px}h2{font-size:21px;margin:0 0 12px}p{margin:8px 0 16px}.eyebrow{color:var(--green);font-size:12px;letter-spacing:2px;font-weight:750}.muted,small{color:var(--muted)}.pill{background:#e2eee6;padding:7px 12px;border-radius:20px;white-space:nowrap;font-size:12px}.toolbar,.panel,.section{background:#fff;border:1px solid var(--line);border-radius:16px}.toolbar{padding:18px;display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin:22px 0 12px}label{font-size:13px;font-weight:600}select,button{font:inherit;border:1px solid #becfc7;background:#fff;border-radius:8px;padding:8px 12px;color:var(--ink)}select{max-width:100%}button{cursor:pointer}button:hover{background:#edf4ef}button.primary{background:var(--green);color:white;border-color:var(--green);min-width:78px}button:focus-visible,select:focus-visible,input:focus-visible{outline:3px solid #dc9233;outline-offset:3px}.transport{display:flex;gap:10px;align-items:center;flex:1;min-width:260px}.transport input{flex:1;min-width:70px;accent-color:var(--green)}#clock{font-variant-numeric:tabular-nums;min-width:108px;font-size:13px}.options{display:flex;gap:18px;flex-wrap:wrap}.scenario-note{min-height:24px;margin:10px 3px 16px;color:var(--muted)}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.panel{padding:16px;overflow:hidden}.panel-head{display:flex;justify-content:space-between;align-items:start;gap:8px}.panel h3{font-size:16px;margin:0}.status{font-size:11px;padding:4px 8px;border-radius:12px;background:#f6ead5;color:#81540b;white-space:nowrap}.status.done{background:#e1f2e9;color:#086543}.scene{display:block;width:100%;aspect-ratio:1.55}.chart{display:block;width:100%;height:94px;margin-top:8px}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;border-top:1px solid var(--line);padding-top:12px}.stats span{font-size:11px;color:var(--muted);display:block}.stats strong{font-size:16px;font-weight:650}.legend{font-size:12px;color:var(--muted);margin:12px 0}.section{padding:22px;margin-top:22px}.scroll{overflow:auto;max-height:510px}table{width:100%;border-collapse:collapse;font-size:13px;white-space:nowrap}th,td{text-align:left;padding:10px;border-bottom:1px solid var(--line)}th{position:sticky;top:0;background:#f3f7f4}tr:hover td{background:#f3f7f4}.formula{background:#f4f7f4;padding:12px 16px;border-radius:8px;font-family:ui-monospace,monospace;overflow:auto;font-size:13px}.columns{display:grid;grid-template-columns:1fr 1fr;gap:24px}summary{cursor:pointer;font-weight:650}footer{margin:24px 0;font-size:12px;color:var(--muted)}@media(max-width:760px){main{padding:20px 12px}.grid,.columns{grid-template-columns:1fr}header{display:block}header .pill{display:inline-block;margin-top:8px}.toolbar{gap:12px}.transport{flex-basis:100%}.stats strong{font-size:14px}.panel-head{flex-wrap:wrap}}
</style></head><body><main>
<header><div><div class="eyebrow">BTP / MULTIROBOT COORDINATION</div><h1>When safe robots get stuck.</h1><p class="muted">Compare CBF controllers, watch circulation change the paths, and inspect the selected runs.</p></div><span class="pill">45 runs · 2–5 robots · offline</span></header>
<div class="toolbar"><label>Scenario<br><select id="scenario" aria-label="Scenario"></select></label><div class="transport"><button id="play" class="primary">Play</button><button id="reset" aria-label="Restart playback">↺</button><input id="time" aria-label="Simulation time" type="range" min="0" step="0.02" value="0"><span id="clock"></span></div><label>Speed<br><select id="speed"><option value="0.5">0.5×</option><option value="1">1×</option><option value="2" selected>2×</option><option value="4">4×</option></select></label><div class="options"><label><input type="checkbox" id="paths" checked> Full paths</label><label><input type="checkbox" id="arrows" checked> Velocity arrows</label></div></div>
<p id="note" class="scenario-note"></p><div id="panels" class="grid"></div>
<p class="legend">Colors identify robots. × = goal; small hollow circle = start; disk radius = 0.30 m. Arrows show applied velocity. Plots share a spatial scale. Completed runs remain at their last recorded frame.</p>
<section class="section"><div class="panel-head"><h2>All results</h2><button id="csv">Download metrics CSV</button></div><p class="muted">Outcomes describe the recorded horizon. An unfinished run is not proof of permanent deadlock.</p><div class="scroll"><table><thead><tr><th>Scenario</th><th>Controller</th><th>Outcome</th><th>End time (s)</th><th>Min separation (m)</th><th>Final max error (m)</th><th>Longest stall (s)</th></tr></thead><tbody id="results"></tbody></table></div></section>
<section class="section"><details><summary>Model, circulation mathematics, and interpretation</summary><div class="columns"><div><h3>Safety filter</h3><p>Single-integrator robots: ṗᵢ = uᵢ. The nominal goal velocity is componentwise clipped to ±1/√2 m/s. The QP minimizes squared deviation from the reference, with the same component limits.</p><div class="formula">hᵢⱼ = ‖pᵢ − pⱼ‖² − 0.6²<br>Centralized: 2(pᵢ−pⱼ)·(uᵢ−uⱼ) ≥ −2hᵢⱼ<br>Decentralized: 2(pᵢ−pⱼ)·uᵢ ≥ −hᵢⱼ</div><p>The decentralized constraints split responsibility equally between the two endpoints. All relevant pair constraints enter each QP together.</p><p>Updates: 0.02 s; goal tolerance: 0.05 m for every robot. Minimum separation metrics account for motion between samples. The little charts display sampled separation.</p></div><div><h3>Circulation reference</h3><div class="formula">eᵢⱼ = (pⱼ − pᵢ) / dᵢⱼ; J(x,y) = (−y,x)<br>cᵢ = Σⱼ βᵢⱼ J eᵢⱼ<br>βᵢⱼ = 1.2 max(0,1−dᵢⱼ/2)<br> × max(0, ĝᵢ·eᵢⱼ) × min(1,‖gᵢ−pᵢ‖/0.3)<br>referenceᵢ = clipped(gᵢ−pᵢ) + cᵢ</div><p>ĝᵢ is the unit direction toward robot i’s goal (zero at the goal). J rotates the neighbor direction by 90° counterclockwise; in a head-on encounter this requests each robot’s own left. The gain uses goal alignment, distance, and a taper near the goal.</p><p>The QP can alter this requested motion. With several neighbors, summed circulation need not be tangent to every pair and can cancel. This is a heuristic, not a general convergence guarantee.</p></div></div><h3>What these experiments establish</h3><p>These are selected open-space cases without obstacles, sensing noise, communication delays, or acceleration constraints. The square speed bounds break rotational invariance. “Stall” means all robot speeds below 0.001 m/s while at least one goal error exceeds 0.05 m.</p><p>The centralized 3-robot baseline is unfinished at 30 s but completes at 32.78 s in the separate extension, after a 12.46 s stall. The extension was allowed up to 60 s and stopped at completion. This distinguishes temporary stalling from a permanent deadlock claim.</p></details></section>
<footer>Self-contained replay of the saved Python simulation outputs. No network access, installation, or server required. Playback speed changes only the display clock; controllers are not re-solved in the browser.</footer>
</main><script id="run-data" type="application/json">__RUN_DATA__</script><script>
'use strict';
const runs=JSON.parse(document.getElementById('run-data').textContent);
const $=id=>document.getElementById(id),colors=['#1779b5','#dc7037','#8a59b5','#139b80','#c54e78'];
const names={head_on:'2 robots · Head-on swap',crossing_90:'2 robots · 90° crossing',crossing_45:'2 robots · 45° crossing',crossing_delayed:'2 robots · Delayed crossing',offset:'2 robots · Offset encounter',oblique_30:'2 robots · Oblique 30°',oblique_120:'2 robots · Oblique 120°',oblique_150:'2 robots · Oblique 150°',multi_3:'3 robots · Antipodal goals',multi_4:'4 robots · Antipodal goals',multi_5:'5 robots · Antipodal goals',multi_3_extended:'3 robots · Extended baseline'};
for(const r of runs)if(r.title)names[r.scenario]=r.title;
const controllerName=c=>c.replace('paper_','Paper QP · ').replace('decentralized','Decentralized').replace(/^centralized/,'Centralized').replace('_baseline',' · Baseline').replace('_circulation',' · Our circulation');
const order=['centralized_baseline','centralized_circulation','decentralized_circulation','paper_decentralized_circulation'];
function decode(s){const bytes=Uint8Array.from(atob(s),c=>c.charCodeAt(0));const view=new DataView(bytes.buffer);return Float64Array.from({length:bytes.length/8},(_,i)=>view.getFloat64(i*8,true));}
for(const r of runs){r.p=decode(r.positions);r.v=decode(r.velocities);r.separation=[];for(let f=0;f<r.frames;f++){let minimum=Infinity;for(let i=0;i<r.n;i++)for(let j=i+1;j<r.n;j++)minimum=Math.min(minimum,Math.hypot(r.p[(f*r.n+i)*2]-r.p[(f*r.n+j)*2],r.p[(f*r.n+i)*2+1]-r.p[(f*r.n+j)*2+1]));r.separation.push(minimum);}}
for(const [id,label] of Object.entries(names)){if(runs.some(r=>r.scenario===id))$('scenario').add(new Option(label,id));}
let current=[],time=0,duration=30,playing=false,last=0,bounds;
function stall(r){return r.metrics.longest_system_stall_s??r.metrics.longest_stall_s;}
function choose(){playing=false;$('play').textContent='Play';time=0;current=runs.filter(r=>r.scenario===$('scenario').value).sort((a,b)=>order.indexOf(a.controller)-order.indexOf(b.controller));duration=Math.max(...current.map(r=>r.metrics.elapsed_s));$('time').max=duration;
let xs=[],ys=[];for(const r of current){for(let i=0;i<r.p.length;i+=2){xs.push(r.p[i]);ys.push(r.p[i+1]);}for(const p of r.goals){xs.push(p[0]);ys.push(p[1]);}}
const minx=xs.reduce((a,b)=>Math.min(a,b),Infinity),maxx=xs.reduce((a,b)=>Math.max(a,b),-Infinity),miny=ys.reduce((a,b)=>Math.min(a,b),Infinity),maxy=ys.reduce((a,b)=>Math.max(a,b),-Infinity);bounds={cx:(minx+maxx)/2,cy:(miny+maxy)/2,span:Math.max(maxx-minx,maxy-miny)+1.4};
$('note').textContent=current[0].note || ($('scenario').value==='multi_3_extended'?'Extended run: completes at 32.78 s after a 12.46 s stall. Only the centralized baseline was extended.':$('scenario').value.startsWith('multi_')?'Equally spaced starts on a radius-2 m circle, with opposite goals. Primary horizon: 30 s.':'Two robots with intersecting or nearby goal paths. Compare the same initial conditions across the displayed controllers.');
$('panels').innerHTML=current.map(r=>`<article class="panel"><div class="panel-head"><h3>${controllerName(r.controller)}</h3><span class="status ${r.metrics.completed?'done':''}">${r.metrics.completed?'Recorded: reached goals':'Recorded: unfinished'}</span></div><canvas class="scene" aria-label="${controllerName(r.controller)} robot trajectory replay"></canvas><div class="stats"><div><span>Recorded end</span><strong>${r.metrics.elapsed_s.toFixed(2)} s</strong></div><div><span>Min separation</span><strong>${r.metrics.min_separation_m.toFixed(3)} m</strong></div><div><span>Longest stall</span><strong>${stall(r).toFixed(2)} s</strong></div></div><canvas class="chart" aria-label="Minimum pair separation over time"></canvas><small class="live"></small></article>`).join('');draw();}
function context(canvas){const w=canvas.clientWidth,h=canvas.clientHeight,dpr=window.devicePixelRatio||1;if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);}const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);return [ctx,w,h];}
function draw(){ $('time').value=time;$('clock').textContent=`${time.toFixed(2)} / ${duration.toFixed(2)} s`;document.querySelectorAll('.panel').forEach((panel,index)=>{const r=current[index],[ctx,w,h]=context(panel.querySelector('.scene')),scale=Math.max(1,Math.min(w,h)-32)/bounds.span,X=x=>w/2+(x-bounds.cx)*scale,Y=y=>h/2-(y-bounds.cy)*scale;
ctx.strokeStyle='#e7ede7';ctx.lineWidth=1;ctx.font='10px system-ui';ctx.fillStyle='#7e8c85';for(let k=Math.floor(bounds.cx-bounds.span/2);k<=bounds.cx+bounds.span/2;k++){ctx.beginPath();ctx.moveTo(X(k),12);ctx.lineTo(X(k),h-12);ctx.stroke();ctx.fillText(k+'',X(k)+3,h-7);}for(let k=Math.floor(bounds.cy-bounds.span/2);k<=bounds.cy+bounds.span/2;k++){ctx.beginPath();ctx.moveTo(12,Y(k));ctx.lineTo(w-12,Y(k));ctx.stroke();ctx.fillText(k+'',13,Y(k)-3);}ctx.fillText('meters',w-45,h-7);
const sample=Math.min(time/r.dt,r.frames-1),frame=Math.min(Math.floor(sample),r.frames-1),next=Math.min(frame+1,r.frames-1),fraction=sample-frame;
for(let robot=0;robot<r.n;robot++){const color=colors[robot],offset=(frame*r.n+robot)*2,nextOffset=(next*r.n+robot)*2,x=r.p[offset]*(1-fraction)+r.p[nextOffset]*fraction,y=r.p[offset+1]*(1-fraction)+r.p[nextOffset+1]*fraction;ctx.strokeStyle=color;ctx.lineWidth=1.5;ctx.globalAlpha=.35;ctx.beginPath();const limit=$('paths').checked?r.frames-1:frame;for(let f=0;f<=limit;f++){const off=(f*r.n+robot)*2;f?ctx.lineTo(X(r.p[off]),Y(r.p[off+1])):ctx.moveTo(X(r.p[off]),Y(r.p[off+1]));}ctx.stroke();ctx.globalAlpha=1;const gx=X(r.goals[robot][0]),gy=Y(r.goals[robot][1]);ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(gx-5,gy-5);ctx.lineTo(gx+5,gy+5);ctx.moveTo(gx-5,gy+5);ctx.lineTo(gx+5,gy-5);ctx.stroke();ctx.beginPath();ctx.arc(X(r.p[robot*2]),Y(r.p[robot*2+1]),3,0,2*Math.PI);ctx.stroke();ctx.fillStyle=color+'20';ctx.beginPath();ctx.arc(X(x),Y(y),.3*scale,0,2*Math.PI);ctx.fill();ctx.stroke();ctx.fillStyle=color;ctx.beginPath();ctx.arc(X(x),Y(y),4,0,2*Math.PI);ctx.fill();ctx.font='bold 11px system-ui';ctx.fillText(String(robot+1),X(x)+6,Y(y)-6);
if($('arrows').checked&&frame<r.frames-1){const vx=r.v[offset],vy=r.v[offset+1],dx=vx*scale*.65,dy=-vy*scale*.65;if(Math.hypot(dx,dy)>2){const ex=X(x)+dx,ey=Y(y)+dy,angle=Math.atan2(dy,dx);ctx.beginPath();ctx.moveTo(X(x),Y(y));ctx.lineTo(ex,ey);ctx.lineTo(ex-5*Math.cos(angle-.5),ey-5*Math.sin(angle-.5));ctx.moveTo(ex,ey);ctx.lineTo(ex-5*Math.cos(angle+.5),ey-5*Math.sin(angle+.5));ctx.stroke();}}}
const [cc,cw,ch]=context(panel.querySelector('.chart')),left=36,right=cw-12,top=15,bottom=ch-20,max=Math.max(1,...r.separation)*1.08,cx=t=>left+(right-left)*t/duration,cy=v=>bottom-(bottom-top)*v/max;cc.font='10px system-ui';cc.fillStyle='#60736f';cc.fillText('Pair separation (m)',left,10);cc.fillText('0',left-4,ch-3);cc.fillText(duration.toFixed(1)+' s',right-32,ch-3);cc.fillText('0.60',1,cy(.6)+3);cc.strokeStyle='#c77747';cc.setLineDash([4,3]);cc.beginPath();cc.moveTo(left,cy(.6));cc.lineTo(right,cy(.6));cc.stroke();cc.setLineDash([]);cc.strokeStyle='#087c68';cc.beginPath();r.separation.forEach((v,f)=>f?cc.lineTo(cx(f*r.dt),cy(v)):cc.moveTo(cx(0),cy(v)));cc.stroke();cc.strokeStyle='#243a35';cc.beginPath();cc.moveTo(cx(time),top);cc.lineTo(cx(time),bottom);cc.stroke();panel.querySelector('.live').textContent=`At ${Math.min(time,r.metrics.elapsed_s).toFixed(2)} s · sampled separation ${r.separation[frame].toFixed(3)} m${time>=r.metrics.elapsed_s?' · end of recording':''}`;
});}
const rows=runs.slice().sort((a,b)=>Object.keys(names).indexOf(a.scenario)-Object.keys(names).indexOf(b.scenario)||order.indexOf(a.controller)-order.indexOf(b.controller));
$('results').innerHTML=rows.map(r=>`<tr><td>${names[r.scenario]}</td><td>${controllerName(r.controller)}</td><td>${r.metrics.completed?'Reached goals':'Unfinished'}</td><td>${r.metrics.elapsed_s.toFixed(2)}</td><td>${r.metrics.min_separation_m.toFixed(6)}</td><td>${r.metrics.final_max_goal_error_m.toFixed(4)}</td><td>${stall(r).toFixed(2)}</td></tr>`).join('');
$('csv').onclick=()=>{const data=[['scenario','controller','completed','elapsed_s','min_separation_m','final_max_goal_error_m','longest_stall_s'],...rows.map(r=>[r.scenario,r.controller,r.metrics.completed,r.metrics.elapsed_s,r.metrics.min_separation_m,r.metrics.final_max_goal_error_m,stall(r)])].map(row=>row.join(',')).join('\n');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([data],{type:'text/csv'}));a.download='cbf_all_metrics.csv';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);};
$('scenario').onchange=choose;$('play').onclick=()=>{if(time>=duration)time=0;playing=!playing;$('play').textContent=playing?'Pause':'Play';last=performance.now();draw();};$('reset').onclick=()=>{time=0;draw();};$('time').oninput=()=>{time=Number($('time').value);draw();};$('paths').onchange=draw;$('arrows').onchange=draw;window.addEventListener('resize',draw);
function tick(now){if(playing){time=Math.min(duration,time+(now-last)/1000*Number($('speed').value));if(time>=duration){playing=false;$('play').textContent='Play';}draw();}last=now;requestAnimationFrame(tick);}choose();requestAnimationFrame(tick);
</script></body></html>'''

if __name__ == '__main__':
    build()
