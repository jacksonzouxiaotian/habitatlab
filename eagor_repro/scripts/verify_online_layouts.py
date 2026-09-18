"""Closed-loop known-geometry fixtures; planner receives only simulated ERP depth.

Ground-truth rectangles belong to this sensor/motion fixture, never the mapper.
They are not MP3D tasks and do not contribute to MP3D success rates.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
from eagor_repro.planning.online_mapper import erp_rays
from eagor_repro.planning.online_navigation import OnlineNavigator
from eagor_repro.planning.frontier_planner import DirectionCue
from eagor_repro.evaluation.online_visualization import map_panel


def rotation(yaw):
    return np.array([[np.cos(yaw),-np.sin(yaw),0.],[0.,0.,1.],[np.sin(yaw),np.cos(yaw),0.]])


def render_depth(position, rot, rectangles):
    _,_,body=erp_rays(64,128)
    rays=body@rot.T
    origin=np.asarray(position)+[0,.88,0]
    depth=np.full((64,128),10.)
    down=rays[...,1]<-1e-6
    depth[down]=np.minimum(depth[down],-.88/rays[...,1][down])
    for x0,z0,x1,z1 in rectangles:
        lo=np.array([x0,0,z0]);hi=np.array([x1,2.,z1])
        direction=np.where(np.abs(rays)<1e-9,1e-9,rays)
        t1=(lo-origin)/direction;t2=(hi-origin)/direction
        near=np.minimum(t1,t2).max(axis=-1);far=np.maximum(t1,t2).min(axis=-1)
        good=(near>0)&(far>=near)
        depth[good]=np.minimum(depth[good],near[good])
    return depth


def run_layout(name,obstacles,output):
    import cv2
    cfg=OmegaConf.to_container(OmegaConf.load('configs/eagor/mp3d_online.yaml').online,resolve=True)
    # Sampling every column keeps synthetic angular density comparable to MP3D.
    cfg['map'].update(row_stride=1,col_stride=1)
    position=np.array([-3.,0.,0.]);target=np.array([3.,0.]);yaw=0.
    nav=OnlineNavigator(cfg,position)
    boundary=[(-5,-4,5,-3.8),(-5,3.8,5,4),(-5,-4,-4.8,4),(4.8,-4,5,4)]
    rectangles=boundary+obstacles
    collided=False;collisions=0;history=[]
    for step in range(300):
        rot=rotation(yaw)
        delta=target-position[[0,2]]
        if np.linalg.norm(delta)<.3:break
        depth=render_depth(position,rot,rectangles)
        action=nav.act(depth,position,rot,DirectionCue(tuple(delta/np.linalg.norm(delta)),True,1),step,collided)
        collided=False
        history.append(dict(step=step,position=position.tolist(),action=action,**nav.telemetry(step)))
        if action is None:break
        if action=='turn_left':yaw+=np.pi/12
        elif action=='turn_right':yaw-=np.pi/12
        elif action=='move_forward':
            end=position+rot[:,0]*.25
            for q in np.linspace(position,end,12):
                for x0,z0,x1,z1 in rectangles:
                    closest=np.clip(q[[0,2]],[x0,z0],[x1,z1])
                    if np.linalg.norm(q[[0,2]]-closest)<.18:collided=True
            if not collided:position=end
            collisions+=int(collided)
    out=output/name;out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'map.npz',**nav.mapper.snapshot(step))
    cv2.imwrite(str(out/'map.png'),cv2.cvtColor(map_panel(nav,step),cv2.COLOR_RGB2BGR))
    (out/'steps.json').write_text(json.dumps(history))
    (out/'frontier_scores.json').write_text(json.dumps(nav.events))
    result=dict(layout=name,success=bool(np.linalg.norm(target-position[[0,2]])<.3),steps=len(history),collisions=collisions,
                final_distance=float(np.linalg.norm(target-position[[0,2]])),criterion='synthetic goal radius .3 m; not MP3D benchmark',**nav.telemetry(step))
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'config.yaml').write_text(Path('configs/eagor/mp3d_online.yaml').read_text())
    sources=[Path(__file__)]+list(Path('eagor_repro/planning').glob('*.py'))
    (args.output/'source_hashes.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2))
    layouts=dict(open=[],single_obstacle=[(-.5,-.7,.5,.7)],door=[(-.1,-4,.1,1),(-.1,2.2,.1,4)],
                 away_first=[(-2.,-1.,-1.8,1.),(-4.,-1.,-2.,-.8),(-4.,.8,-2.,1.)])
    results=[run_layout(name,boxes,args.output) for name,boxes in layouts.items()]
    (args.output/'summary.json').write_text(json.dumps(results,indent=2))
    assert all(r['success'] and r['collisions']==0 for r in results),results


if __name__=='__main__':main()
