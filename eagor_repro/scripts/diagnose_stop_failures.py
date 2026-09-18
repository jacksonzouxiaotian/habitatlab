"""Privileged OFFLINE geometry diagnostics; never imported by online navigation.

Loads frozen sensor/maps and a separate Habitat environment to audit connectivity
and a legal reference path. Does not reroute or edit any evaluated trajectory.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import ndimage
from omegaconf import OmegaConf
from eagor_repro.planning.grid_path_planner import shortest_tree


def reachable(grid,start):
    distance,_=shortest_tree(grid,tuple(start))
    return np.isfinite(distance)


def sample_polyline(points,spacing=.025):
    chunks=[]
    for a,b in zip(points,points[1:]):
        chunks.append(np.linspace(a,b,max(2,int(np.linalg.norm(b-a)/spacing)+1)))
    return np.concatenate(chunks) if chunks else np.asarray(points)


def main():
    import habitat
    import habitat_sim
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();out=args.root/'failure_cases';out.mkdir(exist_ok=False)
    run=args.root/'matrix/oracle_direction__oracle_distance'
    results=json.loads((run/'results.json').read_text())['oracle_direction']
    failures={e['key']:e for e in results if not e['success']}
    cfg=OmegaConf.load(run/'oracle_direction/habitat_config.yaml')
    dataset=habitat.make_dataset(cfg.habitat.dataset.type,config=cfg.habitat.dataset)
    dataset.episodes=[e for e in dataset.episodes if Path(e.scene_id).stem+'_'+str(e.episode_id) in failures]
    audits=[]
    with habitat.Env(config=cfg,dataset=dataset) as env:
        for _ in failures:
            env.reset();episode=env.current_episode;key=Path(episode.scene_id).stem+'_'+str(episode.episode_id)
            source=run/'oracle_direction'/key;dest=out/key;dest.mkdir()
            rows=json.loads((source/'steps.json').read_text());summary=failures[key]
            endpoints=np.array([v.agent_state.position for g in episode.goals for v in g.view_points],dtype=np.float32)
            assert abs(env.get_metrics()['distance_to_goal']-rows[0]['evaluation_distance_to_goal_m'])<1e-4
            def reference(position):
                path=habitat_sim.MultiGoalShortestPath();path.requested_start=np.array(position,dtype=np.float32);path.requested_ends=endpoints
                ok=env.sim.pathfinder.find_path(path)
                return bool(ok),float(path.geodesic_distance),np.array(path.points)
            ok,d0,initial=reference(rows[0]['position'])
            final_position=rows[-1]['position_after'];ok_end,d_end,final=reference(final_position)
            np.savez_compressed(dest/'evaluation_only_navmesh_paths.npz',initial_path=initial,final_path=final,goal_viewpoints=endpoints)
            all_height_deltas=np.abs(endpoints[:,1]-episode.start_position[1])
            audit=dict(key=key,source=str(source),evaluation_only=True,policy_query_count=0,
                separate_environment_initial_distance_matches=True,reference_initial_path_exists=ok,
                reference_initial_distance_m=d0,reference_final_path_exists=ok_end,reference_final_distance_m=d_end,
                reference_initial_y_range_m=float(np.ptp(initial[:,1])) if len(initial) else None,
                closest_viewpoint_height_delta_m=float(all_height_deltas.min()),
                all_viewpoints_outside_start_height_20cm=bool(np.all(all_height_deltas>.2)),
                online_y_range_m=float(np.ptp(np.array([r['position'] for r in rows])[:,1])),
                note='A height-changing feasible reference proves feasibility, not that its exact stair route is necessary. Endpoint height may require some vertical transition; stairs versus ramps remains unverified.',
                summary_counts={k:summary[k] for k in ('astar_calls','astar_failures','invalid_paths','recoveries','state_revisit_count','repeated_backtrack_count','backtrack_selection_count')},snapshots=[])
            snapshots=sorted(source.glob('map_[0-9][0-9][0-9].npz'))
            chosen=sorted(set([0,min(1,len(snapshots)-1),len(snapshots)//2,len(snapshots)-1]))
            for index in chosen:
                file=snapshots[index];step=int(file.stem.split('_')[1]);a=np.load(file)
                depth=np.load(source/f'depth_{step:03d}.npz');row=rows[step]
                origin=a['origin'];res=float(a['resolution']);free=a['free'];safe=a['safe'];occ=a['occupied'];unknown=~a['observed']
                def cells(xz):return np.floor((np.asarray(xz)-origin)/res).astype(int)[...,::-1]
                start=cells(np.array(row['position'])[[0,2]])
                raw_reach=reachable(free,start);safe_reach=reachable(safe,start)
                clearance=ndimage.distance_transform_edt(~occ)*res
                no_extra_margin=free&(clearance>.18+res*.5)
                reduced_reach=reachable(no_extra_margin,start)
                frontier=free&ndimage.binary_dilation(unknown,structure=ndimage.generate_binary_structure(2,1))
                frontier[[0,-1],:]=False;frontier[:,[0,-1]]=False
                approach=ndimage.binary_dilation(frontier,iterations=3)
                ok_now,dist,path=reference(row['position']);dense=sample_polyline(path)
                local=dense[np.linalg.norm(dense-np.array(row['position']),axis=1)<=2.] if len(dense) else dense
                flat=local[np.abs(local[:,1]-row['position'][1])<.1] if len(local) else local
                rc=cells(flat[:,[0,2]]) if len(flat) else np.empty((0,2),int)
                inside=np.all((rc>=0)&(rc<np.array(free.shape)),axis=1);rc=rc[inside]
                sampled={name:int(layer[tuple(rc.T)].sum()) if len(rc) else 0 for name,layer in
                    [('unknown',unknown),('raw_occupied',occ),('free_inflated',free&~safe),('safe',safe)]}
                info=dict(step=step,raw_free_connected_cells=int(raw_reach.sum()),safe_connected_cells=int(safe_reach.sum()),
                    reduced_extra_margin_connected_cells=int(reduced_reach.sum()),raw_frontier_cells=int(frontier.sum()),
                    safe_reachable_frontier_approach_cells=int((approach&safe_reach).sum()),
                    raw_reachable_frontier_approach_cells=int((approach&raw_reach).sum()),
                    reference_path_exists=ok_now,formal_distance_m=dist,
                    local_same_height_reference_samples=len(rc),local_reference_map_status=sampled)
                audit['snapshots'].append(info)
                fig,axs=plt.subplots(2,3,figsize=(15,9))
                rgb=plt.imread(source/f'perspective_{step:03d}.jpg');axs[0,0].imshow(rgb);axs[0,0].set_title('Recorded perspective RGB');axs[0,0].axis('off')
                im=axs[0,1].imshow(depth['radial'],vmin=0,vmax=5,cmap='viridis');axs[0,1].set_title('Recorded radial depth (ERP)');fig.colorbar(im,ax=axs[0,1],shrink=.6)
                extent=[origin[0],origin[0]+free.shape[1]*res,origin[1],origin[1]+free.shape[0]*res]
                raw_image=np.full((*free.shape,3),.5);raw_image[free]=.95;raw_image[occ]=.05
                inflated=raw_image.copy();inflated[free&~safe]=[.65,.45,.6]
                for ax,img,title in [(axs[0,2],raw_image,'Raw occupancy + observed free'),(axs[1,0],inflated,'Inflated map + online route / frontier'),(axs[1,1],safe_reach.astype(float),'Reachable safe component + GT reference (OFFLINE)')]:
                    ax.imshow(img,origin='lower',extent=extent,cmap='gray',vmin=0,vmax=1)
                    trajectory=a['trajectory'];ax.plot(trajectory[:,0],trajectory[:,1],c='green',lw=1)
                    ax.scatter(row['position'][0],row['position'][2],c='red',s=25)
                    ax.set_title(title,fontsize=10);ax.set_aspect('equal')
                route=np.array(row['planned_path_xz'])
                if len(route):axs[1,0].plot(route[:,0],route[:,1],c='blue')
                f_rc=np.argwhere(frontier);f_xz=origin+(f_rc[:,::-1]+.5)*res
                axs[1,0].scatter(f_xz[:,0],f_xz[:,1],c='orange',s=1)
                if row['subgoal'] is not None:axs[1,0].scatter(*row['subgoal'],c='magenta',marker='*',s=50)
                if len(path):axs[1,1].plot(path[:,0],path[:,2],'m--',lw=1,label='navmesh reference EVAL ONLY');axs[1,1].legend(fontsize=7)
                for ax in (axs[0,2],axs[1,0],axs[1,1]):
                    p=np.array(row['position']);ax.set_xlim(p[0]-5,p[0]+5);ax.set_ylim(p[2]-5,p[2]+5)
                axs[1,2].plot([r['step'] for r in rows],[r['position'][1] for r in rows],label='actual Y')
                if len(initial):axs[1,2].axhspan(initial[:,1].min(),initial[:,1].max(),alpha=.2,color='magenta',label='reference Y span, EVAL ONLY')
                events=json.loads((source/'events.json').read_text())
                for e in events:
                    if e['event'] in ('collision_recovery','stall_recovery'):axs[1,2].axvline(e['step'],alpha=.3,color='red')
                axs[1,2].set_title('Height and recovery events');axs[1,2].set_xlabel('Decision index');axs[1,2].legend(fontsize=7)
                fig.suptitle(f'{key} step {step} | GT direction + Oracle stop | no online GT map queries\nRaw connected={info["raw_free_connected_cells"]}, inflated connected={info["safe_connected_cells"]}, A* failures={row["astar_failures"]}; current path cells={len(route)}')
                fig.tight_layout();fig.savefig(dest/f'evidence_{step:03d}.png',dpi=145);plt.close(fig)
            (dest/'audit.json').write_text(json.dumps(audit,indent=2));audits.append(audit)
            print(key,'diagnosed',flush=True)
    (out/'summary.json').write_text(json.dumps(audits,indent=2))


if __name__=='__main__':main()
