"""Offline-only evidence aggregation; none of these truth labels enter policy."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('eagor_outputs/online_navigation_20260911'))
    parser.add_argument('--old-root', type=Path, default=Path('eagor_outputs/failure_attribution_10scenes_corrected_20260910'))
    args = parser.parse_args()
    out = args.root/'analysis'
    out.mkdir(exist_ok=True)
    manifest = json.loads((args.root/'episode_manifest.json').read_text())
    expected = {e['key'] for e in manifest['episodes']}
    all_episodes, aggregates, symptoms, checks = [], [], [], []
    core = ['eagor_repro/planning/'+n+'.py' for n in ('online_mapper','frontier_planner','grid_path_planner','online_navigation')]
    core += ['eagor_repro/evaluation/online_evaluator.py','configs/eagor/mp3d_online.yaml']
    reference = json.loads((args.root/'diagnostic_final/provenance.json').read_text())
    def decision_config(path):
        c=OmegaConf.to_container(OmegaConf.load(path),resolve=True)
        c.pop('output',None);c.pop('episode_keys',None)
        c['controller'].pop('stop_mode',None)
        c['evaluation'].pop('save_video',None)
        return c
    reference_config=decision_config(args.root/'diagnostic_final/config.yaml')
    for run in sorted(args.root.iterdir()):
        if not run.is_dir() or not (run/'results.json').exists():
            continue
        results = json.loads((run/'results.json').read_text())
        prov = json.loads((run/'provenance.json').read_text())
        for method, episodes in results.items():
            hashes_match = all(prov['file_hashes'][p] == reference['file_hashes'][p] for p in core)
            checks.append(dict(run=run.name,method=method,episodes=len(episodes),
                exact_ten={e['key'] for e in episodes} == expected,
                manifest_sha256=hashlib.sha256((run/'episode_manifest.json').read_bytes()).hexdigest(),
                decision_core_matches_final=hashes_match,
                runtime_navigation_config_matches_final=decision_config(run/'config.yaml')==reference_config))
            if run.name.startswith('fair_area'):
                assert checks[-1]['exact_ten'] and hashes_match and checks[-1]['runtime_navigation_config_matches_final'],checks[-1]
            means = {k:float(np.mean([e[k] for e in episodes])) for k in (
                'spl','steps','path_length_m','collision_count','replans','subgoal_switches',
                'no_progress_count','recoveries','free_cells','occupied_cells','visited_cells',
                'frontier_cells','perception_calls','vlm_calls','mean_direction_ms',
                'mean_mapping_ms','mean_planning_ms','mean_following_ms')}
            total_valid = sum(e['direction_valid_samples'] for e in episodes)
            mae = sum((e['direction_mae_deg'] or 0)*e['direction_valid_samples'] for e in episodes)/total_valid if total_valid else None
            aggregates.append(dict(run=run.name,method=method,stop_mode=episodes[0]['stop_mode'],
                n=len(episodes),successes=sum(e['success'] for e in episodes),
                sr=float(np.mean([e['success'] for e in episodes])),
                reached_success_region=sum(e['first_success_region'] is not None for e in episodes),
                direction_mae_deg=mae,direction_valid_samples=total_valid,**means))
            for e in episodes:
                episode = dict(run=run.name,**e)
                all_episodes.append(episode)
                rows = json.loads((run/method/e['key']/'steps.json').read_text())
                events = json.loads((run/method/e['key']/'frontier_scores.json').read_text())
                opposite = sum({a['action'],b['action']}=={'turn_left','turn_right'} for a,b in zip(rows,rows[1:]))
                poses = np.array([r['position'] for r in rows])
                symptoms.append(dict(run=run.name,method=method,key=e['key'],success=e['success'],
                    failure_category=e['failure_category'],stop_reason=e['stop_reason'],
                    final_distance_m=e['final_distance_m'],
                    minimum_observed_evaluation_distance_m=min(r['evaluation_distance_to_goal_m'] for r in rows),
                    position_y_range_m=float(np.ptp(poses[:,1])),
                    consecutive_opposite_turns=opposite,
                    actual_path_invalidation_count=e['invalid_paths'],
                    safety_escape_steps=sum(r.get('safety_escape_active',False) for r in rows),
                    start_not_traversable_events=sum(x['selection']=='start_not_traversable' for x in events),
                    last_trigger_labels_at_replanning=dict(Counter(x['reason'] for x in events)),
                    selection_types=dict(Counter(x['selection'] for x in events))))
    (out/'aggregate.json').write_text(json.dumps(aggregates,indent=2))
    (out/'episodes.json').write_text(json.dumps(all_episodes,indent=2))
    (out/'failure_evidence.json').write_text(json.dumps(symptoms,indent=2))
    (out/'fairness_checks.json').write_text(json.dumps(checks,indent=2))
    write_csv(out/'aggregate.csv',aggregates)
    # Existing successful privileged reference provides evaluation-only evidence
    # of elevation, not a proof that all alternative routes require stairs.
    elevation=[]
    for e in manifest['episodes']:
        path=args.old_root/'I_oracle_navmesh_oracle_stop/oracle_direction/raw'/('episode_'+e['key']+'.csv')
        with path.open() as f: old=list(csv.DictReader(f))
        heights=[float(r['agent_position_y']) for r in old]
        viewpoints=[v['agent_state']['position'][1] for g in e['evaluation_only_goals'] for v in g['view_points']]
        delta=np.abs(np.asarray(viewpoints)-e['start_position'][1])
        elevation.append(dict(key=e['key'],target=e['target_query'],
            oracle_reference_y_range_m=max(heights)-min(heights),
            nearest_viewpoint_height_difference_m=float(min(delta)),
            viewpoints_within_20cm_of_start=int((delta<.2).sum()),
            interpretation='evaluation-only; reference elevation is not proof of all-path necessity'))
    (out/'elevation_audit.json').write_text(json.dumps(elevation,indent=2))
    write_csv(out/'elevation_audit.csv',elevation)
    lines=['# Online navigation: measured results','',
        '| Run | Direction | Stop | SR | SPL | Steps | Path m | Collisions | Entered success region |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for a in aggregates:
        lines.append(f"| {a['run']} | {a['method']} | {a['stop_mode']} | {a['successes']}/{a['n']} | {a['spl']:.3f} | {a['steps']:.1f} | {a['path_length_m']:.2f} | {a['collision_count']:.1f} | {a['reached_success_region']} |")
    (out/'tables.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    main=[a for a in aggregates if a['run'].startswith('fair_area')]
    if main:
        fig,axs=plt.subplots(1,3,figsize=(13,4))
        for ax,metric,title in zip(axs,('sr','collision_count','path_length_m'),('Success rate (same Area stop)','Mean blocked-forward collisions','Mean actual path length (m)')):
            ax.bar([a['method'].replace('_','\n') for a in main],[a[metric] for a in main])
            ax.set_title(title);ax.grid(axis='y',alpha=.2)
        axs[0].set_ylim(0,1)
        fig.suptitle('10 identical MP3D tasks | GT pose + Oracle semantic perception | online map')
        fig.tight_layout();fig.savefig(out/'fair_comparison.png',dpi=170);plt.close(fig)
    diagnostic=[e for e in all_episodes if e['run']=='diagnostic_final']
    if diagnostic:
        fig,axes=plt.subplots(2,5,figsize=(17,7))
        for ax,e in zip(axes.flat,diagnostic):
            a=np.load(args.root/'diagnostic_final/oracle_direction'/e['key']/'map_final.npz')
            raster=np.full((*a['free'].shape,3),.5)
            raster[a['free']]=.95;raster[a['occupied']]=.1
            o=a['origin'];res=float(a['resolution']);h,w=a['free'].shape
            ax.imshow(raster,origin='lower',extent=[o[0],o[0]+w*res,o[1],o[1]+h*res])
            t=a['trajectory'];ax.plot(t[:,0],t[:,1],color='limegreen',lw=1)
            ax.scatter(*t[0],c='blue',s=10);ax.scatter(*t[-1],c='red',s=10)
            ax.set_xlim(t[:,0].min()-2,t[:,0].max()+2);ax.set_ylim(t[:,1].min()-2,t[:,1].max()+2)
            ax.set_title(f"{e['key'].split('_')[0]} {'SUCCESS' if e['success'] else 'FAIL'}\n{e['steps']} steps / {e['collision_count']} collisions",fontsize=10)
            ax.set_aspect('equal');ax.tick_params(labelsize=7)
        fig.suptitle('GT Direction + online planner + Oracle Stop (diagnostic, not deployable SR)\nOnline depth maps only; no GT map overlay')
        fig.tight_layout();fig.savefig(out/'diagnostic_maps.png',dpi=180);plt.close(fig)
    print('\n'.join(lines))
    print('Elevation audit:',json.dumps(elevation))


if __name__=='__main__':main()
