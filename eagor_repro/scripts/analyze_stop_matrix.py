"""Audit all actual runs and produce matched, denominator-explicit tables."""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
from collections import Counter
import numpy as np
from omegaconf import OmegaConf
from eagor_repro.evaluation.stop_diagnosis import paired_prefix,in_success_range

METHODS=['oracle_direction','eagor','grid','centroid','exploration']
STOPS=['area','oracle_distance']


def mean(values):
    return float(np.mean(values)) if values else None


def aggregate_records(records):
    reached=[r for r in records if r['ever_in_success_range']]
    return dict(n=len(records),successes=sum(r['success'] for r in records),
        sr=mean([r['success'] for r in records]),spl=mean([r['spl'] for r in records]),
        reached=len(reached),reach_rate=len(reached)/len(records),
        first_reach_denominator=len(reached),
        first_reach_steps_mean=mean([r['first_reach_step'] for r in reached]),
        first_reach_path_mean_m=mean([r['first_reach_path_m'] for r in reached]),
        mean_action_steps=mean([r['action_steps'] for r in records]),mean_path_m=mean([r['path_length_m'] for r in records]),
        mean_collisions=mean([r['collision_count'] for r in records]),
        mean_recoveries=mean([r['recoveries'] for r in records]),
        total_astar_failures=sum(r['astar_failures'] for r in records),
        mean_state_revisits=mean([r['state_revisit_count'] for r in records]),
        mean_backtrack_loops=mean([r['repeated_backtrack_count'] for r in records]),
        reached_no_stop=sum(r['ever_in_success_range'] and r.get('stop_step') is None for r in records),
        reached_but_stopped_unsuccessfully=sum(r['ever_in_success_range'] and r.get('stop_step') is not None and not r['success'] for r in records),
        **{c:sum(r['stop_classification']==c for r in records) for c in ('correct_stop_success','reached_without_successful_stop','never_reached_false_stop','never_reached_other_termination')})


def csv_write(path,rows):
    if not rows:return
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
        writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in rows)


def normalized_config(path):
    c=OmegaConf.to_container(OmegaConf.load(path),resolve=True)
    c.pop('output');c['controller'].pop('stop_mode')
    return c


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();out=args.root/'analysis';out.mkdir(exist_ok=True)
    manifest=json.loads((args.root/'episode_manifest.json').read_text());keys=[e['key'] for e in manifest['episodes']]
    records=[];lookup={};raw={};provenance=[];aggregates=[]
    first_config=None;first_hashes=None
    baseline=Path('eagor_outputs/online_navigation_20260911/verification_release/source_snapshot')
    frozen=[p for d in ('spherical','policies','planning','controllers','perception','sensors') for p in (Path('eagor_repro')/d).glob('*.py')]
    frozen_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen}
    assert all(p.read_bytes()==(baseline/p).read_bytes() for p in frozen),'Frozen navigation/direction source changed'
    for method,stop in itertools.product(METHODS,STOPS):
        run=args.root/'matrix'/(method+'__'+stop)
        results=json.loads((run/'results.json').read_text())[method]
        assert {e['key'] for e in results}==set(keys) and len(results)==10
        config=normalized_config(run/'config.yaml')
        hashes=json.loads((run/'provenance.json').read_text())['file_hashes']
        checked=list(frozen_hashes)+['eagor_repro/evaluation/online_evaluator.py','eagor_repro/evaluation/stop_diagnosis.py']
        subset={p:hashes[p] for p in checked}
        if first_config is None:first_config=config;first_hashes=subset
        assert config==first_config and subset==first_hashes,'Conditions use different configurations/code'
        assert (run/'episode_manifest.json').read_bytes()==(args.root/'episode_manifest.json').read_bytes()
        provenance.append(dict(method=method,stop_mode=stop,config_equal=True,core_hashes_equal=True,manifest_equal=True,
            source=str(run/'results.json'),reused=False))
        for s in results:
            episode=run/method/s['key'];rows=json.loads((episode/'steps.json').read_text())
            for i,row in enumerate(rows):
                assert row['formal_success_after']==(row['action']=='stop' and in_success_range(row['evaluation_distance_after_m'],s['formal_success_threshold_m']))
                if i+1<len(rows):assert abs(row['evaluation_distance_after_m']-rows[i+1]['evaluation_distance_to_goal_m'])<1e-6
                if stop=='oracle_distance':
                    assert (row['action']=='stop')==in_success_range(row['evaluation_distance_to_goal_m'],s['formal_success_threshold_m'])
                    if row['oracle_suppressed_area']:assert row['action']==row['navigation_action']
            assert s['success']==rows[-1]['formal_success_after'] and abs(s['spl']-rows[-1]['formal_spl_after'])<1e-7
            # Cross-check Habitat SPL arithmetic, never replace its official value.
            d0=rows[0]['evaluation_distance_to_goal_m'];expected=float(s['success'])*d0/max(d0,s['path_length_m'])
            assert abs(s['spl']-expected)<1e-5
            if s['ever_in_success_range']:
                index=min(s['first_reach_step'],len(rows)-1)
                s['first_reach_free_cells']=rows[index]['free_cells']
                s['first_reach_visited_cells']=rows[index]['visited_cells']
                s['first_reach_replans']=rows[index]['replans']
            else:
                s.update(first_reach_free_cells=None,first_reach_visited_cells=None,first_reach_replans=None)
            s['source_episode']=str(episode);records.append(s);lookup[method,stop,s['key']]=s;raw[method,stop,s['key']]=rows
        aggregates.append(dict(method=method,stop_mode=stop,**aggregate_records(results)))
    assert len(records)==100
    prefixes=[];stop_pairs=[]
    for method,key in itertools.product(METHODS,keys):
        a=lookup[method,'area',key];b=lookup[method,'oracle_distance',key]
        check=paired_prefix(raw[method,'area',key],raw[method,'oracle_distance',key])
        assert check['prefix_consistent'],(method,key,check)
        prefixes.append(dict(method=method,key=key,**check))
        stop_pairs.append(dict(method=method,key=key,area_success=a['success'],oracle_success=b['success'],
            area_reached=a['ever_in_success_range'],oracle_reached=b['ever_in_success_range'],
            area_first_reach=a['first_reach_step'],oracle_first_reach=b['first_reach_step'],
            area_class=a['stop_classification'],oracle_class=b['stop_classification'],
            rescued=not a['success'] and b['success'],area_spl=a['spl'],oracle_spl=b['spl']))
    common_rows=[];common_summary=[]
    for stop in STOPS:
        for a,b in itertools.combinations(METHODS,2):
            matched=[key for key in keys if lookup[a,stop,key]['ever_in_success_range'] and lookup[b,stop,key]['ever_in_success_range']]
            subset=[]
            for key in matched:
                x=lookup[a,stop,key];y=lookup[b,stop,key]
                row=dict(stop_mode=stop,method_a=a,method_b=b,key=key,
                    reach_step_a=x['first_reach_step'],reach_step_b=y['first_reach_step'],
                    reach_path_a_m=x['first_reach_path_m'],reach_path_b_m=y['first_reach_path_m'],
                    paired_step_delta_a_minus_b=x['first_reach_step']-y['first_reach_step'],
                    paired_path_delta_a_minus_b_m=x['first_reach_path_m']-y['first_reach_path_m'],
                    reach_visited_cells_a=x['first_reach_visited_cells'],reach_visited_cells_b=y['first_reach_visited_cells'],
                    reach_replans_a=x['first_reach_replans'],reach_replans_b=y['first_reach_replans'])
                subset.append(row);common_rows.append(row)
            common_summary.append(dict(stop_mode=stop,method_a=a,method_b=b,common_n=len(matched),keys=matched,
                mean_paired_steps_a_minus_b=mean([r['paired_step_delta_a_minus_b'] for r in subset]),
                mean_paired_path_a_minus_b_m=mean([r['paired_path_delta_a_minus_b_m'] for r in subset])))
    # Historical observations are verified separately, never reused as matrix runs.
    historical=[]
    for method in METHODS:
        oldrun='fair_area_direction' if method in ('eagor','oracle_direction') else 'fair_area_baselines'
        oldroot=Path('eagor_outputs/online_navigation_20260911')/oldrun/method
        for key in keys:
            old=json.loads((oldroot/key/'steps.json').read_text());new=raw[method,'area',key]
            historical.append(dict(method=method,key=key,same_rows=len(old)==len(new),
                actions_same=[r['action'] for r in old]==[r['action'] for r in new],
                poses_same=len(old)==len(new) and bool(np.allclose([r['position'] for r in old],[r['position'] for r in new],atol=1e-5)),
                historical_source=str(oldroot/key)))
    for name,data in [('episodes',records),('aggregate',aggregates),('stop_pairs',stop_pairs),
        ('prefix_checks',prefixes),('common_reach_pairs',common_rows),('common_reach_summary',common_summary),
        ('provenance_checks',provenance),('historical_area_checks',historical)]:
        (out/(name+'.json')).write_text(json.dumps(data,indent=2,ensure_ascii=False));csv_write(out/(name+'.csv'),data)
    (out/'frozen_source_hashes.json').write_text(json.dumps(frozen_hashes,indent=2))
    differences=[]
    for key in ('2azQ1b91cZZ_0','TbHJrupSAjP_0','QUCTc6BB5sX_0'):
        a=raw['eagor','oracle_distance',key];b=raw['grid','oracle_distance',key]
        divergence=next((i for i,(x,y) in enumerate(zip(a,b)) if x['navigation_action']!=y['navigation_action']),None)
        record=dict(key=key,first_eagor_grid_navigation_divergence=divergence,conditions=[])
        for method in ('eagor','grid','oracle_direction'):
            s=lookup[method,'oracle_distance',key]
            info={k:s[k] for k in ('method','success','first_reach_step','minimum_formal_distance_m','path_length_m','replans','recoveries')}
            if divergence is not None and divergence<len(raw[method,'oracle_distance',key]):
                row=raw[method,'oracle_distance',key][divergence]
                info['at_divergence']={k:row[k] for k in ('position','navigation_action','subgoal','direction','direction_error_deg','confidence')}
                nav_events=json.loads((Path(s['source_episode'])/'frontier_scores.json').read_text())
                preceding=[e for e in nav_events if e['step']<=divergence]
                info['last_frontier_decision']=preceding[-1] if preceding else None
            record['conditions'].append(info)
        differences.append(record)
    (out/'direction_difference_cases.json').write_text(json.dumps(differences,indent=2))
    lines=['# 实测 5×2 对照','', '|方向|停止|SR|SPL|到达率|首次到达均值（仅到达者）|分母|', '|---|---|---:|---:|---:|---:|---:|']
    for a in aggregates:
        first='N/A' if a['first_reach_steps_mean'] is None else f"{a['first_reach_steps_mean']:.1f}"
        lines.append(f"|{a['method']}|{a['stop_mode']}|{a['successes']}/10|{a['spl']:.4f}|{a['reached']}/10|{first}|{a['first_reach_denominator']}|")
    lines+=['','首次到达均值来自不同子集，不用于跨方法效率排名。配对效率见 common_reach_pairs.csv。','',
        '## 逐任务条件表','', '|任务|条件|SR|到达|首次到达动作数|STOP 动作序号|最小正式距离 m|分类|','|---|---|---:|---:|---:|---:|---:|---|']
    for key in keys:
        for method,stop in itertools.product(METHODS,STOPS):
            r=lookup[method,stop,key]
            lines.append(f"|{key}|{method}/{stop}|{int(r['success'])}|{int(r['ever_in_success_range'])}|{r['first_reach_step']}|{r['stop_step']}|{r['minimum_formal_distance_m']:.4f}|{r['stop_classification']}|")
    (out/'tables.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4));x=np.arange(5)
    for ax,metric,title in zip(axes,('sr','spl','reach_rate'),('Official SR','Official SPL','Ever in formal distance range')):
        for i,stop in enumerate(STOPS):
            values=[next(a[metric] for a in aggregates if a['method']==m and a['stop_mode']==stop) for m in METHODS]
            ax.bar(x+(i-.5)*.35,values,.35,label=stop)
        ax.set_xticks(x);ax.set_xticklabels(['GT direction','EAGOR','Grid','Centroid','Explore'],rotation=20)
        ax.set_ylim(0,1);ax.set_title(title);ax.grid(axis='y',alpha=.25)
    axes[0].legend();fig.suptitle('10 fixed MP3D tasks | GT pose + Oracle Semantic | navigation frozen')
    fig.tight_layout();fig.savefig(out/'matrix_results.png',dpi=170);plt.close(fig)
    for key in ('2azQ1b91cZZ_0','TbHJrupSAjP_0','QUCTc6BB5sX_0'):
        fig,axes=plt.subplots(4,3,figsize=(14,15))
        for column,method in enumerate(('oracle_direction','eagor','grid')):
            s=lookup[method,'oracle_distance',key];source=Path(s['source_episode'])
            file=sorted(source.glob('map_[0-9][0-9][0-9].npz'))[-1];step=int(file.stem.split('_')[1]);a=np.load(file)
            row=raw[method,'oracle_distance',key][step];depth=np.load(source/f'depth_{step:03d}.npz')
            axes[0,column].imshow(plt.imread(source/f'perspective_{step:03d}.jpg'));axes[0,column].axis('off')
            axes[0,column].set_title(f"{method}: success={s['success']}\nrecorded decision {step}")
            axes[1,column].imshow(depth['radial'],vmin=0,vmax=5,cmap='viridis');axes[1,column].set_title('Radial ERP depth (0-5m)')
            res=float(a['resolution']);o=a['origin'];h,w=a['free'].shape;extent=[o[0],o[0]+w*res,o[1],o[1]+h*res]
            raster=np.full((h,w,3),.5);raster[a['free']]=.95;raster[a['occupied']]=.05
            for line in (2,3):
                im=raster.copy()
                if line==3:im[a['free']&~a['safe']]=[.65,.45,.6]
                ax=axes[line,column];ax.imshow(im,origin='lower',extent=extent)
                t=a['trajectory'];ax.plot(t[:,0],t[:,1],c='green',lw=1);ax.scatter(row['position'][0],row['position'][2],c='red',s=15)
                path=np.array(row['planned_path_xz'])
                if len(path):ax.plot(path[:,0],path[:,1],c='blue',lw=1)
                if row['subgoal'] is not None:ax.scatter(*row['subgoal'],c='magenta',marker='*',s=25)
                ax.set_title('Raw occupancy' if line==2 else 'Inflated safe space / trajectory / subgoal')
                ax.set_aspect('equal')
        fig.suptitle(f'{key} / Oracle Stop / different actual trajectories, NOT matched RGB observations')
        fig.tight_layout();fig.savefig(out/(key+'_direction_cases.png'),dpi=135);plt.close(fig)
    print('\n'.join(lines[:14]));print('Verified 100 actual episodes, 50 paired prefixes; see',out)


if __name__=='__main__':main()
