"""Evaluation boundary for online planning: policy never receives env or goals."""
from __future__ import annotations
import csv
import hashlib
import json
import time
import copy
from contextlib import nullcontext
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
from eagor_repro.config import make_backend, make_grid, make_policy, make_stop_criterion
from eagor_repro.planning.online_navigation import OnlineNavigator
from eagor_repro.planning.online_mapper import cubemap_axial_to_radial
from eagor_repro.planning.frontier_planner import DirectionCue
from eagor_repro.policies.common import DirectionPrediction
from eagor_repro.sensors.panorama_sensor import extract_panorama, extract_depth_panorama, PERSPECTIVE_UUID, SEMANTIC_UUID, normalize_native_equirectangular
from eagor_repro.spherical.rotation import habitat_rotation_world_from_eagor_body, relative_view_rotation
from eagor_repro.evaluation.metrics import angular_error_deg
from eagor_repro.evaluation.video_renderer import VideoStreamWriter
from eagor_repro.evaluation.online_visualization import frame, map_panel
from eagor_repro.evaluation.stop_diagnosis import (NavigationObserver, arbitrate_stop,
    episode_stop_metrics, diagnostic_events, in_success_range)


def truth_direction_adapter(policy, position, rotation, goals):
    """Diagnostic GT adapter drops Euclidean range and instance index here."""
    prediction, _, _ = policy.predict_nearest(position, rotation, goals)
    return prediction


def cue_from_prediction(prediction, rotation, enabled=True):
    direction = np.asarray(prediction.direction_xyz,float)
    valid=bool(enabled and prediction.valid and np.isfinite(direction).all() and np.linalg.norm(direction)>1e-6)
    world=np.asarray(rotation) @ direction if valid else np.zeros(3)
    return DirectionCue(tuple(world[[0,2]]),valid,float(prediction.confidence))


def metric_distance_stop(metrics, radius):
    # Exactly one boolean crosses the evaluator -> final-stop boundary.
    return float(metrics['distance_to_goal']) < radius


def evaluate_online(config, method):
    import habitat
    import cv2
    from habitat.config import read_write
    from eagor_repro.evaluation.habitat_evaluator import build_habitat_config, _target_mask_from_observations
    manifest=json.loads(Path(config.episode_manifest).read_text())
    hc=build_habitat_config(config)
    with read_write(hc):
        hc.habitat.environment.iterator_options.shuffle=False
    dataset=habitat.make_dataset(hc.habitat.dataset.type,config=hc.habitat.dataset)
    serialized_goals=json.loads(dataset.to_json())['goals_by_category']
    lookup={(Path(e.scene_id).stem,str(e.episode_id)):e for e in dataset.episodes}
    selected=manifest['episodes']
    keys=list(config.get('episode_keys',[]))
    if keys:
        selected=[e for e in selected if e['key'] in keys]
        if len(selected)!=len(keys):
            raise ValueError('Unknown manifest episode key')
    selected=selected[:int(config.evaluation.num_episodes)]
    entries={e['key']:e for e in selected}
    dataset.episodes=[lookup[(Path(e['scene_id']).stem,e['episode_id'])] for e in selected]
    for ep,entry in zip(dataset.episodes,selected):
        np.testing.assert_allclose(ep.start_position,entry['start_position'],atol=1e-6)
        np.testing.assert_allclose(ep.start_rotation,entry['start_rotation'],atol=1e-6)
        assert ep.object_category==entry['target_query']
        digest=hashlib.sha256(json.dumps(serialized_goals[ep.goals_key],sort_keys=True).encode()).hexdigest()
        assert digest==entry['goals_sha256'], 'Task goal definitions changed since recovered manifest'
    # Fail rather than silently changing the inherited action/sensor protocol.
    old=manifest['protocol']['habitat']
    for field in ('forward_step_size','turn_angle'):
        assert hc.habitat.simulator[field]==old['simulator'][field]
    for field in ('radius','height','sim_sensors'):
        assert OmegaConf.to_container(hc.habitat.simulator.agents.main_agent,resolve=True)[field]==old['simulator']['agents']['main_agent'][field]
    assert hc.habitat.task.measurements.success.success_distance==old['task']['measurements']['success']['success_distance']
    root=Path(config.output.root)/method
    root.mkdir(parents=True,exist_ok=True)
    OmegaConf.save(hc,root/'habitat_config.yaml')
    grid=make_grid(config)
    policy=make_policy('centroid' if method=='exploration' else method,config,grid)
    backend=make_backend(config)
    stop=make_stop_criterion(config)
    diagnostic=bool(config.get('stop_diagnosis',{}).get('enabled',False))
    area_stop=copy.copy(stop);area_stop.mode='area'
    formal_threshold=float(hc.habitat.task.measurements.success.success_distance)
    summaries=[]
    with habitat.Env(config=hc,dataset=dataset) as env:
        for _ in selected:
            obs=env.reset()
            ep=env.current_episode
            key=Path(ep.scene_id).stem+'_'+str(ep.episode_id)
            entry=entries[key]
            out=root/key
            out.mkdir(exist_ok=False)
            policy.reset()
            if hasattr(backend,'reset'): backend.reset()
            st=env.sim.get_agent_state()
            nav=OnlineNavigator(OmegaConf.to_container(config.online,resolve=True),st.position,float(config.controller.forward_step_m),float(config.controller.turn_step_deg))
            np.testing.assert_allclose(st.position,entry['start_position'],atol=1e-5)
            truth_positions=[g.position for g in ep.goals]
            # Exact episode-instance visibility for evaluation only. Never used
            # by direction, stop, mapper or navigator.
            goal_ids={int(g.object_id) for g in ep.goals if str(g.object_id).isdigit()}
            old_rotation=None
            previous_collision=False
            path_length=0.; collisions=0; rows=[]; errors=[]
            observer=NavigationObserver()
            first_detection=None; first_instance_visible=None; first_success_region=None
            end_reason='budget_exhausted'; calls=0
            video=VideoStreamWriter(out/'navigation.mp4',int(config.evaluation.video_fps)) if config.evaluation.save_video else None
            for step in range(int(config.evaluation.max_episode_steps)):
                state=env.sim.get_agent_state()
                position=np.asarray(state.position,float)
                rotation=habitat_rotation_world_from_eagor_body(state.rotation)
                panorama=extract_panorama(obs); depth=extract_depth_panorama(obs)
                ts=time.perf_counter()
                result=backend.infer(panorama,entry['target_query'],observations=obs,simulator_state=env.sim)
                calls+=1
                if result.target_visible and first_detection is None: first_detection=step
                # RGB/semantic backend is fixed across methods; its raw IDs are
                # confined to the pre-existing perception/stop/evaluation layer.
                mask=_target_mask_from_observations(result,obs)
                if method=='oracle_direction':
                    prediction=truth_direction_adapter(policy,position,rotation,truth_positions)
                else:
                    prediction=policy.update(result.likelihood,result.target_visible,None if old_rotation is None else relative_view_rotation(old_rotation,rotation))
                direction_ms=(time.perf_counter()-ts)*1000
                cue=cue_from_prediction(prediction,rotation,method!='exploration')
                # Offline-style evaluation computations: only the boolean stop
                # result may affect execution in explicitly Oracle-stop runs.
                metrics=env.get_metrics()
                task_distance=float(metrics['distance_to_goal'])
                in_region=task_distance<float(config.evaluation.success_distance_m)
                if in_region and first_success_region is None: first_success_region=step
                semantic=np.asarray(normalize_native_equirectangular(obs[SEMANTIC_UUID])).squeeze()
                exact_visible=bool(np.isin(semantic,list(goal_ids)).any())
                if exact_visible and first_instance_visible is None: first_instance_visible=step
                error=None
                if cue.unit() is not None:
                    deltas=np.asarray(truth_positions)-position
                    body=deltas @ rotation
                    error=min(angular_error_deg(prediction.direction_xyz,d) for d in body)
                    errors.append(error)
                area_fraction=float(np.mean(result.likelihood>=.5))
                # Original Area inputs only. GT metrics are deliberately absent.
                area_assessment=area_stop.assess(target_visible=result.target_visible,confidence=prediction.confidence,
                    target_area_fraction=area_fraction,target_mask=mask,depth=depth) if diagnostic else None
                if stop.mode=='oracle_distance':
                    should_stop=metric_distance_stop(metrics,float(config.evaluation.success_distance_m))
                    stop_reason='oracle_distance_stop'
                else:
                    assessment=stop.assess(target_visible=result.target_visible,confidence=prediction.confidence,
                        target_area_fraction=float(np.mean(result.likelihood>=.5)),target_mask=mask,depth=depth)
                    should_stop=assessment.should_stop
                    stop_reason=assessment.reason
                radial_depth=cubemap_axial_to_radial(depth)
                with observer.observe_astar(step) if diagnostic else nullcontext():
                    action=nav.act(radial_depth,position,rotation,cue,step,previous_collision)
                navigation_action=action
                arbitration=None
                if diagnostic:
                    arbitration=arbitrate_stop(navigation_action,area_assessment.should_stop,
                        in_success_range(task_distance,formal_threshold),stop.mode)
                    action=arbitration['action']
                if should_stop:
                    action='stop'; end_reason=stop_reason
                telemetry=nav.telemetry(step)
                row=dict(step=step,position=position.tolist(),rotation_world_from_body=rotation.tolist(),
                    direction=prediction.direction_xyz.tolist(),direction_valid=cue.unit() is not None,
                    direction_error_deg=error,confidence=float(prediction.confidence),
                    perception_target_visible=bool(result.target_visible),evaluation_instance_visible=exact_visible,
                    evaluation_distance_to_goal_m=task_distance,subgoal=None if nav.goal is None else nav.goal.tolist(),
                    action=action,commanded_distance_m=float(config.controller.forward_step_m) if action=='move_forward' else 0.,
                    direction_ms=direction_ms,backend_calls=calls,**telemetry)
                if diagnostic:
                    row.update(direction_method=method,stop_mode=stop.mode,seed=int(config.seed),
                        scene_id=ep.scene_id,episode_id=str(ep.episode_id),navigation_action=navigation_action,
                        action_ordinal=step+1 if action is not None else None,
                        area_stop_proposal=bool(area_assessment.should_stop),area_stop_reason=area_assessment.reason,
                        area_fraction=area_fraction,area_confidence_threshold=area_stop.confidence_threshold,
                        area_fraction_threshold=area_stop.area_fraction_threshold,
                        area_target_pixels=area_assessment.target_pixels,
                        formal_success_threshold_m=formal_threshold,
                        oracle_stop_triggered=arbitration['oracle_stop_triggered'],oracle_suppressed_area=arbitration['oracle_suppressed_area'],
                        astar_calls=observer.astar_calls,astar_failures=observer.astar_failures,
                        planned_path_xz=[np.asarray(p).tolist() for p in nav.path])
                if video:
                    video.append(frame(nav,step,panorama,obs.get(PERSPECTIVE_UUID,panorama),[
                        f'{key} / {entry["target_query"]} / step {step}',f'{method} | {action} | {nav.status}',
                        f'stop={stop.mode} / pose=simulator GT',f'direction valid={cue.valid} / semantic backend=ORACLE',
                        f'replans={nav.replans} collisions={collisions}',
                        'No goal coordinates / navmesh queries in planner']))
                if step%25==0 or (diagnostic and (should_stop or (in_region and first_success_region==step))):
                    np.savez_compressed(out/f'map_{step:03d}.npz',**nav.mapper.snapshot(step),path=np.asarray(nav.path),subgoal=np.asarray(nav.goal if nav.goal is not None else []))
                    np.savez_compressed(out/f'depth_{step:03d}.npz',raw_face_axial=depth,radial=radial_depth,position=position,rotation=rotation)
                    if diagnostic:
                        cv2.imwrite(str(out/f'rgb_{step:03d}.jpg'),cv2.cvtColor(panorama,cv2.COLOR_RGB2BGR))
                        cv2.imwrite(str(out/f'perspective_{step:03d}.jpg'),cv2.cvtColor(obs.get(PERSPECTIVE_UUID,panorama)[...,:3],cv2.COLOR_RGB2BGR))
                old_rotation=rotation
                if action is None:
                    row.update(actual_displacement_m=0.,collision=False)
                    if diagnostic:row.update(evaluation_distance_after_m=task_distance,path_length_after_m=path_length,
                        position_after=position.tolist(),formal_success_after=bool(metrics['success']),formal_spl_after=float(metrics['spl']))
                    rows.append(row); end_reason=nav.status; break
                obs=env.step(action)
                new_position=np.asarray(env.sim.get_agent_state().position,float)
                displacement=float(np.linalg.norm(new_position-position));path_length+=displacement
                previous_collision=action=='move_forward' and displacement<1e-4
                collisions+=int(previous_collision)
                row.update(actual_displacement_m=displacement,collision=previous_collision)
                if diagnostic:
                    after_metrics=env.get_metrics()
                    row.update(evaluation_distance_after_m=float(after_metrics['distance_to_goal']),path_length_after_m=path_length,
                        position_after=new_position.tolist(),formal_success_after=bool(after_metrics['success']),formal_spl_after=float(after_metrics['spl']))
                rows.append(row)
                if env.episode_over: break
            if video:video.close()
            metrics=env.get_metrics();success=bool(metrics['success'])
            if float(metrics['distance_to_goal'])<float(config.evaluation.success_distance_m) and first_success_region is None: first_success_region=len(rows)
            # Symptoms and end conditions are not asserted to be causal diagnoses.
            failure='none' if success else ('stop' if rows[-1]['action']=='stop' else ('budget_exhausted' if len(rows)>=int(config.evaluation.max_episode_steps) else 'unresolved'))
            summary=dict(key=key,scene_id=ep.scene_id,episode_id=str(ep.episode_id),seed=int(config.seed),
                start_position=entry['start_position'],start_rotation=entry['start_rotation'],target_query=entry['target_query'],
                method=method,planning_mode='online_frontier',pose_source='simulator_gt_pose',
                oracle_direction=method=='oracle_direction',oracle_stop=stop.mode=='oracle_distance',
                oracle_semantic=result.backend_name=='oracle_semantic',planner_privileged_queries=0,
                privilege_audit='capability-limited planner inputs + AST banned import/call test; evaluator metrics remain privileged',
                depth_convention='native cubemap-face axial meters converted to radial meters for mapper; original stop depth preserved',
                success=success,spl=float(metrics['spl']),steps=len(rows),path_length_m=path_length,collision_count=collisions,
                final_distance_m=float(metrics['distance_to_goal']),direction_mae_deg=float(np.mean(errors)) if errors else None,
                direction_valid_samples=len(errors),first_perception_detection=first_detection,
                first_evaluation_instance_visible=first_instance_visible,first_success_region=first_success_region,
                evaluation_visibility_definition='>=1 ERP pixel of episode object_id; official task has no separate visibility metric',
                perception_calls=calls,vlm_calls=calls if result.backend_name=='qwen_grounding' else 0,
                stop_mode=stop.mode,stop_reason=end_reason,failure_category=failure,
                max_vertical_displacement_m=max(abs(r['position'][1]-entry['start_position'][1]) for r in rows),
                **nav.telemetry(len(rows)-1))
            for name in ('direction_ms','mapping_ms','planning_ms','following_ms'):
                summary['mean_'+name]=float(np.mean([r[name] for r in rows]))
            if diagnostic:
                summary.update(episode_stop_metrics(rows,success,formal_threshold))
                events,counts=diagnostic_events(rows,nav.events,observer.events,formal_threshold)
                summary.update(counts,astar_calls=observer.astar_calls,astar_failures=observer.astar_failures,
                    termination_reason=end_reason,formal_success_threshold_m=formal_threshold,
                    formal_distance_definition=str(hc.habitat.task.measurements.distance_to_goal.distance_to),
                    state_revisit_definition='0.5m XZ cells + 30deg heading bin, revisits separated by >=20 decisions',
                    repeated_backtrack_definition='backtrack selected within 0.5m of a prior backtrack target, >=10 decisions apart',
                    recovery_outcome_definition='translation >0.25m within next 12 decisions; observational, not global progress')
                tags=[]
                if not success and not summary['ever_in_success_range']:tags.append('not_reached')
                if nav.invalid_paths:tags.append('online_path_invalidation_observed')
                if observer.astar_failures:tags.append('astar_no_path_observed')
                if counts['repeated_backtrack_count']:tags.append('repeated_backtrack_observed')
                if end_reason=='exploration_exhausted':tags.append('reachable_candidates_exhausted')
                summary['planning_issue_tags']=tags
                (out/'events.json').write_text(json.dumps(events,indent=2,allow_nan=False))
            (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
            (out/'steps.json').write_text(json.dumps(rows,allow_nan=False))
            with (out/'steps.csv').open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            (out/'frontier_scores.json').write_text(json.dumps(nav.events,allow_nan=False))
            np.savez_compressed(out/'map_final.npz',**nav.mapper.snapshot(len(rows)-1),path=np.asarray(nav.path))
            cv2.imwrite(str(out/'map_final.png'),cv2.cvtColor(map_panel(nav,len(rows)-1),cv2.COLOR_RGB2BGR))
            (out/'task.json').write_text(json.dumps(entry,indent=2))
            summaries.append(summary)
            (root/'summary.json').write_text(json.dumps(summaries,indent=2))
            print(f'{method} {key}: success={success}, steps={len(rows)}, path={path_length:.2f}m, collisions={collisions}, end={end_reason}',flush=True)
    return summaries
