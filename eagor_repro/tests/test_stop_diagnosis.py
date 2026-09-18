"""Tests for stop semantics, post-action reach accounting, and prefix audits."""
import inspect
from types import SimpleNamespace
import numpy as np
from omegaconf import OmegaConf
from eagor_repro.controllers.stop_criteria import StopCriterion
from eagor_repro.evaluation.stop_diagnosis import (in_success_range,arbitrate_stop,
    episode_stop_metrics,stop_classification,paired_prefix,NavigationObserver)


def row(step,before,after,action='move_forward',length=.25):
    return dict(step=step,evaluation_distance_to_goal_m=before,evaluation_distance_after_m=after,
        action=action,action_ordinal=step+1 if action else None,actual_displacement_m=length,
        direction_method='eagor',area_stop_proposal=False,oracle_suppressed_area=False,
        position=[0,0,-step*.25],rotation_world_from_body=np.eye(3).tolist(),navigation_action='move_forward')


def test_oracle_strict_comparison_matches_actual_habitat_success():
    from habitat.tasks.nav.nav import Success
    threshold=.123
    measure=Success(sim=None,config=OmegaConf.create({'success_distance':threshold}))
    for distance in (threshold-1e-6,threshold,threshold+1e-6,float('inf')):
        task=SimpleNamespace(is_stop_called=True,measurements=SimpleNamespace(measures={
            'distance_to_goal':SimpleNamespace(get_metric=lambda:distance)}))
        measure.update_metric(None,task)
        assert bool(measure.get_metric())==in_success_range(distance,threshold)
        task.is_stop_called=False;measure.update_metric(None,task)
        assert not measure.get_metric()


def test_oracle_suppresses_area_but_executes_real_navigation_action():
    for action in ('move_forward','turn_left','turn_right',None):
        r=arbitrate_stop(action,True,False,'oracle_distance')
        assert r['action']==action and r['oracle_suppressed_area']
        assert arbitrate_stop(action,False,True,'oracle_distance')['action']=='stop'
        assert arbitrate_stop(action,True,False,'area')['action']=='stop'


def test_original_area_is_invariant_to_diagnostic_gt_distance():
    stop=StopCriterion(mode='area')
    for visible,conf,area in [(False,1.,.9),(True,.1,.9),(True,1.,.01),(True,1.,.9)]:
        decisions=[stop.assess(target_visible=visible,confidence=conf,target_area_fraction=area,oracle_distance_m=d).should_stop for d in (0.,.1,100.)]
        assert len(set(decisions))==1
    # Evaluator's explicit area call has no metric/goal argument.
    from eagor_repro.evaluation import online_evaluator
    import ast
    tree=ast.parse(inspect.getsource(online_evaluator))
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='area_stop']
    assert len(calls)==1
    assert {k.arg for k in calls[0].keywords}=={'target_visible','confidence','target_area_fraction','target_mask','depth'}


def test_post_last_action_reach_is_not_lost_or_counted_as_stop_success():
    rows=[row(0,1.,.05)]
    s=episode_stop_metrics(rows,False,.1)
    assert s['ever_in_success_range'] and s['first_reach_step']==1 and s['first_reach_path_m']==.25
    assert s['stop_step'] is None and s['reach_to_stop_steps'] is None
    assert s['stop_classification']=='reached_without_successful_stop'
    rows.append(row(1,.05,.05,'stop',0.))
    s=episode_stop_metrics(rows,True,.1)
    assert s['stop_step']==2 and s['reach_to_stop_steps']==1 and s['reach_to_stop_decisions']==0


def test_stop_outcome_classes_and_unreached_nulls():
    assert len({stop_classification(True,True,True),stop_classification(False,True,False),
        stop_classification(False,False,True),stop_classification(False,False,False)})==4
    s=episode_stop_metrics([row(0,1.,1.,'stop',0)],False,.1)
    assert s['first_reach_step'] is None and s['first_reach_path_m'] is None
    assert s['reach_to_stop_steps'] is None and s['stop_classification']=='never_reached_false_stop'
    s=episode_stop_metrics([row(0,.05,.2)],False,.1)
    assert s['first_reach_step']==0 and s['exited_success_range_count']==1


def test_prefix_allows_stop_divergence_but_rejects_navigation_or_pose_change():
    a=[row(0,1.,1.),row(1,1.,1.,'stop',0)]
    b=[row(0,1.,1.),row(1,1.,.8)]
    assert paired_prefix(a,b)['prefix_consistent']
    b[1]['navigation_action']='turn_left'
    assert not paired_prefix(a,b)['prefix_consistent']
    b[1]['navigation_action']='move_forward';b[0]['position'][0]=.1
    assert not paired_prefix(a,b)['prefix_consistent']


def test_astar_observer_is_non_mutating_and_restores_function():
    import eagor_repro.planning.online_navigation as module
    original=module.astar;observer=NavigationObserver()
    grid=np.array([[1,0],[0,1]],bool)
    with observer.observe_astar(3):
        assert module.astar(grid,(0,0),(1,1))==original(grid,(0,0),(1,1))==[]
    assert module.astar is original and observer.astar_calls==observer.astar_failures==1


def test_aggregation_keeps_unreached_tasks_in_denominator_not_as_zero_steps():
    from eagor_repro.scripts.analyze_stop_matrix import aggregate_records
    template=dict(success=False,spl=0.,ever_in_success_range=False,first_reach_step=None,
        first_reach_path_m=None,action_steps=300,path_length_m=10.,collision_count=0,
        recoveries=0,astar_failures=0,state_revisit_count=0,repeated_backtrack_count=0,
        stop_classification='never_reached_other_termination')
    a=dict(template);b=dict(template,success=True,spl=.5,ever_in_success_range=True,
        first_reach_step=80,first_reach_path_m=7.,stop_classification='correct_stop_success')
    result=aggregate_records([a,b])
    assert result['n']==2 and result['reach_rate']==.5 and result['sr']==.5
    assert result['first_reach_denominator']==1 and result['first_reach_steps_mean']==80
    assert aggregate_records([a])['first_reach_steps_mean'] is None
