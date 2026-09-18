"""Evaluation-only stop arbitration, event accounting and paired-prefix audit.

Nothing in this module is an input to the map, direction belief or recovery.
Step convention: state k follows k executed actions; STOP ordinal is k+1.
"""
from __future__ import annotations
from contextlib import contextmanager
import numpy as np


def in_success_range(distance, threshold):
    # Match Habitat Success's strict comparison, not planar/Euclidean distance.
    return bool(float(distance) < float(threshold))


def arbitrate_stop(navigation_action, area_proposal, oracle_reached, mode):
    """Arbitration cannot generate or change the underlying navigation action."""
    if mode not in ('area','oracle_distance'):
        raise ValueError(mode)
    stop = bool(oracle_reached if mode=='oracle_distance' else area_proposal)
    return dict(action='stop' if stop else navigation_action,
                oracle_stop_triggered=mode=='oracle_distance' and stop,
                oracle_suppressed_area=mode=='oracle_distance' and bool(area_proposal) and not stop)


def stop_classification(success, ever_reached, stopped):
    if success:
        return 'correct_stop_success'
    if ever_reached:
        return 'reached_without_successful_stop'
    if stopped:
        return 'never_reached_false_stop'
    return 'never_reached_other_termination'


class NavigationObserver:
    """Observation-only A* instrumentation; original implementation is called once."""
    def __init__(self):
        self.astar_calls=0
        self.astar_failures=0
        self.events=[]

    @contextmanager
    def observe_astar(self, step):
        import eagor_repro.planning.online_navigation as module
        original=module.astar
        def recorded(safe,start,goal):
            path=original(safe,start,goal)
            self.astar_calls+=1
            self.astar_failures+=int(not path)
            self.events.append(dict(step=step,event='astar_path' if path else 'astar_no_path',
                start=np.asarray(start).tolist(),goal=np.asarray(goal).tolist(),path_cells=len(path)))
            return path
        module.astar=recorded
        try:
            yield
        finally:
            module.astar=original


def episode_stop_metrics(rows, success, threshold):
    """Only actual observed official distances; never counterfactual SR/SPL."""
    states=[(0,rows[0]['evaluation_distance_to_goal_m'],0.)]
    executed=0;path=0.;stop_row=None
    for row in rows:
        if row['action'] is not None:
            executed+=1
            path+=row['actual_displacement_m']
        states.append((executed,row['evaluation_distance_after_m'],path))
        if row['action']=='stop':stop_row=row
    reached=[s for s in states if in_success_range(s[1],threshold)]
    first=reached[0] if reached else None
    stop_step=stop_row['action_ordinal'] if stop_row is not None else None
    classification=stop_classification(success,bool(reached),stop_row is not None)
    return dict(direction_method=rows[0]['direction_method'],ever_in_success_range=bool(reached),
        first_reach_step=first[0] if first else None,first_reach_path_m=first[2] if first else None,
        stop_step=stop_step,stop_decision_index=stop_row['step'] if stop_row else None,
        stop_distance_m=stop_row['evaluation_distance_to_goal_m'] if stop_row else None,
        reach_to_stop_steps=stop_step-first[0] if first and stop_row else None,
        reach_to_stop_decisions=stop_row['step']-first[0] if first and stop_row else None,
        reach_stop_status='reached_and_stopped' if first and stop_row else ('reached_no_stop' if first else ('not_reached_stopped' if stop_row else 'not_reached_no_stop')),
        minimum_formal_distance_m=min(s[1] for s in states),action_steps=executed,
        stop_classification=classification,
        exited_success_range_count=sum(in_success_range(a[1],threshold) and not in_success_range(b[1],threshold) for a,b in zip(states,states[1:])),
        area_proposal_count=sum(r['area_stop_proposal'] for r in rows),
        oracle_suppressed_area_count=sum(r['oracle_suppressed_area'] for r in rows))


def diagnostic_events(rows, nav_events, astar_events, threshold):
    events=list(astar_events)
    previous_recoveries=0;previous_stall=0;previous_switches=0
    pending=[];past_bins={};revisit_count=0;backtracks=[];loop_count=0
    previously_in=False;ever=False
    for row in rows:
        step=row['step'];p=np.array(row['position']);rotation=np.array(row['rotation_world_from_body'])
        if step==0 and in_success_range(row['evaluation_distance_to_goal_m'],threshold):
            events.append(dict(step=0,event='first_reach',path_m=0.));ever=True;previously_in=True
        if row['area_stop_proposal'] or row['oracle_stop_triggered']:
            events.append(dict(step=step,event='stop_proposals',area=row['area_stop_proposal'],
                oracle_trigger=row['oracle_stop_triggered'],oracle_suppressed_area=row['oracle_suppressed_area'],
                visible=row['perception_target_visible'],confidence=row['confidence'],
                area_fraction=row['area_fraction'],distance=row['evaluation_distance_to_goal_m']))
        for field,old,label in [('recoveries',previous_recoveries,'collision_recovery'),('no_progress_count',previous_stall,'stall_recovery')]:
            if row[field]>old:
                event=dict(step=step,event=label,position=p.tolist(),subgoal=row['subgoal'],outcome='pending')
                pending.append((event,p.copy()));events.append(event)
        previous_recoveries=row['recoveries'];previous_stall=row['no_progress_count']
        for event,origin in pending:
            if event['outcome']=='pending':
                if np.linalg.norm(p-origin)>.25:
                    event.update(outcome='moved_gt_0.25m',outcome_step=step)
                elif step-event['step']>=12:
                    event.update(outcome='no_0.25m_progress_within_12_steps',outcome_step=step)
        if row['subgoal_switches']>previous_switches:
            events.append(dict(step=step,event='subgoal_switch',subgoal=row['subgoal']))
        previous_switches=row['subgoal_switches']
        # Deterministic observational bin: .5m horizontal cells and 30deg yaw.
        # At least 20 steps apart; consecutive turns do not count as new loops.
        yaw=np.arctan2(rotation[2,0],rotation[0,0])
        state_bin=tuple(np.floor(p[[0,2]]/.5).astype(int))+(int(np.floor((yaw+np.pi)/(np.pi/6))),)
        if state_bin in past_bins and step-past_bins[state_bin]>=20:
            revisit_count+=1;events.append(dict(step=step,event='state_revisit',previous_step=past_bins[state_bin],position=p.tolist()))
            past_bins[state_bin]=step
        elif state_bin not in past_bins:past_bins[state_bin]=step
        after=in_success_range(row['evaluation_distance_after_m'],threshold)
        if after and not ever:
            events.append(dict(step=row['action_ordinal'],event='first_reach',path_m=row['path_length_after_m']));ever=True
        if previously_in and not after:events.append(dict(step=row['action_ordinal'],event='left_success_range'))
        previously_in=after
    for event,_ in pending:
        if event['outcome']=='pending':event['outcome']='episode_ended_before_12_step_observation'
    for event in nav_events:
        if event['selection']=='backtrack' and event['selected'] is not None:
            point=np.array(event['selected'])
            if any(np.linalg.norm(point-p)<.5 and event['step']-s>=10 for s,p in backtracks):
                loop_count+=1;events.append(dict(step=event['step'],event='repeated_backtrack',selected=point.tolist()))
            backtracks.append((event['step'],point))
    counts=dict(state_revisit_count=revisit_count,repeated_backtrack_count=loop_count,
                backtrack_selection_count=len(backtracks),recovery_event_count=sum(e['event'] in ('collision_recovery','stall_recovery') for e in events))
    return sorted(events,key=lambda x:x['step']),counts


def paired_prefix(area_rows, oracle_rows):
    common=min(len(area_rows),len(oracle_rows));divergence=None
    max_position_error=0.;max_rotation_error=0.;nav_equal=True
    for index,(a,b) in enumerate(zip(area_rows,oracle_rows)):
        max_position_error=max(max_position_error,float(np.max(np.abs(np.array(a['position'])-b['position']))))
        max_rotation_error=max(max_rotation_error,float(np.max(np.abs(np.array(a['rotation_world_from_body'])-b['rotation_world_from_body']))))
        nav_equal=nav_equal and a['navigation_action']==b['navigation_action']
        if a['action']!=b['action']:
            divergence=index
            break
    stop_only=divergence is None or 'stop' in (area_rows[divergence]['action'],oracle_rows[divergence]['action'])
    return dict(checked_states=common if divergence is None else divergence+1,
                first_stop_divergence_decision=divergence,max_position_error_m=max_position_error,
                max_rotation_matrix_error=max_rotation_error,underlying_navigation_actions_equal=bool(nav_equal),
                divergence_is_stop_only=stop_only,
                prefix_consistent=bool(nav_equal and stop_only and max_position_error<1e-5 and max_rotation_error<1e-5))
