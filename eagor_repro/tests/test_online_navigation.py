"""Behavioral geometry, planning and information-boundary tests."""
from __future__ import annotations
import ast
from pathlib import Path
import numpy as np
from eagor_repro.planning.online_mapper import MapConfig, OnlineOccupancyMap, erp_rays, cubemap_axial_to_radial
from eagor_repro.planning.grid_path_planner import astar, segment_safe
from eagor_repro.planning.frontier_planner import DirectionCue, FrontierPlanner
from eagor_repro.planning.online_navigation import OnlineNavigator
from eagor_repro.evaluation.online_evaluator import cue_from_prediction, truth_direction_adapter
from eagor_repro.policies.common import DirectionPrediction

ROT=np.array([[0.,-1.,0.],[0.,0.,1.],[-1.,0.,0.]])


def test_cubemap_face_depth_conversion_and_clipped_samples():
    _,_,rays=erp_rays(32,64)
    depth=np.full((32,64),3.)
    radial=cubemap_axial_to_radial(depth)
    assert np.allclose(radial*np.max(np.abs(rays),axis=-1),3.)
    assert radial.max()>4.8 and radial.min()<3.01
    depth[0,:4]=[0,10,np.inf,np.nan]
    assert np.isnan(cubemap_axial_to_radial(depth)[0,:4]).all()


def test_erp_cardinal_axes_and_metric_floor_height():
    _,_,rays=erp_rays(64,128)
    assert rays[32,64,0]>.99
    assert rays[32,96,1]>.99
    assert rays[32,32,1]<-.99
    assert rays[32,0,0]<-.99
    assert rays[0,64,2]>.99
    world=rays@ROT.T
    depth=np.full((64,128),np.nan)
    down=world[...,1]<-.1
    depth[down]=.88/-world[...,1][down]
    point=world[down]*depth[down,None]+[0,.88,0]
    assert np.max(np.abs(point[:,1]))<1e-6
    m=OnlineOccupancyMap(MapConfig(row_stride=1,col_stride=1),[0,0,0])
    m.update(depth,[0,0,0],ROT,0)
    free,occ,_,_=m.layers(0)
    assert free.sum()>100 and occ.sum()==0


def test_ray_carving_low_obstacle_and_overhead_filter():
    cfg=MapConfig(resolution=.1,row_stride=1,col_stride=1)
    m=OnlineOccupancyMap(cfg,[0,0,0])
    _,_,rays=erp_rays(64,128)
    depth=np.full((64,128),np.nan)
    # Three measured rays: forward horizontal wall, low object, ceiling.
    for r,c,d in [(32,64,2.),(40,80,1.5),(10,64,2.)]:depth[r,c]=d
    for step in range(3):m.update(depth,[0,0,0],ROT,step)
    points=(rays@ROT.T)*np.nan_to_num(depth)[...,None]+[0,.88,0]
    free,occ,_,_=m.layers(2)
    for r,c in [(32,64),(40,80)]:assert occ[tuple(m.cells(points[r,c,[0,2]]))]
    assert not occ[tuple(m.cells(points[10,64,[0,2]]))]
    assert free[tuple(m.cells(points[32,64,[0,2]]*.5))]
    assert not m.observed[tuple(m.cells([0,-3.]))]


def test_map_growth_keeps_old_evidence_and_visits():
    m=OnlineOccupancyMap(MapConfig(),[0,0,0])
    rc=tuple(m.cells([1.,1.]));m.evidence[rc]=4;m.observed[rc]=True;m.visits[rc]=7
    m.ensure_bounds(np.array([[-40,-40],[40,40]]))
    rc=tuple(m.cells([1.,1.]));assert m.evidence[rc]==4 and m.visits[rc]==7


def test_inflated_door_width_and_unknown_conservatism():
    m=OnlineOccupancyMap(MapConfig(resolution=.1,initial_size_m=8),[0,0,0])
    m.observed[:]=True;m.evidence[:]=-3
    m.evidence[30:33,:]=4;m.evidence[30:33,35:45]=-3
    _,_,safe,_=m.layers(0)
    assert astar(safe,(20,40),(40,40))
    m.evidence[30:33,:]=4;m.evidence[30:33,39:42]=-3
    assert not astar(m.layers(0)[2],(20,40),(40,40))
    m.observed[10:20,10:20]=False
    assert not m.layers(0)[2][15,15]


def test_astar_detours_and_disallows_diagonal_corner_cutting():
    safe=np.ones((20,20),bool);safe[3:18,10]=False
    path=astar(safe,(10,5),(10,15))
    assert path and any(r<3 or r>=18 for r,c in path)
    assert not segment_safe(safe,(10,5),(10,15))
    tiny=np.array([[1,0],[0,1]],bool)
    assert not astar(tiny,(0,0),(1,1))
    assert not segment_safe(tiny,(0,0),(1,1))


def test_frontier_ignores_storage_boundary_and_unreachable_island():
    m=OnlineOccupancyMap(MapConfig(resolution=.1,initial_size_m=8),[0,0,0])
    m.observed[:]=True;m.evidence[:]=-3
    p=FrontierPlanner({'bearing_probes':False})
    goal,path,status=p.choose(m,np.array([0.,0.]),DirectionCue(),0)
    assert goal is None and p.frontier_count==0
    m.observed[20:25,20:25]=False
    goal,path,status=p.choose(m,np.array([0.,0.]),DirectionCue(),0)
    assert goal is not None and path and m.observed[tuple(m.cells(goal))]
    m.evidence[15:30,15]=4;m.evidence[15:30,29]=4;m.evidence[15,15:30]=4;m.evidence[29,15:30]=4
    goal,path,status=p.choose(m,np.array([0.,0.]),DirectionCue(),1)
    assert goal is None


def test_soft_direction_allows_away_from_goal_and_cooldown_expires():
    m=OnlineOccupancyMap(MapConfig(resolution=.1,initial_size_m=8),[0,0,0])
    # The sole observed corridor leads WEST, opposite target bearing EAST.
    m.evidence[38:43,10:44]=-3;m.observed[38:43,10:44]=True
    p=FrontierPlanner({'bearing_probes':False,'cooldown_steps':5})
    goal,path,status=p.choose(m,np.array([0.,0.]),DirectionCue((1.,0.),True,1.),0)
    assert goal is not None and goal[0] < 0
    p.cool(goal,0)
    assert p.cooldowns
    p.choose(m,np.array([0.,0.]),DirectionCue(),6)
    assert not p.cooldowns


def test_discrete_follower_and_stall_feedback():
    nav=OnlineNavigator({'map':{'resolution':.1}},[0,0,0])
    nav.mapper.observed[:]=True;nav.mapper.evidence[:]=-3
    nav.path=[np.array([0.,0.]),np.array([0.,-.5])]
    assert nav._action([0,0,0],ROT,nav.mapper.layers(0)[2])=='move_forward'
    nav.previous_position=np.array([0.,0.,0.]);nav.previous_rotation=ROT;nav.previous_action='move_forward';nav.goal=np.array([0,-2])
    nav.feedback([0,0,0],True,1)
    assert nav.goal is None and nav.recoveries==1 and nav.frontiers.cooldowns


def test_zero_and_invalid_bearings_never_become_forward():
    p=DirectionPrediction(np.zeros(3),0,0,1,np.zeros((2,4)),True,'fake')
    assert cue_from_prediction(p,ROT).unit() is None
    p=DirectionPrediction(np.array([1.,0,0]),0,0,0,np.zeros((2,4)),False,'fake')
    assert cue_from_prediction(p,ROT).unit() is None


def test_quantized_detour_turn_commits_safe_forward_step():
    nav=OnlineNavigator({'map':{'resolution':.025,'initial_size_m':4}},[0,0,0])
    nav.mapper.observed[:]=True;nav.mapper.evidence[:]=-3
    safe=nav.mapper.layers(0)[2]
    safe[tuple(nav.mapper.cells([0,-.25]))]=False
    nav.path=[np.array([0.,0.]),np.array([0.,-.5])]
    assert nav._action([0,0,0],ROT,safe)=='turn_left'
    assert nav.pending_forward
    rotated=ROT.copy();a=np.deg2rad(15)
    rotated[:,0]=np.cos(a)*ROT[:,0]+np.sin(a)*ROT[:,1]
    rotated[:,1]=-np.sin(a)*ROT[:,0]+np.cos(a)*ROT[:,1]
    assert nav._action([0,0,0],rotated,safe)=='move_forward'
    assert not nav.pending_forward


def test_stall_timeout_and_unknown_exhaustion_are_not_success():
    nav=OnlineNavigator({'observation_budget':3},[0,0,0])
    nav.previous_position=np.zeros(3);nav.previous_action='turn_left'
    nav.goal=np.array([0.,-2.])
    nav.feedback([0,0,0],False,24)
    assert nav.no_progress==1 and nav.goal is None and nav.frontiers.cooldowns
    nav=OnlineNavigator({'observation_budget':3},[0,0,0])
    actions=[nav.act(np.full((32,64),np.nan),[0,0,0],ROT,DirectionCue(),i) for i in range(4)]
    assert actions==['turn_left','turn_left','turn_left',None]
    assert nav.status=='exploration_exhausted'


def test_legal_start_inside_margin_can_escape_but_not_approach_wall():
    m=OnlineOccupancyMap(MapConfig(resolution=.075),[0,0,0])
    m.observed[:]=True;m.evidence[:]=-3
    r,c=m.cells([0,0]);m.evidence[:,c+3]=4
    m.trajectory.append(np.array([0.,0.]))
    safe=m.layers(0)[2]
    assert safe[r,c] and m.safety_escape_active
    assert segment_safe(safe,(r,c),(r,c-4))
    assert not segment_safe(safe,(r,c),(r,c+2))


def test_planner_source_has_no_privileged_imports_or_calls():
    files=['online_mapper.py','grid_path_planner.py','frontier_planner.py','online_navigation.py']
    banned={'habitat','habitat_sim','pathfinder','find_path','snap_point','get_topdown_view','ShortestPathFollower','geodesic_distance','semantic_scene','get_metrics'}
    for name in files:
        tree=ast.parse((Path('eagor_repro/planning')/name).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Attribute):assert node.attr not in banned
            if isinstance(node,ast.Name):assert node.id not in banned
            if isinstance(node,ast.Import):assert all(a.name.split('.')[0] not in banned for a in node.names)
            if isinstance(node,ast.ImportFrom):assert (node.module or '').split('.')[0] not in banned
    assert set(DirectionCue.__dataclass_fields__)=={'world_xz','valid','confidence'}


def test_gt_adapter_discards_position_distance_and_instance_id():
    class Fake:
        def predict_nearest(self,*args):return 'direction_only',17.2,5
    assert truth_direction_adapter(Fake(),None,None,None)=='direction_only'
