"""Verify native ERP face-axial to radial conversion in a known six-wall box."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from eagor_repro.planning.online_mapper import erp_rays, cubemap_axial_to_radial
from eagor_repro.sensors.panorama_sensor import normalize_native_equirectangular


def main():
    import habitat_sim
    import magnum as mn
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    stage=args.output.parent/'audit_box.obj'
    vertices=[(-2,-1,-5),(3,-1,-5),(3,4,-5),(-2,4,-5),(-2,-1,6),(3,-1,6),(3,4,6),(-2,4,6)]
    faces=[(1,2,3,4),(5,8,7,6),(1,5,6,2),(4,3,7,8),(1,4,8,5),(2,6,7,3)]
    stage.write_text('\n'.join(['v '+' '.join(map(str,v)) for v in vertices]+['f '+' '.join(map(str,f)) for f in faces]+['f '+' '.join(map(str,f[::-1])) for f in faces]))
    cfg=habitat_sim.SimulatorConfiguration();cfg.scene_id=str(stage.resolve())
    camera=habitat_sim.EquirectangularSensorSpec();camera.uuid='depth';camera.sensor_type=habitat_sim.SensorType.DEPTH
    camera.resolution=[128,256];camera.position=[0.,0.,0.];camera.channels=1
    pinhole=habitat_sim.CameraSensorSpec();pinhole.uuid='pinhole';pinhole.sensor_type=habitat_sim.SensorType.DEPTH;pinhole.resolution=[128,256];pinhole.position=[0.,0.,0.]
    agent=habitat_sim.agent.AgentConfiguration();agent.sensor_specifications=[camera,pinhole]
    with habitat_sim.Simulator(habitat_sim.Configuration(cfg,[agent])) as sim:
        state=habitat_sim.AgentState();state.position=[0.,0.,0.];sim.initialize_agent(0,state)
        observations=sim.get_sensor_observations()
        print('pinhole',np.quantile(observations['pinhole'],[0,.5,1]),'agent',sim.get_agent(0).get_state().position,flush=True)
        native=observations['depth']
        depth=normalize_native_equirectangular(native).squeeze()
        _,_,rays=erp_rays(*depth.shape)
        rot=np.array([[0.,-1.,0.],[0.,0.,1.],[-1.,0.,0.]])
        world=rays@rot.T
        expected=np.min(np.where(world>0,np.array([3,4,6])/world,np.array([-2,-1,-5])/world),axis=-1)
        converted=cubemap_axial_to_radial(depth)
        error=np.abs(converted-expected)
        stats=dict(shape=list(depth.shape),raw_depth_convention='cubemap_face_axial_meters',converted_depth_convention='radial_meters',median_error_m=float(np.median(error)),p99_error_m=float(np.quantile(error,.99)),raw_radial_assumption_median_error=float(np.median(np.abs(depth-expected))),
                   description='asymmetric stage bounds X[-2,3],Y[-1,4],Z[-5,6]; world axes and all six face ranges checked')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(stats,indent=2))
        np.savez_compressed(args.output.with_suffix('.npz'),depth=depth,converted=converted,expected=expected,error=error)
        print(stats)
        assert stats['median_error_m']<.03 and stats['p99_error_m']<.12, stats


if __name__=='__main__':main()
