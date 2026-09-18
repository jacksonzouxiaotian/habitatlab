"""Validate saved videos and paired stop interventions, and render real frames."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('eagor_outputs/online_navigation_20260911'))
    args=parser.parse_args()
    out=args.root/'analysis';out.mkdir(exist_ok=True)
    videos=[];pairs=[]
    for run in ('case_videos_final','stop_diagnostic_final'):
        for video in sorted((args.root/run).glob('*/*/navigation.mp4')):
            summary=json.loads((video.parent/'summary.json').read_text())
            capture=cv2.VideoCapture(str(video))
            assert capture.isOpened(),video
            n=int(capture.get(cv2.CAP_PROP_FRAME_COUNT));fps=capture.get(cv2.CAP_PROP_FPS)
            assert n==summary['steps'] and fps>0,(video,n,summary['steps'])
            images=[]
            indices=np.linspace(0,n-1,8).astype(int)
            for i in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES,int(i));ok,frame=capture.read()
                assert ok,(video,i)
                small=cv2.resize(frame,(640,360))
                cv2.putText(small,f'frame {i} / {i/fps:.1f}s',(8,352),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,255),1,cv2.LINE_AA)
                images.append(small)
            capture.release()
            sheet=np.concatenate([np.concatenate(images[i:i+2],axis=1) for i in range(0,8,2)],axis=0)
            name=run+'_'+summary['method']+'_'+summary['key']+'_frames.jpg'
            cv2.imwrite(str(out/name),sheet)
            videos.append(dict(path=str(video),method=summary['method'],key=summary['key'],frames=n,fps=fps,
                seconds=n/fps,success=summary['success'],contact_sheet=str(out/name)))
            other_run='diagnostic_final' if run=='case_videos_final' else ('fair_area_direction' if summary['method']=='eagor' else 'fair_area_baselines')
            other=args.root/other_run/summary['method']/summary['key']
            rows=json.loads((video.parent/'steps.json').read_text());reference=json.loads((other/'steps.json').read_text())
            # The stop intervention must not change any preceding navigation.
            common=min(len(rows),len(reference))
            prefix_position=all(np.allclose(a['position'],b['position'],atol=1e-5) for a,b in zip(rows[:common],reference[:common]))
            prefix_action=all(a['action']==b['action'] for a,b in zip(rows[:common-1],reference[:common-1]))
            pairs.append(dict(run=run,method=summary['method'],key=summary['key'],reference_run=other_run,
                shared_frames=common,identical_pose_prefix=prefix_position,identical_action_prefix_before_stop=prefix_action,
                success=summary['success'],reference_success=json.loads((other/'summary.json').read_text())['success']))
            assert prefix_position and prefix_action,pairs[-1]
    (out/'video_audit.json').write_text(json.dumps(videos,indent=2))
    (out/'paired_stop_and_replay_audit.json').write_text(json.dumps(pairs,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a=np.load(args.root/'geometry/native_depth_verified.npz')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,key,title in zip(axes,('depth','expected','error'),('Native face-axial depth (m)','Analytic radial depth (m)','Corrected radial absolute error (m)')):
        im=ax.imshow(a[key],vmin=0,vmax=.12 if key=='error' else 8,cmap='magma' if key=='error' else 'viridis');ax.set_title(title);fig.colorbar(im,ax=ax,shrink=.7)
    fig.tight_layout();fig.savefig(out/'depth_geometry_audit.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(videos=videos,paired_checks=pairs),indent=2))


if __name__=='__main__':main()
