"""Side-by-side actual Area/Oracle replays; held terminal frames are labeled."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from eagor_repro.evaluation.video_renderer import VideoStreamWriter


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();out=args.root/'comparison_videos';out.mkdir(exist_ok=False)
    cases=[('eagor','pLe4wQe7qrG_0'),('oracle_direction','EU6Fwq7SyZv_0'),('oracle_direction','8194nk5LbLH_0')]
    checks=[];index=[]
    for method,key in cases:
        sources=[];data=[];summaries=[];captures=[]
        for stop in ('area','oracle_distance'):
            source=args.root/'videos'/(method+'__'+stop+'__'+key)/method/key
            rows=json.loads((source/'steps.json').read_text());s=json.loads((source/'summary.json').read_text())
            reference=args.root/'matrix'/(method+'__'+stop)/method/key
            other=json.loads((reference/'steps.json').read_text());other_s=json.loads((reference/'summary.json').read_text())
            identical=len(rows)==len(other) and [r['action'] for r in rows]==[r['action'] for r in other] and np.allclose([r['position'] for r in rows],[r['position'] for r in other],atol=1e-5)
            assert identical and s['success']==other_s['success'] and abs(s['spl']-other_s['spl'])<1e-6
            cap=cv2.VideoCapture(str(source/'navigation.mp4'));assert cap.isOpened()
            frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));assert frames==len(rows)
            checks.append(dict(method=method,key=key,stop=stop,source=str(source),matrix_reference=str(reference),same_actual_replay=True,frames=frames))
            sources.append(source);data.append(rows);summaries.append(s);captures.append(cap)
        path=out/(method+'_'+key+'_comparison.mp4');writer=VideoStreamWriter(path,10)
        last=[None,None];n=max(map(len,data));selected={0,n-1,min(len(data[0]),len(data[1]))-1};sheets=[]
        for frame in range(n):
            canvas=np.zeros((528,1280,3),np.uint8)
            for side in (0,1):
                if frame<len(data[side]):
                    ok,actual=captures[side].read();assert ok
                    last[side]=cv2.resize(actual,(640,360))
                canvas[88:448,side*640:(side+1)*640]=last[side]
                row=data[side][min(frame,len(data[side])-1)];s=summaries[side]
                labels=[f'{method} / {key} / '+('AREA STOP' if side==0 else 'ORACLE STOP'),
                    f'decision={row["step"]} action={row["action"]} / formal d(pre)={row["evaluation_distance_to_goal_m"]:.3f}m',
                    f'Area proposal={row["area_stop_proposal"]} Oracle suppressed={row["oracle_suppressed_area"]}']
                for j,label in enumerate(labels):cv2.putText(canvas,label,(side*640+8,21+j*25),cv2.FONT_HERSHEY_SIMPLEX,.49,(245,245,245),1,cv2.LINE_AA)
                done=frame>=len(data[side])-1
                text=f'SR={int(s["success"])} / reached={s["ever_in_success_range"]} / first reach={s["first_reach_step"]}' if done else f'GT pose + Oracle Semantic | post-action success={int(row["formal_success_after"])}'
                cv2.putText(canvas,text,(side*640+8,473),cv2.FONT_HERSHEY_SIMPLEX,.48,(200,255,200),1,cv2.LINE_AA)
                if frame>=len(data[side]):cv2.putText(canvas,'TERMINATED: held final frame; NO extra actions',(side*640+8,499),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,190,70),1,cv2.LINE_AA)
            cv2.putText(canvas,'Evaluation-only comparison. Same navigation policy; no goal-distance input to ordinary planning.',(8,522),cv2.FONT_HERSHEY_SIMPLEX,.45,(220,220,220),1)
            # Loaded frames use BGR; this helper accepts RGB.
            writer.append(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB))
            if frame in selected:sheets.append(canvas.copy())
        writer.close()
        for cap in captures:cap.release()
        sheet=np.concatenate(sheets,axis=0);cv2.imwrite(str(path.with_suffix('.jpg')),sheet)
        verify=cv2.VideoCapture(str(path));assert int(verify.get(cv2.CAP_PROP_FRAME_COUNT))==n;verify.release()
        index.append(dict(path=str(path),method=method,key=key,frames=n,fps=10,seconds=n/10,
            area_success=summaries[0]['success'],oracle_success=summaries[1]['success'],held_terminal_frames_explicitly_labeled=True))
    (out/'replay_checks.json').write_text(json.dumps(checks,indent=2));(out/'index.json').write_text(json.dumps(index,indent=2))
    print(json.dumps(index,indent=2))


if __name__=='__main__':main()
