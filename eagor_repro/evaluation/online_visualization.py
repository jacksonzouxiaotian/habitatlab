"""Online-only map panels. No truth map or simulator access."""
from __future__ import annotations
import cv2
import numpy as np


def map_panel(nav, step, size=720):
    m = nav.mapper
    free, occ, safe, _ = m.layers(step)
    view = np.full((*free.shape,3),75,np.uint8)
    view[free] = (225,225,225)
    view[free & ~safe] = (155,155,180)
    view[occ] = (25,25,25)
    seen = np.argwhere(m.observed)
    if len(seen):
        low = np.maximum(seen.min(axis=0)-8,0)
        high = np.minimum(seen.max(axis=0)+9,free.shape)
    else:
        low=np.array([0,0]); high=np.array(free.shape)
    def xy(p):
        return tuple(m.cells(p)[::-1].astype(int))
    for candidate in nav.frontiers.last_scores:
        cv2.circle(view,xy(candidate['point']),2,(255,180,0),-1)
    if len(m.trajectory)>1:
        cv2.polylines(view,[np.array([xy(p) for p in m.trajectory])],False,(40,160,50),1)
    if nav.path:
        # Canvas is RGB throughout (VideoStreamWriter converts at encoding).
        cv2.polylines(view,[np.array([xy(p) for p in nav.path])],False,(30,70,250),1)
    if nav.goal is not None:
        cv2.circle(view,xy(nav.goal),4,(255,20,190),-1)
    if m.trajectory:
        cv2.circle(view,xy(m.trajectory[-1]),3,(255,20,20),-1)
    view=view[low[0]:high[0],low[1]:high[1]]
    scale=min(size/view.shape[0],size/view.shape[1])
    small=cv2.resize(view,(int(view.shape[1]*scale),int(view.shape[0]*scale)),interpolation=cv2.INTER_NEAREST)
    canvas=np.full((size,size,3),55,np.uint8)
    canvas[:small.shape[0],:small.shape[1]]=small
    cv2.putText(canvas,'ONLINE RGB-D MAP (GT pose, no prior map)',(12,size-34),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1)
    cv2.putText(canvas,'green: trajectory / blue: path / pink: subgoal',(12,size-12),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
    return canvas


def frame(nav, step, panorama, perspective, telemetry):
    canvas=np.zeros((720,1280,3),np.uint8)
    canvas[:,:720]=map_panel(nav,step)
    canvas[:260,720:]=cv2.resize(np.asarray(perspective)[...,:3],(560,260))
    canvas[260:500,720:]=cv2.resize(panorama[...,:3],(560,240))
    for i,text in enumerate(telemetry[:10]):
        cv2.putText(canvas,str(text),(730,522+19*i),cv2.FONT_HERSHEY_SIMPLEX,.47,(235,235,235),1)
    return canvas
