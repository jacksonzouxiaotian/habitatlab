"""Reachable frontier clusters and normalized soft bearing preferences."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import ndimage
from eagor_repro.planning.grid_path_planner import shortest_tree, reconstruct


@dataclass(frozen=True)
class DirectionCue:
    # Only a unit bearing is exposed. No target position, distance, mask or ID.
    world_xz: tuple[float, float] = (0., 0.)
    valid: bool = False
    confidence: float = 0.

    def unit(self):
        x = np.asarray(self.world_xz, float)
        norm = np.linalg.norm(x)
        return x/norm if self.valid and np.isfinite(x).all() and norm > 1e-6 else None


class FrontierPlanner:
    def __init__(self, config):
        self.config = config
        self.cooldowns = []
        self.last_scores = []
        self.frontier_count = 0

    def cool(self, point, step):
        self.cooldowns.append((np.array(point), step+int(self.config.get('cooldown_steps', 35))))

    def choose(self, mapper, position_xz, cue, step):
        free, occupied, safe, clearance = mapper.layers(step)
        start = tuple(mapper.cells(position_xz))
        if not safe[start]:
            self.last_scores = []
            return None, [], 'start_not_traversable'
        distance, parent = shortest_tree(safe, start)
        unknown = ~mapper.observed
        frontier = free & ndimage.binary_dilation(unknown, structure=ndimage.generate_binary_structure(2,1))
        frontier[[0,-1], :] = False
        frontier[:, [0,-1]] = False
        self.frontier_count = int(frontier.sum())
        labels, n = ndimage.label(frontier, structure=np.ones((3,3)))
        candidates = []
        for label in range(1, n+1):
            region = labels == label
            count = int(region.sum())
            if count < int(self.config.get('min_frontier_cells', 3)):
                continue
            # Approach unknown boundaries from reachable known free space.
            near = ndimage.binary_dilation(region, iterations=3) & safe & np.isfinite(distance)
            rc = np.argwhere(near & (distance*mapper.resolution >= .45))
            if not len(rc):
                continue
            center = np.argwhere(region).mean(axis=0)
            metric = np.linalg.norm(rc-center, axis=1) + .3*distance[rc[:,0],rc[:,1]]
            best = rc[int(np.argmin(metric))]
            candidates.append((best, count, 'frontier'))
        direction = cue.unit()
        # Bearing-only approach probes: distances are fixed exploration horizons,
        # never estimated object ranges. They remain on reachable known cells.
        if direction is not None and bool(self.config.get('bearing_probes', True)):
            for radius in (.5, 1., 2., 3., 4.):
                rc = mapper.cells(np.asarray(position_xz)+radius*direction)
                if np.all(rc >= 0) and np.all(rc < np.array(safe.shape)) and safe[tuple(rc)] and np.isfinite(distance[tuple(rc)]):
                    candidates.append((rc, 0, 'bearing_probe'))
        self.cooldowns = [(p,t) for p,t in self.cooldowns if t > step]
        self.last_scores = []
        for rc, size, kind in candidates:
            point = mapper.world(rc)
            if any(np.linalg.norm(point-p) < .65 for p,t in self.cooldowns):
                continue
            cost = float(distance[tuple(rc)]*mapper.resolution)
            delta = point-position_xz
            alignment = (1+float(np.dot(delta/np.linalg.norm(delta), direction)))/2 if direction is not None else 0.
            r,c = rc
            visits = float(mapper.visits[max(0,r-4):r+5,max(0,c-4):c+5].sum())
            components = dict(gain=size/(size+20), cost=cost/(cost+3), risk=1-min(float(clearance[r,c])/.6,1), revisit=visits/(visits+5), direction=alignment)
            weights = dict(gain=1., cost=-.7, risk=-.3, revisit=-1.2, direction=float(self.config.get('direction_weight',.8)))
            score = sum(weights[k]*v for k,v in components.items())
            # Avoid repeatedly selecting a visited bearing probe in known space.
            if kind == 'bearing_probe':
                score += .2
            self.last_scores.append(dict(point=point.tolist(), cell=rc.tolist(), kind=kind, score=score, path_cost_m=cost, components=components, weights=weights))
        self.last_scores.sort(key=lambda x:(-x['score'], x['point']))
        if not self.last_scores:
            # A bounded safe backtrack can revisit a junction after new evidence.
            for old in mapper.trajectory[::max(1,len(mapper.trajectory)//20)][::-1]:
                rc = tuple(mapper.cells(old))
                if safe[rc] and np.isfinite(distance[rc]) and np.linalg.norm(old-position_xz) > 1. and not any(np.linalg.norm(old-p)<.65 for p,t in self.cooldowns):
                    return old, [mapper.world(c) for c in reconstruct(parent,start,rc)], 'backtrack'
            return None, [], 'exploration_exhausted'
        best = self.last_scores[0]
        return np.array(best['point']), [mapper.world(c) for c in reconstruct(parent,start,best['cell'])], best['kind']
