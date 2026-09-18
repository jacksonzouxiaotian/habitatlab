"""Single-floor, growing RGB-D occupancy map. No simulator dependency.

World axes: Habitat X/Z horizontal, Y up. ERP input is normalized positive-left
and contains metric radial depth. A ray at (theta, phi) has EAGOR-body direction
(cos(phi)cos(theta), cos(phi)sin(theta), sin(phi)).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import ndimage


@dataclass
class MapConfig:
    resolution: float = 0.075
    initial_size_m: float = 18.0
    max_range_m: float = 5.0
    sensor_max_m: float = 10.0
    min_depth_m: float = 0.08
    sensor_height_m: float = 0.88
    robot_radius_m: float = 0.18
    robot_height_m: float = 0.88
    safety_margin_m: float = 0.02
    floor_tolerance_m: float = 0.07
    hit: float = 1.5
    miss: float = 0.45
    occupied_threshold: float = 0.6
    row_stride: int = 3
    col_stride: int = 2


def erp_rays(height, width, row_stride=1, col_stride=1):
    rr, cc = np.meshgrid(np.arange(0, height, row_stride), np.arange(0, width, col_stride), indexing='ij')
    theta = (cc + .5) * (2*np.pi / width) - np.pi
    phi = np.pi/2 - (rr+.5) * np.pi/height
    rays = np.stack((np.cos(phi)*np.cos(theta), np.cos(phi)*np.sin(theta), np.sin(phi)), -1)
    return rr, cc, rays


def cubemap_axial_to_radial(depth, clipped_max_m=10.):
    """Habitat-Sim 0.3.3 native ERP retains cubemap-face axial Z.

    Verified against six asymmetric analytic stage planes. Radial distance is
    face-Z / max(abs(unit ray)); this is not a pinhole ERP projection.
    """
    depth=np.asarray(depth,float)
    _,_,rays=erp_rays(*depth.shape)
    radial=depth/np.max(np.abs(rays),axis=-1)
    radial[(~np.isfinite(depth)) | (depth<=0) | (depth>=clipped_max_m-.01)]=np.nan
    return radial.astype(np.float32)


class OnlineOccupancyMap:
    def __init__(self, config: MapConfig, start_position):
        self.config = config
        self.resolution = config.resolution
        n = int(np.ceil(config.initial_size_m / self.resolution))
        self.origin = np.asarray(start_position, float)[[0, 2]] - n*self.resolution/2
        self.floor_y = float(start_position[1])
        self.evidence = np.zeros((n, n), np.float32)
        self.observed = np.zeros((n, n), bool)
        self.visits = np.zeros((n, n), np.int32)
        self.updated = np.full((n, n), -1, np.int32)
        self.contacts = np.full((n, n), -10000, np.int32)
        self.trajectory = []
        self._rays = {}
        self.frame_stats = {}

    def cells(self, xz):
        return np.floor((np.asarray(xz)-self.origin)/self.resolution).astype(int)[..., ::-1]

    def world(self, rc):
        return self.origin + (np.asarray(rc)[..., ::-1]+.5)*self.resolution

    def ensure_bounds(self, points):
        rc = self.cells(points)
        low = np.maximum(0, 2-rc.min(axis=0))
        high = np.maximum(0, rc.max(axis=0)+3-np.array(self.evidence.shape))
        if not np.any(low+high):
            return
        # Expand in chunks; all history and backtracking paths are preserved.
        low = np.where(low > 0, ((low+63)//64)*64, 0)
        high = np.where(high > 0, ((high+63)//64)*64, 0)
        pad = tuple(zip(low, high))
        for name, default in [('evidence', 0), ('observed', False), ('visits', 0), ('updated', -1), ('contacts', -10000)]:
            setattr(self, name, np.pad(getattr(self, name), pad, constant_values=default))
        self.origin -= low[::-1]*self.resolution

    def update(self, depth, position, rotation_world_from_body, step):
        cfg = self.config
        position = np.asarray(position, float)
        self.ensure_bounds(position[[0, 2]] + np.array([[-1,-1],[1,1]])*(cfg.max_range_m+1))
        h, w = depth.shape
        if (h, w) not in self._rays:
            self._rays[h, w] = erp_rays(h, w, cfg.row_stride, cfg.col_stride)
        rr, cc, rays = self._rays[h, w]
        ranges = np.asarray(depth)[rr, cc].reshape(-1)
        directions = rays.reshape(-1, 3) @ np.asarray(rotation_world_from_body).T
        valid = np.isfinite(ranges) & (ranges > cfg.min_depth_m) & (ranges < cfg.sensor_max_m-.01)
        ranges, directions = ranges[valid], directions[valid]
        camera = position + np.asarray(rotation_world_from_body)[:, 2]*cfg.sensor_height_m
        clipped = np.minimum(ranges, cfg.max_range_m)
        endpoints = camera + directions * clipped[:, None]
        height = endpoints[:, 1]-self.floor_y
        hits = (ranges <= cfg.max_range_m) & (height > cfg.floor_tolerance_m) & (height < cfg.robot_height_m)
        hit_cells = self.cells(endpoints[hits][:, [0,2]])
        free_mask = np.zeros_like(self.observed)
        occupied_mask = np.zeros_like(self.observed)
        if len(hit_cells):
            occupied_mask[tuple(hit_cells.T)] = True
        # Carve only ray segments inside the robot's collision-height slab.
        # Rays above the robot do not establish floor traversability.
        lengths = np.linalg.norm(endpoints[:, [0,2]]-camera[[0,2]], axis=1)
        ns = max(1, int(np.ceil(cfg.max_range_m / (self.resolution*.7))))
        for j in range(ns):
            dist = j*self.resolution*.7
            t = dist/np.maximum(lengths, 1e-9)
            points = camera + (endpoints-camera)*t[:, None]
            useful = (t < 1-self.resolution/np.maximum(lengths, self.resolution)) & (points[:,1]-self.floor_y >= -cfg.floor_tolerance_m) & (points[:,1]-self.floor_y <= cfg.robot_height_m)
            cells = self.cells(points[useful][:, [0,2]])
            if len(cells):
                free_mask[tuple(cells.T)] = True
        # Floor returns themselves provide free-space evidence, not obstacles.
        floor = (ranges <= cfg.max_range_m) & (np.abs(height) <= cfg.floor_tolerance_m)
        fc = self.cells(endpoints[floor][:, [0,2]])
        if len(fc):
            free_mask[tuple(fc.T)] = True
        free_mask &= ~occupied_mask
        self.evidence[free_mask] -= cfg.miss
        self.evidence[occupied_mask] += cfg.hit
        np.clip(self.evidence, -3, 5, out=self.evidence)
        changed = free_mask | occupied_mask
        self.observed[changed] = True
        self.updated[changed] = step
        # The body's actually occupied disk is known safe from odometry, never
        # from a privileged map query. Do not clear the surrounding safety ring.
        r, c = self.cells(position[[0,2]])
        k = int(np.ceil(cfg.robot_radius_m/self.resolution))
        yy, xx = np.mgrid[-k:k+1, -k:k+1]
        disk = (xx*xx+yy*yy)*self.resolution**2 <= cfg.robot_radius_m**2
        sub = np.s_[r-k:r+k+1, c-k:c+k+1]
        self.evidence[sub][disk] = -3
        self.observed[sub][disk] = True
        self.updated[sub][disk] = step
        self.visits[r, c] += 1
        self.trajectory.append(position[[0,2]].copy())
        self.frame_stats = dict(valid_depth_samples=int(valid.sum()), obstacle_endpoints=int(hits.sum()), floor_endpoints=int(floor.sum()))

    def mark_contact(self, position, forward_world, step):
        point = np.asarray(position)[[0,2]] + np.asarray(forward_world)[[0,2]]*(self.config.robot_radius_m+.06)
        self.ensure_bounds(np.asarray([point]))
        rc = tuple(self.cells(point))
        self.contacts[rc] = step
        self.evidence[rc] = 5
        self.observed[rc] = True
        self.updated[rc] = step

    def layers(self, step):
        occupied = (self.evidence >= self.config.occupied_threshold) | (step-self.contacts < 35)
        free = self.observed & (self.evidence < 0) & ~occupied
        clearance = ndimage.distance_transform_edt(~occupied)*self.resolution
        safe = free & (clearance > self.config.robot_radius_m+self.config.safety_margin_m+self.resolution*.5)
        self.safety_escape_active = False
        if self.trajectory:
            p=self.trajectory[-1]
            r,c=self.cells(p)
            if free[r,c] and not safe[r,c] and clearance[r,c] >= self.config.robot_radius_m:
                # A legal measured pose may lie inside the EXTRA safety margin.
                # Permit a local exit through free cells of no smaller clearance.
                # Occupied cells and the physical radius remain hard constraints.
                yy,xx=np.ogrid[:free.shape[0],:free.shape[1]]
                escape=free & (clearance>=clearance[r,c]-1e-8) & (((yy-r)**2+(xx-c)**2)*self.resolution**2<=.45**2)
                safe |= escape
                self.safety_escape_active = True
        safe[[0,-1], :] = False
        safe[:, [0,-1]] = False
        return free, occupied, safe, clearance

    def snapshot(self, step):
        free, occupied, safe, clearance = self.layers(step)
        return dict(origin=self.origin, resolution=self.resolution, floor_y=self.floor_y,
                    evidence=self.evidence, observed=self.observed, updated=self.updated,
                    visits=self.visits, contacts=self.contacts, free=free, occupied=occupied,
                    safe=safe, trajectory=np.asarray(self.trajectory))
