"""A* and reachable distance tree on observed free cells only."""
from __future__ import annotations
import heapq
import numpy as np


def neighbors(safe, node):
    r, c = node
    h, w = safe.shape
    for dr, dc in ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)):
        rr, cc = r+dr, c+dc
        if 0 <= rr < h and 0 <= cc < w and safe[rr, cc]:
            if dr and dc and (not safe[r+dr,c] or not safe[r,c+dc]):
                continue
            yield (rr, cc), (2**.5 if dr and dc else 1.)


def shortest_tree(safe, start, goal=None):
    start = tuple(start)
    distances = np.full(safe.shape, np.inf)
    parent = {}
    if not safe[start]:
        return distances, parent
    distances[start] = 0
    heuristic = (lambda p: float(np.linalg.norm(np.subtract(p,goal)))) if goal is not None else (lambda p: 0.)
    queue = [(heuristic(start), 0., start)]
    while queue:
        _, cost, node = heapq.heappop(queue)
        if cost > distances[node]+1e-9:
            continue
        if goal is not None and node == tuple(goal):
            break
        for nxt, dc in neighbors(safe, node):
            candidate = cost+dc
            if candidate+1e-9 < distances[nxt]:
                distances[nxt] = candidate
                parent[nxt] = node
                heapq.heappush(queue, (candidate+heuristic(nxt), candidate, nxt))
    return distances, parent


def reconstruct(parent, start, goal):
    node, start = tuple(goal), tuple(start)
    path = [node]
    while node != start:
        if node not in parent:
            return []
        node = parent[node]
        path.append(node)
    return path[::-1]


def astar(safe, start, goal):
    distances, parent = shortest_tree(safe, start, goal)
    return reconstruct(parent, start, goal) if np.isfinite(distances[tuple(goal)]) else []


def segment_safe(safe, a, b):
    # Dense sampling plus adjacent diagonal side checks (supercover).
    points = np.linspace(a, b, max(2, int(np.linalg.norm(np.subtract(a,b))*4)+2))
    cells = np.rint(points).astype(int)
    h,w = safe.shape
    if np.any(cells < 0) or np.any(cells[:,0] >= h) or np.any(cells[:,1] >= w):
        return False
    if not np.all(safe[cells[:,0], cells[:,1]]):
        return False
    for p,q in zip(cells[:-1], cells[1:]):
        if np.all(p != q) and not (safe[p[0],q[1]] and safe[q[0],p[1]]):
            return False
    return True
