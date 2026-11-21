"""Line art as a clean, continuous network of lines (reports/Embroidery digitizing deep dive.md,
rules 5-7, 11, 22, 53).

A traced drawing of a building is a web of thin dark lines. Sewing the raw skeleton of that web
gives wobbly lines, little kinks at every junction and a trim at every loose end. What a
digitizer draws instead:

  * one point where lines meet, not a cluster of pixels
  * no stubs shorter than the smallest sewable detail
  * every wall dead straight, and walls that share a level or an upright lined up exactly
    (windows in a row sit on one line)
  * the whole web sewn as one thread: out along each branch and back, so every line gets
    exactly two passes (even weight, no jumps, no trims). This is Erich Campbell's redwork
    "tour" and the same idea as Ink/Stitch's routing (re-sew on the way back), written as our
    own code.
"""

from __future__ import annotations

import math

import numpy as np
from shapely.geometry import LineString, Point

from . import geometry

MERGE_MM = 0.6           # junction pixels closer than this are one junction
SPUR_MM = 1.0            # loose stubs shorter than this are dropped (smallest detail)
ALIGN_MM = 0.25          # levels/uprights closer than this are made exactly equal
LINE_TOL_MM = 0.3        # a centre-line edge this close to straight is drawn straight
AXIS_DEG = 8.0           # a straight edge this close to level/upright is made exactly so
MIRROR_TOL_MM = 0.7      # a junction this close to another's mirror image is its twin
MIRROR_SHARE = 0.7       # a drawing is symmetric when this share of its junctions have twins
MIRROR_MIN_NODES = 12


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def build(paths_mm: list[np.ndarray]) -> tuple[np.ndarray, list[dict]]:
    """A graph from centre-line paths: (node positions, edges {a, b, pts}). Path ends closer
    than MERGE_MM become one node."""
    ends = []
    for p in paths_mm:
        ends.append(p[0])
        ends.append(p[-1])
    ends = np.asarray(ends, float)
    lengths = [LineString(p).length for p in paths_mm]
    # Group ends into junctions around a seed point (no chaining, so a line made of many
    # short pieces is never collapsed into one point). A path's own two ends only share a
    # junction when the path is a tiny blob inside that junction.
    seeds, members = [], []
    node_of = {}
    for i in range(len(ends)):
        best = None
        for k, s in enumerate(seeds):
            d = float(np.linalg.norm(ends[i] - s))
            if d < MERGE_MM / 2 and (best is None or d < best[0]):
                other = i ^ 1                              # the other end of the same path
                if node_of.get(other) == k and lengths[i // 2] >= MERGE_MM:
                    continue
                best = (d, k)
        if best is None:
            seeds.append(ends[i])
            members.append([i])
            node_of[i] = len(seeds) - 1
        else:
            members[best[1]].append(i)
            node_of[i] = best[1]
    nodes = np.asarray([ends[m].mean(axis=0) for m in members])
    edges = []
    for n, p in enumerate(paths_mm):
        a, b = node_of[2 * n], node_of[2 * n + 1]
        pts = np.asarray(p, float).copy()
        pts[0], pts[-1] = nodes[a], nodes[b]
        if a == b and LineString(pts).length < 2 * MERGE_MM:
            continue                                   # a junction blob's own little loop
        edges.append({"a": a, "b": b, "pts": pts})
    return nodes, edges


def prune(nodes: np.ndarray, edges: list[dict], spur: float = SPUR_MM) -> list[dict]:
    """Drop loose stubs: a whole dangling branch (from a free end to the junction it hangs
    off) shorter than `spur`. A long line made of many short pieces is measured as one
    branch, so it is never eaten away piece by piece."""
    if len(edges) < 2:
        return edges
    adj = {}
    for k, e in enumerate(edges):
        adj.setdefault(e["a"], []).append(k)
        adj.setdefault(e["b"], []).append(k)
    deg = {n: len(ks) for n, ks in adj.items()}
    drop = set()
    for start, ks in adj.items():
        if deg[start] != 1:
            continue
        branch, total, node, prev = [], 0.0, start, None
        while True:
            nxt = [k for k in adj[node] if k != prev and k not in branch]
            if not nxt:
                break
            k = nxt[0]
            e = edges[k]
            branch.append(k)
            total += LineString(e["pts"]).length
            node = e["b"] if e["a"] == node else e["a"]
            prev = k
            if deg[node] != 2:
                break
        if deg.get(node, 0) >= 3 and total < spur:
            drop.update(branch)
    return [e for k, e in enumerate(edges) if k not in drop]


def straighten(nodes: np.ndarray, edges: list[dict], axes: list[float]) -> np.ndarray:
    """Fit every edge (straight where it is straight), make near-level and near-upright
    straight edges exactly so by giving their two nodes the same y (or x), and line up
    levels and uprights that are within ALIGN_MM of each other across the whole drawing."""
    nodes = nodes.copy()
    for e in edges:
        # centre lines from a skeleton are noisier than outlines: allow a little more before
        # calling an edge a curve (still under one thread width)
        pieces = geometry.fit(e["pts"], False, tol=LINE_TOL_MM)
        e["straight"] = len(pieces) == 1 and pieces[0]["kind"] == "line"
        if not e["straight"] and pieces:
            # (centre lines lean more than outlines do: junctions drag their ends about)
            e["pts"] = geometry._join(geometry._snap(pieces, axes, False, snap_deg=6.0,
                                                     snap_max_deg=AXIS_DEG))
    nodes, twin, mid = symmetrize(nodes)
    if twin is not None:
        _mirror_edges(edges, twin, mid)
    for axis in (1, 0):                               # 1: level edges share y; 0: uprights share x
        uf = _UF(len(nodes))
        for e in edges:
            if not e["straight"] or e["a"] == e["b"]:
                continue
            d = nodes[e["b"]] - nodes[e["a"]]
            ang = math.degrees(math.atan2(abs(d[1]), abs(d[0])))      # 0 = level, 90 = upright
            if (axis == 1 and ang <= AXIS_DEG) or (axis == 0 and ang >= 90 - AXIS_DEG):
                uf.union(e["a"], e["b"])
        groups = {}
        for i in range(len(nodes)):
            groups.setdefault(uf.find(i), []).append(i)
        groups = [g for g in groups.values() if len(g) > 1]
        values = [float(np.mean(nodes[g, axis])) for g in groups]
        # line up rows that are within ALIGN_MM (windows on one level)
        order = np.argsort(values)
        merged, cur = [], []
        for k in order:
            if cur and values[k] - values[cur[-1]] > ALIGN_MM:
                merged.append(cur)
                cur = []
            cur.append(int(k))
        if cur:
            merged.append(cur)
        for ks in merged:
            members = [i for k in ks for i in groups[k]]
            v = float(np.mean(nodes[members, axis]))
            nodes[members, axis] = v
    for e in edges:
        a, b = nodes[e["a"]], nodes[e["b"]]
        if e["straight"]:
            e["pts"] = np.vstack([a, b])
        else:
            # move a curve's ends to the new node positions, bending it gently to suit
            p = e["pts"]
            t = np.linspace(0.0, 1.0, len(p))[:, None]
            e["pts"] = p + (1 - t) * (a - p[0]) + t * (b - p[-1])
    return nodes


def symmetrize(nodes: np.ndarray) -> np.ndarray:
    """Make a drawing that is nearly mirror-symmetric (a building front, an emblem) exactly
    so. Tracing leaves each side a little different: one tower leans, a window sits lower.
    Those are the first things the eye catches, and a digitizer draws the two sides alike.
    Finds the upright axis that pairs the most junctions with a twin on the other side; if
    most have one, each pair is moved to its average position and junctions on the axis are
    put exactly on it."""
    n = len(nodes)
    if n < MIRROR_MIN_NODES:
        return nodes, None, 0.0
    xs, ys = nodes[:, 0], nodes[:, 1]

    def twins(c):
        mx = 2.0 * c - xs
        d = np.hypot(mx[:, None] - xs[None, :], ys[:, None] - ys[None, :])
        j = d.argmin(axis=1)
        ok = d[np.arange(n), j] <= MIRROR_TOL_MM
        mutual = ok & (j[j] == np.arange(n))          # each is the other's nearest twin
        return j, mutual

    c0 = (xs.min() + xs.max()) / 2.0
    best = None
    for c in c0 + np.linspace(-1.5, 1.5, 31):
        j, mutual = twins(c)
        score = int(mutual.sum())
        if best is None or score > best[0]:
            best = (score, c)
    j, mutual = twins(best[1])
    pairs = [(i, int(j[i])) for i in range(n) if mutual[i] and i < j[i]]
    if pairs:                                          # the axis the pairs themselves agree on
        c = float(np.mean([(xs[a] + xs[b]) / 2.0 for a, b in pairs]))
        j, mutual = twins(c)
    else:
        c = best[1]
    if mutual.sum() < MIRROR_SHARE * n:
        return nodes, None, 0.0
    out = nodes.copy()
    for i in range(n):
        k = int(j[i])
        if not mutual[i] or k < i:
            continue
        if k == i:                                     # on the axis
            out[i, 0] = c
            continue
        half = (abs(xs[i] - c) + abs(xs[k] - c)) / 2.0
        y = (ys[i] + ys[k]) / 2.0
        out[i] = (c - half if xs[i] < c else c + half, y)
        out[k] = (c - half if xs[k] < c else c + half, y)
    twin = np.where(mutual, j, -1)
    return out, twin, c


def _resampled(pts: np.ndarray, n: int) -> np.ndarray:
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    t = np.linspace(0.0, d[-1], n)
    return np.column_stack([np.interp(t, d, pts[:, 0]), np.interp(t, d, pts[:, 1])])


def _mirror_edges(edges: list[dict], twin: np.ndarray, c: float) -> None:
    """Give the two sides of a symmetric drawing the same lines: each edge takes the shape
    of its mirror twin. If either is straight, both are; otherwise both take the simpler
    of the two curves. An edge that crosses the axis is averaged with its own reflection."""
    by = {}
    for k, e in enumerate(edges):
        by.setdefault((min(e["a"], e["b"]), max(e["a"], e["b"])), []).append(k)
    for k, e in enumerate(edges):
        ta, tb = int(twin[e["a"]]), int(twin[e["b"]])
        if ta < 0 or tb < 0:
            continue
        mine, theirs = (min(e["a"], e["b"]), max(e["a"], e["b"])), (min(ta, tb), max(ta, tb))
        if len(by.get(mine, [])) != 1 or len(by.get(theirs, [])) != 1:
            continue                                  # two lines between the same junctions
        if mine == theirs:                            # crosses the axis: its own twin
            if not e["straight"] and e["a"] != e["b"] and len(e["pts"]) >= 2:
                n = max(8, len(e["pts"]))
                fwd = _resampled(np.asarray(e["pts"], float), n)
                back = fwd[::-1].copy()
                back[:, 0] = 2.0 * c - back[:, 0]
                e["pts"] = (fwd + back) / 2.0
            continue
        m = by[theirs][0]
        if m < k:
            continue
        o = edges[m]
        if e["straight"] or o["straight"]:
            e["straight"] = o["straight"] = True
            continue
        src, dst = (e, o) if len(e["pts"]) <= len(o["pts"]) else (o, e)
        pts = np.asarray(src["pts"], float).copy()
        pts[:, 0] = 2.0 * c - pts[:, 0]
        if int(twin[src["a"]]) != dst["a"]:
            pts = pts[::-1]
        dst["pts"] = pts


def tours(nodes: np.ndarray, edges: list[dict], start_hint=None) -> list[list[np.ndarray]]:
    """One continuous route per connected part of the network that sews every edge twice:
    out along each branch and back. Returns, per part, the list of edge polylines in sewing
    order (each already oriented in the direction it is sewn)."""
    adj = {}
    for k, e in enumerate(edges):
        adj.setdefault(e["a"], []).append(k)
        adj.setdefault(e["b"], []).append(k)
    used = [False] * len(edges)
    seen_node = set()
    out = []
    remaining = set(adj)
    while remaining:
        # start each part at its left-most bottom node (sewing works left to right)
        part = _component(adj, edges, next(iter(remaining)))
        remaining -= part
        start = min(part, key=lambda i: (nodes[i][0], -nodes[i][1]))
        route = []
        seen_node.add(start)
        order = {start: iter(sorted(adj[start], key=lambda k: _edge_angle(edges[k], start, nodes)))}
        stack = [(start, None)]                       # (node, the way we came in)
        while stack:
            u, came = stack[-1]
            k = next((k for k in order[u] if not used[k]), None)
            if k is None:
                stack.pop()
                if came is not None:
                    route.append(came[::-1])          # back along the way we came
                continue
            used[k] = True
            e = edges[k]
            v = e["b"] if e["a"] == u else e["a"]
            there = e["pts"] if e["a"] == u else e["pts"][::-1]
            route.append(there)
            if v in seen_node:
                route.append(there[::-1])             # a loop closes: straight back
            else:
                seen_node.add(v)
                order[v] = iter(sorted(adj[v], key=lambda q: _edge_angle(edges[q], v, nodes)))
                stack.append((v, there))
        if route:
            out.append(route)
    return out


def _component(adj, edges, s):
    seen, stack = {s}, [s]
    while stack:
        u = stack.pop()
        for k in adj[u]:
            e = edges[k]
            for v in (e["a"], e["b"]):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
    return seen


def _edge_angle(e, u, nodes):
    there = e["pts"] if e["a"] == u else e["pts"][::-1]
    d = there[min(len(there) - 1, 1)] - there[0]
    return math.atan2(d[1], d[0])


def network(paths_mm: list[np.ndarray], axes: list[float], spur: float = SPUR_MM):
    """Centre-line paths -> list of routes (each a list of oriented edge polylines)."""
    paths_mm = [np.asarray(p, float) for p in paths_mm if len(p) >= 2]
    if not paths_mm:
        return []
    nodes, edges = build(paths_mm)
    edges = prune(nodes, edges, spur)
    if not edges:
        return []
    nodes = straighten(nodes, edges, axes)
    edges = [e for e in edges if LineString(e["pts"]).length >= 0.15 or e["a"] != e["b"]]
    return tours(nodes, edges)


def route_points(route: list[np.ndarray], run_points, rules: dict,
                 back_pass=None) -> list[tuple[float, float]]:
    """Needle points along a route. Each edge is stitched once and the return pass reuses
    the same needle holes, so the two passes lie exactly on top of each other. With
    `back_pass` (the bean function) the way back is sewn as a bean: four passes in all, a
    bold line that still ends where it started."""
    cache = {}
    pts: list = []
    for poly in route:
        key = np.round(np.asarray(poly, float), 4).tobytes()
        rkey = np.round(np.asarray(poly, float)[::-1], 4).tobytes()
        if rkey in cache:
            seg = cache[rkey][::-1]
            if back_pass is not None:
                seg = back_pass(seg)
        else:
            seg = run_points(LineString(poly), rules)
            cache[key] = seg
        if pts and math.dist(pts[-1], seg[0]) < 1e-6:
            pts.extend(seg[1:])
        else:
            pts.extend(seg)
    return pts


def length(route: list[np.ndarray]) -> float:
    return float(sum(LineString(p).length for p in route if len(p) >= 2))


def covers(route: list[np.ndarray]):
    return [LineString(p) for p in route if len(p) >= 2]


__all__ = ["network", "route_points", "length", "covers", "Point"]
