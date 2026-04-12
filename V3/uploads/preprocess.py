# Copyright 2022 Matthias Mueller - Ten Minute Physics
# https://www.youtube.com/channel/UCTG_vrRdKYfrpqCv_WV4eyA
# www.matthiasMueller.info/tenMinutePhysics
#
# Standalone OBJ -> Tetrahedralization -> JSON converter
# Ported from the original Blender plugin by Matthias Mueller.
# Requires no Blender or bpy dependencies — only the Python standard library.
#
# Usage:
#   python obj_to_tet_json.py input.obj output.json [options]
#
# Options:
#   --resolution INT      Interior grid resolution (default: 10). Use >=5 for solid
#                         volumes; 0 = surface vertices only (thin shells, not solids).
#   --min-quality FLOAT   Minimum tet quality threshold (default: 0.001)
#   --one-face-per-tet    Store shared vertex indices (default: True)
#   --scale FLOAT         Tet shrink scale for exploded view (default: 1.0)
#
# Output JSON schema:
#   {
#     "tet": {
#       "verts": [x,y,z,x,y,z,...],
#       "tetIds": [i0,i1,i2,i3,...],
#       "edgeIds": [a,b,a,b,...],
#       "surfTriIds": [i0,i1,i2,...]
#     },
#     "vis": {
#       "verts": [x,y,z,x,y,z,...],
#       "triIds": [i0,i1,i2,...],
#       "skinning": [tetIndex,w0,w1,w2,w3,...]  # per vis vertex
#     },
#     "meta": {
#       "nominalRadius": float
#     }
#   }

import argparse
import json
import math
import os
import sys
from random import random
from functools import cmp_to_key


# ---------------------------------------------------------------------------
# Minimal 3-D vector (replaces mathutils.Vector)
# ---------------------------------------------------------------------------

class Vec3:
    __slots__ = ("x", "y", "z")

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x); self.y = float(y); self.z = float(z)

    def __add__(self, o): return Vec3(self.x+o.x, self.y+o.y, self.z+o.z)
    def __sub__(self, o): return Vec3(self.x-o.x, self.y-o.y, self.z-o.z)
    def __mul__(self, s): return Vec3(self.x*s, self.y*s, self.z*s)
    def __rmul__(self, s): return self.__mul__(s)
    def __truediv__(self, s): return Vec3(self.x/s, self.y/s, self.z/s)
    def __getitem__(self, i): return (self.x, self.y, self.z)[i]
    def __setitem__(self, i, v):
        if i == 0: self.x = float(v)
        elif i == 1: self.y = float(v)
        else: self.z = float(v)

    def dot(self, o): return self.x*o.x + self.y*o.y + self.z*o.z

    def cross(self, o):
        return Vec3(self.y*o.z - self.z*o.y,
                    self.z*o.x - self.x*o.z,
                    self.x*o.y - self.y*o.x)

    @property
    def magnitude(self):
        return math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)

    def normalize(self):
        m = self.magnitude
        if m > 0: self.x /= m; self.y /= m; self.z /= m

    def copy(self): return Vec3(self.x, self.y, self.z)


# ---------------------------------------------------------------------------
# BVH tree for ray-casting (replaces mathutils.bvhtree.BVHTree)
# ---------------------------------------------------------------------------

class BVHTree:
    """Triangle-soup BVH for interior point testing via ray casting."""

    def __init__(self, triangles):
        self.tris = triangles
        self.normals = []
        for p0, p1, p2 in triangles:
            n = (p1 - p0).cross(p2 - p0)
            n.normalize()
            self.normals.append(n)

    @staticmethod
    def _ray_tri(orig, dirv, p0, p1, p2):
        """Möller–Trumbore intersection. Returns t or None."""
        eps = 1e-9
        e1 = p1 - p0; e2 = p2 - p0
        h = dirv.cross(e2); a = e1.dot(h)
        if abs(a) < eps: return None
        f = 1.0 / a; s = orig - p0; u = f * s.dot(h)
        if u < 0.0 or u > 1.0: return None
        q = s.cross(e1); v = f * dirv.dot(q)
        if v < 0.0 or u + v > 1.0: return None
        t = f * e2.dot(q)
        return t if t > eps else None

    def ray_cast(self, origin, direction):
        """Returns (location, normal, index, distance) or (None,None,None,None)."""
        best_t = float('inf'); best_i = -1
        for i, (p0, p1, p2) in enumerate(self.tris):
            t = self._ray_tri(origin, direction, p0, p1, p2)
            if t is not None and t < best_t:
                best_t = t; best_i = i
        if best_i < 0:
            return None, None, None, None
        return origin + direction * best_t, self.normals[best_i], best_i, best_t


# ---------------------------------------------------------------------------
# Helpers (direct ports)
# ---------------------------------------------------------------------------

# Fix this in preprocess.py
DIRS = [
    Vec3(1,0,0), Vec3(-1,0,0), # Must have both positive and negative X
    Vec3(0,1,0), Vec3(0,-1,0), 
    Vec3(0,0,1), Vec3(0,0,-1),
]

TET_FACES = [[2,1,0], [0,1,3], [1,2,3], [2,0,3]]


def is_inside(tree, p, min_dist=0.0):
    num_in = 0
    for d in DIRS:
        loc, normal, idx, dist = tree.ray_cast(p, d)
        if normal is not None:
            if normal.dot(d) > 0.0:
                num_in += 1
            if min_dist > 0.0 and dist < min_dist:
                return False
    return num_in > 3


def get_circum_center(p0, p1, p2, p3):
    b = p1-p0; c = p2-p0; d = p3-p0
    det = 2.0*(b.x*(c.y*d.z-c.z*d.y)-b.y*(c.x*d.z-c.z*d.x)+b.z*(c.x*d.y-c.y*d.x))
    if det == 0.0: return p0.copy()
    v = c.cross(d)*b.dot(b) + d.cross(b)*c.dot(c) + b.cross(c)*d.dot(d)
    return p0 + v/det


def tet_quality(p0, p1, p2, p3):
    d0=p1-p0; d1=p2-p0; d2=p3-p0
    d3=p2-p1; d4=p3-p2; d5=p1-p3
    ms=(d0.magnitude**2+d1.magnitude**2+d2.magnitude**2
       +d3.magnitude**2+d4.magnitude**2+d5.magnitude**2)/6.0
    rms = math.sqrt(ms)
    if rms == 0: return 0.0
    vol = d0.dot(d1.cross(d2)) / 6.0
    return (12.0/math.sqrt(2.0)) * vol / (rms**3)


def compare_edges(e0, e1):
    return -1 if (e0[0]<e1[0] or (e0[0]==e1[0] and e0[1]<e1[1])) else 1

def equal_edges(e0, e1):
    return e0[0]==e1[0] and e0[1]==e1[1]

def rand_eps():
    eps = 0.0001
    return -eps + 2.0*random()*eps


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def _flatten_vec3_list(vecs):
    out = []
    for v in vecs:
        out.extend([float(v.x), float(v.y), float(v.z)])
    return out


def _flatten_xyz_list(xyz):
    # xyz: [[x,y,z], ...] or tuples
    out = []
    for p in xyz:
        out.extend([float(p[0]), float(p[1]), float(p[2])])
    return out


def _tet_edges_from_tet_ids(tet_ids):
    # Returns deduped edge list as flat [a,b,a,b,...] with a<b.
    edges = set()
    nt = len(tet_ids) // 4
    for ti in range(nt):
        a, b, c, d = tet_ids[4*ti:4*ti+4]
        pairs = ((a,b),(a,c),(a,d),(b,c),(b,d),(c,d))
        for u, v in pairs:
            if u == v:
                continue
            if u < v:
                edges.add((u, v))
            else:
                edges.add((v, u))
    out = []
    for u, v in sorted(edges):
        out.extend([u, v])
    return out


def _surface_tris_from_tet_ids(tet_ids):
    """
    Returns surface triangles as flat triples [i0,i1,i2,...].
    A face is on the surface if it belongs to exactly one tet.
    """
    face_count = {}
    face_first = {}
    nt = len(tet_ids) // 4
    for ti in range(nt):
        ids = tet_ids[4*ti:4*ti+4]
        for fi in range(4):
            tri = (ids[TET_FACES[fi][0]], ids[TET_FACES[fi][1]], ids[TET_FACES[fi][2]])
            key = tuple(sorted(tri))
            face_count[key] = face_count.get(key, 0) + 1
            if key not in face_first:
                face_first[key] = tri
    out = []
    for key, cnt in face_count.items():
        if cnt == 1:
            tri = face_first[key]
            out.extend([tri[0], tri[1], tri[2]])
    return out


def _compact_tris(verts_flat, tri_ids):
    """
    Build a compacted vertex buffer containing only vertices referenced by tri_ids.
    - verts_flat: [x,y,z,...] (tet space indices)
    - tri_ids: [i0,i1,i2,...] referencing tet vertex indices
    Returns (vis_verts_flat, vis_tri_ids_flat, tet_to_vis_index)
    """
    used = set(tri_ids)
    mapping = {}
    vis_verts = []
    for old_i in sorted(used):
        mapping[old_i] = len(mapping)
        base = 3 * old_i
        vis_verts.extend([verts_flat[base], verts_flat[base+1], verts_flat[base+2]])
    vis_tris = [mapping[i] for i in tri_ids]
    return vis_verts, vis_tris, mapping


def _incident_tet_for_vertex(tet_ids):
    """Returns list mapping vertex index -> some incident tet index."""
    nt = len(tet_ids) // 4
    inc = {}
    for ti in range(nt):
        a, b, c, d = tet_ids[4*ti:4*ti+4]
        for v in (a, b, c, d):
            if v not in inc:
                inc[v] = ti
    return inc


def _skinning_for_surface_vis(mapping_tet_to_vis, tet_ids):
    """
    For each vis vertex (which is a subset of tet verts), pick an incident tet and
    use exact corner barycentric weights.
    Output format: [tetIndex,w0,w1,w2,w3] per vis vertex (flat).
    """
    inc = _incident_tet_for_vertex(tet_ids)
    # invert mapping to get visIndex -> tetVertexIndex
    vis_to_tet = [0] * len(mapping_tet_to_vis)
    for tet_i, vis_i in mapping_tet_to_vis.items():
        vis_to_tet[vis_i] = tet_i

    skin = []
    for vis_i, tet_vert in enumerate(vis_to_tet):
        ti = inc.get(tet_vert, 0)
        a, b, c, d = tet_ids[4*ti:4*ti+4]
        # Corner weights (exact)
        if tet_vert == a:
            w = (1.0, 0.0, 0.0, 0.0)
        elif tet_vert == b:
            w = (0.0, 1.0, 0.0, 0.0)
        elif tet_vert == c:
            w = (0.0, 0.0, 1.0, 0.0)
        elif tet_vert == d:
            w = (0.0, 0.0, 0.0, 1.0)
        else:
            # Shouldn't happen for surface-compacted mesh, but keep sane defaults.
            w = (0.25, 0.25, 0.25, 0.25)
        skin.extend([int(ti), float(w[0]), float(w[1]), float(w[2]), float(w[3])])
    return skin


def _nominal_radius_from_edges(verts_flat, edge_ids):
    # Heuristic: quarter of mean unique edge length.
    if not edge_ids:
        return 1.0
    total = 0.0
    cnt = 0
    for i in range(0, len(edge_ids), 2):
        a = edge_ids[i]
        b = edge_ids[i+1]
        ax, ay, az = verts_flat[3*a], verts_flat[3*a+1], verts_flat[3*a+2]
        bx, by, bz = verts_flat[3*b], verts_flat[3*b+1], verts_flat[3*b+2]
        dx, dy, dz = ax - bx, ay - by, az - bz
        total += math.sqrt(dx*dx + dy*dy + dz*dz)
        cnt += 1
    mean_len = total / max(1, cnt)
    r = mean_len * 0.25
    return float(r if r > 0.0 else 1.0)


# ---------------------------------------------------------------------------
# Delaunay tetrahedralization (direct port from createTetIds)
# ---------------------------------------------------------------------------

def create_tet_ids(verts, tree, min_quality):
    tet_ids=[]; neighbors=[]; tet_marks=[]; tet_mark=0; first_free=-1
    planes_n=[]; planes_d=[]
    first_big = len(verts)-4

    tet_ids += [first_big, first_big+1, first_big+2, first_big+3]
    tet_marks.append(0)
    for i in range(4):
        neighbors.append(-1)
        p0=verts[first_big+TET_FACES[i][0]]
        p1=verts[first_big+TET_FACES[i][1]]
        p2=verts[first_big+TET_FACES[i][2]]
        n=(p1-p0).cross(p2-p0); n.normalize()
        planes_n.append(n); planes_d.append(p0.dot(n))

    print("--- tetrahedralization ---")
    for i in range(first_big):
        p = verts[i]
        if i % 100 == 0:
            print(f"  inserting vertex {i+1}/{first_big}")

        tet_nr = 0
        while tet_ids[4*tet_nr] < 0:
            tet_nr += 1

        tet_mark += 1; found = False
        while not found:
            if tet_nr < 0 or tet_marks[tet_nr] == tet_mark: break
            tet_marks[tet_nr] = tet_mark
            id0,id1,id2,id3 = (tet_ids[4*tet_nr+k] for k in range(4))
            center = (verts[id0]+verts[id1]+verts[id2]+verts[id3])*0.25
            min_t=float('inf'); min_face=-1
            for j in range(4):
                n=planes_n[4*tet_nr+j]; d=planes_d[4*tet_nr+j]
                hp=n.dot(p)-d; hc=n.dot(center)-d
                t=hp-hc
                if t==0: continue
                t=-hc/t
                if t>=0.0 and t<min_t: min_t=t; min_face=j
            if min_t >= 1.0: found=True
            else: tet_nr=neighbors[4*tet_nr+min_face]

        if not found:
            print(f"  WARNING: failed to insert vertex {i}"); continue

        tet_mark += 1; violating=[]; stack=[tet_nr]
        while stack:
            tet_nr=stack.pop()
            if tet_marks[tet_nr]==tet_mark: continue
            tet_marks[tet_nr]=tet_mark; violating.append(tet_nr)
            for j in range(4):
                nb=neighbors[4*tet_nr+j]
                if nb<0 or tet_marks[nb]==tet_mark: continue
                id0,id1,id2,id3=(tet_ids[4*nb+k] for k in range(4))
                c=get_circum_center(verts[id0],verts[id1],verts[id2],verts[id3])
                r=(verts[id0]-c).magnitude
                if (p-c).magnitude < r: stack.append(nb)

        edges=[]
        for j in range(len(violating)):
            tet_nr=violating[j]
            ids=[tet_ids[4*tet_nr+k] for k in range(4)]
            ns=[neighbors[4*tet_nr+k] for k in range(4)]
            tet_ids[4*tet_nr]=-1; tet_ids[4*tet_nr+1]=first_free; first_free=tet_nr

            for k in range(4):
                nb=ns[k]
                if nb>=0 and tet_marks[nb]==tet_mark: continue
                new_tet=first_free
                if new_tet>=0:
                    first_free=tet_ids[4*first_free+1]
                else:
                    new_tet=len(tet_ids)//4; tet_marks.append(0)
                    for _ in range(4):
                        tet_ids.append(-1); neighbors.append(-1)
                        planes_n.append(Vec3()); planes_d.append(0.0)

                id0=ids[TET_FACES[k][2]]; id1=ids[TET_FACES[k][1]]; id2=ids[TET_FACES[k][0]]
                tet_ids[4*new_tet]=id0; tet_ids[4*new_tet+1]=id1
                tet_ids[4*new_tet+2]=id2; tet_ids[4*new_tet+3]=i
                neighbors[4*new_tet]=nb
                if nb>=0:
                    for l in range(4):
                        if neighbors[4*nb+l]==tet_nr: neighbors[4*nb+l]=new_tet
                neighbors[4*new_tet+1]=-1; neighbors[4*new_tet+2]=-1; neighbors[4*new_tet+3]=-1

                for l in range(4):
                    pp0=verts[tet_ids[4*new_tet+TET_FACES[l][0]]]
                    pp1=verts[tet_ids[4*new_tet+TET_FACES[l][1]]]
                    pp2=verts[tet_ids[4*new_tet+TET_FACES[l][2]]]
                    nn=(pp1-pp0).cross(pp2-pp0); nn.normalize()
                    planes_n[4*new_tet+l]=nn; planes_d[4*new_tet+l]=nn.dot(pp0)

                a,b2=id0,id1
                if a<b2: edges.append((a,b2,new_tet,1))
                else:    edges.append((b2,a,new_tet,1))
                a,b2=id1,id2
                if a<b2: edges.append((a,b2,new_tet,2))
                else:    edges.append((b2,a,new_tet,2))
                a,b2=id2,id0
                if a<b2: edges.append((a,b2,new_tet,3))
                else:    edges.append((b2,a,new_tet,3))

        sorted_edges=sorted(edges,key=cmp_to_key(compare_edges))
        nr=0; ne=len(sorted_edges)
        while nr<ne:
            e0=sorted_edges[nr]; nr+=1
            if nr<ne and equal_edges(sorted_edges[nr],e0):
                e1=sorted_edges[nr]
                neighbors[4*e0[2]+e0[3]]=e1[2]; neighbors[4*e1[2]+e1[3]]=e0[2]; nr+=1

    # filter
    num_tets=len(tet_ids)//4; out=[]; num_bad=0
    for i in range(num_tets):
        id0,id1,id2,id3=(tet_ids[4*i+k] for k in range(4))
        if id0<0 or id0>=first_big or id1>=first_big or id2>=first_big or id3>=first_big: continue
        p0,p1,p2,p3=verts[id0],verts[id1],verts[id2],verts[id3]
        if tet_quality(p0,p1,p2,p3)<min_quality: num_bad+=1; continue
        c=(p0+p1+p2+p3)*0.25
        if not is_inside(tree,c): continue
        out+=[id0,id1,id2,id3]

    print(f"  {num_bad} bad tets removed, {len(out)//4} tets kept")
    return out


# ---------------------------------------------------------------------------
# OBJ reader
# ---------------------------------------------------------------------------

def _resolve_path(p: str) -> str:
    """
    Make CLI paths robust to different working directories.

    - Absolute paths are kept as-is.
    - Relative paths are first tried relative to this script's directory
      (so `uploads\\foo.obj` works regardless of where Python is launched).
    - If that doesn't exist, fall back to the original relative path.
    """
    if not p:
        return p
    p = os.path.expandvars(os.path.expanduser(p))
    if os.path.isabs(p):
        return os.path.normpath(p)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    v3_dir = os.path.dirname(script_dir)  # preprocess.py is typically in V3/uploads
    cwd = os.getcwd()

    candidates = [
        os.path.normpath(os.path.join(script_dir, p)),
        os.path.normpath(os.path.join(v3_dir, p)),
        os.path.normpath(os.path.join(cwd, p)),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return os.path.normpath(p)

def _resolve_output_path(p: str) -> str:
    """
    Resolve an output path in a user-friendly way, without requiring it to exist.

    Heuristic:
    - Absolute paths are kept as-is.
    - Relative paths with a directory component are interpreted relative to `V3/`
      (parent of this script's directory, typically `V3/uploads`).
    - Bare filenames (no directory component) are written relative to cwd.
    """
    if not p:
        return p
    p = os.path.expandvars(os.path.expanduser(p))
    if os.path.isabs(p):
        return os.path.normpath(p)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    v3_dir = os.path.dirname(script_dir)
    dir_part = os.path.dirname(p)
    base_dir = os.getcwd() if dir_part in ("", ".", None) else v3_dir
    return os.path.normpath(os.path.join(base_dir, p))

def load_obj(path):
    raw_verts=[]; raw_faces=[]
    path = _resolve_path(path)
    try:
        f = open(path, encoding="utf-8")
    except FileNotFoundError as e:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        v3_dir = os.path.dirname(script_dir)
        raise FileNotFoundError(
            f"{e}\nTried path: {path}\n(For relative paths, this tool searches relative to: cwd, {v3_dir}, {script_dir})"
        ) from None
    with f:
        for line in f:
            line=line.strip()
            if not line or line.startswith('#'): continue
            parts=line.split()
            if parts[0]=='v':
                raw_verts.append(Vec3(float(parts[1]),float(parts[2]),float(parts[3])))
            elif parts[0]=='f':
                indices=[int(p.split('/')[0])-1 for p in parts[1:]]
                raw_faces.append(indices)

    triangles=[]
    for face in raw_faces:
        for k in range(1,len(face)-1):
            triangles.append((raw_verts[face[0]],raw_verts[face[k]],raw_verts[face[k+1]]))

    return raw_verts, triangles


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def obj_to_tet_json(obj_path, json_path, resolution=10, min_quality=0.001,
                    one_face_per_tet=True, scale=1.0):
    print(f"Loading {obj_path} …")
    raw_verts, triangles = load_obj(obj_path)
    if not triangles:
        sys.exit("No triangles found — check that the OBJ has 'f' face entries.")
    print(f"  {len(raw_verts)} vertices, {len(triangles)} triangles")

    tree = BVHTree(triangles)

    tet_verts=[Vec3(v.x+rand_eps(),v.y+rand_eps(),v.z+rand_eps()) for v in raw_verts]

    inf=float('inf'); bmin=Vec3(inf,inf,inf); bmax=Vec3(-inf,-inf,-inf)
    center=Vec3()
    for p in tet_verts:
        center=center+p
        for ax in range(3):
            bmin[ax]=min(bmin[ax],p[ax]); bmax[ax]=max(bmax[ax],p[ax])
    center=center/len(tet_verts)
    radius=max((p-center).magnitude for p in tet_verts)

    if resolution == 0:
        print("  Warning: --resolution 0 adds no interior seeds (surface verts only). "
              "Closed solids often get few or no valid tets; use --resolution >= 5 for solids.")
    if resolution>0:
        dims=bmax-bmin
        dim=max(dims.x,dims.y,dims.z); h=dim/resolution
        xi=0
        while xi*h<=dims.x:
            x=bmin.x+xi*h+rand_eps(); xi+=1
            yi=0
            while yi*h<=dims.y:
                y=bmin.y+yi*h+rand_eps(); yi+=1
                zi=0
                while zi*h<=dims.z:
                    z=bmin.z+zi*h+rand_eps(); zi+=1
                    q=Vec3(x,y,z)
                    if is_inside(tree,q,0.5*h): tet_verts.append(q)

    print(f"  {len(tet_verts)} sample points total")

    s = 5.0 * radius
    tet_verts += [
        Vec3(center.x - s, center.y, center.z - s),
        Vec3(center.x + s, center.y, center.z - s),
        Vec3(center.x, center.y + s, center.z + s),
        Vec3(center.x, center.y - s, center.z + s),
    ]

    tet_id_list=create_tet_ids(tet_verts,tree,min_quality)
    num_tets=len(tet_id_list)//4
    num_src=len(raw_verts); num_pts=len(tet_verts)-4

    if one_face_per_tet:
        out_verts=[[v.x,v.y,v.z] for v in raw_verts]
        for i in range(num_src,num_pts):
            v=tet_verts[i]; out_verts.append([v.x,v.y,v.z])
        out_tet_ids=tet_id_list[:]
    else:
        out_verts=[]; out_tet_ids=[]
        for i in range(num_tets):
            c=(tet_verts[tet_id_list[4*i]]+tet_verts[tet_id_list[4*i+1]]
              +tet_verts[tet_id_list[4*i+2]]+tet_verts[tet_id_list[4*i+3]])*0.25
            base=len(out_verts)
            for j in range(4):
                for k in range(3):
                    p=tet_verts[tet_id_list[4*i+TET_FACES[j][k]]]
                    p=c+(p-c)*scale; out_verts.append([p.x,p.y,p.z])
            for j in range(4):
                out_tet_ids+=[base+j*3,base+j*3+1,base+j*3+2]

    # Build required derived data for the tool.
    tet_verts_flat = _flatten_xyz_list(out_verts)
    tet_edge_ids = _tet_edges_from_tet_ids(out_tet_ids)
    tet_surf_tri_ids = _surface_tris_from_tet_ids(out_tet_ids)

    # Starter visual mesh: the tet surface (compacted).
    vis_verts_flat, vis_tri_ids, tet_to_vis = _compact_tris(tet_verts_flat, tet_surf_tri_ids)
    vis_skinning = _skinning_for_surface_vis(tet_to_vis, out_tet_ids)

    # This tool treats nominalRadius as the physical particle radius used for
    # broadphase/contact spacing. Use the mesh bounding-sphere radius (from the
    # original vertex pass) instead of a tet edge-length heuristic.
    nominal_radius = float(radius if radius > 0.0 else 1.0)

    result = {
        "tet": {
            "verts": tet_verts_flat,
            "tetIds": out_tet_ids,
            "edgeIds": tet_edge_ids,
            "surfTriIds": tet_surf_tri_ids,
        },
        "vis": {
            "verts": vis_verts_flat,
            "triIds": vis_tri_ids,
            "skinning": vis_skinning,
        },
        "meta": {
            "nominalRadius": nominal_radius,
        }
    }
    json_path = _resolve_output_path(json_path)
    out_dir = os.path.dirname(json_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result,f)

    print(f"Saved -> {json_path}")
    print(f"  {len(out_verts)} tet vertices, {num_tets} tetrahedra")
    print(f"  {len(tet_edge_ids)//2} unique edges, {len(tet_surf_tri_ids)//3} surface triangles")
    print(f"  {len(vis_verts_flat)//3} visual vertices, {len(vis_tri_ids)//3} visual triangles")
    print(f"  nominalRadius={nominal_radius:g}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser=argparse.ArgumentParser(
        description="Convert an OBJ mesh to a tetrahedralized JSON (no Blender required).")
    parser.add_argument("input",  help="Input .obj file")
    parser.add_argument("output", help="Output .json file")
    parser.add_argument("--resolution",       type=int,   default=10,
                        help="Interior grid resolution (default: 10; >=5 for solids; 0=surface only, for thin shells)")
    parser.add_argument("--min-quality",      type=float, default=0.001,
                        help="Minimum tet quality 0–1 (default: 0.001)")
    parser.add_argument("--one-face-per-tet", action="store_true", default=True,
                        help="Shared vertex indices mode (default: True)")
    parser.add_argument("--scale",            type=float, default=1.0,
                        help="Tet shrink scale for exploded view (default: 1.0)")
    args=parser.parse_args()

    obj_to_tet_json(args.input, args.output,
                    resolution=args.resolution,
                    min_quality=args.min_quality,
                    one_face_per_tet=args.one_face_per_tet,
                    scale=args.scale)

if __name__=="__main__":
    main()