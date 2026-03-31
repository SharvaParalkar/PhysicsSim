#!/usr/bin/env python3
"""
preprocess_particle.py
======================
Converts a particle OBJ file into a simulation-ready JSON for the
syringe gel-particle demo (Examples/30-syringeGelParticles.html).

Output JSON contains:
  - A coarse tetrahedral mesh  (tet verts, tetIds, edgeIds, tet-surface triIds)
  - The original visual surface mesh (vis verts, vis triIds)
  - Skinning weights (per vis-vertex: [tetIdx, b0, b1, b2])
  - Meta information (units, bounding radius, etc.)

Usage:
    python preprocess_particle.py input.obj [output.json]
                                  [--res N] [--scale S]

Options:
    --res N     Tet-lattice resolution (default 6, range 4-12).
                Higher = finer tet mesh; diminishing returns above 8 for
                particles ~600 µm diam.
    --scale S   Multiply OBJ coordinates by S to get millimetres.
                Default 0.001  (OBJ in microns → mm).

Example:
    python preprocess_particle.py ../V2/Particles/Cyl600M.obj cylinder.json --res 6

The output JSON can be loaded via the "Load particle JSON" button in the demo.
"""

import sys
import math
import json
import os
import argparse
from collections import defaultdict


# ── Math helpers ────────────────────────────────────────────────────────────

def sub(a, b):   return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]
def add(a, b):   return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]
def scl(a, s):   return [a[0]*s,    a[1]*s,    a[2]*s]
def dot(a, b):   return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
def norm2(a):    return a[0]*a[0]+a[1]*a[1]+a[2]*a[2]
def cross(a, b):
    return [a[1]*b[2]-a[2]*b[1],
            a[2]*b[0]-a[0]*b[2],
            a[0]*b[1]-a[1]*b[0]]


def tet_volume(p0, p1, p2, p3):
    e1 = sub(p1, p0)
    e2 = sub(p2, p0)
    e3 = sub(p3, p0)
    return dot(e1, cross(e2, e3)) / 6.0


def solve_3x3(A_cols, b):
    """Solve A*x = b (Cramer's rule). A_cols = [col0, col1, col2]."""
    c01 = cross(A_cols[1], A_cols[2])
    det = dot(A_cols[0], c01)
    if abs(det) < 1e-14:
        return None
    inv = 1.0 / det
    d0 = dot(b,         cross(A_cols[1], A_cols[2])) * inv
    d1 = dot(A_cols[0], cross(b,         A_cols[2])) * inv
    d2 = dot(A_cols[0], cross(A_cols[1], b        )) * inv
    return [d0, d1, d2]


# ── OBJ parsing ──────────────────────────────────────────────────────────────

def parse_obj(filepath):
    """Return (vertices [[x,y,z]...], faces [[i,j,k]...]) (0-indexed)."""
    vertices, faces = [], []
    with open(filepath, 'r', encoding='utf-8', errors='replace') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if parts[0] == 'v' and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f' and len(parts) >= 4:
                idxs = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                # fan-triangulate
                for k in range(1, len(idxs) - 1):
                    faces.append([idxs[0], idxs[k], idxs[k+1]])
    return vertices, faces


def center_and_scale(vertices, scale):
    """Translate centroid to origin, scale by *scale* to get mm."""
    if not vertices:
        return []
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    zs = [v[2] for v in vertices]
    cx = (min(xs)+max(xs))*0.5
    cy = (min(ys)+max(ys))*0.5
    cz = (min(zs)+max(zs))*0.5
    return [[(v[0]-cx)*scale, (v[1]-cy)*scale, (v[2]-cz)*scale]
            for v in vertices]


def bounding_radius(vertices):
    return max(math.sqrt(norm2(v)) for v in vertices) if vertices else 0.0


def bounding_box(vertices):
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    zs = [v[2] for v in vertices]
    return (min(xs),max(xs)), (min(ys),max(ys)), (min(zs),max(zs))


# ── Point-in-mesh (ray casting) ──────────────────────────────────────────────

def ray_tri_intersect(orig, rd, v0, v1, v2):
    """Möller-Trumbore. Returns (hit:bool, t:float)."""
    EPSILON = 1e-9
    e1 = sub(v1, v0); e2 = sub(v2, v0)
    h  = cross(rd, e2)
    a  = dot(e1, h)
    if abs(a) < EPSILON:
        return False, 0.0
    f = 1.0/a; s = sub(orig, v0)
    u = f*dot(s, h)
    if u < 0.0 or u > 1.0:
        return False, 0.0
    q = cross(s, e1)
    v = f*dot(rd, q)
    if v < 0.0 or u+v > 1.0:
        return False, 0.0
    t = f*dot(e2, q)
    return t > EPSILON, t


def is_inside(point, vertices, faces):
    """Majority-vote ray-casting over 3 slightly off-axis directions."""
    rays = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.01],
        [0.01, 0.0, -1.0],
    ]
    votes = 0
    for rd in rays:
        rlen = math.sqrt(norm2(rd))
        rd = [x/rlen for x in rd]
        cnt = sum(1 for f in faces
                  if ray_tri_intersect(point, rd,
                                       vertices[f[0]], vertices[f[1]], vertices[f[2]])[0])
        if cnt % 2 == 1:
            votes += 1
    return votes >= 2


# ── Tetrahedral-mesh generation ───────────────────────────────────────────────

# Standard decomposition of a unit cube into 6 tetrahedra (vertex offsets)
_CUBE_TO_6_TETS = [
    [(0,0,0),(1,0,0),(1,1,0),(1,1,1)],
    [(0,0,0),(1,1,0),(0,1,0),(1,1,1)],
    [(0,0,0),(0,1,0),(0,1,1),(1,1,1)],
    [(0,0,0),(0,1,1),(0,0,1),(1,1,1)],
    [(0,0,0),(0,0,1),(1,0,1),(1,1,1)],
    [(0,0,0),(1,0,1),(1,0,0),(1,1,1)],
]


def generate_tet_mesh(vis_verts, faces, resolution=6):
    """
    Builds an interior tetrahedral mesh by lattice sampling.
    Returns (tet_verts, tet_ids, edge_ids, surf_tri_ids) or falls back to
    an icosahedron cage if the lattice produces no tets.
    """
    (xmin,xmax),(ymin,ymax),(zmin,zmax) = bounding_box(vis_verts)
    span = max(xmax-xmin, ymax-ymin, zmax-zmin)
    margin = 0.04 * span
    xmin -= margin; xmax += margin
    ymin -= margin; ymax += margin
    zmin -= margin; zmax += margin

    N  = resolution + 1
    dx = (xmax - xmin) / resolution
    dy = (ymax - ymin) / resolution
    dz = (zmax - zmin) / resolution

    total = N*N*N
    sys.stdout.write(f"  Testing {total} lattice points for interior ... ")
    sys.stdout.flush()

    # Map (xi,yi,zi) → index in grid_verts (or -1 if outside)
    vertex_map = {}   # (xi,yi,zi) → index
    grid_verts = []

    for xi in range(N):
        for yi in range(N):
            for zi in range(N):
                x = xmin + xi*dx
                y = ymin + yi*dy
                z = zmin + zi*dz
                if is_inside([x,y,z], vis_verts, faces):
                    vertex_map[(xi,yi,zi)] = len(grid_verts)
                    grid_verts.append([x, y, z])
                else:
                    vertex_map[(xi,yi,zi)] = -1

    inside_count = len(grid_verts)
    print(f"{inside_count} inside")

    if inside_count < 4:
        print("  Too few interior points – using icosahedron fallback.")
        r = bounding_radius(vis_verts)
        return icosahedron_tet_mesh(r)

    # Build tets from grid cubes; keep only tets where ALL 4 vertices are inside
    raw_tet_ids = []
    for xi in range(resolution):
        for yi in range(resolution):
            for zi in range(resolution):
                for offsets in _CUBE_TO_6_TETS:
                    corners = [(xi+ox, yi+oy, zi+oz) for ox,oy,oz in offsets]
                    idxs = [vertex_map.get(c, -1) for c in corners]
                    if all(i >= 0 for i in idxs):
                        raw_tet_ids.extend(idxs)

    if not raw_tet_ids:
        print("  No complete-interior tets – using icosahedron fallback.")
        r = bounding_radius(vis_verts)
        return icosahedron_tet_mesh(r)

    # Remove unused vertices; remap
    used = set(raw_tet_ids)
    old2new = {old: new for new, old in enumerate(sorted(used))}
    tet_verts = [grid_verts[i] for i in sorted(used)]
    tet_ids   = [old2new[i] for i in raw_tet_ids]

    print(f"  Raw: {len(tet_ids)//4} tets from {len(tet_verts)} verts")

    # Fix orientations so volumes are positive
    corrected = []
    neg = 0
    for t in range(0, len(tet_ids), 4):
        ids = tet_ids[t:t+4]
        p0,p1,p2,p3 = [tet_verts[i] for i in ids]
        vol = tet_volume(p0,p1,p2,p3)
        if vol < 0:
            corrected.extend([ids[0],ids[2],ids[1],ids[3]])
            neg += 1
        else:
            corrected.extend(ids)

    if neg:
        print(f"  Fixed {neg} inverted tets")

    tet_ids = corrected
    return _finalize_tet_mesh(tet_verts, tet_ids)


def _finalize_tet_mesh(tet_verts, tet_ids):
    """Extracts unique edges and surface triangles from a tet mesh."""
    # Unique edges
    edge_set = set()
    for t in range(0, len(tet_ids), 4):
        ids = tet_ids[t:t+4]
        for i in range(4):
            for j in range(i+1, 4):
                a,b = ids[i], ids[j]
                edge_set.add((min(a,b), max(a,b)))
    edge_ids = [x for pair in sorted(edge_set) for x in pair]

    # Surface triangles: tet faces with exactly one adjacent tet
    face_count   = defaultdict(int)
    face_orient  = {}   # canonical key → oriented (a,b,c)
    tet_face_orders = [(1,3,2),(0,2,3),(0,3,1),(0,1,2)]
    for t in range(0, len(tet_ids), 4):
        ids = tet_ids[t:t+4]
        for order in tet_face_orders:
            a,b,c = ids[order[0]], ids[order[1]], ids[order[2]]
            key   = tuple(sorted((a,b,c)))
            face_count[key]  += 1
            face_orient[key]  = (a,b,c)

    surf_tri_ids = []
    for key, cnt in face_count.items():
        if cnt == 1:
            surf_tri_ids.extend(face_orient[key])

    print(f"  Final: {len(tet_ids)//4} tets, {len(edge_ids)//2} edges, "
          f"{len(surf_tri_ids)//3} surface tris")
    return tet_verts, tet_ids, edge_ids, surf_tri_ids


# ── Icosahedron fallback ──────────────────────────────────────────────────────

def icosahedron_tet_mesh(radius):
    """
    13-vertex (12 outer + 1 center), 20-tet mesh approximating a sphere.
    Used as fallback when lattice tetrahedralization yields no tets.
    """
    phi  = (1.0 + math.sqrt(5.0)) * 0.5
    norm = math.sqrt(1.0 + phi*phi)
    r    = radius

    outer = [
        [0, 1/norm, phi/norm], [0, -1/norm, phi/norm],
        [0, 1/norm,-phi/norm], [0, -1/norm,-phi/norm],
        [1/norm, phi/norm, 0], [-1/norm, phi/norm, 0],
        [1/norm,-phi/norm, 0], [-1/norm,-phi/norm, 0],
        [phi/norm, 0, 1/norm], [-phi/norm, 0, 1/norm],
        [phi/norm, 0,-1/norm], [-phi/norm, 0,-1/norm],
    ]
    verts = [[x*r, y*r, z*r] for x,y,z in outer] + [[0,0,0]]  # 12: center

    ico_faces = [
        (0,8,4),(0,4,5),(0,5,9),(0,9,1),(0,1,8),
        (3,11,2),(3,2,10),(3,10,6),(3,6,7),(3,7,11),
        (1,9,7),(9,11,7),(9,5,11),(5,2,11),(5,4,2),
        (4,10,2),(4,8,10),(8,6,10),(8,1,6),(1,7,6),
    ]

    tet_ids = []
    for a,b,c in ico_faces:
        vol = tet_volume(verts[12], verts[a], verts[b], verts[c])
        if vol < 0:
            tet_ids.extend([12, a, c, b])
        else:
            tet_ids.extend([12, a, b, c])

    surf_tri_ids = [x for face in ico_faces for x in face]
    return _finalize_tet_mesh(verts, tet_ids)


# ── Skinning (barycentric weights) ───────────────────────────────────────────

def compute_skinning(vis_verts, tet_verts, tet_ids):
    """
    For each visual vertex, find the best enclosing/nearest tet and store
    (tetIdx, b0, b1, b2) where b3 = 1 - b0 - b1 - b2.
    Returns flat list of floats length = 4 * numVisVerts.
    """
    num_tets = len(tet_ids) // 4
    num_vis  = len(vis_verts)
    skinning = []

    # Pre-compute tet bounding spheres for quick rejection
    tet_centers = []
    tet_radii2  = []
    for ti in range(num_tets):
        ids = tet_ids[4*ti:4*ti+4]
        pts = [tet_verts[i] for i in ids]
        cx  = sum(p[0] for p in pts) / 4.0
        cy  = sum(p[1] for p in pts) / 4.0
        cz  = sum(p[2] for p in pts) / 4.0
        r2  = max(norm2(sub(p,[cx,cy,cz])) for p in pts) * 1.21  # 10% margin²
        tet_centers.append([cx,cy,cz])
        tet_radii2.append(r2)

    progress_step = max(1, num_vis // 20)

    for vi, vv in enumerate(vis_verts):
        if vi % progress_step == 0:
            pct = vi * 100 // num_vis
            sys.stdout.write(f"\r  Skinning … {pct:3d}%")
            sys.stdout.flush()

        best_tet  = -1
        best_dist = float('inf')   # negative = vertex is inside
        best_bary = [0.25, 0.25, 0.25]

        for ti in range(num_tets):
            # Quick bounding-sphere reject
            dv = sub(vv, tet_centers[ti])
            if norm2(dv) > tet_radii2[ti]:
                continue

            ids = tet_ids[4*ti:4*ti+4]
            p0,p1,p2,p3 = [tet_verts[i] for i in ids]

            # Solve for barycentric coords:
            #   vv = b0*p0 + b1*p1 + b2*p2 + b3*p3, b0+b1+b2+b3=1
            #   => vv - p3 = b0*(p0-p3) + b1*(p1-p3) + b2*(p2-p3)
            A = [sub(p0,p3), sub(p1,p3), sub(p2,p3)]
            sol = solve_3x3(A, sub(vv, p3))
            if sol is None:
                continue

            b0, b1, b2 = sol
            b3 = 1.0 - b0 - b1 - b2
            # Penetration metric (0 = on surface, positive = fully inside)
            dist = -min(b0, b1, b2, b3)   # dist<0 → inside

            if dist < best_dist:
                best_dist = dist
                best_tet  = ti
                best_bary = [b0, b1, b2]

        if best_tet < 0:
            best_tet  = 0
            best_bary = [0.25, 0.25, 0.25]

        skinning.extend([float(best_tet),
                         best_bary[0], best_bary[1], best_bary[2]])

    sys.stdout.write("\r  Skinning … 100%\n")
    return skinning


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess a particle OBJ into simulation JSON for "
                    "Examples/30-syringeGelParticles.html")
    parser.add_argument('input',  help='Input OBJ file path')
    parser.add_argument('output', nargs='?', default='particle_preprocessed.json',
                        help='Output JSON file (default: particle_preprocessed.json)')
    parser.add_argument('--res',   type=int,   default=6,
                        help='Tet grid resolution (default 6, range 4-12)')
    parser.add_argument('--scale', type=float, default=0.001,
                        help='Scale factor OBJ→mm (default 0.001 = microns→mm)')
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: File not found: {args.input}")
        sys.exit(1)

    print(f"\n=== Particle Preprocessor ===")
    print(f"Input  : {args.input}")
    print(f"Output : {args.output}")
    print(f"Scale  : {args.scale} (1 OBJ unit = {args.scale} mm)")
    print(f"Tet res: {args.res}\n")

    # 1. Parse OBJ
    print("Step 1: Parsing OBJ …")
    raw_verts, faces = parse_obj(args.input)
    print(f"  {len(raw_verts)} vertices, {len(faces)} triangles")

    # 2. Center and scale to mm
    print("Step 2: Centering & scaling …")
    vis_verts = center_and_scale(raw_verts, args.scale)
    nom_r     = bounding_radius(vis_verts)
    bb   = bounding_box(vis_verts)
    dims = [bb[i][1]-bb[i][0] for i in range(3)]
    print(f"  Bounding box: {dims[0]:.4f} × {dims[1]:.4f} × {dims[2]:.4f} mm")
    print(f"  Bounding radius: {nom_r:.4f} mm")

    # 3. Generate tet mesh
    print("Step 3: Generating tet mesh …")
    tet_verts, tet_ids, edge_ids, surf_tri_ids = generate_tet_mesh(
        vis_verts, faces, args.res)

    # 4. Compute skinning
    print("Step 4: Computing skinning weights …")
    skinning = compute_skinning(vis_verts, tet_verts, tet_ids)

    # 5. Assemble JSON
    print("Step 5: Writing JSON …")
    out = {
        "tet": {
            "verts":      [round(c, 6) for v in tet_verts  for c in v],
            "tetIds":     tet_ids,
            "edgeIds":    edge_ids,
            "surfTriIds": surf_tri_ids,
        },
        "vis": {
            "verts":    [round(c, 6) for v in vis_verts for c in v],
            "triIds":   [i for f in faces for i in f],
            "skinning": [round(x, 6) for x in skinning],
        },
        "meta": {
            "source":        os.path.basename(args.input),
            "units":         "mm",
            "scale":         args.scale,
            "nominalRadius": round(nom_r, 6),
            "tetResolution": args.res,
            "numTetVerts":   len(tet_verts),
            "numTets":       len(tet_ids) // 4,
            "numEdges":      len(edge_ids) // 2,
            "numVisVerts":   len(vis_verts),
            "numVisTris":    len(faces),
        }
    }

    with open(args.output, 'w') as fh:
        json.dump(out, fh, separators=(',', ':'))

    size_kb = os.path.getsize(args.output) / 1024
    print(f"\n=== Done ===")
    print(f"Output: {args.output}  ({size_kb:.1f} KB)")
    print(f"  Tet mesh : {len(tet_verts)} verts, {len(tet_ids)//4} tets, "
          f"{len(edge_ids)//2} edges")
    print(f"  Vis mesh : {len(vis_verts)} verts, {len(faces)} tris")
    print(f"  Bounding radius: {nom_r:.4f} mm")
    print()
    print("Load the JSON via the 'Load particle JSON' button in")
    print("Examples/30-syringeGelParticles.html")


if __name__ == '__main__':
    main()
