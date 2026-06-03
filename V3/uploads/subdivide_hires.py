"""
subdivide_hires.py
==================
Midpoint-subdivides an OBJ surface mesh N times, then tetrahedralizes it
via preprocess.py at a higher interior resolution.  Produces a high-quality
particle JSON suitable for nanoindentation experiments.

Usage
-----
  python subdivide_hires.py input.obj output.json [options]

Options
-------
  --subdivisions INT    Surface subdivision levels (default: 3).
                        Each level multiplies triangles by 4.
                        cube 12->48->192->768 tris.
  --resolution   INT    Interior tet-grid resolution passed to preprocess.py
                        (default: 20).
  --max-dim      FLOAT  Scale OBJ so longest dimension = this value (mm).
  --recenter            Centre mesh at origin before processing (default on).
  --nominal-radius FLOAT Override meta.nominalRadius in output JSON.

Example
-------
  python subdivide_hires.py uploads/600mCube.obj assets/Cube600M_hires.json \\
         --subdivisions 3 --resolution 20 --max-dim 1.0
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------

def _vadd(a, b): return (a[0]+b[0], a[1]+b[1], a[2]+b[2])
def _vsub(a, b): return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
def _vmul(a, s): return (a[0]*s, a[1]*s, a[2]*s)
def _vdot(a, b): return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
def _vlen(a):    return math.sqrt(_vdot(a, a))
def _vnorm(a):
    l = _vlen(a)
    return (a[0]/l, a[1]/l, a[2]/l) if l > 1e-15 else (0.0, 0.0, 0.0)

# ---------------------------------------------------------------------------
# OBJ I/O
# ---------------------------------------------------------------------------

def load_obj(path):
    """Returns (verts: list of (x,y,z), faces: list of (a,b,c) 0-indexed)."""
    verts, faces = [], []
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split()
            if parts[0] == "v":
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif parts[0] == "f":
                indices = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                # Fan triangulation for polygonal faces
                for k in range(1, len(indices) - 1):
                    faces.append((indices[0], indices[k], indices[k + 1]))
    return verts, faces


def save_obj(path, verts, faces):
    with open(path, "w", encoding="utf-8") as fh:
        for x, y, z in verts:
            fh.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        for a, b, c in faces:
            fh.write(f"f {a+1} {b+1} {c+1}\n")

# ---------------------------------------------------------------------------
# Mesh transforms
# ---------------------------------------------------------------------------

def recenter(verts):
    n = len(verts)
    if n == 0:
        return verts
    cx = sum(v[0] for v in verts) / n
    cy = sum(v[1] for v in verts) / n
    cz = sum(v[2] for v in verts) / n
    return [(v[0]-cx, v[1]-cy, v[2]-cz) for v in verts]


def scale_to_max_dim(verts, target):
    if not verts:
        return verts
    xs, ys, zs = [v[0] for v in verts], [v[1] for v in verts], [v[2] for v in verts]
    cur = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs))
    if cur <= 1e-15:
        return verts
    s = target / cur
    return [(v[0]*s, v[1]*s, v[2]*s) for v in verts]

# ---------------------------------------------------------------------------
# Midpoint subdivision
# ---------------------------------------------------------------------------

def _midpoint_key(a, b):
    return (min(a, b), max(a, b))


def subdivide(verts, faces, n=1):
    """
    Midpoint subdivision: each triangle -> 4 triangles.
    Runs n times.  Newly created edge-midpoints are shared (welded).
    """
    for _ in range(n):
        edge_mid = {}          # (min_i, max_i) -> new_vertex_index
        new_verts = list(verts)
        new_faces = []

        def get_mid(a, b):
            key = _midpoint_key(a, b)
            if key not in edge_mid:
                va, vb = new_verts[a], new_verts[b]
                mid = ((va[0]+vb[0])*0.5, (va[1]+vb[1])*0.5, (va[2]+vb[2])*0.5)
                edge_mid[key] = len(new_verts)
                new_verts.append(mid)
            return edge_mid[key]

        for (a, b, c) in faces:
            ab = get_mid(a, b)
            bc = get_mid(b, c)
            ca = get_mid(c, a)
            new_faces.extend([(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)])

        verts = new_verts
        faces = new_faces

    return verts, faces


def project_to_sphere(verts, faces, center, radius):
    """
    Project all vertices radially onto a sphere of given radius.
    Useful for subdivided icospheres / curved shapes to prevent flat artefacts.
    Only projects if the original shape is approximately spherical.
    """
    cx, cy, cz = center
    out = []
    for (x, y, z) in verts:
        dx, dy, dz = x-cx, y-cy, z-cz
        d = math.sqrt(dx*dx + dy*dy + dz*dz)
        if d < 1e-15:
            out.append((x, y, z))
        else:
            s = radius / d
            out.append((cx+dx*s, cy+dy*s, cz+dz*s))
    return out


def _mesh_approx_spherical(verts, threshold=0.15):
    """Return True if all vertices lie within threshold * radius of a sphere."""
    if not verts:
        return False
    cx = sum(v[0] for v in verts) / len(verts)
    cy = sum(v[1] for v in verts) / len(verts)
    cz = sum(v[2] for v in verts) / len(verts)
    dists = [math.sqrt((v[0]-cx)**2+(v[1]-cy)**2+(v[2]-cz)**2) for v in verts]
    if not dists:
        return False
    rmax, rmin = max(dists), min(dists)
    return (rmax - rmin) / max(rmax, 1e-15) < threshold


def _mesh_center_and_radius(verts):
    if not verts:
        return (0, 0, 0), 1.0
    cx = sum(v[0] for v in verts) / len(verts)
    cy = sum(v[1] for v in verts) / len(verts)
    cz = sum(v[2] for v in verts) / len(verts)
    r = max(math.sqrt((v[0]-cx)**2+(v[1]-cy)**2+(v[2]-cz)**2) for v in verts)
    return (cx, cy, cz), r or 1.0

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _repo_dir():
    return Path(__file__).resolve().parent.parent   # /V3


def subdivide_and_tet(
    in_obj: str,
    out_json: str,
    *,
    subdivisions: int = 3,
    resolution: int = 20,
    max_dim: float = None,
    do_recenter: bool = True,
    nominal_radius: float = None,
):
    print(f"\n{'='*60}")
    print(f"subdivide_hires: {Path(in_obj).name}  ->  {Path(out_json).name}")
    print(f"  subdivisions={subdivisions}  resolution={resolution}"
          + (f"  max_dim={max_dim}" if max_dim else ""))

    verts, faces = load_obj(in_obj)
    print(f"  Loaded: {len(verts)} verts, {len(faces)} triangles")

    if not verts or not faces:
        raise SystemExit(f"No geometry in {in_obj}")

    if do_recenter:
        verts = recenter(verts)
    if max_dim is not None:
        verts = scale_to_max_dim(verts, max_dim)

    verts, faces = subdivide(verts, faces, n=subdivisions)
    print(f"  After {subdivisions} subdivision: {len(verts)} verts, {len(faces)} triangles")

    # If mesh is approximately spherical, project back to sphere to avoid
    # the flat-midpoint artefact (subdivided icospheres become bumpy)
    ctr, rad = _mesh_center_and_radius(verts)
    if _mesh_approx_spherical(verts):
        verts = project_to_sphere(verts, faces, ctr, rad)
        print(f"  Applied spherical projection (r={rad:.4f})")

    repo = _repo_dir()
    preprocess_py = repo / "uploads" / "preprocess.py"
    if not preprocess_py.exists():
        raise FileNotFoundError(f"Cannot find {preprocess_py}")

    out_json_path = Path(out_json).resolve()
    out_json_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        tmp_obj = Path(td) / "subdivided.obj"
        save_obj(str(tmp_obj), verts, faces)
        cmd = [
            sys.executable, str(preprocess_py),
            str(tmp_obj), str(out_json_path),
            "--resolution", str(resolution),
            "--min-quality", "0.001",
        ]
        print(f"  Running preprocess.py --resolution {resolution} ")
        ret = subprocess.call(cmd)
        if ret != 0:
            raise SystemExit(f"preprocess.py failed (exit {ret})")

    # Optionally override nominalRadius
    if nominal_radius is not None and nominal_radius > 0:
        data = json.loads(out_json_path.read_text(encoding="utf-8"))
        data.setdefault("meta", {})["nominalRadius"] = float(nominal_radius)
        out_json_path.write_text(json.dumps(data), encoding="utf-8")
        print(f"  Overrode nominalRadius -> {nominal_radius}")

    # Print summary
    try:
        d = json.loads(out_json_path.read_text(encoding="utf-8"))
        t, v2 = d.get("tet", {}), d.get("vis", {})
        ntv = len(t.get("verts", [])) // 3
        ntt = len(t.get("tetIds", [])) // 4
        nst = len(t.get("surfTriIds", [])) // 3
        nvv = len(v2.get("verts", [])) // 3
        print(f"  -> {ntv} tet verts, {ntt} tets, {nst} surf tris, {nvv} vis verts")
        print(f"  -> nominalRadius = {d.get('meta',{}).get('nominalRadius',''):.4f}")
    except Exception:
        pass

    print(f"  Saved -> {out_json_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input",   help="Source .obj file")
    ap.add_argument("output",  help="Output particle .json file")
    ap.add_argument("--subdivisions", type=int, default=3,
                    help="Surface subdivision levels (default: 3 -> 64 more triangles)")
    ap.add_argument("--resolution", type=int, default=20,
                    help="Interior tet grid resolution for preprocess.py (default: 20)")
    ap.add_argument("--max-dim", type=float, default=None,
                    help="Scale OBJ so its longest dimension = this value (mm)")
    ap.add_argument("--no-recenter", action="store_true",
                    help="Skip recentering mesh at origin")
    ap.add_argument("--nominal-radius", type=float, default=None,
                    help="Override meta.nominalRadius in output JSON (mm)")
    args = ap.parse_args()

    subdivide_and_tet(
        args.input, args.output,
        subdivisions=args.subdivisions,
        resolution=args.resolution,
        max_dim=args.max_dim,
        do_recenter=not args.no_recenter,
        nominal_radius=args.nominal_radius,
    )


if __name__ == "__main__":
    main()
