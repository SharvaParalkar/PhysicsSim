#!/usr/bin/env python3
"""
Prepare an OBJ for V3 syringeGelParticles "Fill volume OBJ" (invisible box mode).

Open or non-watertight meshes (e.g. airway / lung shells with open tube ends) need
shell-style containment in the sim. This script welds vertices, fixes normals,
optionally subdivides and smooths for smoother collision/rendering, and reports
whether the mesh is watertight.

Usage (from repo root):
  python V3/uploads/prepare_fill_volume.py path/to/lungshape.obj path/to/lungshape_fill.obj
  python V3/uploads/prepare_fill_volume.py lung.obj out.obj --max-dim 80 --subdivide 1 --smooth 2
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import trimesh
except ImportError:
    print("prepare_fill_volume.py requires trimesh: pip install trimesh", file=sys.stderr)
    raise


def _load_obj_triangles(path: Path) -> tuple[np.ndarray, np.ndarray]:
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v" and len(parts) >= 4:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                idx: list[int] = []
                for tok in parts[1:]:
                    vi = int(tok.split("/")[0])
                    if vi < 0:
                        vi = len(verts) + vi + 1
                    idx.append(vi - 1)
                for k in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[k], idx[k + 1]])
    if not faces:
        raise SystemExit(f"No triangles in {path}")
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _boundary_edge_count(mesh: trimesh.Trimesh) -> int:
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for a, b, c in mesh.faces:
        for e in ((a, b), (b, c), (c, a)):
            counts[tuple(sorted(e))] += 1
    return sum(1 for c in counts.values() if c == 1)


def _scale_to_max_dim(mesh: trimesh.Trimesh, max_dim: float) -> trimesh.Trimesh:
    extents = mesh.bounds[1] - mesh.bounds[0]
    cur = float(np.max(extents))
    if cur <= 0:
        return mesh
    s = float(max_dim) / cur
    mesh = mesh.copy()
    mesh.apply_scale(s)
    return mesh


def _recenter(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.vertices -= mesh.centroid
    return mesh


def prepare(
    in_path: Path,
    out_path: Path,
    *,
    max_dim: float | None,
    subdivide: int,
    smooth: int,
    recenter: bool,
) -> dict:
    v, f = _load_obj_triangles(in_path)
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.fix_normals()

    raw_boundary = _boundary_edge_count(mesh)
    raw_watertight = mesh.is_watertight

    if max_dim is not None and max_dim > 0:
        mesh = _scale_to_max_dim(mesh, max_dim)
    if recenter:
        mesh = _recenter(mesh)

    for _ in range(max(0, subdivide)):
        try:
            mesh = mesh.subdivide()
        except Exception as exc:
            print(f"  Warning: subdivision skipped ({exc})", file=sys.stderr)
            break
    if smooth > 0 and mesh.is_volume and mesh.volume > 1e-6:
        try:
            trimesh.smoothing.filter_laplacian(mesh, iterations=int(smooth))
        except Exception as exc:
            print(f"  Warning: smoothing skipped ({exc})", file=sys.stderr)

    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    mesh.fix_normals()
    trimesh.repair.fill_holes(mesh)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_path))

    bb = mesh.bounds
    extents = np.nan_to_num(bb[1] - bb[0], nan=0.0, posinf=0.0, neginf=0.0)
    thickness = float(max(extents)) * 0.45 if np.any(extents > 0) else 20.0

    report = {
        "input": str(in_path),
        "output": str(out_path),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "boundary_edges_before": int(raw_boundary),
        "watertight_before": bool(raw_watertight),
        "extents_mm": [float(x) for x in extents],
        "recommended_sim_mode": "volume" if mesh.is_watertight else "shell",
        "suggested_max_interior_mm": max(8.0, min(40.0, thickness)),
    }
    return report, mesh


def export_spawn_slots(
    mesh: "trimesh.Trimesh",
    path: Path,
    *,
    particle_spacing_mm: float = 0.65,
    max_interior_mm: float | None = None,
    max_slots: int = 1200,
) -> int:
    """Voxel-band interior points for shell meshes (avoids browser grid freeze)."""
    bb = mesh.bounds
    extents = bb[1] - bb[0]
    max_in = max_interior_mm
    if max_in is None:
        max_in = max(8.0, min(40.0, float(np.max(extents)) * 0.45))

    sp = max(particle_spacing_mm * 1.1, max_in / 8.0, 0.9)
    mins, maxs = bb[0], bb[1]
    slots: list[list[float]] = []

    # cap grid size
    while True:
        xs = np.linspace(mins[0], maxs[0], max(2, int((maxs[0] - mins[0]) / sp) + 1))
        ys = np.linspace(mins[1], maxs[1], max(2, int((maxs[1] - mins[1]) / sp) + 1))
        zs = np.linspace(mins[2], maxs[2], max(2, int((maxs[2] - mins[2]) / sp) + 1))
        if len(xs) * len(ys) * len(zs) <= 45000:
            break
        sp *= 1.2

    pts = np.array(
        [(x, y, z) for y in ys for x in xs for z in zs],
        dtype=np.float64,
    )
    closest, dist, face_id = trimesh.proximity.closest_point(mesh, pts)
    for i, p in enumerate(pts):
        cp = closest[i]
        d = float(dist[i])
        if d > max_in:
            continue
        n = mesh.face_normals[int(face_id[i])]
        if float(np.dot(p - cp, n)) < 0:
            slots.append([float(p[0]), float(p[1]), float(p[2])])

    if len(slots) > max_slots:
        idx = np.random.choice(len(slots), size=max_slots, replace=False)
        slots = [slots[int(i)] for i in idx]

    payload = {
        "slots": slots,
        "maxInteriorMm": float(max_in),
        "particleSpacingMm": float(sp),
        "count": len(slots),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return len(slots)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare a fill-volume OBJ for V3 PhysicsSim.")
    ap.add_argument("input_obj")
    ap.add_argument("output_obj")
    ap.add_argument(
        "--max-dim",
        type=float,
        default=80.0,
        help="Uniform scale so max extent becomes this size in mm (default: 80). Use 0 to skip.",
    )
    ap.add_argument("--subdivide", type=int, default=0, help="Loop subdivision passes (default: 0; 1+ is slow in-browser).")
    ap.add_argument("--smooth", type=int, default=0, help="Laplacian smoothing iterations (default: 0).")
    ap.add_argument("--no-recenter", action="store_true", help="Keep original world coordinates.")
    ap.add_argument(
        "--spawn-json",
        default=None,
        help="Write spawn slot JSON for fast browser load (recommended for lung/airway shells).",
    )
    args = ap.parse_args()

    report, mesh = prepare(
        Path(args.input_obj),
        Path(args.output_obj),
        max_dim=None if args.max_dim == 0 else float(args.max_dim),
        subdivide=max(0, args.subdivide),
        smooth=max(0, args.smooth),
        recenter=not args.no_recenter,
    )

    if args.spawn_json:
        n = export_spawn_slots(
            mesh,
            Path(args.spawn_json),
            max_interior_mm=report["suggested_max_interior_mm"],
        )
        print(f"  Spawn JSON: {args.spawn_json} ({n} slots)")

    print(f"Wrote {report['output']}")
    print(f"  {report['vertices']} verts, {report['faces']} faces")
    print(f"  Watertight: {report['watertight_before']} -> {report['watertight']}")
    print(f"  Boundary edges (input): {report['boundary_edges_before']}")
    print(f"  Extents mm: {[round(x, 2) for x in report['extents_mm']]}")
    print(f"  Sim: use Container -> Invisible box + load prepared OBJ")
    print(f"  Recommended containment: {report['recommended_sim_mode']}")
    if report["recommended_sim_mode"] == "shell":
        print(
            "  (Open tubes / thin shell - the browser sim auto-uses shell containment; "
            f"max interior depth ~ {report['suggested_max_interior_mm']:.1f} mm)"
        )


if __name__ == "__main__":
    main()
