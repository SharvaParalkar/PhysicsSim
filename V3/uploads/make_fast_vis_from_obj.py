import argparse
import json
import math
import re
from pathlib import Path


_V_RE = re.compile(r"^v\s+([-0-9.eE]+)\s+([-0-9.eE]+)\s+([-0-9.eE]+)")
_F_RE = re.compile(r"^f\s+(.+)$")


def load_obj_verts_faces(obj_path: Path):
    verts = []
    tris = []
    with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            m = _V_RE.match(ln)
            if m:
                verts.extend([float(m.group(1)), float(m.group(2)), float(m.group(3))])
                continue
            m = _F_RE.match(ln)
            if m:
                parts = m.group(1).split()
                # triangulate fan for polygons, but we expect triangles already
                ids = []
                for p in parts:
                    # formats: v, v/vt, v//vn, v/vt/vn
                    v_str = p.split("/")[0]
                    ids.append(int(v_str) - 1)
                if len(ids) < 3:
                    continue
                for i in range(1, len(ids) - 1):
                    tris.extend([ids[0], ids[i], ids[i + 1]])
    return verts, tris


def incident_tet_for_vertex(tet_ids):
    inc = {}
    nt = len(tet_ids) // 4
    for ti in range(nt):
        a, b, c, d = tet_ids[4 * ti : 4 * ti + 4]
        for v in (a, b, c, d):
            if v not in inc:
                inc[v] = ti
    return inc


def main():
    ap = argparse.ArgumentParser(
        description="Replace particle JSON vis mesh with a low-poly OBJ surface, and attach via nearest tet-vertex skinning."
    )
    ap.add_argument("input_particle_json")
    ap.add_argument("input_obj_surface", help="OBJ whose verts/faces will become vis.verts/vis.triIds")
    ap.add_argument("output_particle_json")
    args = ap.parse_args()

    pj = json.loads(Path(args.input_particle_json).read_text(encoding="utf-8"))
    tet = pj["tet"]
    tet_verts = tet["verts"]
    tet_ids = tet["tetIds"]

    vis_verts, vis_tris = load_obj_verts_faces(Path(args.input_obj_surface))

    # Build kNN by brute force (small vis meshes); find nearest tet vertex per vis vertex
    n_tet = len(tet_verts) // 3
    inc = incident_tet_for_vertex(tet_ids)

    skin = []
    for vi in range(len(vis_verts) // 3):
        x, y, z = vis_verts[3 * vi], vis_verts[3 * vi + 1], vis_verts[3 * vi + 2]
        best = 0
        best_d = float("inf")
        for ti in range(n_tet):
            dx = tet_verts[3 * ti] - x
            dy = tet_verts[3 * ti + 1] - y
            dz = tet_verts[3 * ti + 2] - z
            d = dx * dx + dy * dy + dz * dz
            if d < best_d:
                best_d = d
                best = ti

        tet_index = inc.get(best, 0)
        a, b, c, d = tet_ids[4 * tet_index : 4 * tet_index + 4]
        if best == a:
            w = (1.0, 0.0, 0.0, 0.0)
        elif best == b:
            w = (0.0, 1.0, 0.0, 0.0)
        elif best == c:
            w = (0.0, 0.0, 1.0, 0.0)
        elif best == d:
            w = (0.0, 0.0, 0.0, 1.0)
        else:
            w = (0.25, 0.25, 0.25, 0.25)
        skin.extend([int(tet_index), float(w[0]), float(w[1]), float(w[2]), float(w[3])])

    pj["vis"] = {"verts": vis_verts, "triIds": vis_tris, "skinning": skin}

    Path(args.output_particle_json).write_text(json.dumps(pj, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

