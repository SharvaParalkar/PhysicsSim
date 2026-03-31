import argparse
import json
import math
import re
from pathlib import Path


def _parse_number_list(s: str, kind: str):
    parts = re.split(r"[,\s]+", s.strip())
    out = []
    for p in parts:
        if not p:
            continue
        if kind == "float":
            out.append(float(p))
        else:
            # tolerate "12.0" in integer lists
            out.append(int(float(p)))
    return out


def _extract_array_block(text: str, key: str) -> str:
    # Matches: key : [ ... ] possibly across lines.
    # Non-greedy up to the first closing bracket followed by optional spaces/comma.
    m = re.search(rf"\b{re.escape(key)}\b\s*:\s*\[(.*?)\]\s*,?", text, re.IGNORECASE | re.DOTALL)
    if not m:
        raise ValueError(f"Could not find array '{key}: [...]' in source.")
    return m.group(1)


def _estimate_nominal_radius(verts_flat):
    n = len(verts_flat) // 3
    if n <= 0:
        return 0.3
    cx = sum(verts_flat[0::3]) / n
    cy = sum(verts_flat[1::3]) / n
    cz = sum(verts_flat[2::3]) / n
    r = 0.0
    for i in range(n):
        dx = verts_flat[3 * i] - cx
        dy = verts_flat[3 * i + 1] - cy
        dz = verts_flat[3 * i + 2] - cz
        r = max(r, math.sqrt(dx * dx + dy * dy + dz * dz))
    return r


def _compact_vis_from_surface(tet_verts_flat, surf_tri_ids):
    used = sorted(set(int(i) for i in surf_tri_ids))
    old_to_new = {old: new for new, old in enumerate(used)}
    vis_verts = []
    for old in used:
        b = 3 * old
        vis_verts.extend([tet_verts_flat[b], tet_verts_flat[b + 1], tet_verts_flat[b + 2]])
    vis_tris = [old_to_new[int(i)] for i in surf_tri_ids]
    return vis_verts, vis_tris


def main():
    ap = argparse.ArgumentParser(
        description="Convert Examples/10-softBodies.html bunnyMesh-style arrays to V3 particle JSON."
    )
    ap.add_argument("input_html", help="Path to Examples/10-softBodies.html (or any file containing the mesh object)")
    ap.add_argument("output_json", help="Path to write the particle JSON")
    ap.add_argument("--name", default="softBodies10_mesh", help="Source label for meta.source")
    ap.add_argument("--scale", type=float, default=1.0, help="Multiply all vertex coordinates by this factor")
    ap.add_argument(
        "--nominal-radius",
        type=float,
        default=None,
        help="Override meta.nominalRadius (otherwise estimated from vertex bounds)",
    )
    args = ap.parse_args()

    src_path = Path(args.input_html)
    text = src_path.read_text(encoding="utf-8", errors="ignore")

    verts_raw = _extract_array_block(text, "verts")
    tet_ids_raw = _extract_array_block(text, "tetIds")
    edge_ids_raw = _extract_array_block(text, "tetEdgeIds")
    surf_ids_raw = _extract_array_block(text, "tetSurfaceTriIds")

    verts = _parse_number_list(verts_raw, "float")
    if args.scale != 1.0:
        verts = [v * args.scale for v in verts]

    tet_ids = _parse_number_list(tet_ids_raw, "int")
    edge_ids = _parse_number_list(edge_ids_raw, "int")
    surf_tri_ids = _parse_number_list(surf_ids_raw, "int")

    vis_verts, vis_tri_ids = _compact_vis_from_surface(verts, surf_tri_ids)

    nominal_radius = args.nominal_radius
    if nominal_radius is None:
        nominal_radius = _estimate_nominal_radius(verts)

    out = {
        "tet": {
            "verts": verts,
            "tetIds": tet_ids,
            "edgeIds": edge_ids,
            "surfTriIds": surf_tri_ids,
        },
        # No skinning: V3 loader now accepts missing/empty skinning and will render from surface tris.
        "vis": {
            "verts": vis_verts,
            "triIds": vis_tri_ids,
            "skinning": [],
        },
        "meta": {
            "source": args.name,
            "nominalRadius": float(nominal_radius),
        },
    }

    out_path = Path(args.output_json)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

