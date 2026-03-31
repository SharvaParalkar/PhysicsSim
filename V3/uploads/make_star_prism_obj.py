import argparse
import math
from pathlib import Path


def star_polygon(n_points: int, r_outer: float, r_inner: float):
    # Returns list of (x,y) vertices, length 2*n_points, CCW.
    verts = []
    for i in range(2 * n_points):
        a = (i * math.pi) / n_points  # 0..2pi
        r = r_outer if (i % 2 == 0) else r_inner
        verts.append((r * math.cos(a), r * math.sin(a)))
    return verts


def triangulate_fan(center_idx: int, ring_indices, ccw=True):
    # ring_indices must be ordered around the ring.
    tris = []
    n = len(ring_indices)
    for i in range(n):
        a = ring_indices[i]
        b = ring_indices[(i + 1) % n]
        tris.append((center_idx, a, b) if ccw else (center_idx, b, a))
    return tris


def main():
    ap = argparse.ArgumentParser(description="Generate a watertight star prism OBJ (mm units).")
    ap.add_argument("output_obj", help="Where to write the OBJ")
    ap.add_argument("--points", type=int, default=5)
    ap.add_argument("--outer", type=float, required=True, help="Tip distance to center (mm)")
    ap.add_argument("--inner", type=float, required=True, help="Valley distance to center (mm)")
    ap.add_argument("--thickness", type=float, required=True, help="Extrusion thickness (mm)")
    ap.add_argument(
        "--max-dim",
        type=float,
        default=0.6,
        help="Uniformly scale so max(dx,dy,dz) equals this (mm). Default 0.6.",
    )
    args = ap.parse_args()

    poly = star_polygon(args.points, args.outer, args.inner)  # CCW in XY
    t = args.thickness
    z0 = -t / 2.0
    z1 = +t / 2.0

    # Build vertices: bottom ring, top ring, bottom center, top center.
    v = []
    for x, y in poly:
        v.append((x, y, z0))
    for x, y in poly:
        v.append((x, y, z1))
    bottom_center_idx = len(v)
    v.append((0.0, 0.0, z0))
    top_center_idx = len(v)
    v.append((0.0, 0.0, z1))

    n = len(poly)
    bottom_ring = list(range(0, n))
    top_ring = list(range(n, 2 * n))

    # Faces (1-based in OBJ)
    f = []

    # Bottom cap: normal should face -Z, so winding is CW when viewed from +Z.
    f.extend(triangulate_fan(bottom_center_idx, bottom_ring, ccw=False))

    # Top cap: normal +Z, winding CCW.
    f.extend(triangulate_fan(top_center_idx, top_ring, ccw=True))

    # Side walls: connect bottom i->i+1 with top i->i+1
    for i in range(n):
        b0 = bottom_ring[i]
        b1 = bottom_ring[(i + 1) % n]
        t0 = top_ring[i]
        t1 = top_ring[(i + 1) % n]
        # Quad split into two triangles. Choose winding outward.
        f.append((b0, b1, t1))
        f.append((b0, t1, t0))

    # Normalize scale so max dimension is args.max_dim.
    xs = [p[0] for p in v]
    ys = [p[1] for p in v]
    zs = [p[2] for p in v]
    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    dz = max(zs) - min(zs)
    cur = max(dx, dy, dz)
    scale = (args.max_dim / cur) if cur > 0 else 1.0
    v = [(x * scale, y * scale, z * scale) for (x, y, z) in v]

    out = Path(args.output_obj)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fp:
        fp.write("# star prism generated (units: mm)\n")
        fp.write(f"# points={args.points} outer={args.outer} inner={args.inner} thickness={args.thickness}\n")
        fp.write(f"# scaled_to_max_dim={args.max_dim} scale={scale}\n")
        for x, y, z in v:
            fp.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        for a, b, c in f:
            fp.write(f"f {a+1} {b+1} {c+1}\n")


if __name__ == "__main__":
    main()

