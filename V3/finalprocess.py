import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path


_V_RE = re.compile(r"^(v)\s+([-0-9.eE]+)\s+([-0-9.eE]+)\s+([-0-9.eE]+)(.*)$")


def _load_vertices(obj_path: Path):
    vs = []
    with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            m = _V_RE.match(ln.rstrip("\n"))
            if m:
                vs.append((float(m.group(2)), float(m.group(3)), float(m.group(4))))
    return vs


def _centroid(vs):
    n = len(vs)
    if n == 0:
        return (0.0, 0.0, 0.0)
    sx = sum(v[0] for v in vs)
    sy = sum(v[1] for v in vs)
    sz = sum(v[2] for v in vs)
    return (sx / n, sy / n, sz / n)


def _bounds(vs):
    if not vs:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    xs = [v[0] for v in vs]
    ys = [v[1] for v in vs]
    zs = [v[2] for v in vs]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def _write_transformed_obj(src_obj: Path, dst_obj: Path, *, scale: float, translate):
    tx, ty, tz = translate
    with src_obj.open("r", encoding="utf-8", errors="ignore") as fin, dst_obj.open(
        "w", encoding="utf-8"
    ) as fout:
        for ln in fin:
            m = _V_RE.match(ln.rstrip("\n"))
            if not m:
                fout.write(ln)
                continue
            x = (float(m.group(2)) + tx) * scale
            y = (float(m.group(3)) + ty) * scale
            z = (float(m.group(4)) + tz) * scale
            tail = m.group(5) or ""
            fout.write(f"v {x:.9g} {y:.9g} {z:.9g}{tail}\n")


def _override_nominal_radius(json_path: Path, nominal_radius: float):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    meta = data.get("meta") or {}
    meta["nominalRadius"] = float(nominal_radius)
    data["meta"] = meta
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "OBJ -> V3 syringeGelParticles-ready particle JSON.\n"
            "Pipeline: (optional recenter/scale) -> tetrahedralize via V3/uploads/preprocess.py -> "
            "(optional replace vis using low-poly OBJ)."
        )
    )
    ap.add_argument("input_obj", help="Input .obj (surface mesh). Units should be mm for the syringe sim.")
    ap.add_argument("output_json", help="Output particle .json file (loadable by V3/syringeGelParticles.html).")
    ap.add_argument(
        "--input-scale",
        type=float,
        default=1.0,
        help="Multiply OBJ vertex coordinates by this factor before processing (default: 1.0).",
    )
    ap.add_argument(
        "--recenter",
        action="store_true",
        default=True,
        help="Translate vertices so centroid is at origin (default: True).",
    )
    ap.add_argument(
        "--no-recenter",
        action="store_false",
        dest="recenter",
        help="Do not recenter the mesh.",
    )
    ap.add_argument(
        "--max-dim",
        type=float,
        default=None,
        help="Uniformly scale so max(dx,dy,dz) becomes this value (in the OBJ's units after --input-scale).",
    )
    ap.add_argument("--resolution", type=int, default=10, help="Interior sampling resolution (default: 10).")
    ap.add_argument(
        "--min-quality",
        type=float,
        default=0.001,
        help="Minimum tet quality 0..1 (default: 0.001).",
    )
    ap.add_argument(
        "--tet-shrink",
        type=float,
        default=1.0,
        help="Passed to preprocess.py --scale (exploded-view shrink; leave at 1.0 for this sim).",
    )
    ap.add_argument(
        "--fast-vis-obj",
        default=None,
        help="Optional low-poly OBJ surface to use for vis mesh (keeps tet mesh for physics).",
    )
    ap.add_argument(
        "--nominal-radius",
        type=float,
        default=None,
        help="Override meta.nominalRadius in the output JSON (mm). If omitted, uses preprocess.py's estimate.",
    )
    args = ap.parse_args()

    repo_v3_dir = Path(__file__).resolve().parent
    preprocess_py = repo_v3_dir / "uploads" / "preprocess.py"
    fast_vis_py = repo_v3_dir / "uploads" / "make_fast_vis_from_obj.py"

    if not preprocess_py.exists():
        raise FileNotFoundError(f"Missing {preprocess_py}")
    if args.fast_vis_obj and not fast_vis_py.exists():
        raise FileNotFoundError(f"Missing {fast_vis_py}")

    in_obj = Path(args.input_obj)
    out_json = Path(args.output_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    vs = _load_vertices(in_obj)
    if not vs:
        raise SystemExit("No vertices found — check that the OBJ has 'v' lines.")

    cx, cy, cz = _centroid(vs)
    translate = (-cx, -cy, -cz) if args.recenter else (0.0, 0.0, 0.0)

    scale = float(args.input_scale)
    if args.max_dim is not None:
        (mnx, mny, mnz), (mxx, mxy, mxz) = _bounds(vs)
        dx = (mxx - mnx) * scale
        dy = (mxy - mny) * scale
        dz = (mxz - mnz) * scale
        cur = max(dx, dy, dz)
        if cur > 0:
            scale *= float(args.max_dim) / cur

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        tmp_obj = td / (in_obj.stem + "_finalprocess.obj")
        _write_transformed_obj(in_obj, tmp_obj, scale=scale, translate=translate)

        cmd = [
            sys.executable,
            str(preprocess_py),
            str(tmp_obj),
            str(out_json),
            "--resolution",
            str(args.resolution),
            "--min-quality",
            str(args.min_quality),
            "--scale",
            str(args.tet_shrink),
        ]
        code = subprocess.call(cmd)
        if code != 0:
            raise SystemExit(code)

        if args.fast_vis_obj:
            vis_cmd = [
                sys.executable,
                str(fast_vis_py),
                str(out_json),
                str(Path(args.fast_vis_obj).resolve()),
                str(out_json),
            ]
            code = subprocess.call(vis_cmd)
            if code != 0:
                raise SystemExit(code)

    if args.nominal_radius is not None:
        if not (args.nominal_radius > 0.0 and math.isfinite(args.nominal_radius)):
            raise SystemExit("--nominal-radius must be a finite, positive number.")
        _override_nominal_radius(out_json, args.nominal_radius)


if __name__ == "__main__":
    main()

