import argparse
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


def main():
    ap = argparse.ArgumentParser(
        description="Recenter/rescale an OBJ, then convert to V3 particle JSON using V3/uploads/preprocess.py."
    )
    ap.add_argument("input_obj", help="Input .obj file")
    ap.add_argument("output_json", help="Output .json file (V3 particle schema)")
    ap.add_argument(
        "--input-scale",
        type=float,
        default=1.0,
        help="Multiply OBJ vertex coordinates by this factor before tetrahedralization (default: 1.0).",
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
    ap.add_argument("--resolution", type=int, default=10)
    ap.add_argument("--min-quality", type=float, default=0.001)
    ap.add_argument(
        "--tet-shrink",
        type=float,
        default=1.0,
        help="Passed through as preprocess.py --scale (exploded view shrink).",
    )
    args = ap.parse_args()

    in_obj = Path(args.input_obj)
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    vs = _load_vertices(in_obj)
    cx, cy, cz = _centroid(vs)
    # We translate by (-centroid) to recenter.
    translate = (-cx, -cy, -cz) if args.recenter else (0.0, 0.0, 0.0)

    preprocess_py = Path(__file__).resolve().parent / "uploads" / "preprocess.py"
    if not preprocess_py.exists():
        raise FileNotFoundError(f"Missing preprocessor at {preprocess_py}")

    with tempfile.TemporaryDirectory() as td:
        tmp_obj = Path(td) / (in_obj.stem + "_recentered.obj")
        _write_transformed_obj(in_obj, tmp_obj, scale=args.input_scale, translate=translate)

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
        # Let preprocess.py stream its own logs/warnings.
        raise_code = subprocess.call(cmd)
        if raise_code != 0:
            raise SystemExit(raise_code)


if __name__ == "__main__":
    main()

