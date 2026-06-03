"""
generate_hires_assets.py
========================
Batch-generates all high-resolution particle JSON files needed for
nanoindentation experiments.  For each 600m geometry it:

  1. Creates the star OBJ if missing.
  2. Subdivides the source OBJ surface mesh 3 times (64 triangles).
  3. Tetrahedralizes with --resolution 20 via preprocess.py.
  4. Saves to V3/assets/<Name>_hires.json.
  5. Updates V3/assets/manifest.json.

Run from the V3 directory:
  python uploads/generate_hires_assets.py

Or from the uploads directory:
  python generate_hires_assets.py

Expected output resolutions (before tet quality filtering):
  Cube  : 12 -> 768 surf tris,  ~400-800 tets
  Cyl   : 32 -> 2048 surf tris, ~400-900 tets
  Hex   : 18 -> 1152 surf tris, ~400-800 tets
  Tri   :  8 -> 512 surf tris,  ~200-500 tets
  Star  : 48 -> 3072 surf tris, ~300-700 tets
"""
import json
import math
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Locate project directories
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent         # V3/uploads/
V3_DIR     = SCRIPT_DIR.parent                       # V3/
ASSETS_DIR = V3_DIR / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

SUBDIVIDE_SCRIPT = SCRIPT_DIR / "subdivide_hires.py"
STAR_GEN_SCRIPT  = SCRIPT_DIR / "create_star_obj.py"
PREPROCESS_PY    = SCRIPT_DIR / "preprocess.py"

# ---------------------------------------------------------------------------
# Shape definitions
# ---------------------------------------------------------------------------
# Each entry: (label, source_obj, output_json, subdivisions, resolution)
# subdivisions=3 -> multiplies triangles by 4^3 = 64
# resolution=20  -> dense interior tet grid
SHAPES = [
    {
        # 12 surf tris -> 768 after 3 subdiv.  resolution=5 -> ~400-900 interior tets.
        "label":        "Cube",
        "source_obj":   SCRIPT_DIR / "600mCube.obj",
        "output_json":  ASSETS_DIR / "Cube600M_hires.json",
        "subdivisions": 3,
        "resolution":   5,
        "nominal_radius": None,
    },
    {
        # 32 surf tris -> 2048 after 3 subdiv.  resolution=5.
        "label":        "Cylinder",
        "source_obj":   SCRIPT_DIR / "600mCyl.obj",
        "output_json":  ASSETS_DIR / "Cyl600M_hires.json",
        "subdivisions": 3,
        "resolution":   5,
        "nominal_radius": None,
    },
    {
        # 18/24 surf tris -> ~1150 after 3 subdiv.  resolution=5.
        "label":        "Hexagon",
        "source_obj":   SCRIPT_DIR / "600mHex.obj",
        "output_json":  ASSETS_DIR / "Hex600M_hires.json",
        "subdivisions": 3,
        "resolution":   5,
        "nominal_radius": None,
    },
    {
        # 8 surf tris -> 512 after 3 subdiv (extra level because shape is tiny).
        "label":        "Triangle",
        "source_obj":   SCRIPT_DIR / "600mTri.obj",
        "output_json":  ASSETS_DIR / "Tri600M_hires.json",
        "subdivisions": 3,
        "resolution":   5,
        "nominal_radius": None,
    },
    {
        # Generated star prism.  48 surf tris -> 768 after 3 subdiv.
        "label":        "Star",
        "source_obj":   SCRIPT_DIR / "600mStar.obj",
        "output_json":  ASSETS_DIR / "Star600M_hires.json",
        "subdivisions": 3,
        "resolution":   5,
        "nominal_radius": None,
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd, label=""):
    print(f"\n  $ {' '.join(str(c) for c in cmd)}")
    ret = subprocess.call([str(c) for c in cmd])
    if ret != 0:
        print(f"  ERROR: command failed (exit {ret}) for {label}")
        return False
    return True


def summarise_json(path):
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        t  = d.get("tet", {})
        v2 = d.get("vis", {})
        ntv = len(t.get("verts",   [])) // 3
        ntt = len(t.get("tetIds",  [])) // 4
        nst = len(t.get("surfTriIds", [])) // 3
        nvv = len(v2.get("verts",  [])) // 3
        nr  = d.get("meta", {}).get("nominalRadius", "")
        nr_str = f"{nr:.4f}" if isinstance(nr, float) else str(nr)
        return (f"tet_verts={ntv}  tets={ntt}  surf_tris={nst}  "
                f"vis_verts={nvv}  nomR={nr_str}")
    except Exception as e:
        return f"(could not parse: {e})"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("generate_hires_assets.py  high-resolution particle batch")
    print("=" * 60)

    # Step 1: Create 600mStar.obj (always regenerate to ensure correct 5-arm geometry)
    star_src = SCRIPT_DIR / "600mStar.obj"
    print(f"\n[Star] Generating 5-arm star OBJ -> {star_src.name}")
    ok = run([sys.executable, str(STAR_GEN_SCRIPT),
              str(star_src),
              "--arms", "5",
              "--outer", "0.332",
              "--inner", "0.129",
              "--height", "0.60"],
             "create_star_obj")
    if not ok:
        print("  WARNING: star OBJ generation failed  skipping star shape.")

    # Step 2: Process each shape
    results = {}
    for shape in SHAPES:
        label = shape["label"]
        src   = shape["source_obj"]
        out   = shape["output_json"]
        subs  = shape["subdivisions"]
        res   = shape["resolution"]
        nr    = shape["nominal_radius"]

        print(f"\n{'-'*50}")
        print(f"[{label}] {src.name} -> {out.name}")

        if not src.exists():
            print(f"  SKIP: source OBJ not found ({src})")
            results[label] = "SKIPPED  source OBJ missing"
            continue

        cmd = [
            sys.executable, str(SUBDIVIDE_SCRIPT),
            str(src), str(out),
            "--subdivisions", str(subs),
            "--resolution",   str(res),
            "--max-dim",      "0.6",
        ]
        if nr is not None:
            cmd += ["--nominal-radius", str(nr)]

        ok = run(cmd, label)
        if ok and out.exists():
            summary = summarise_json(out)
            results[label] = f"OK  {summary}"
        else:
            results[label] = "FAILED"

    # Step 3: Update manifest.json
    manifest_path = ASSETS_DIR / "manifest.json"
    try:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        existing = []

    new_entries = [
        "Cube600M_hires.json",
        "Cyl600M_hires.json",
        "Hex600M_hires.json",
        "Tri600M_hires.json",
        "Star600M_hires.json",
    ]
    updated = list(existing)
    for entry in new_entries:
        out_path = ASSETS_DIR / entry
        if out_path.exists() and entry not in updated:
            updated.append(entry)

    manifest_path.write_text(json.dumps(updated, indent=2), encoding="utf-8")
    print(f"\n{'='*60}")
    print(f"Updated manifest.json ({len(updated)} entries)")

    # Step 4: Print summary table
    print(f"\n{'-'*60}")
    print("RESULTS")
    print(f"{'-'*60}")
    for label, status in results.items():
        print(f"  {label:<12} {status}")
    print(f"{'-'*60}")
    print("\nDone.  Load the _hires.json files via the particle library in the sim.")


if __name__ == "__main__":
    main()
