"""
Genesis + dependencies installation checker.
Run with:  python check_genesis.py
"""

import sys
import importlib

# ── Colour helpers (no external deps) ─────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

OK   = f"{GREEN}  ✔  OK{RESET}"
FAIL = f"{RED}  ✗  MISSING / BROKEN{RESET}"
WARN = f"{YELLOW}  ⚠  WARNING{RESET}"

def check(label, fn, hint=""):
    try:
        result = fn()
        status = OK
        extra  = f"  ({result})" if result else ""
    except Exception as exc:
        status = FAIL
        extra  = f"  → {exc}"
        if hint:
            extra += f"\n       {YELLOW}hint: {hint}{RESET}"
    print(f"  {label:<38}{status}{extra}")
    return "OK" in status


def section(title):
    print(f"\n{BOLD}{CYAN}{'─'*55}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*55}{RESET}")


# ── 1. Python version ──────────────────────────────────────────────────────────
section("Python")
py = sys.version_info
ok_py = (3, 10) <= (py.major, py.minor) <= (3, 13)
tag   = f"{py.major}.{py.minor}.{py.micro}"
flag  = OK if ok_py else WARN
print(f"  {'Python version':<38}{flag}  ({tag})"
      + ("" if ok_py else "  ← Genesis requires 3.10–3.13"))


# ── 2. Core numeric / graph deps ──────────────────────────────────────────────
section("Core dependencies")
core_deps = [
    ("numpy",       "numpy",    lambda: importlib.import_module("numpy").__version__,
     "pip install numpy"),
    ("scipy",       "scipy",    lambda: importlib.import_module("scipy").__version__,
     "pip install scipy"),
    ("pandas",      "pandas",   lambda: importlib.import_module("pandas").__version__,
     "pip install pandas"),
    ("networkx",    "networkx", lambda: importlib.import_module("networkx").__version__,
     "pip install networkx"),
    ("h5py",        "h5py",     lambda: importlib.import_module("h5py").__version__,
     "pip install h5py"),
    ("trimesh",     "trimesh",  lambda: importlib.import_module("trimesh").__version__,
     "pip install trimesh"),
    ("coacd",       "coacd",    lambda: importlib.import_module("coacd").__version__,
     "pip install coacd"),
    ("open3d",      "open3d",   lambda: importlib.import_module("open3d").__version__,
     "pip install open3d"),
]

results = {}
for label, mod, ver_fn, hint in core_deps:
    results[label] = check(label, ver_fn, hint)


# ── 3. PyTorch ────────────────────────────────────────────────────────────────
section("PyTorch  (required by Genesis)")

def _torch_info():
    import torch
    cuda = torch.cuda.is_available()
    mps  = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    accel = "CUDA" if cuda else ("MPS" if mps else "CPU-only")
    return f"{torch.__version__}  [{accel}]"

torch_ok = check(
    "torch",
    _torch_info,
    "pip install torch  (see pytorch.org/get-started for GPU variants)"
)


# ── 4. Genesis ────────────────────────────────────────────────────────────────
section("Genesis")

def _genesis_import():
    import genesis as gs
    return getattr(gs, "__version__", "installed (version attr not found)")

genesis_import_ok = check(
    "genesis-world (import)",
    _genesis_import,
    "pip install genesis-world"
)

def _genesis_init_cpu():
    import genesis as gs
    import io, contextlib
    # Suppress Genesis's own startup banner so output stays clean
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            gs.init(backend=gs.cpu, logging_level="warning")
        except Exception:
            gs.init(backend=gs.cpu)
    return "gs.init(backend=gs.cpu) succeeded"

genesis_init_ok = False
if genesis_import_ok:
    genesis_init_ok = check(
        "genesis init (CPU backend)",
        _genesis_init_cpu,
        "Check torch install; Genesis needs torch to be importable first"
    )

def _genesis_scene():
    import genesis as gs
    scene = gs.Scene(gravity=(0, -9.81, 0), dt=1/240, substeps=4)
    return "gs.Scene() created OK"

genesis_scene_ok = False
if genesis_init_ok:
    genesis_scene_ok = check(
        "genesis Scene creation",
        _genesis_scene,
        "Genesis may need re-initialising — run gs.init() before gs.Scene()"
    )

def _genesis_rigid_entity():
    import genesis as gs
    scene = gs.Scene(gravity=(0, -9.81, 0), dt=1/240, substeps=4)
    floor = scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(friction=0.5),
        fixed=True,
    )
    return f"entity id={floor.id}"

genesis_entity_ok = False
if genesis_init_ok:
    genesis_entity_ok = check(
        "genesis add rigid entity",
        _genesis_rigid_entity,
        "Ensure genesis-world is up to date: pip install -U genesis-world"
    )

def _genesis_fem():
    import genesis as gs, numpy as np
    scene = gs.Scene(gravity=(0, -9.81, 0), dt=1/240, substeps=4)
    mat   = gs.materials.FEM(E=1_000_000, nu=0.45, rho=1200)
    return "gs.materials.FEM() created OK"

genesis_fem_ok = False
if genesis_import_ok:
    genesis_fem_ok = check(
        "genesis FEM material",
        _genesis_fem,
        "FEM solver requires genesis-world >= 0.4"
    )

def _genesis_mpm():
    import genesis as gs
    mat = gs.materials.MPM(E=1_000, nu=0.45, rho=1200)
    return "gs.materials.MPM() created OK"

genesis_mpm_ok = False
if genesis_import_ok:
    genesis_mpm_ok = check(
        "genesis MPM material",
        _genesis_mpm,
        "MPM solver requires genesis-world >= 0.4"
    )


# ── 5. Mesh pipeline smoke test ───────────────────────────────────────────────
section("Mesh pipeline  (trimesh → coacd → Genesis)")

def _mesh_pipeline():
    import numpy as np, trimesh, coacd

    # Build a simple icosphere in-memory (no file needed)
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.05)
    sphere.apply_translation(-sphere.centroid)

    cm    = coacd.Mesh(
        np.array(sphere.vertices, dtype=np.float64),
        np.array(sphere.faces,    dtype=np.int32),
    )
    parts = coacd.run_coacd(cm, max_convex_hull=4)
    if not parts:
        raise RuntimeError("coacd returned 0 parts")
    return f"{len(sphere.vertices)} verts → {len(parts)} convex hull(s)"

check(
    "trimesh + coacd decomposition",
    _mesh_pipeline,
    "pip install trimesh coacd"
)

def _genesis_mesh_entity():
    import numpy as np, trimesh, genesis as gs

    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.05)
    sphere.apply_translation(-sphere.centroid)

    scene = gs.Scene(gravity=(0, -9.81, 0), dt=1/240, substeps=4)
    mat   = gs.materials.Rigid(friction=0.4)

    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as tf:
        tmp_path = tf.name
    try:
        sphere.export(tmp_path)
        e = scene.add_entity(
            gs.morphs.Mesh(file=tmp_path),
            material=mat,
            pos=(0, 0.1, 0),
        )
        return f"mesh entity id={e.id}"
    finally:
        os.unlink(tmp_path)

if genesis_init_ok:
    check(
        "genesis load mesh entity",
        _genesis_mesh_entity,
        "Ensure genesis-world is up to date"
    )


# ── 6. Optional / web frontend deps ───────────────────────────────────────────
section("Optional dependencies")
opt_deps = [
    ("fastapi",    lambda: importlib.import_module("fastapi").__version__,
     "pip install fastapi"),
    ("uvicorn",    lambda: importlib.import_module("uvicorn").__version__,
     "pip install uvicorn"),
    ("websockets", lambda: importlib.import_module("websockets").__version__,
     "pip install websockets"),
    ("imageio",    lambda: importlib.import_module("imageio").__version__,
     "pip install imageio[ffmpeg]"),
]
for label, fn, hint in opt_deps:
    check(label, fn, hint)


# ── Summary ───────────────────────────────────────────────────────────────────
section("Summary")

critical = [
    ("numpy",                   results.get("numpy",   False)),
    ("trimesh",                 results.get("trimesh", False)),
    ("coacd",                   results.get("coacd",   False)),
    ("torch",                   torch_ok),
    ("genesis import",          genesis_import_ok),
    ("genesis init (CPU)",      genesis_init_ok),
    ("genesis Scene",           genesis_scene_ok),
    ("genesis FEM material",    genesis_fem_ok),
]

all_ok = all(v for _, v in critical)

if all_ok:
    print(f"\n  {GREEN}{BOLD}✔  All critical checks passed — ready to run simulation.py{RESET}\n")
else:
    print(f"\n  {RED}{BOLD}✗  Some checks failed. Fix the items marked above, then re-run.{RESET}")
    print(f"\n  {YELLOW}Quick install (CPU):{RESET}")
    print("    pip install torch --index-url https://download.pytorch.org/whl/cpu")
    print("    pip install genesis-world trimesh coacd numpy scipy networkx pandas h5py open3d\n")