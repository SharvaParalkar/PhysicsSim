# PhysicsSim - How to Use

This project runs a Genesis-based granular particle simulation with an optional local web viewer (React/Three.js).

You have two ways to get started:

## Option A: Download the `PhysicsSimUpload` folder

1. Copy `PhysicsSimUpload/` to any location on your machine.
2. Open a terminal in the `PhysicsSimUpload/` folder.
3. Install the Python backend dependencies:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Start the backend (FastAPI):
   ```powershell
   python server.py
   ```
5. In a second terminal, start the frontend (Vite):
   ```powershell
   cd frontend
   npm install
   npm run dev
   ```
6. Open the URL Vite prints (usually `http://localhost:5173`).

Outputs are written into `results/` (CSV/HDF5).

## Option B: Clone the git repo

1. Clone the repository:
   ```powershell
   git clone <repo-url>
   cd PhysicsSim
   ```
2. Then follow the same steps as Option A:
   - `python -m venv .venv` + `pip install -r requirements.txt`
   - `python server.py`
   - `cd frontend && npm install && npm run dev`

## Optional: Run the simulation script directly (no web UI)

From the project root (`PhysicsSimUpload/` or the cloned repo):
```powershell
.\.venv\Scripts\Activate.ps1
python simulation.py --backend auto --n 100 --duration 5.0
```

# Granular Particle Simulation & Contact Analysis

### Implementation Guide — Genesis + React/Three.js Stack

---

## Overview

This guide describes how to build a granular particle jamming simulation and contact-analysis tool that runs **fully locally** on your machine. The tool loads a single 3D particle geometry from an OBJ or STL file, duplicates it N times, drops the particles from a configurable height into an environment under standard gravity, and extracts a full suite of contact metrics — with support for variable material stiffness from fully rigid down to gel-soft.

The simulation environment is configurable: either a flat rigid plate for open-pile tests, or an open-top cylinder with adjustable diameter and height for confined packing tests.

### Why Genesis

Genesis is the only Python-accessible physics engine that satisfies all requirements simultaneously:

- Runs entirely locally — no cloud, no external services required after install
- Accepts arbitrary mesh geometry from OBJ and STL files directly
- True deformable objects (FEM solver) with a continuously tunable stiffness parameter `E`
- Coupled rigid + deformable interactions in the same scene
- GPU acceleration available but CPU-only mode fully supported
- Mesh-based contact enabling real contact area computation for soft objects
- Full contact state accessible per-frame: position, normal, depth, force

---

## Local-First Requirement

This tool is designed to run entirely on your local machine. No internet connection is needed after installation.

```
Genesis simulation  →  runs in Python locally
Open3D viewer       →  local 3D window (no browser required)
CSV / HDF5 output   →  written to local disk
Web frontend        →  optional, also runs locally via npm run dev
```

The web frontend (React + Three.js) is optional. Running `python simulation.py` from the terminal is sufficient for a complete workflow including 3D visualisation.

---

## Stack

|Layer|Technology|
|---|---|
|Physics engine|`genesis-world` (Rigid + FEM + MPM solvers)|
|Mesh loading|`trimesh` — OBJ and STL native support|
|Mesh preprocessing|`coacd` — convex decomposition|
|Numerics & analysis|`numpy`, `scipy`, `networkx`, `pandas`|
|Local 3D viewer|`pyvista`|
|Data storage|`h5py` (HDF5 for replay), CSV export|
|Backend API (optional)|`fastapi`, `websockets`, `uvicorn`|
|Web 3D viewer (optional)|`Three.js` via `react-three-fiber`|
|Contact network viz (optional)|`D3.js` force-directed graph|
|Metric charts (optional)|`Recharts`|
|Frontend shell (optional)|`React` + `TypeScript`|

---

## Installation

Everything installs locally via pip. No Docker required for basic use.

```bash
# Python 3.10–3.13 required

# 1. Install PyTorch — choose the variant for your hardware

# CPU only (works on any machine, no GPU needed):
pip install torch --index-url https://download.pytorch.org/whl/cpu

# NVIDIA GPU — match your CUDA version, e.g. CUDA 12.6:
pip install torch --index-url https://download.pytorch.org/whl/cu126

# Apple Silicon (Metal):
pip install torch

# 2. Install Genesis
pip install genesis-world

# 3. Install simulation dependencies
pip install trimesh coacd numpy scipy networkx pandas h5py pyvista

# 4. Optional — web frontend server
pip install fastapi websockets uvicorn

# 5. Optional — web frontend (requires Node.js 18+)
cd frontend
npm install react react-dom three @react-three/fiber @react-three/drei d3 recharts
npm install -D typescript @types/react @types/three
```

**Verify your install before running:**

```bash
python -c "import genesis as gs; gs.init(backend=gs.cpu); print('Genesis OK')"
python -c "import trimesh; print('trimesh OK')"
python -c "import pyvista as pv; print('PyVista OK')"
```

---

## Adjustable Parameters

All simulation behaviour is controlled from one block at the top of `simulation.py`. Edit these before each run — no other part of the code needs to change.

```python
# ── Input ──────────────────────────────────────────────────────────────
PARTICLE_FILE        = "particle.obj"  # path to OBJ or STL file
N_PARTICLES          = 100             # number of particle copies to spawn
SCALE_FACTOR         = 1.0             # uniform scale (1.0 = no change)
                                        # e.g. 0.001 converts mm mesh to metres

# ── Material (stiffness spectrum) ──────────────────────────────────────
# YOUNGS_MODULUS is the single dial for rigidness.
# The correct internal solver is selected automatically from this value.
#
#   Steel / ceramic:   E = 200_000_000  (200 MPa)  → Rigid solver
#   Hard plastic:      E =  50_000_000  (50 MPa)   → Rigid solver
#   Hard rubber:       E =   1_000_000  (1 MPa)    → FEM solver
#   Soft silicone:     E =      10_000  (10 kPa)   → FEM solver
#   Gel / putty:       E =       1_000  (1 kPa)    → MPM solver
YOUNGS_MODULUS       = 1_000_000       # Pa — main rigidness dial
POISSON_RATIO        = 0.45            # 0.5 = fully incompressible (rubber-like)
DENSITY              = 1200            # kg/m³

# ── Environment ────────────────────────────────────────────────────────
# ENVIRONMENT_TYPE sets what the particles fall into:
#   "cylinder"  — open-top cylinder, adjustable diameter and height (default)
#   "plate"     — flat rigid plate, open in all directions
ENVIRONMENT_TYPE     = "cylinder"

# Plate settings (active when ENVIRONMENT_TYPE = "plate")
PLATE_SIZE           = 0.6             # metres — side length of square plate

# Cylinder settings (active when ENVIRONMENT_TYPE = "cylinder")
CYLINDER_DIAMETER    = 0.20            # metres — inner diameter
CYLINDER_HEIGHT      = 0.30            # metres — wall height (open top)
CYLINDER_SEGMENTS    = 32             # facets used to approximate the cylinder wall
                                        # 24+ recommended; lower values leave gaps
WALL_THICKNESS       = 0.008          # metres

# ── Drop configuration ─────────────────────────────────────────────────
# DROP_HEIGHT is measured from the top edge of the container (or plate surface).
# Particles are placed in a grid at this height and released under gravity.
DROP_HEIGHT          = 0.15            # metres above the container top edge
DROP_SPREAD          = 0.85            # fraction of container inner diameter
                                        # used for initial XZ placement (0–1)

# ── Gravity ────────────────────────────────────────────────────────────
GRAVITY              = (0, -9.81, 0)   # m/s² — standard Earth gravity
                                        # Y is the vertical axis (up = +Y)
                                        # Moon: (0, -1.62, 0)
                                        # Mars: (0, -3.72, 0)

# ── Simulation ─────────────────────────────────────────────────────────
DT                   = 1 / 240         # timestep in seconds
SUBSTEPS             = 4               # solver substeps per timestep
SIM_DURATION         = 5.0             # maximum simulated time in seconds
SETTLE_THRESHOLD     = 1e-4            # max particle speed (m/s) to declare settled
                                        # lower = wait longer for full settlement

# ── Contact analysis ───────────────────────────────────────────────────
CONTACT_SAMPLE_EVERY = 10              # extract contacts every N steps
CONTACT_DEPTH_TOL    = 1e-5            # minimum penetration depth to count as contact

# ── Output ─────────────────────────────────────────────────────────────
OUTPUT_DIR           = "./results"
SAVE_HDF5            = True            # saves full replay to simulation.h5
SAVE_CSV             = True            # saves CSV files for particles and contacts
```

---

## Material Stiffness — Solver Assignment

Genesis automatically routes each object to the correct internal solver based on the material you assign. You do not manage this manually — setting `YOUNGS_MODULUS` is all that is needed.

```
E > ~100 MPa   →  Rigid body solver   metal, ceramic, glass, hard plastic
E   1 kPa–100 MPa  →  FEM solver      rubber, foam, silicone, soft plastic
E < ~1 kPa     →  MPM solver          gel, clay, putty, very soft tissue
```

In a mixed scene (e.g. some rigid particles and some rubber ones), Genesis resolves cross-solver contacts automatically. A rigid particle landing on an FEM particle will correctly deform the soft one and transfer contact forces.

---

## Environment Types

### Open-top cylinder (default)

An upright cylinder with a solid bottom disc and open top. The wall is built from `CYLINDER_SEGMENTS` flat rectangular panels arranged in a ring — at 32 segments the interior is visually smooth. The cylinder is fixed in place and, in the web viewer, rendered as a transparent mesh so the interior is visible at all times.

```
    │             │  ← open top — particles fall in freely
    │  particles  │
    │  pack here  │
    └─────────────┘  ← solid bottom disc
```

`CYLINDER_DIAMETER` controls the inner usable diameter. `CYLINDER_HEIGHT` controls the wall height. Increase either to accommodate more or larger particles.

### Flat plate

A square rigid plate lying on the horizontal plane. Particles fall onto the open surface and form a free pile with no lateral confinement. Use this for unconfined packing tests or to observe free-pile geometry.

---

## Code Structure

### 1. `load_particle_mesh()`

Handles both OBJ and STL formats. Applies scale, centres the mesh, and runs convex decomposition so Genesis can use it for collision detection.

```python
import trimesh, coacd, numpy as np, math

def load_particle_mesh(filepath: str, scale: float = 1.0):
    """
    Load an OBJ or STL file, apply uniform scale, and run convex decomposition.
    Returns (physics_mesh, original_mesh):
      physics_mesh   — decomposed compound mesh passed to Genesis
      original_mesh  — original geometry used for visualisation only

    Supported formats: .obj, .stl (binary and ASCII)

    Convex decomposition (coacd) is required for any concave shape.
    Genesis's collision pipeline needs convex geometry; coacd splits a
    concave mesh into a set of convex sub-meshes (a compound shape).

    Accuracy:
      Convex particles (spheres, cubes, ellipsoids) — exact collision
      Mildly concave particles                      — small approximation error
      Deeply concave (donut, hook, shell)            — use max_convex_hull=64
    """
    ext = filepath.lower().rsplit(".", 1)[-1]
    if ext not in ("obj", "stl"):
        raise ValueError(f"Unsupported format '.{ext}'. Provide an OBJ or STL file.")

    original_mesh = trimesh.load(filepath, force="mesh")
    if original_mesh.is_empty:
        raise ValueError(f"Mesh loaded from '{filepath}' is empty. Check the file.")

    original_mesh.apply_scale(scale)
    original_mesh.apply_translation(-original_mesh.centroid)

    parts        = coacd.run_coacd(original_mesh, max_convex_hull=32)
    physics_mesh = trimesh.util.concatenate(parts)

    print(f"Loaded: {filepath}")
    print(f"  Vertices : {len(original_mesh.vertices)}")
    print(f"  Faces    : {len(original_mesh.faces)}")
    print(f"  Extents  : {original_mesh.extents.round(5)} m")
    print(f"  Volume   : {original_mesh.volume:.6f} m³")
    print(f"  Convex parts after decomposition: {len(parts)}")

    return physics_mesh, original_mesh
```

---

### 2. `create_environment()`

Builds whichever environment is selected and returns all container entity IDs for contact classification, plus geometry facts used by `spawn_particles()`.

```python
import genesis as gs

def create_environment(scene: gs.Scene,
                       kind: str,
                       plate_size: float     = 0.6,
                       cyl_diameter: float   = 0.20,
                       cyl_height: float     = 0.30,
                       cyl_segments: int     = 32,
                       wall_thickness: float = 0.008) -> tuple[set, dict]:
    """
    Build the simulation environment.

    Returns:
      container_ids  — set of entity IDs; used to classify P-container contacts
      env_info       — geometry facts consumed by spawn_particles()
        Keys present for both types:
          surface_y      — Y coordinate of the lowest interior surface
          top_y          — Y coordinate of the top edge (spawn reference)
          spread_radius  — max XZ radius for initial particle placement
        Cylinder only:
          inner_radius   — inner radius of the cylinder
    """
    mat           = gs.materials.Rigid(friction=0.55, restitution=0.05)
    container_ids = set()

    if kind == "plate":
        t     = wall_thickness
        plate = scene.add_entity(
            gs.morphs.Box(size=(plate_size, t, plate_size)),
            material = mat,
            pos      = (0, t / 2, 0),
            fixed    = True,
        )
        container_ids.add(plate.id)
        env_info = {
            "surface_y":     t,
            "top_y":         t,
            "spread_radius": plate_size / 2 * 0.9,
        }

    elif kind == "cylinder":
        r_inner = cyl_diameter / 2
        t       = wall_thickness

        # Bottom disc
        bottom = scene.add_entity(
            gs.morphs.Cylinder(radius=r_inner + t, height=t),
            material = mat,
            pos      = (0, t / 2, 0),
            fixed    = True,
        )
        container_ids.add(bottom.id)

        # Wall panels arranged in a ring
        angle_step  = 2 * math.pi / cyl_segments
        chord_width = 2 * r_inner * math.sin(angle_step / 2)

        for i in range(cyl_segments):
            angle = i * angle_step
            cx    = (r_inner + t / 2) * math.cos(angle)
            cz    = (r_inner + t / 2) * math.sin(angle)
            cy    = t + cyl_height / 2      # sit flush on the bottom disc

            panel = scene.add_entity(
                gs.morphs.Box(size=(chord_width, cyl_height, t)),
                material = mat,
                pos      = (cx, cy, cz),
                quat     = gs.utils.euler_to_quat((0, math.degrees(angle), 0)),
                fixed    = True,
            )
            container_ids.add(panel.id)

        env_info = {
            "surface_y":     t,
            "top_y":         t + cyl_height,
            "inner_radius":  r_inner,
            "spread_radius": r_inner * 0.85,   # margin from walls
        }

    else:
        raise ValueError(
            f"Unknown ENVIRONMENT_TYPE '{kind}'. Choose 'plate' or 'cylinder'."
        )

    return container_ids, env_info
```

---

### 3. `spawn_particles()`

Places N particle copies at `DROP_HEIGHT` metres above the container top edge, spread within the inner diameter.

```python
def spawn_particles(scene: gs.Scene,
                    physics_mesh: trimesh.Trimesh,
                    n: int,
                    env_info: dict,
                    drop_height: float,
                    drop_spread: float,
                    E: float,
                    nu: float,
                    rho: float) -> list:
    """
    Spawn N copies of the particle above the environment.

    Spawn Y  = env_info["top_y"] + drop_height
    XZ range = env_info["spread_radius"] * drop_spread

    Grid-jitter layout minimises initial mesh overlaps.
    Each particle receives a uniform random orientation.

    Solver auto-selected from E:
      E > 1e8 Pa  →  Rigid   (metal, ceramic, hard plastic)
      E > 1e3 Pa  →  FEM     (rubber, foam, silicone)
      E ≤ 1e3 Pa  →  MPM     (gel, clay, putty)
    """
    if E > 1e8:
        material = gs.materials.Rigid(friction=0.4, restitution=0.2)
    elif E > 1e3:
        material = gs.materials.FEM(E=E, nu=nu, rho=rho)
    else:
        material = gs.materials.MPM(E=E, nu=nu, rho=rho)

    extents  = physics_mesh.bounds[1] - physics_mesh.bounds[0]
    p_radius = float(np.linalg.norm(extents)) / 2.0
    spacing  = p_radius * 2.4

    spawn_r  = env_info["spread_radius"] * drop_spread
    spawn_y0 = env_info["top_y"] + drop_height

    cols     = int(math.ceil(math.sqrt(n)))
    entities = []

    for i in range(n):
        row, col = divmod(i, cols)
        x = (col - cols / 2 + 0.5) * spacing
        z = (row - cols / 2 + 0.5) * spacing

        # Clamp XZ to spawn radius (important for cylinder)
        dist = math.sqrt(x**2 + z**2)
        if dist > spawn_r:
            s  = spawn_r / dist * np.random.uniform(0.7, 1.0)
            x *= s
            z *= s

        x += np.random.uniform(-p_radius * 0.2, p_radius * 0.2)
        z += np.random.uniform(-p_radius * 0.2, p_radius * 0.2)

        # Stagger drop heights slightly to reduce simultaneous impacts
        y = spawn_y0 + (i % cols) * p_radius * 0.3

        quat = gs.utils.mat_to_quat(
            trimesh.transformations.random_rotation_matrix()[:3, :3]
        )

        entity = scene.add_entity(
            gs.morphs.Mesh(mesh=physics_mesh),
            material = material,
            pos      = (x, y, z),
            quat     = quat,
        )
        entities.append(entity)

    print(f"Spawned {n} particles")
    print(f"  Drop height above container top: {drop_height:.3f} m")
    print(f"  Spawn Y range: {spawn_y0:.3f} – {spawn_y0 + p_radius:.3f} m")
    return entities
```

---

### 4. `run_simulation()`

Steps the simulation until all particles fall below the settle threshold or the time limit is reached.

```python
def run_simulation(scene: gs.Scene,
                   dt: float,
                   substeps: int,
                   duration: float,
                   settle_threshold: float) -> int:
    """
    Step the simulation. Checks for settlement every 60 steps.
    Returns the number of steps executed.
    """
    scene.build()
    total_steps = int(duration / dt)

    for step in range(total_steps):
        scene.step()

        if step % 60 == 0:
            velocities = [
                float(np.linalg.norm(e.get_vel()))
                for e in scene.entities
                if not getattr(e, "fixed", False)
            ]
            max_vel = max(velocities) if velocities else 0.0
            print(f"  t={step * dt:.2f}s  max_vel={max_vel:.5f} m/s", end="\r")

            if max_vel < settle_threshold:
                print(f"\nSettled at t={step * dt:.3f}s  (step {step})")
                return step

    print(f"\nReached max duration ({duration}s)")
    return total_steps
```

---

### 5. `extract_contacts()` — Unified Contact Extraction

Normalises contact data from all Genesis solvers into a single schema and classifies each contact as particle-particle or particle-container.

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class NormalizedContact:
    entity_a:              int
    entity_b:              int
    is_particle_particle:  bool
    is_particle_container: bool
    position:              np.ndarray    # (3,) world coordinates in metres
    normal:                np.ndarray    # (3,) unit vector, A → B direction
    depth:                 float         # penetration depth in metres
    force:                 Optional[float]   # magnitude in Newtons (None for MPM)
    contact_area:          Optional[float]   # m² (FEM-FEM contacts only)


def extract_contacts(scene: gs.Scene,
                     particle_ids: set,
                     container_ids: set,
                     depth_tol: float = 1e-5) -> list[NormalizedContact]:
    """
    Pull all contacts from Genesis, classify them, and filter noise.
    Contacts with depth below depth_tol are discarded.
    """
    raw     = scene.get_contacts()
    results = []

    for c in raw:
        if abs(c.depth) < depth_tol:
            continue

        a, b  = c.entity_a.id, c.entity_b.id
        is_pp = (a in particle_ids) and (b in particle_ids)
        is_pc = ((a in particle_ids) and (b in container_ids)) or \
                ((b in particle_ids) and (a in container_ids))

        if not (is_pp or is_pc):
            continue

        results.append(NormalizedContact(
            entity_a              = a,
            entity_b              = b,
            is_particle_particle  = is_pp,
            is_particle_container = is_pc,
            position              = np.array(c.pos),
            normal                = np.array(c.normal),
            depth                 = float(c.depth),
            force                 = float(c.force) if hasattr(c, "force") else None,
            contact_area          = float(c.area)  if hasattr(c, "area")  else None,
        ))

    return results
```

---

### 6. `compute_metrics()`

Derives all required metrics from the normalised contact list.

```python
import networkx as nx

def compute_metrics(contacts: list[NormalizedContact],
                    particle_ids: set) -> dict:
    pp = [c for c in contacts if c.is_particle_particle]
    pc = [c for c in contacts if c.is_particle_container]
    n  = len(particle_ids)

    G = nx.Graph()
    G.add_nodes_from(particle_ids)
    for c in pp:
        G.add_edge(c.entity_a, c.entity_b,
                   depth = c.depth,
                   force = c.force,
                   area  = c.contact_area)

    contact_counts     = dict(G.degree())
    isolated           = [pid for pid, deg in contact_counts.items() if deg == 0]
    container_touching = {
        c.entity_a if c.entity_a in particle_ids else c.entity_b
        for c in pc
    }

    return {
        # Summary counts
        "total_pp_contacts":            len(pp),
        "total_pc_contacts":            len(pc),
        "avg_contacts_per_particle":    len(pp) * 2 / n if n else 0,
        "Z":                            len(pp) * 2 / n if n else 0,
        "n_isolated_particles":         len(isolated),
        "n_container_touching":         len(container_touching),

        # Per-contact geometry
        "contact_points":               [c.position for c in pp],
        "contact_normals":              [c.normal   for c in pp],
        "contact_depths":               [c.depth    for c in pp],
        "contact_forces":               [c.force    for c in pp if c.force  is not None],
        "contact_areas":                [c.contact_area for c in pp
                                         if c.contact_area is not None],

        # Graph
        "contact_graph":                G,
        "contact_graph_dict":           {pid: list(G.neighbors(pid))
                                         for pid in particle_ids},
        "contact_counts_per_particle":  contact_counts,
    }
```

---

### 7. `export_results()`

Writes particle states, contact pairs, and contact-point data to disk.

```python
import os, pandas as pd, h5py

def export_results(scene, entities, metrics, container_ids,
                   output_dir: str, save_hdf5: bool, save_csv: bool):
    os.makedirs(output_dir, exist_ok=True)

    # Particle positions, orientations, contact counts
    rows = []
    for e in entities:
        pos  = e.get_pos()
        quat = e.get_quat()
        rows.append({
            "id": e.id,
            "x": pos[0], "y": pos[1], "z": pos[2],
            "qx": quat[0], "qy": quat[1], "qz": quat[2], "qw": quat[3],
            "n_contacts": metrics["contact_counts_per_particle"].get(e.id, 0),
        })
    df_particles = pd.DataFrame(rows)

    # Contact pairs
    G = metrics["contact_graph"]
    df_contacts = pd.DataFrame([
        {"particle_a": a, "particle_b": b,
         "depth": d.get("depth"), "force": d.get("force"), "contact_area": d.get("area")}
        for a, b, d in G.edges(data=True)
    ])

    # Contact points and normals
    pts, nrms = metrics["contact_points"], metrics["contact_normals"]
    df_points = pd.DataFrame([
        {"x": p[0], "y": p[1], "z": p[2], "nx": n[0], "ny": n[1], "nz": n[2]}
        for p, n in zip(pts, nrms)
    ])

    if save_csv:
        df_particles.to_csv(f"{output_dir}/particles.csv",      index=False)
        df_contacts.to_csv( f"{output_dir}/contact_pairs.csv",  index=False)
        df_points.to_csv(   f"{output_dir}/contact_points.csv", index=False)
        print(f"CSV files written to {output_dir}/")

    if save_hdf5:
        with h5py.File(f"{output_dir}/simulation.h5", "w") as f:
            f.create_dataset("particle_positions",
                             data=df_particles[["x","y","z"]].values)
            f.create_dataset("contact_points",
                             data=np.array(pts)  if pts  else np.empty((0, 3)))
            f.create_dataset("contact_normals",
                             data=np.array(nrms) if nrms else np.empty((0, 3)))
            f.attrs["Z"]                 = metrics["Z"]
            f.attrs["total_pp"]          = metrics["total_pp_contacts"]
            f.attrs["total_pc"]          = metrics["total_pc_contacts"]
            f.attrs["n_isolated"]        = metrics["n_isolated_particles"]
            f.attrs["n_container_touch"] = metrics["n_container_touching"]
        print(f"HDF5 replay written to {output_dir}/simulation.h5")

    # Console summary
    print("\n── Contact Analysis Summary ────────────────────────────────")
    print(f"  Particles simulated:               {len(entities)}")
    print(f"  Total particle-particle contacts:  {metrics['total_pp_contacts']}")
    print(f"  Total particle-container contacts: {metrics['total_pc_contacts']}")
    print(f"  Avg contacts per particle (Z):     {metrics['Z']:.3f}")
    print(f"  Isolated particles (Z=0):          {metrics['n_isolated_particles']}")
    print(f"  Particles touching container:      {metrics['n_container_touching']}")
    if metrics["contact_forces"]:
        forces = np.array(metrics["contact_forces"])
        print(f"  Mean contact force:                {forces.mean():.4f} N")
        print(f"  Max  contact force:                {forces.max():.4f} N")
    if metrics["contact_areas"]:
        areas = np.array(metrics["contact_areas"])
        print(f"  Mean contact area:                 {areas.mean() * 1e6:.4f} mm²")
    print("────────────────────────────────────────────────────────────\n")
```

---

### 8. `visualize_results()`

Local 3D viewer via Open3D. Particles are coloured by contact count. Contact points appear as yellow spheres.

```python
import pyvista as pv

def visualize_results(entities, metrics, original_mesh: trimesh.Trimesh):
    """
    Opens a local 3D window. Particles coloured blue (0 contacts) → red (max).
    Yellow spheres mark contact point locations.
    """
    geoms  = []
    counts = metrics["contact_counts_per_particle"]
    max_c  = max(counts.values()) if counts and max(counts.values()) > 0 else 1

    for e in entities:
        pos  = e.get_pos()
        quat = e.get_quat()
        c    = counts.get(e.id, 0)

        m = o3d.geometry.TriangleMesh(
            vertices  = o3d.utility.Vector3dVector(original_mesh.vertices),
            triangles = o3d.utility.Vector3iVector(original_mesh.faces),
        )
        t = c / max_c
        m.paint_uniform_color([t, 0.2, 1.0 - t])
        m.rotate(o3d.geometry.get_rotation_matrix_from_quaternion(quat),
                 center=(0, 0, 0))
        m.translate(pos)
        m.compute_vertex_normals()
        geoms.append(m)

    for pt in metrics["contact_points"]:
        s = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
        s.translate(pt)
        s.paint_uniform_color([1.0, 1.0, 0.0])
        geoms.append(s)

    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05))

    o3d.visualization.draw_geometries(
        geoms,
        window_name = "Particle Contact Analysis",
        width=1280, height=800,
    )
```

---

### 9. `main()`

```python
import genesis as gs, math

def main():
    # Init — use gs.gpu if a GPU is available, gs.cpu otherwise
    gs.init(backend=gs.cpu)
    scene = gs.Scene(gravity=GRAVITY, dt=DT, substeps=SUBSTEPS)

    physics_mesh, original_mesh = load_particle_mesh(PARTICLE_FILE, SCALE_FACTOR)

    container_ids, env_info = create_environment(
        scene,
        kind           = ENVIRONMENT_TYPE,
        plate_size     = PLATE_SIZE,
        cyl_diameter   = CYLINDER_DIAMETER,
        cyl_height     = CYLINDER_HEIGHT,
        cyl_segments   = CYLINDER_SEGMENTS,
        wall_thickness = WALL_THICKNESS,
    )

    entities = spawn_particles(
        scene, physics_mesh,
        n           = N_PARTICLES,
        env_info    = env_info,
        drop_height = DROP_HEIGHT,
        drop_spread = DROP_SPREAD,
        E           = YOUNGS_MODULUS,
        nu          = POISSON_RATIO,
        rho         = DENSITY,
    )
    particle_ids = {e.id for e in entities}

    run_simulation(scene, DT, SUBSTEPS, SIM_DURATION, SETTLE_THRESHOLD)

    contacts = extract_contacts(scene, particle_ids, container_ids,
                                 depth_tol=CONTACT_DEPTH_TOL)
    metrics  = compute_metrics(contacts, particle_ids)

    export_results(scene, entities, metrics, container_ids,
                   OUTPUT_DIR, SAVE_HDF5, SAVE_CSV)

    visualize_results(entities, metrics, original_mesh)


if __name__ == "__main__":
    main()
```

---

## Running the Simulation

```bash
# Edit parameters at the top of simulation.py, then:
python simulation.py

# Switch to a cylindrical container (default):
# ENVIRONMENT_TYPE = "cylinder"

# Switch to a flat plate:
# ENVIRONMENT_TYPE = "plate"

# Change cylinder dimensions:
# CYLINDER_DIAMETER = 0.15   (150 mm diameter)
# CYLINDER_HEIGHT   = 0.40   (400 mm tall)

# Adjust where particles start falling from:
# DROP_HEIGHT = 0.30   (drop from 300 mm above the top edge)
```

---

## Optional Web Frontend (Local)

For an interactive browser-based view, run Genesis as a headless local server and connect React via WebSocket. Both run on your local machine — no internet needed.

```
localhost:3000  (React)                  localhost:8000  (Python)
─────────────────────────                ─────────────────────────────────
Three.js 3D viewer   ←── WS frames ───  FastAPI + WebSocket
D3 contact graph     ←── metrics   ───  Genesis simulation
Recharts dashboard                       Contact extraction layer
Material editor      ──── cmds  ──────→ Per-frame metrics
Sim controls                             HDF5 replay storage
```

The cylinder renders as a transparent mesh in Three.js so the interior is visible during simulation.

**Material preset slider (TypeScript):**

```typescript
const MATERIAL_PRESETS = [
  { label: "Steel",        E: 200_000_000 },
  { label: "Hard plastic", E:  50_000_000 },
  { label: "Hard rubber",  E:   1_000_000 },
  { label: "Silicone",     E:      10_000 },
  { label: "Gel",          E:       1_000 },
];
// Use a logarithmic slider: value = log10(E), range 3–8
```

---

## Contact Metrics Reference

|Metric|How it's computed|Notes|
|---|---|---|
|Total P-P contacts|`len(pp_contacts)`|After depth-tolerance filter|
|Total P-container contacts|`len(pc_contacts)`|Same filter|
|Avg contacts per particle (Z)|`2 × pp / N`|Jamming transition at Z ≈ 6|
|Contact graph|`networkx.Graph`|Nodes = particles, edges = contacts|
|Which particles touch which|`G.neighbors(pid)`|Adjacency list per particle|
|Contact positions|`c.position`|3D world coordinates (metres)|
|Contact normals|`c.normal`|Unit vector, particle A → B|
|Penetration depth|`c.depth`|Metres|
|Contact force|`c.force`|Newtons — Rigid and FEM only|
|Contact area|`c.contact_area`|m² — FEM-FEM pairs only|
|Isolated particles|`G.degree(pid) == 0`||
|Particles touching container|from P-container contact set||

**Contact area:** Rigid-rigid contacts are point contacts — `contact_area` is `None`. FEM-FEM contacts produce a real contact patch area from mesh geometry. FEM-rigid contacts approximate area from FEM surface deformation.

**MPM contacts:** The MPM solver (gel/putty) does not produce discrete contact pairs. Force and depth will be `None`. If contact metrics are critical for very soft materials, use FEM with a low `E` (e.g. 500 Pa) instead of MPM.

---

## Extending to Contact Efficiency

```python
def contact_efficiency(metrics, entities, mesh):
    """
    contact_efficiency = stable P-P contacts / total particle volume

    Higher values mean the particle shape packs more contacts
    into a given volume. Use to compare different geometries.
    """
    volume = mesh.volume * len(entities)
    return metrics["total_pp_contacts"] / volume


def weighted_contact_efficiency(metrics, entities, mesh):
    """Force-weighted: total contact force / total particle volume (N/m³)."""
    total_force = sum(f for f in metrics["contact_forces"] if f)
    return total_force / (mesh.volume * len(entities))
```

---

## Limitations & Known Constraints

**Concave mesh approximation** — coacd splits concave shapes into convex hulls. Deep concavities (interlocking features, hollow shells) will not generate contacts inside the concave region. Increase `max_convex_hull=64` in `coacd.run_coacd()` for higher fidelity at the cost of simulation speed.

**Cylinder wall gaps** — at low `CYLINDER_SEGMENTS` values (below 20) the panels leave small gaps at their seams. Particles smaller than the gap may escape the container. Use 24 or more segments, or increase `WALL_THICKNESS`.

**CPU performance** — FEM simulations with N > 200 will be slow on CPU. Reduce mesh resolution, lower `N_PARTICLES`, or use `gs.init(backend=gs.gpu)` when a GPU is available.

**MPM contact metrics** — gel/putty materials do not produce discrete contact pairs. Use FEM at low `E` if contact metrics matter for very soft objects.

**Genesis API maturity** — Genesis is at v0.4. The `scene.get_contacts()` API is functional but still evolving. If a contact field is unavailable in a given release, the extraction layer sets it to `None` gracefully. Check the Genesis release notes when upgrading.

---

## File Structure

```
project/
├── simulation.py           # complete simulation script
├── requirements.txt
├── particle.obj            # your input geometry (OBJ or STL)
├── results/
│   ├── particles.csv           # positions, orientations, contact counts
│   ├── contact_pairs.csv       # which particles touch which + force/depth/area
│   ├── contact_points.csv      # 3D contact positions and normals
│   └── simulation.h5           # full replay for web viewer (optional)
└── frontend/               # optional React web viewer (local)
    ├── src/
    │   ├── App.tsx
    │   ├── Viewer3D.tsx            # react-three-fiber scene
    │   ├── ContactGraph.tsx        # D3 force-directed contact network
    │   └── Dashboard.tsx           # Recharts metric charts
    └── package.json
``` 

---

### Prompt 1 — Project scaffold + configuration block

```
Create the complete file and folder structure for a granular particle 
simulation tool with the following layout:

project/
├── simulation.py
├── requirements.txt
├── particle.obj            (placeholder — empty file is fine)
└── results/                (empty directory)

In requirements.txt, list these exact dependencies:
  genesis-world
  trimesh
  coacd
  numpy
  scipy
  networkx
  pandas
  h5py
  pyvista
  fastapi
  websockets
  uvicorn

In simulation.py, write only the following — no functions yet:

1. All import statements at the top:
     import os, math
     import numpy as np
     import pandas as pd
     import h5py
     import trimesh
     import coacd
     import networkx as nx
     import genesis as gs
     import pyvista as pv
     from dataclasses import dataclass
     from typing import Optional

2. The complete parameter block with every variable listed below.
   Group them under these exact comment headers and use these exact
   variable names and default values:

   # ── Input ──────────────────────────────────────────────────────────
   PARTICLE_FILE        = "particle.obj"
   N_PARTICLES          = 100
   SCALE_FACTOR         = 1.0

   # ── Material ───────────────────────────────────────────────────────
   YOUNGS_MODULUS       = 1_000_000
   POISSON_RATIO        = 0.45
   DENSITY              = 1200

   # ── Environment ────────────────────────────────────────────────────
   ENVIRONMENT_TYPE     = "cylinder"
   PLATE_SIZE           = 0.6
   CYLINDER_DIAMETER    = 0.20
   CYLINDER_HEIGHT      = 0.30
   CYLINDER_SEGMENTS    = 32
   WALL_THICKNESS       = 0.008

   # ── Drop configuration ─────────────────────────────────────────────
   DROP_HEIGHT          = 0.15
   DROP_SPREAD          = 0.85

   # ── Gravity ────────────────────────────────────────────────────────
   GRAVITY              = (0, -9.81, 0)

   # ── Simulation ─────────────────────────────────────────────────────
   DT                   = 1 / 240
   SUBSTEPS             = 4
   SIM_DURATION         = 5.0
   SETTLE_THRESHOLD     = 1e-4

   # ── Contact analysis ───────────────────────────────────────────────
   CONTACT_SAMPLE_EVERY = 10
   CONTACT_DEPTH_TOL    = 1e-5

   # ── Output ─────────────────────────────────────────────────────────
   OUTPUT_DIR           = "./results"
   SAVE_HDF5            = True
   SAVE_CSV             = True

3. Add an inline comment on every parameter explaining what it does
   and what a sensible alternative value would be. For YOUNGS_MODULUS
   specifically, include the full spectrum comment showing steel through
   gel with solver assignment notes.

4. At the bottom of the file, add this verification block inside
   if __name__ == "__main__": that prints each parameter name and 
   value so the config can be confirmed before any simulation runs:

     if __name__ == "__main__":
         import inspect, sys
         this = sys.modules[__name__]
         params = {k: v for k, v in vars(this).items()
                   if k.isupper() and not k.startswith("_")}
         for k, v in params.items():
             print(f"  {k:<25} = {v}")

Do not write any functions yet. The file should be importable and 
the __main__ block should print all parameters cleanly.
```

---

### Prompt 2 — Mesh loading + convex decomposition

```
Add the load_particle_mesh() function to simulation.py. The function
signature must be:

  def load_particle_mesh(filepath: str, scale: float = 1.0)
      -> tuple[trimesh.Trimesh, trimesh.Trimesh]:

Implement it with this exact behavior:

1. Extract the file extension and raise a ValueError with a clear
   message if the extension is not "obj" or "stl" (case-insensitive).

2. Load the mesh using trimesh.load() with force="mesh". After loading,
   check if the mesh is empty and raise a ValueError if so, including
   the filepath in the message.

3. Apply the scale factor using mesh.apply_scale(scale).

4. Center the mesh at the origin using 
   mesh.apply_translation(-mesh.centroid).

5. Run convex decomposition using coacd.run_coacd() with 
   max_convex_hull=32. Concatenate the parts into a single mesh
   using trimesh.util.concatenate(). This is the physics_mesh.
   Keep the original pre-decomposition mesh as original_mesh.

6. Print the following to stdout after successful loading:
     Loaded: <filepath>
       Vertices : <count>
       Faces    : <count>
       Extents  : <x y z rounded to 5 decimal places> m
       Volume   : <value to 6 decimal places> m³
       Convex parts after decomposition: <count>

7. Return (physics_mesh, original_mesh) in that order.

Then replace the if __name__ == "__main__": block with a test that:
- Creates a small in-memory trimesh box (trimesh.creation.box()) 
  and saves it as /tmp/test_particle.obj using mesh.export()
- Calls load_particle_mesh("/tmp/test_particle.obj", scale=0.5)
- Prints "load_particle_mesh: OK" if no exception is raised
- Catches and prints any exception with a clear error message

The rest of simulation.py (imports, parameter block) must remain
completely unchanged.
```

---

### Prompt 3 — Environment builder

```
Add the create_environment() function to simulation.py. The function
signature must be:

  def create_environment(
      scene: gs.Scene,
      kind: str,
      plate_size: float     = 0.6,
      cyl_diameter: float   = 0.20,
      cyl_height: float     = 0.30,
      cyl_segments: int     = 32,
      wall_thickness: float = 0.008,
  ) -> tuple[set, dict]:

Implement it with this exact behavior:

1. Define a shared rigid material at the top of the function:
     mat = gs.materials.Rigid(friction=0.55, restitution=0.05)

2. Initialize an empty set called container_ids.

3. If kind == "plate":
   - Add a single Box entity with size=(plate_size, wall_thickness, 
     plate_size), positioned so its top surface sits at Y=0. 
     Set fixed=True. Add its .id to container_ids.
   - Build and return env_info with keys:
       surface_y     = wall_thickness
       top_y         = wall_thickness
       spread_radius = plate_size / 2 * 0.9

4. If kind == "cylinder":
   - Add a bottom disc as a Cylinder with radius=(cyl_diameter/2 + 
     wall_thickness) and height=wall_thickness, positioned so its 
     top surface sits at Y=0. Set fixed=True. Add its .id to 
     container_ids.
   - Build CYLINDER_SEGMENTS wall panels arranged in a ring using 
     this exact geometry:
       angle_step  = 2 * math.pi / cyl_segments
       chord_width = 2 * (cyl_diameter/2) * math.sin(angle_step / 2)
     For each panel i:
       angle = i * angle_step
       cx = (cyl_diameter/2 + wall_thickness/2) * math.cos(angle)
       cz = (cyl_diameter/2 + wall_thickness/2) * math.sin(angle)
       cy = wall_thickness + cyl_height / 2
     Each panel is a Box with size=(chord_width, cyl_height, 
     wall_thickness). Rotate each panel around Y by math.degrees(angle)
     using gs.utils.euler_to_quat((0, math.degrees(angle), 0)).
     Set fixed=True. Add every panel's .id to container_ids.
   - Build and return env_info with keys:
       surface_y     = wall_thickness
       top_y         = wall_thickness + cyl_height
       inner_radius  = cyl_diameter / 2
       spread_radius = (cyl_diameter / 2) * 0.85

5. If kind is anything else, raise a ValueError naming the bad value
   and listing the two valid options.

6. Return (container_ids, env_info).

Then replace the if __name__ == "__main__": block with a test that:
- Calls gs.init(backend=gs.cpu)
- Creates a gs.Scene with gravity=GRAVITY, dt=DT, substeps=SUBSTEPS
- Calls create_environment() with kind="cylinder" using the global 
  cylinder parameters
- Prints the returned env_info dict
- Calls scene.build() to confirm no Genesis errors
- Prints "create_environment: OK"
- Repeats the same test for kind="plate"

All prior code (imports, parameters, load_particle_mesh) must remain
completely unchanged.
```

---

### Prompt 4 — Particle spawning + frozen scene verification

```
Add the spawn_particles() function to simulation.py. The function
signature must be:

  def spawn_particles(
      scene: gs.Scene,
      physics_mesh: trimesh.Trimesh,
      n: int,
      env_info: dict,
      drop_height: float,
      drop_spread: float,
      E: float,
      nu: float,
      rho: float,
  ) -> list:

Implement it with this exact behavior:

1. Select the Genesis material based on E:
     if E > 1e8:   material = gs.materials.Rigid(friction=0.4, 
                                                  restitution=0.2)
     elif E > 1e3: material = gs.materials.FEM(E=E, nu=nu, rho=rho)
     else:         material = gs.materials.MPM(E=E, nu=nu, rho=rho)

2. Compute particle bounding radius and grid spacing:
     extents  = physics_mesh.bounds[1] - physics_mesh.bounds[0]
     p_radius = float(np.linalg.norm(extents)) / 2.0
     spacing  = p_radius * 2.4

3. Compute spawn parameters:
     spawn_r  = env_info["spread_radius"] * drop_spread
     spawn_y0 = env_info["top_y"] + drop_height
     cols     = int(math.ceil(math.sqrt(n)))

4. Loop i from 0 to n-1:
   - Compute grid row and column with divmod(i, cols)
   - Compute x and z from grid position * spacing, offset by -cols/2+0.5
   - If sqrt(x²+z²) > spawn_r, scale x and z down:
       s = spawn_r / dist * np.random.uniform(0.7, 1.0)
       x *= s;  z *= s
   - Add jitter: x and z each get += np.random.uniform(-p_radius*0.2, 
     p_radius*0.2)
   - Stagger Y: y = spawn_y0 + (i % cols) * p_radius * 0.3
   - Generate a uniformly random orientation quaternion using:
       trimesh.transformations.random_rotation_matrix()[:3, :3]
     then convert with gs.utils.mat_to_quat()
   - Add the entity to the scene using gs.morphs.Mesh(mesh=physics_mesh)
     with the computed pos and quat
   - Append to entities list

5. Print after the loop:
     Spawned <n> particles
       Drop height above container top: <drop_height> m
       Spawn Y range: <spawn_y0> – <spawn_y0 + p_radius> m

6. Return entities list.

Then replace the if __name__ == "__main__": block with a test that
runs the FULL pipeline up to and including scene.build() but does NOT
call scene.step(). This confirms the complete geometry loads without
errors. The test must:
- Call gs.init(backend=gs.cpu)
- Build a scene with global gravity/dt/substeps
- Call load_particle_mesh(PARTICLE_FILE, SCALE_FACTOR) — if 
  PARTICLE_FILE doesn't exist, create a box mesh and export it 
  to that path first
- Call create_environment() with global parameters
- Call spawn_particles() with global parameters
- Call scene.build()
- Print the number of entities in the scene
- Print "spawn_particles: OK — scene built without stepping"

All prior code must remain completely unchanged.
```

---

### Prompt 5 — Simulation loop + settlement detection

```
Add the run_simulation() function to simulation.py. The function 
signature must be:

  def run_simulation(
      scene: gs.Scene,
      entities: list,
      dt: float,
      substeps: int,
      duration: float,
      settle_threshold: float,
  ) -> int:

Note that entities is passed in (not retrieved from scene) so the 
velocity check only measures particle entities, not fixed container 
walls. This is important — checking fixed entities would prevent 
settlement from ever being detected.

Implement it with this exact behavior:

1. Compute total_steps = int(duration / dt).

2. Loop step from 0 to total_steps-1:
   - Call scene.step() on every iteration.
   - Every 60 steps, compute the velocity magnitude of every entity
     in the entities list using e.get_vel(). Build a list of 
     float(np.linalg.norm(vel)) for each. If the list is empty, 
     max_vel = 0.0, otherwise max_vel = max of the list.
   - Print on the same line (end="\r"):
       t=<step*dt:.2f>s  max_vel=<max_vel:.5f> m/s  
       step=<step>/<total_steps>
   - If max_vel < settle_threshold, print a newline then:
       Settled at t=<step*dt:.3f>s  (step <step>)
     and return step immediately.

3. After the loop completes without settling, print a newline then:
     Reached max duration (<duration>s) without settling
   and return total_steps.

Then replace the if __name__ == "__main__": block with a test that
runs the simulation end-to-end for a short duration to confirm 
stepping works. The test must:
- Create a small scene with 5 particles (override N_PARTICLES=5)
  using all other global parameters
- Set SIM_DURATION=1.0 for a fast test
- Run the full pipeline: init → load mesh → build environment → 
  spawn particles → build scene → run_simulation()
- Print the returned step count
- Print "run_simulation: OK"

Do not add any contact extraction yet. All prior code must remain
completely unchanged.
```

---

### Prompt 6 — Contact extraction + metrics

```
Add three things to simulation.py in this order:

── Part A: NormalizedContact dataclass ──────────────────────────────

Add this dataclass before the functions (after the imports and 
parameter block):

  @dataclass
  class NormalizedContact:
      entity_a:              int
      entity_b:              int
      is_particle_particle:  bool
      is_particle_container: bool
      position:              np.ndarray
      normal:                np.ndarray
      depth:                 float
      force:                 Optional[float]
      contact_area:          Optional[float]

── Part B: extract_contacts() ───────────────────────────────────────

Function signature:

  def extract_contacts(
      scene: gs.Scene,
      particle_ids: set,
      container_ids: set,
      depth_tol: float = 1e-5,
  ) -> list[NormalizedContact]:

Implement it with this exact behavior:

1. Call scene.get_contacts() to get raw contacts.
2. For each raw contact c:
   - Skip if abs(c.depth) < depth_tol.
   - Extract a = c.entity_a.id, b = c.entity_b.id.
   - Compute is_pp: both a and b are in particle_ids.
   - Compute is_pc: one of (a,b) is in particle_ids and the other 
     is in container_ids.
   - Skip if neither is_pp nor is_pc.
   - For force: use float(c.force) if hasattr(c, "force") else None.
   - For area: use float(c.area) if hasattr(c, "area") else None.
   - Append a NormalizedContact with all fields populated.
3. Return the list.

── Part C: compute_metrics() ────────────────────────────────────────

Function signature:

  def compute_metrics(
      contacts: list[NormalizedContact],
      particle_ids: set,
  ) -> dict:

Implement it with this exact behavior:

1. Separate pp (particle-particle) and pc (particle-container) lists.
2. Build a NetworkX Graph G:
   - Add all particle_ids as nodes.
   - For each contact in pp, add an edge between entity_a and entity_b
     with attributes depth, force, and area (use contact_area for area).
3. Compute contact_counts = dict(G.degree()).
4. Compute isolated = particles where degree == 0.
5. Compute container_touching = set of particle IDs that appear in 
   any pc contact.
6. Return a dict with ALL of these keys:
     total_pp_contacts
     total_pc_contacts
     avg_contacts_per_particle   (= 2 * len(pp) / n, where n = len(particle_ids))
     Z                           (same value as avg_contacts_per_particle)
     n_isolated_particles
     n_container_touching
     contact_points              (list of c.position for pp contacts)
     contact_normals             (list of c.normal for pp contacts)
     contact_depths              (list of c.depth for pp contacts)
     contact_forces              (list of c.force for pp contacts where force is not None)
     contact_areas               (list of c.contact_area for pp where contact_area is not None)
     contact_graph               (the NetworkX Graph object)
     contact_graph_dict          (dict of pid -> list(G.neighbors(pid)) for all pids)
     contact_counts_per_particle (the contact_counts dict)

── Part D: contact efficiency extensions ────────────────────────────

Add these two standalone functions immediately after compute_metrics():

  def contact_efficiency(metrics: dict, entities: list, 
                         mesh: trimesh.Trimesh) -> float:
      volume = mesh.volume * len(entities)
      return metrics["total_pp_contacts"] / volume

  def weighted_contact_efficiency(metrics: dict, entities: list,
                                  mesh: trimesh.Trimesh) -> float:
      total_force = sum(f for f in metrics["contact_forces"] if f)
      volume = mesh.volume * len(entities)
      return total_force / volume

── Part E: test block ───────────────────────────────────────────────

Replace if __name__ == "__main__": with a test that runs the full 
pipeline through contact extraction:
- Use N_PARTICLES=10 and SIM_DURATION=2.0 for speed
- Run init → load mesh → build environment → spawn → build → simulate
- Call extract_contacts() with the correct particle_ids and 
  container_ids
- Call compute_metrics()
- Print every key in the returned metrics dict and its value (or 
  length if it's a list)
- Print "extract_contacts + compute_metrics: OK"

All prior code must remain completely unchanged.
```

---

### Prompt 7 — Export, visualization, main(), and optional frontend

```
Complete simulation.py by adding the final three functions and wiring
everything together.

── Part A: export_results() ─────────────────────────────────────────

Function signature:

  def export_results(
      entities: list,
      metrics: dict,
      output_dir: str,
      save_hdf5: bool,
      save_csv: bool,
  ) -> None:

Implement it with this exact behavior:

1. Call os.makedirs(output_dir, exist_ok=True).

2. Build df_particles: one row per entity with columns:
     id, x, y, z, qx, qy, qz, qw, n_contacts
   Get pos from e.get_pos(), quat from e.get_quat(), n_contacts from
   metrics["contact_counts_per_particle"].get(e.id, 0).

3. Build df_contacts from metrics["contact_graph"] edges with columns:
     particle_a, particle_b, depth, force, contact_area
   Use G.edges(data=True) and pull depth, force, area from edge data.

4. Build df_points from metrics["contact_points"] and 
   metrics["contact_normals"] zipped together, with columns:
     x, y, z, nx, ny, nz

5. If save_csv is True:
   - Write df_particles to <output_dir>/particles.csv
   - Write df_contacts to <output_dir>/contact_pairs.csv
   - Write df_points to <output_dir>/contact_points.csv
   - Print "CSV files written to <output_dir>/"

6. If save_hdf5 is True:
   - Open <output_dir>/simulation.h5 for writing
   - Dataset "particle_positions": df_particles[["x","y","z"]].values
   - Dataset "contact_points": np.array of contact points, or 
     np.empty((0,3)) if the list is empty
   - Dataset "contact_normals": same pattern
   - Attributes on the file: Z, total_pp, total_pc, n_isolated, 
     n_container_touch — all from metrics
   - Print "HDF5 replay written to <output_dir>/simulation.h5"

7. Print this exact console summary:
     ── Contact Analysis Summary ────────────────────────────────
       Particles simulated:               <count>
       Total particle-particle contacts:  <total_pp_contacts>
       Total particle-container contacts: <total_pc_contacts>
       Avg contacts per particle (Z):     <Z:.3f>
       Isolated particles (Z=0):          <n_isolated_particles>
       Particles touching container:      <n_container_touching>
   Then if contact_forces is non-empty, add:
       Mean contact force:                <mean:.4f> N
       Max  contact force:                <max:.4f> N
   Then if contact_areas is non-empty, add:
       Mean contact area:                 <mean*1e6:.4f> mm²
   Close with:
     ────────────────────────────────────────────────────────────

── Part B: visualize_results() ──────────────────────────────────────

Function signature:

  def visualize_results(
      entities: list,
      metrics: dict,
      original_mesh: trimesh.Trimesh,
  ) -> None:

Implement it with this exact behavior:

1. Build a list of Open3D TriangleMesh objects, one per entity:
   - Get pos from e.get_pos(), quat from e.get_quat()
   - Get contact count c from metrics["contact_counts_per_particle"]
   - Compute t = c / max_c where max_c = max of all contact counts 
     (or 1 if all are zero)
   - Color each mesh [t, 0.2, 1.0 - t] (blue=0 contacts, red=max)
   - Apply rotation using 
     o3d.geometry.get_rotation_matrix_from_quaternion(quat)
   - Translate to pos
   - Call compute_vertex_normals()

2. For each point in metrics["contact_points"]:
   - Create a sphere with radius=0.003
   - Translate to the point
   - Color it [1.0, 1.0, 0.0] (yellow)

3. Add a coordinate frame with size=0.05.

4. Call o3d.visualization.draw_geometries() with all geometries,
   window_name="Particle Contact Analysis", width=1280, height=800.

── Part C: main() ───────────────────────────────────────────────────

Write the complete main() function that calls everything in order:

  def main():
      gs.init(backend=gs.cpu)
      scene = gs.Scene(gravity=GRAVITY, dt=DT, substeps=SUBSTEPS)

      physics_mesh, original_mesh = load_particle_mesh(
          PARTICLE_FILE, SCALE_FACTOR)

      container_ids, env_info = create_environment(
          scene, kind=ENVIRONMENT_TYPE,
          plate_size=PLATE_SIZE, cyl_diameter=CYLINDER_DIAMETER,
          cyl_height=CYLINDER_HEIGHT, cyl_segments=CYLINDER_SEGMENTS,
          wall_thickness=WALL_THICKNESS)

      entities = spawn_particles(
          scene, physics_mesh, n=N_PARTICLES, env_info=env_info,
          drop_height=DROP_HEIGHT, drop_spread=DROP_SPREAD,
          E=YOUNGS_MODULUS, nu=POISSON_RATIO, rho=DENSITY)

      particle_ids = {e.id for e in entities}

      run_simulation(scene, entities, DT, SUBSTEPS, 
                     SIM_DURATION, SETTLE_THRESHOLD)

      contacts = extract_contacts(scene, particle_ids, container_ids,
                                   depth_tol=CONTACT_DEPTH_TOL)
      metrics  = compute_metrics(contacts, particle_ids)

      export_results(entities, metrics, OUTPUT_DIR, SAVE_HDF5, SAVE_CSV)
      visualize_results(entities, metrics, original_mesh)

Replace the if __name__ == "__main__": block with simply:

  if __name__ == "__main__":
      main()

── Part D: optional frontend files ──────────────────────────────────

Create these four files. They are self-contained and do not affect 
simulation.py.

frontend/package.json:
  Standard React + TypeScript package.json with dependencies:
  react, react-dom, three, @react-three/fiber, @react-three/drei, 
  d3, recharts and devDependencies: typescript, @types/react, 
  @types/three. Set "scripts": { "dev": "vite", "build": "vite build" }
  and include vite and @vitejs/plugin-react in devDependencies.

frontend/src/types.ts:
  Export TypeScript interfaces for:
    ParticleData  { id, x, y, z, qx, qy, qz, qw, n_contacts }
    ContactPair   { particle_a, particle_b, depth, force, contact_area }
    ContactPoint  { x, y, z, nx, ny, nz }
    SimMetrics    { Z, total_pp, total_pc, n_isolated, n_container_touch }

frontend/src/MaterialPresets.tsx:
  Export a React component MaterialPresetSlider that renders a 
  labeled logarithmic range input (min=3, max=8, step=0.01) and 
  displays the currently selected preset name and E value. The 
  presets array must contain exactly:
    { label: "Steel",        E: 200_000_000 }
    { label: "Hard plastic", E:  50_000_000 }
    { label: "Hard rubber",  E:   1_000_000 }
    { label: "Silicone",     E:      10_000 }
    { label: "Gel",          E:       1_000 }
  The slider value maps to E via E = 10 ** sliderValue. On change, 
  call an onChangeE(value: number) prop with the computed E.

server.py (project root, not inside frontend/):
  A FastAPI app with:
  - GET /results — reads particles.csv, contact_pairs.csv, 
    contact_points.csv from OUTPUT_DIR and returns them as JSON
  - GET /metrics — reads simulation.h5 and returns its attributes 
    (Z, total_pp, total_pc, n_isolated, n_container_touch) as JSON
  - A WebSocket endpoint at /ws that accepts connections and, when 
    it receives the message "start", calls subprocess.Popen to launch 
    simulation.py and streams its stdout line by line back to the 
    client until the process exits
  - An if __name__ == "__main__": block that calls 
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
```