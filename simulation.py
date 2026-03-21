import os, math
import argparse
import sys
import numpy as np
import pandas as pd
import h5py
import trimesh
import coacd
import networkx as nx
from dataclasses import dataclass
from typing import Callable, Optional

# Avoid Windows console UnicodeEncodeError when libraries print box-drawing/emoji.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Optional: hide CUDA before importing Torch when forcing CPU (`--backend cpu` or GENESIS_USE_CPU=1).
_use_cpu_cli = False
if "--backend" in sys.argv:
    try:
        i = sys.argv.index("--backend")
        if i + 1 < len(sys.argv) and str(sys.argv[i + 1]).strip().lower() == "cpu":
            _use_cpu_cli = True
    except ValueError:
        pass
if _use_cpu_cli or os.environ.get("GENESIS_USE_CPU", "").strip().lower() in ("1", "true", "yes"):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import genesis as gs

# ── STEP 2: Parameter block ────────────────────────────────────────────────────  # parameters section
# ── Input ──────────────────────────────────────────────────────────────────  # input settings
PARTICLE_FILE        = "particle.obj"   # OBJ or STL path
N_PARTICLES          = 100              # number of particle copies to drop
SCALE_FACTOR         = 1.0             # 0.001 converts mm mesh → metres
# ── Material ───────────────────────────────────────────────────────────────  # material settings
# YOUNGS_MODULUS: high-stiffness FEM jamming (E ≤ 1e8 Pa → FEM in Genesis).
# E > 1e8 Pa → Rigid (avoid unless needed). Server/UI clamp to FEM_JAMMING_E_MAX for jamming runs.
FEM_JAMMING_E_MAX  = 100_000_000.0   # Pa — upper FEM bound (1e8)
YOUNGS_MODULUS       = 100_000_000.0  # Pa — high-stiffness FEM (≈1e8)
POISSON_RATIO        = 0.45            # 0.5 = fully incompressible; use 0.45+ to limit volume loss
DENSITY              = 1200            # kg/m³
# Rigid-body restitution (Genesis maps this to internal coupling restitution).
# Non-zero values can trigger a Genesis WARNING and may reduce stability; 0 = inelastic.
PARTICLE_RESTITUTION = 0.0             # particle–contact bounciness (was 0.2)
ENV_RESTITUTION      = 0.0             # floor/walls (was 0.05)
# ── Environment ────────────────────────────────────────────────────────────  # environment settings
ENVIRONMENT_TYPE     = "plate"      # "cylinder" or "plate"
PLATE_SIZE           = 0.6             # square plate side length (m)
CYLINDER_DIAMETER    = 0.20            # inner diameter (m)
CYLINDER_HEIGHT      = 0.30            # wall height (m)
CYLINDER_SEGMENTS    = 32              # wall facets — use 24+ to avoid gaps
WALL_THICKNESS       = 0.008           # m
# ── Drop ───────────────────────────────────────────────────────────────────  # drop settings
DROP_HEIGHT          = 0.15            # metres above container top edge
DROP_SPREAD          = 0.85            # fraction of inner radius used for XZ spread
# ── Gravity ────────────────────────────────────────────────────────────────  # gravity settings
GRAVITY              = (0, -9.81, 0)   # Y is up; change to (0,-1.62,0) for Moon
# ── Simulation ─────────────────────────────────────────────────────────────  # simulation settings
DT                   = 1 / 400        # timestep (s) — smaller dt for stiff FEM contacts
SUBSTEPS             = 10               # substeps per timestep — stability for jamming
SIM_DURATION         = 5.0             # max simulated time (s)
SETTLE_THRESHOLD     = 1e-4            # m/s — stop early when all particles slow
# ── Runtime / performance ───────────────────────────────────────────────────  # runtime settings
# Genesis’s viewer can dominate runtime on CPU (the FPS log you saw is from it).
# Keep it off by default so simulation runs as fast as possible.
SHOW_VIEWER          = False
BACKEND              = "auto"           # "auto" → GPU if available; "cpu" or "gpu"
# ── Contact analysis ───────────────────────────────────────────────────────  # contact analysis settings
CONTACT_SAMPLE_EVERY = 10              # extract contacts every N steps
CONTACT_DEPTH_TOL    = 1e-5            # min penetration depth to count as contact
# ── Output ─────────────────────────────────────────────────────────────────  # output settings
OUTPUT_DIR           = "./results"     # output directory path
SAVE_HDF5            = True            # enable HDF5 output
SAVE_CSV             = True            # enable CSV output

DEFAULT_CONFIG = {
    "PARTICLE_FILE": PARTICLE_FILE,
    "N_PARTICLES": N_PARTICLES,
    "SCALE_FACTOR": SCALE_FACTOR,
    "FEM_JAMMING_E_MAX": FEM_JAMMING_E_MAX,
    "YOUNGS_MODULUS": YOUNGS_MODULUS,
    "POISSON_RATIO": POISSON_RATIO,
    "DENSITY": DENSITY,
    "PARTICLE_RESTITUTION": PARTICLE_RESTITUTION,
    "ENV_RESTITUTION": ENV_RESTITUTION,
    "ENVIRONMENT_TYPE": ENVIRONMENT_TYPE,
    "PLATE_SIZE": PLATE_SIZE,
    "CYLINDER_DIAMETER": CYLINDER_DIAMETER,
    "CYLINDER_HEIGHT": CYLINDER_HEIGHT,
    "CYLINDER_SEGMENTS": CYLINDER_SEGMENTS,
    "WALL_THICKNESS": WALL_THICKNESS,
    "DROP_HEIGHT": DROP_HEIGHT,
    "DROP_SPREAD": DROP_SPREAD,
    "GRAVITY": GRAVITY,
    "DT": DT,
    "SUBSTEPS": SUBSTEPS,
    "SIM_DURATION": SIM_DURATION,
    "SETTLE_THRESHOLD": SETTLE_THRESHOLD,
    "CONTACT_DEPTH_TOL": CONTACT_DEPTH_TOL,
    "OUTPUT_DIR": OUTPUT_DIR,
    "SAVE_HDF5": SAVE_HDF5,
    "SAVE_CSV": SAVE_CSV,
}


def build_runtime_config(payload: Optional[dict]) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    raw = payload or {}
    for key in cfg.keys():
        if key in raw:
            cfg[key] = raw[key]

    cfg["N_PARTICLES"] = int(cfg["N_PARTICLES"])
    cfg["FEM_JAMMING_E_MAX"] = float(cfg["FEM_JAMMING_E_MAX"])
    cfg["YOUNGS_MODULUS"] = float(cfg["YOUNGS_MODULUS"])
    # Clamp Young's modulus to FEM jamming band (avoids accidental Rigid regime when E > 1e8).
    if cfg["YOUNGS_MODULUS"] > cfg["FEM_JAMMING_E_MAX"]:
        cfg["YOUNGS_MODULUS"] = cfg["FEM_JAMMING_E_MAX"]
    cfg["POISSON_RATIO"] = float(cfg["POISSON_RATIO"])
    cfg["PARTICLE_RESTITUTION"] = float(cfg["PARTICLE_RESTITUTION"])
    cfg["ENV_RESTITUTION"] = float(cfg["ENV_RESTITUTION"])
    cfg["CYLINDER_DIAMETER"] = float(cfg["CYLINDER_DIAMETER"])
    cfg["DROP_HEIGHT"] = float(cfg["DROP_HEIGHT"])
    cfg["DT"] = float(cfg["DT"])
    cfg["SUBSTEPS"] = int(cfg["SUBSTEPS"])
    cfg["SIM_DURATION"] = float(cfg["SIM_DURATION"])
    cfg["SETTLE_THRESHOLD"] = float(cfg["SETTLE_THRESHOLD"])
    cfg["ENVIRONMENT_TYPE"] = str(cfg["ENVIRONMENT_TYPE"]).strip().lower()
    return cfg


@dataclass
class NormalizedContact:
    entity_a: int
    entity_b: int
    is_particle_particle: bool
    is_particle_container: bool
    position: np.ndarray
    normal: np.ndarray
    depth: float
    force: Optional[float]
    contact_area: Optional[float]


def _entity_id(e) -> int:
    # Genesis versions use either `.id` or `.idx`.
    if hasattr(e, "id"):
        return int(getattr(e, "id"))
    return int(getattr(e, "idx"))


def _contact_pos(c):
    # Genesis versions use either `.pos` or `.position`.
    if hasattr(c, "pos"):
        return c.pos
    return c.position


def _entity_pose(e) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """
    Return (pos_xyz, quat_xyzw) for both RigidEntity-like and FEM/MPM-like entities.

    - RigidEntity: use get_pos/get_quat
    - FEMEntity (Genesis 0.4.3): no get_pos/get_quat; approximate with centroid of vertex positions
    """
    # Genesis FEM/MPM entities can expose get_pos() but it may not correspond
    # to the world-space rigid transform you expect for visualization.
    # For these types we prefer extracting centroid from get_state().
    entity_type = type(e).__name__
    force_state_pose = ("FEMEntity" in entity_type) or ("MPMEntity" in entity_type)

    # Prefer explicit pose getters when available (rigid-like entities).
    if (not force_state_pose) and hasattr(e, "get_pos"):
        x, y, z = e.get_pos()
        # If pose getters return something clearly not in world space (e.g.
        # very large magnitudes), fall back to state-based heuristics below.
        if np.isfinite([x, y, z]).all() and float(max(abs(x), abs(y), abs(z))) <= 10.0:
            if hasattr(e, "get_quat"):
                qw, qx, qy, qz = e.get_quat()  # (w,x,y,z) -> (x,y,z,w)
                quat = (float(qx), float(qy), float(qz), float(qw))
            else:
                quat = (0.0, 0.0, 0.0, 1.0)
            return (np.asarray([x, y, z], dtype=float), quat)
        # else: fall through to fallback extraction
        if hasattr(e, "get_quat"):
            qw, qx, qy, qz = e.get_quat()  # (w,x,y,z) -> (x,y,z,w)
            quat = (float(qx), float(qy), float(qz), float(qw))
        else:
            quat = (0.0, 0.0, 0.0, 1.0)
        # Keep the existing return for completeness; most callers will hit
        # the magnitude guard above.
        return (np.asarray([x, y, z], dtype=float), quat)

    # Fallback: try entity state (post scene.build()) and compute centroid.
    try:
        if hasattr(e, "get_state"):
            st = e.get_state()
            candidate_positions: list[np.ndarray] = []
            for attr in ("pos", "x", "p"):
                if not hasattr(st, attr):
                    continue
                arr = np.asarray(getattr(st, attr), dtype=float)
                # Common shapes: (N,3) or (B,N,3).
                if arr.ndim == 3:
                    arr = arr[0]
                if arr.ndim == 2 and arr.shape[-1] == 3 and arr.shape[0] > 0:
                    pos = arr.mean(axis=0)
                    if np.isfinite(pos).all():
                        candidate_positions.append(pos)

            # Heuristic: in a correctly-scaled scene, positions should stay
            # within a small neighborhood of the container (order ~0.1-1m).
            # If the state tensor we picked is actually velocity/momentum-like
            # data, its magnitude can be wildly larger and break visualization.
            if candidate_positions:
                best_pos = min(candidate_positions, key=lambda p: float(np.abs(p).sum()))
                return (best_pos, (0.0, 0.0, 0.0, 1.0))
    except Exception:
        pass

    return (np.zeros((3,), dtype=float), (0.0, 0.0, 0.0, 1.0))


def load_particle_mesh(filepath: str, scale: float = 1.0):
    ext = os.path.splitext(filepath)[1].lower().lstrip(".")
    if ext not in {"obj", "stl"}:
        raise ValueError(f"Unsupported mesh extension '.{ext}'. Expected 'obj' or 'stl'.")

    original_mesh = trimesh.load(filepath, force="mesh")
    if original_mesh is None or (hasattr(original_mesh, "is_empty") and original_mesh.is_empty):
        raise ValueError(f"Mesh is empty: {filepath!r}")
    if isinstance(original_mesh, trimesh.Scene):
        geoms = [g for g in original_mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        original_mesh = trimesh.util.concatenate(geoms) if len(geoms) > 1 else (geoms[0] if geoms else None)
        if original_mesh is None:
            raise ValueError(f"Mesh is empty: {filepath!r}")
    if not isinstance(original_mesh, trimesh.Trimesh) or len(original_mesh.vertices) == 0 or len(original_mesh.faces) == 0:
        raise ValueError(f"Mesh is empty: {filepath!r}")

    original_mesh = original_mesh.copy()
    original_mesh.apply_scale(scale)
    original_mesh.apply_translation(-original_mesh.centroid)

    parts = []
    try:
        parts = coacd.run_coacd(original_mesh, max_convex_hull=32)
    except Exception:
        parts = []

    if not parts:
        parts = [original_mesh.convex_hull]

    physics_mesh = trimesh.util.concatenate(parts)

    volume = float(physics_mesh.volume) if physics_mesh.is_watertight else float(physics_mesh.convex_hull.volume)
    print(
        f"{filepath} | verts={len(physics_mesh.vertices)} | faces={len(physics_mesh.faces)} | "
        f"extents={physics_mesh.extents} | volume={volume:.6g} | parts={len(parts)}"
    )
    return (physics_mesh, original_mesh)


def _rigid_material(friction: float, restitution: float):
    """Genesis versions disagree on `restitution` vs `coup_restitution`; support both."""
    try:
        return gs.materials.Rigid(friction=friction, restitution=restitution)
    except TypeError:
        return gs.materials.Rigid(friction=friction, coup_restitution=restitution)


def create_environment(scene, kind, plate_size=0.6, cyl_diameter=0.20,
cyl_height=0.30, cyl_segments=32,
wall_thickness=0.008, env_restitution: float = ENV_RESTITUTION) -> tuple[set, dict]:
    container_ids = set()
    mat = _rigid_material(0.55, float(env_restitution))

    if kind == "plate":
        plate = scene.add_entity(
            gs.morphs.Box(
                size=(plate_size, wall_thickness, plate_size),
                pos=(0, wall_thickness / 2, 0),
                fixed=True,
            ),
            material=mat,
        )
        container_ids.add(plate)
        env_info = {
            "surface_y": wall_thickness,
            "top_y": wall_thickness,
            "spread_radius": plate_size / 2,
        }
        return (container_ids, env_info)

    if kind == "cylinder":
        r_inner = cyl_diameter / 2
        bottom = scene.add_entity(
            gs.morphs.Cylinder(
                radius=r_inner + wall_thickness,
                height=wall_thickness,
                pos=(0, wall_thickness / 2, 0),
                fixed=True,
            ),
            material=mat,
        )
        container_ids.add(bottom)

        angle_step = 2 * math.pi / cyl_segments
        chord_width = 2 * (cyl_diameter / 2) * math.sin(angle_step / 2)
        for i in range(cyl_segments):
            angle = 2 * math.pi * i / cyl_segments
            cx = (cyl_diameter / 2 + wall_thickness / 2) * math.cos(angle)
            cz = (cyl_diameter / 2 + wall_thickness / 2) * math.sin(angle)
            cy = wall_thickness + cyl_height / 2
            # Genesis quaternion helper expects Euler angles in degrees.
            angle_deg = angle * 180.0 / math.pi
            quat = gs.utils.geom.euler_to_quat((0.0, angle_deg, 0.0))
            panel = scene.add_entity(
                gs.morphs.Box(
                    size=(chord_width, cyl_height, wall_thickness),
                    pos=(cx, cy, cz),
                    quat=quat,
                    fixed=True,
                ),
                material=mat,
            )
            container_ids.add(panel)

        env_info = {
            "surface_y": wall_thickness,
            "top_y": wall_thickness + cyl_height,
            "inner_radius": r_inner,
            "spread_radius": r_inner,
        }
        return (container_ids, env_info)

    raise ValueError(f"Unknown environment kind: {kind!r}")


def spawn_particles(scene, physics_mesh, n, env_info, drop_height,
drop_spread, E, nu, rho, particle_file=PARTICLE_FILE, scale_factor=SCALE_FACTOR,
particle_restitution: float = PARTICLE_RESTITUTION) -> list:
    if E > 1e8:
        material = _rigid_material(0.4, float(particle_restitution))
    elif E > 1e3:
        try:
            material = gs.materials.FEM(E=E, nu=nu, rho=rho)
        except TypeError:
            material = gs.materials.FEM.Elastic(E=E, nu=nu, rho=rho)
    else:
        try:
            material = gs.materials.MPM(E=E, nu=nu, rho=rho)
        except TypeError:
            material = gs.materials.MPM.Elastic(E=E, nu=nu, rho=rho)

    extents = physics_mesh.bounds[1] - physics_mesh.bounds[0]
    p_radius = float(np.linalg.norm(extents) / 2)
    spacing = p_radius * 2.4
    spawn_r = float(env_info["spread_radius"] * drop_spread)
    spawn_y0 = float(env_info["top_y"] + drop_height)
    cols = int(math.ceil(math.sqrt(n)))

    entities = []
    ys = []
    for i in range(n):
        row, col = divmod(i, cols)
        x = (col - cols / 2 + 0.5) * spacing
        z = (row - cols / 2 + 0.5) * spacing
        dist = math.sqrt(x * x + z * z)
        if dist > spawn_r and dist > 1e-12:
            scale = (spawn_r / dist) * float(np.random.uniform(0.7, 1.0))
            x *= scale
            z *= scale
        x += float(np.random.uniform(-p_radius * 0.2, p_radius * 0.2))
        z += float(np.random.uniform(-p_radius * 0.2, p_radius * 0.2))
        y = spawn_y0 + (i % cols) * p_radius * 0.3
        ys.append(y)
        R = trimesh.transformations.random_rotation_matrix()[:3, :3]
        quat = gs.utils.geom.R_to_quat(R)
        ent = scene.add_entity(
            gs.morphs.Mesh(
                file=particle_file,
                scale=float(scale_factor),
                pos=(x, y, z),
                quat=quat,
            ),
            material=material,
        )
        entities.append(ent)

    y_min = float(min(ys)) if ys else spawn_y0
    y_max = float(max(ys)) if ys else spawn_y0
    print(f"Spawned {len(entities)} particles | drop_height={drop_height} | y_range=({y_min:.6g}, {y_max:.6g})")
    return entities


def run_simulation(scene, entities, dt, substeps, duration, settle_threshold) -> int:
    total_steps = int(duration / dt)
    for step in range(total_steps):
        scene.step()
        if step % 60 == 0:
            max_vel = None
            if entities:
                speeds: list[float] = []
                for e in entities:
                    if hasattr(e, "get_vel"):
                        v = np.asarray(e.get_vel(), dtype=float)
                        speeds.append(float(np.linalg.norm(v)))
                        continue

                    if hasattr(e, "get_state"):
                        st = e.get_state()
                        if hasattr(st, "vel"):
                            v = np.asarray(st.vel, dtype=float)
                            if v.size == 0:
                                continue
                            if v.ndim == 1:
                                speeds.append(float(np.linalg.norm(v)))
                            else:
                                speeds.append(float(np.linalg.norm(v, axis=-1).max()))
                        continue
                max_vel = max(speeds) if speeds else None
            print(
                f"t={step * dt:.2f}s  max_vel={(max_vel if max_vel is not None else 0.0):.5f} m/s  step={step}/{total_steps}",
                end="\r",
            )
            # Only early-exit if we successfully measured velocities.
            if max_vel is not None and max_vel < settle_threshold:
                print()
                print(f"Settled at t={step * dt:.3f}s  (step {step})")
                return step
    print()
    print(f"Reached max duration ({duration}s)")
    return total_steps


def compute_max_velocity(entities) -> Optional[float]:
    speeds: list[float] = []
    for e in entities:
        if hasattr(e, "get_vel"):
            v = np.asarray(e.get_vel(), dtype=float)
            speeds.append(float(np.linalg.norm(v)))
            continue
        if hasattr(e, "get_state"):
            st = e.get_state()
            if hasattr(st, "vel"):
                v = np.asarray(st.vel, dtype=float)
                if v.size == 0:
                    continue
                if v.ndim == 1:
                    speeds.append(float(np.linalg.norm(v)))
                else:
                    speeds.append(float(np.linalg.norm(v, axis=-1).max()))
    if not speeds:
        return None
    return max(speeds)


def _tensor_to_numpy(x) -> np.ndarray:
    if x is None:
        return np.zeros((0,), dtype=float)
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


def compute_fem_vertex_force_stress(scene, entities) -> tuple[dict[int, np.ndarray], float]:
    """
    Per-vertex |F| from FEM nodal forces (proxy for contact / internal loading).
    Returns (entity_id -> array shape (n_vertices,)), global max for normalization.
    """
    fs = getattr(scene.sim, "fem_solver", None)
    if fs is None or not getattr(fs, "is_active", False):
        return {}, 1.0
    forces = fs.get_forces()
    if forces is None:
        return {}, 1.0
    forces = _tensor_to_numpy(forces)
    if forces.ndim != 3 or forces.shape[0] < 1:
        return {}, 1.0
    forces_b = forces[0]
    per_entity: dict[int, np.ndarray] = {}
    all_norms: list[float] = []
    for e in entities:
        if type(e).__name__ != "FEMEntity":
            continue
        if not hasattr(e, "v_start") or not hasattr(e, "n_vertices"):
            continue
        vs = int(e.v_start)
        nv = int(e.n_vertices)
        if vs + nv > forces_b.shape[0]:
            continue
        sub = forces_b[vs : vs + nv, :]
        norms = np.linalg.norm(sub, axis=1)
        per_entity[int(_entity_id(e))] = norms.astype(float)
        all_norms.extend(float(x) for x in norms.tolist())
    gmax = max(all_norms) if all_norms else 1.0
    if gmax <= 1e-18:
        gmax = 1.0
    return per_entity, float(gmax)


def compute_particle_stress_map(contacts, particle_ids: set[int]) -> dict[int, float]:
    # Aggregate per-particle contact "intensity" from force and depth.
    raw_scores: dict[int, float] = {int(pid): 0.0 for pid in particle_ids}
    for c in contacts:
        score = abs(float(c.force)) if c.force is not None else abs(float(c.depth))
        if c.entity_a in raw_scores:
            raw_scores[int(c.entity_a)] += score
        if c.entity_b in raw_scores:
            raw_scores[int(c.entity_b)] += score

    max_score = max(raw_scores.values()) if raw_scores else 0.0
    if max_score <= 0.0:
        return {pid: 0.0 for pid in raw_scores}
    return {pid: float(min(1.0, val / max_score)) for pid, val in raw_scores.items()}


def _collect_particle_transforms(
    entities,
    stress_map: Optional[dict[int, float]] = None,
    *,
    fem_vertex_norms: Optional[dict[int, np.ndarray]] = None,
    fem_norm_global_max: float = 1.0,
    max_vertex_stress_floats: int = 8,
) -> list[dict]:
    out: list[dict] = []
    smap = stress_map or {}
    gmax = float(fem_norm_global_max) if fem_norm_global_max > 1e-18 else 1.0
    for e in entities:
        pos, (qx, qy, qz, qw) = _entity_pose(e)
        if not (np.isfinite(pos).all() and all(np.isfinite([qx, qy, qz, qw]))):
            pos = np.zeros((3,), dtype=float)
            qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
        eid = int(_entity_id(e))
        contact_s = float(smap.get(eid, 0.0))
        vertex_stress: Optional[list[float]] = None
        stress_intensity = contact_s
        if fem_vertex_norms and eid in fem_vertex_norms:
            arr = np.clip(np.asarray(fem_vertex_norms[eid], dtype=float) / gmax, 0.0, 1.0)
            stress_intensity = float(np.mean(arr)) if arr.size else contact_s
            n = min(int(arr.shape[0]), int(max_vertex_stress_floats))
            padded = np.zeros((max_vertex_stress_floats,), dtype=float)
            padded[:n] = arr[:n]
            vertex_stress = [float(x) for x in padded.tolist()]
        row = {
            "id": eid,
            "x": float(pos[0]),
            "y": float(pos[1]),
            "z": float(pos[2]),
            "qx": float(qx),
            "qy": float(qy),
            "qz": float(qz),
            "qw": float(qw),
            "stress_intensity": stress_intensity,
        }
        if vertex_stress is not None:
            row["vertex_stress"] = vertex_stress
        out.append(row)
    return out


def run_simulation_stream(
    scene,
    entities,
    dt: float,
    duration: float,
    settle_threshold: float,
    *,
    frame_every: int = 10,
    log_every: int = 60,
    on_log: Optional[Callable[[str], None]] = None,
    on_frame: Optional[Callable[[dict], None]] = None,
) -> int:
    """
    Headless simulation loop that can stream:
    - logs: human-readable status lines (via on_log)
    - frames: full particle transforms every N steps (via on_frame)
    """
    if frame_every <= 0:
        frame_every = 1
    if log_every <= 0:
        log_every = 1

    total_steps = int(duration / dt)
    for step in range(total_steps):
        # Emit a frame at the *start* of the step (t = step*dt), so the
        # first frame represents the spawned configuration.
        if on_frame and (step % frame_every == 0):
            on_frame(
                {
                    "type": "frame",
                    "step": int(step),
                    "t": float(step * dt),
                    "particles": _collect_particle_transforms(entities),
                }
            )

        scene.step()
        current_t = float((step + 1) * dt)

        if (step % log_every == 0) or (step == total_steps - 1):
            max_vel: Optional[float] = None
            if entities:
                speeds: list[float] = []
                for e in entities:
                    if hasattr(e, "get_vel"):
                        v = np.asarray(e.get_vel(), dtype=float)
                        speeds.append(float(np.linalg.norm(v)))
                        continue

                    if hasattr(e, "get_state"):
                        st = e.get_state()
                        if hasattr(st, "vel"):
                            v = np.asarray(st.vel, dtype=float)
                            if v.size == 0:
                                continue
                            if v.ndim == 1:
                                speeds.append(float(np.linalg.norm(v)))
                            else:
                                speeds.append(float(np.linalg.norm(v, axis=-1).max()))
                        continue
                max_vel = max(speeds) if speeds else None

            line = f"t={step * dt:.2f}s  max_vel={(max_vel if max_vel is not None else 0.0):.5f} m/s  step={step}/{total_steps}"
            if on_log:
                on_log(line)

            # Only early-exit if we successfully measured velocities.
            if max_vel is not None and max_vel < settle_threshold:
                if on_log:
                    on_log(f"Settled at t={current_t:.3f}s  (step {step + 1})")
                return step + 1

    if on_log:
        on_log(f"Reached max duration ({duration}s)")
    return total_steps


def extract_contacts(scene, particle_ids, container_ids, depth_tol=1e-5) -> list:
    # Genesis API differs by entity regime/backends; in some configurations
    # (e.g., rigid-only) `get_contacts()` may not exist.
    if not hasattr(scene, "get_contacts"):
        return []
    raw = scene.get_contacts()
    out: list[NormalizedContact] = []
    for c in raw:
        if abs(float(c.depth)) < depth_tol:
            continue

        a = _entity_id(c.entity_a)
        b = _entity_id(c.entity_b)
        is_pp = (a in particle_ids) and (b in particle_ids)
        is_pc = ((a in particle_ids) and (b in container_ids)) or ((b in particle_ids) and (a in container_ids))
        if not (is_pp or is_pc):
            continue

        force = float(c.force) if hasattr(c, "force") else None
        area = float(c.area) if hasattr(c, "area") else None

        out.append(
            NormalizedContact(
                entity_a=a,
                entity_b=b,
                is_particle_particle=is_pp,
                is_particle_container=is_pc,
                position=np.asarray(_contact_pos(c), dtype=float),
                normal=np.asarray(c.normal, dtype=float),
                depth=float(c.depth),
                force=force,
                contact_area=area,
            )
        )
    return out


def compute_metrics(contacts, particle_ids) -> dict:
    pp = [c for c in contacts if c.is_particle_particle]
    pc = [c for c in contacts if c.is_particle_container]

    G = nx.Graph()
    for pid in particle_ids:
        G.add_node(int(pid))
    for c in pp:
        G.add_edge(
            int(c.entity_a),
            int(c.entity_b),
            depth=float(c.depth),
            force=c.force,
            area=c.contact_area,
        )

    n = len(particle_ids) if particle_ids else 0
    Z = (2 * len(pp) / n) if n else 0.0
    contact_counts = dict(G.degree())
    n_isolated = sum(1 for pid in particle_ids if contact_counts.get(int(pid), 0) == 0)
    n_container_touching = len({int(c.entity_a) for c in pc if int(c.entity_a) in particle_ids}.union(
        {int(c.entity_b) for c in pc if int(c.entity_b) in particle_ids}
    ))

    contact_points = [np.asarray(c.position, dtype=float) for c in contacts]
    contact_normals = [np.asarray(c.normal, dtype=float) for c in contacts]
    contact_depths = [float(c.depth) for c in contacts]
    contact_forces = [float(c.force) for c in contacts if c.force is not None]
    contact_areas = [float(c.contact_area) for c in contacts if c.contact_area is not None]

    return {
        "total_pp_contacts": len(pp),
        "total_pc_contacts": len(pc),
        "avg_contacts_per_particle": Z,
        "Z": Z,
        "n_isolated_particles": n_isolated,
        "n_container_touching": n_container_touching,
        "contact_points": contact_points,
        "contact_normals": contact_normals,
        "contact_depths": contact_depths,
        "contact_forces": contact_forces,
        "contact_areas": contact_areas,
        "contact_graph": G,
        "contact_graph_dict": nx.to_dict_of_dicts(G),
        "contact_counts_per_particle": contact_counts,
    }


def contact_efficiency(metrics, entities, mesh) -> float:
    return metrics["total_pp_contacts"] / (mesh.volume * len(entities))


def weighted_contact_efficiency(metrics, entities, mesh) -> float:
    total_force = sum(f for f in metrics["contact_forces"] if f)
    return total_force / (mesh.volume * len(entities))


def export_results(entities, metrics, output_dir, save_hdf5, save_csv):
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    counts = metrics["contact_counts_per_particle"]
    for e in entities:
        pos, (qx, qy, qz, qw) = _entity_pose(e)
        eid = _entity_id(e)
        rows.append(
            {
                "id": int(eid),
                "x": float(pos[0]),
                "y": float(pos[1]),
                "z": float(pos[2]),
                "qx": float(qx),
                "qy": float(qy),
                "qz": float(qz),
                "qw": float(qw),
                "n_contacts": int(counts.get(int(eid), 0)),
            }
        )
    df_particles = pd.DataFrame(rows)

    G = metrics["contact_graph"]
    contact_rows = []
    for a, b, data in G.edges(data=True):
        contact_rows.append(
            {
                "particle_a": int(a),
                "particle_b": int(b),
                "depth": float(data.get("depth", 0.0)),
                "force": (float(data["force"]) if data.get("force", None) is not None else np.nan),
                "contact_area": (float(data["area"]) if data.get("area", None) is not None else np.nan),
            }
        )
    df_contacts = pd.DataFrame(contact_rows)

    pts = np.asarray(metrics["contact_points"], dtype=float) if metrics["contact_points"] else np.zeros((0, 3), dtype=float)
    nrm = np.asarray(metrics["contact_normals"], dtype=float) if metrics["contact_normals"] else np.zeros((0, 3), dtype=float)
    if len(pts) and pts.shape[1] != 3:
        pts = pts.reshape((-1, 3))
    if len(nrm) and nrm.shape[1] != 3:
        nrm = nrm.reshape((-1, 3))
    df_points = pd.DataFrame(
        {
            "x": pts[:, 0] if len(pts) else [],
            "y": pts[:, 1] if len(pts) else [],
            "z": pts[:, 2] if len(pts) else [],
            "nx": nrm[:, 0] if len(nrm) else [],
            "ny": nrm[:, 1] if len(nrm) else [],
            "nz": nrm[:, 2] if len(nrm) else [],
        }
    )

    if save_csv:
        p_path = os.path.join(output_dir, "particles.csv")
        c_path = os.path.join(output_dir, "contact_pairs.csv")
        pts_path = os.path.join(output_dir, "contact_points.csv")
        df_particles.to_csv(p_path, index=False)
        df_contacts.to_csv(c_path, index=False)
        df_points.to_csv(pts_path, index=False)
        print(f"Wrote CSV: {p_path}, {c_path}, {pts_path}")

    if save_hdf5:
        h5_path = os.path.join(output_dir, "results.h5")
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("particle_positions", data=df_particles[["x", "y", "z"]].to_numpy(dtype=float))
            f.create_dataset("contact_points", data=pts)
            f.create_dataset("contact_normals", data=nrm)
            f.attrs["Z"] = float(metrics["Z"])
            f.attrs["total_pp"] = int(metrics["total_pp_contacts"])
            f.attrs["total_pc"] = int(metrics["total_pc_contacts"])
            f.attrs["n_isolated"] = int(metrics["n_isolated_particles"])
            f.attrs["n_container_touch"] = int(metrics["n_container_touching"])
        print(f"Wrote HDF5: {h5_path}")

    print("── Contact Analysis Summary ─────────────────────────────────")
    print(f"Particles simulated:               {len(entities)}")
    print(f"Total particle-particle contacts:  {metrics['total_pp_contacts']}")
    print(f"Total particle-container contacts: {metrics['total_pc_contacts']}")
    print(f"Avg contacts per particle (Z):     {metrics['Z']:.3f}")
    print(f"Isolated particles (Z=0):          {metrics['n_isolated_particles']}")
    print(f"Particles touching container:      {metrics['n_container_touching']}")
    if metrics["contact_forces"]:
        forces = np.asarray(metrics["contact_forces"], dtype=float)
        print(f"Mean contact force:                {forces.mean():.4f} N")
        print(f"Max  contact force:                {forces.max():.4f} N")
    if metrics["contact_areas"]:
        areas = np.asarray(metrics["contact_areas"], dtype=float)
        print(f"Mean contact area:                 {(areas.mean() * 1e6):.4f} mm²")
    print("────────────────────────────────────────────────────────────")


def visualize_results(entities, metrics, original_mesh):
    try:
        import pyvista as pv
    except ModuleNotFoundError:
        print("Skipping visualization (optional dependency missing: pyvista).")
        return
    from scipy.spatial.transform import Rotation

    plotter = pv.Plotter(title="Particle Contact Analysis", window_size=[1280, 800])
    plotter.set_background("black")

    counts = metrics["contact_counts_per_particle"]
    max_c = max(counts.values()) if counts else 0
    max_c = max_c if max_c > 0 else 1

    faces = np.hstack(
        [
            np.full((len(original_mesh.faces), 1), 3, dtype=np.int64),
            np.asarray(original_mesh.faces, dtype=np.int64),
        ]
    ).ravel()
    base_poly = pv.PolyData(np.asarray(original_mesh.vertices, dtype=float).copy(), faces)

    for e in entities:
        pos = e.get_pos()
        qw, qx, qy, qz = e.get_quat()
        c = int(counts.get(_entity_id(e), 0))
        t = c / max_c

        poly = base_poly.copy(deep=True)
        rot = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        verts = (rot @ poly.points.T).T + np.asarray(pos, dtype=float)
        poly.points = verts
        poly["t"] = np.full((poly.n_points,), float(t), dtype=float)
        plotter.add_mesh(
            poly,
            scalars="t",
            cmap="coolwarm",
            clim=[0, 1],
            opacity=0.85,
            show_scalar_bar=False,
        )

    for pt in metrics["contact_points"]:
        plotter.add_mesh(pv.Sphere(radius=0.003, center=np.asarray(pt, dtype=float)), color="yellow")

    plotter.add_axes()
    plotter.show()


def main():
    parser = argparse.ArgumentParser(description="Genesis particle drop simulation")
    parser.add_argument("--backend", choices=["auto", "cpu", "gpu"], default=BACKEND)
    parser.add_argument("--show-viewer", action="store_true", default=SHOW_VIEWER)
    parser.add_argument("--n", type=int, default=N_PARTICLES, help="number of particles")
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--substeps", type=int, default=SUBSTEPS)
    parser.add_argument("--duration", type=float, default=SIM_DURATION)
    parser.add_argument("--no-hdf5", action="store_true", help="disable HDF5 output")
    parser.add_argument("--no-csv", action="store_true", help="disable CSV output")
    args = parser.parse_args()

    if args.backend == "cpu":
        backend = gs.cpu
    elif args.backend == "gpu":
        backend = gs.gpu
    else:
        # "auto" → prefer GPU (CUDA / Metal), same as Genesis default intent.
        backend = gs.gpu

    try:
        gs.init(backend=backend)
    except Exception as e:
        if args.backend in {"auto", "gpu"}:
            print(f"[WARNING] gs.init(GPU) failed, falling back to CPU: {e}")
            gs.init(backend=gs.cpu)
        else:
            raise

    sim_options = gs.options.SimOptions(dt=args.dt, substeps=args.substeps, gravity=GRAVITY)
    rigid_options = gs.options.RigidOptions(use_gjk_collision=True)

    # Avoid building the visualizer unless explicitly requested.
    # This prevents the viewer from throttling the run (e.g., ~0.1 FPS on CPU).
    scene = gs.Scene(sim_options=sim_options, rigid_options=rigid_options, show_viewer=bool(args.show_viewer))

    physics_mesh, original_mesh = load_particle_mesh(PARTICLE_FILE, SCALE_FACTOR)
    container_ids, env_info = create_environment(
        scene,
        ENVIRONMENT_TYPE,
        PLATE_SIZE,
        CYLINDER_DIAMETER,
        CYLINDER_HEIGHT,
        CYLINDER_SEGMENTS,
        WALL_THICKNESS,
        env_restitution=ENV_RESTITUTION,
    )
    container_ids = {_entity_id(e) for e in container_ids}
    entities = spawn_particles(
        scene,
        physics_mesh,
        args.n,
        env_info,
        DROP_HEIGHT,
        DROP_SPREAD,
        YOUNGS_MODULUS,
        POISSON_RATIO,
        DENSITY,
        particle_restitution=PARTICLE_RESTITUTION,
    )
    particle_ids = {_entity_id(e) for e in entities}
    scene.build()
    run_simulation(scene, entities, args.dt, args.substeps, args.duration, SETTLE_THRESHOLD)
    contacts = extract_contacts(scene, particle_ids, container_ids, CONTACT_DEPTH_TOL)
    metrics = compute_metrics(contacts, particle_ids)
    export_results(
        entities,
        metrics,
        OUTPUT_DIR,
        save_hdf5=(SAVE_HDF5 and (not args.no_hdf5)),
        save_csv=(SAVE_CSV and (not args.no_csv)),
    )
    if args.show_viewer:
        visualize_results(entities, metrics, original_mesh)


if __name__ == "__main__":
    main()
