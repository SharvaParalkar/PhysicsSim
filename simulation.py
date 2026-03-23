import os, math
import argparse
import logging
import sys
import numpy as np
import torch
import pandas as pd
import h5py
import trimesh
import coacd
import networkx as nx
from scipy.spatial.transform import Rotation
from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import Any, Callable, Optional

# Avoid Windows console UnicodeEncodeError when libraries print box-drawing/emoji.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Default Genesis backend for `python simulation.py`, `server.py`, and GENESIS_USE_CPU.
# "auto" → prefer GPU; "cpu" / "gpu" force that backend unless GENESIS_USE_CPU=0|1 overrides.
BACKEND = "cpu"

# Optional: hide CUDA before importing Torch when the effective backend is CPU.
_cli_backend = None
if "--backend" in sys.argv:
    try:
        i = sys.argv.index("--backend")
        if i + 1 < len(sys.argv):
            _cli_backend = str(sys.argv[i + 1]).strip().lower()
    except ValueError:
        pass
_gc = os.environ.get("GENESIS_USE_CPU", "").strip().lower()
if _gc in ("1", "true", "yes"):
    _effective_cpu = True
elif _gc in ("0", "false", "no"):
    _effective_cpu = False
elif _cli_backend in ("cpu", "gpu", "auto"):
    _effective_cpu = _cli_backend == "cpu"
else:
    _effective_cpu = BACKEND.strip().lower() == "cpu"
if _effective_cpu:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


class _GenesisLazy:
    """Lazy `genesis` import so Taichi/LLVM bind to the thread that first imports (server worker)."""

    _mod = None

    @classmethod
    def _load(cls):
        if cls._mod is None:
            import genesis as g

            cls._mod = g
        return cls._mod

    def __getattr__(self, name):
        return getattr(self._load(), name)


gs = _GenesisLazy()

# ── STEP 2: Parameter block ────────────────────────────────────────────────────  # parameters section
# ── Input ──────────────────────────────────────────────────────────────────  # input settings
PARTICLE_FILE        = "particle.obj"   # OBJ or STL path
N_PARTICLES          = 50               # number of particle copies to drop
SCALE_FACTOR         = 1.0             # 0.001 converts mm mesh → metres
# ── Material ───────────────────────────────────────────────────────────────  # material settings
# YOUNGS_MODULUS: FEM jamming uses E ≤ 1e8; above 1e8 Genesis uses rigid particles (see spawn_particles).
FEM_JAMMING_E_MAX  = 100_000_000.0   # Pa — upper FEM bound (1e8)
YOUNGS_MODULUS       = 200_000_000.0  # Pa — forces Rigid solver path (E > 1e8)
POISSON_RATIO        = 0.45            # 0.5 = fully incompressible; use 0.45+ to limit volume loss
DENSITY              = 1200            # kg/m³
# Rigid-body restitution (Genesis maps this to internal coupling restitution).
# Non-zero values can trigger a Genesis WARNING and may reduce stability; 0 = inelastic.
PARTICLE_RESTITUTION = 0.0             # particle–contact bounciness (was 0.2)
ENV_RESTITUTION      = 0.0             # floor/walls (was 0.05)
# ── Environment ────────────────────────────────────────────────────────────  # environment settings
ENVIRONMENT_TYPE     = "plate"      # "cylinder" or "plate"
PLATE_SIZE           = 0.25            # square plate side length (m)
CYLINDER_DIAMETER    = 0.20            # inner diameter (m)
CYLINDER_HEIGHT      = 0.30            # wall height (m)
CYLINDER_SEGMENTS    = 32              # wall facets — use 24+ to avoid gaps
WALL_THICKNESS       = 0.02            # m — plate slab thickness (too thin + coarse dt → FEM tunneling)
# Rim height for ENVIRONMENT_TYPE="plate" — keeps particles on the plate (0 = flat open plate).
PLATE_WALL_HEIGHT    = 0.15            # m — vertical walls along the square perimeter
# ── Drop ───────────────────────────────────────────────────────────────────  # drop settings
DROP_HEIGHT          = 0.05            # metres above container top edge (plate / cylinder rim)
# 0 = stack all particles in a vertical column at (0, ·, 0); >0 = Vogel disk on XZ up to this fraction of spread radius
DROP_SPREAD          = 0.5
# ── Gravity ────────────────────────────────────────────────────────────────  # gravity settings
GRAVITY              = (0, -9.81, 0)   # Y is up; change to (0,-1.62,0) for Moon
# ── Simulation ─────────────────────────────────────────────────────────────  # simulation settings
# Throughput-first defaults: larger outer dt, fewer substeps; implicit FEM + low Newton count (see FEMOptions).
# For thin rigid plates, use Analytical Mode (large dt while falling, then rebuild at 500 Hz / 16 substeps).
DT                   = 1 / 240       # s — outer step (with SUBSTEPS)
SUBSTEPS             = 4              # inner substeps per dt (higher → less rigid/FEM tunneling)
SIM_DURATION         = 10.0            # max simulated time (s)
SETTLE_THRESHOLD     = 1e-3            # m/s — stop early when all particles slow
# ── Runtime / performance ───────────────────────────────────────────────────  # runtime settings
# Genesis’s viewer can dominate runtime on CPU (the FPS log you saw is from it).
# Keep it off by default so simulation runs as fast as possible.
SHOW_VIEWER          = False
# ── Contact analysis ───────────────────────────────────────────────────────  # contact analysis settings
CONTACT_SAMPLE_EVERY = 20              # legacy doc alignment: prefer CONTACT_EXTRACT_FALLING_EVERY for live runs
# During fast motion, skip scene.get_contacts() most steps (see extract_contacts_resampled).
CONTACT_EXTRACT_FALLING_EVERY = 20
CONTACT_DEPTH_TOL    = 5e-5            # min penetration depth to count as contact
STRESS_FLOOR         = 0.05            # normalized intensities below this are clamped to 0 (kills noise on non-touching particles)
# Strong PP count (reference): |F| above this; Z uses depth > CONTACT_DEPTH_TOL only (see compute_metrics).
Z_CONTACT_FORCE_MIN_N = 1e-3
# ── Output ─────────────────────────────────────────────────────────────────  # output settings
OUTPUT_DIR           = "./results"     # output directory path
SAVE_HDF5            = True            # enable HDF5 output
SAVE_CSV             = True            # enable CSV output

# ── Physics normalisation ───────────────────────────────────────────────────
# Micro-scale meshes (e.g. 600M variants authored in µm) have characteristic
# sizes in the sub-mm range after applying their display scale factor (1e-6).
# Genesis's rigid solver rejects masses below an EPS (~1e-6 kg), and CoACD
# fails with "Weights sum to zero" for sub-mm vertex coordinates.  Both issues
# are solved by temporarily scaling the mesh UP to PHYSICS_NORM_TARGET before
# physics and CoACD, then dividing all output positions back down by the same
# factor so the viewer sees the correct µm-scale geometry.
PHYSICS_NORM_THRESHOLD = 5e-3   # m — normalise when char size < 5 mm
PHYSICS_NORM_TARGET    = 0.025  # m — target char size (≈ default particle.obj)

# ── Genesis option bundles (throughput vs analytical phases) ───────────────
@dataclass(frozen=True)
class ThroughputSimTuning:
    """Default SimOptions tuning: prioritize wall-clock throughput."""

    dt: float = 1.0 / 240.0
    substeps: int = 8


@dataclass(frozen=True)
class ThroughputFEMTuning:
    """Default FEMOptions tuning: fewer Newton iterations per implicit step."""

    n_newton_iterations: int = 4


@dataclass(frozen=True)
class AnalyticalFallingTuning:
    """Falling phase timestep — match throughput defaults so thin plate contacts are not skipped (tunneling)."""

    dt: float = 1.0 / 240.0
    substeps: int = 8


@dataclass(frozen=True)
class AnalyticalPrecisionTuning:
    """Settling phase once max speed drops below ANALYTICAL_VEL_THRESHOLD: 500 Hz, 16 substeps."""

    dt: float = 1.0 / 500.0
    substeps: int = 16
    n_newton_iterations: int = 8


def make_sim_options(gs_mod, cfg: dict):
    """Build `gs.options.SimOptions` from a runtime config dict."""
    g = getattr(gs_mod, "options", gs_mod)
    return g.SimOptions(
        dt=float(cfg["DT"]),
        substeps=int(cfg["SUBSTEPS"]),
        gravity=cfg.get("GRAVITY", (0, -9.81, 0)),
    )


def make_fem_options(gs_mod, cfg: dict):
    """Build `gs.options.FEMOptions` from a runtime config dict."""
    g = getattr(gs_mod, "options", gs_mod)
    return g.FEMOptions(
        use_implicit_solver=True,
        n_newton_iterations=int(cfg.get("FEM_NEWTON_ITERATIONS", ThroughputFEMTuning.n_newton_iterations)),
    )


def make_rigid_options(gs_mod, _cfg: Optional[dict] = None):
    """
    Rigid solver options for mesh particles against fixed box/cylinder containers.

    `box_box_detection` improves box–box contact; stiffer `constraint_timeconst` reduces penetration.
    Increased `iterations` (300) reduces residual penetration for dense concave multi-hull packing.
    Tighter `constraint_timeconst` (0.001) shrinks per-step penetration residual before it accumulates.
    With compound-hull collision proxies the solver sees accurate geometry, so tighter settings
    converge cleanly without instability.
    """
    g = getattr(gs_mod, "options", gs_mod)
    return g.RigidOptions(
        use_gjk_collision=True,
        box_box_detection=True,
        iterations=300,
        constraint_timeconst=0.001,
    )


def snapshot_fem_entities(entities) -> list[dict]:
    """Per-vertex pos/vel for FEM entities (for scene rebuild handoff). Skips non-FEM."""
    out: list[dict] = []
    for e in entities:
        if type(e).__name__ != "FEMEntity":
            continue
        st = e.get_state()
        pos = _tensor_to_numpy(st.pos).astype(float)
        vel = _tensor_to_numpy(st.vel).astype(float)
        if pos.ndim == 3:
            pos = pos[0]
        if vel.ndim == 3:
            vel = vel[0]
        out.append({"pos": np.ascontiguousarray(pos), "vel": np.ascontiguousarray(vel)})
    return out


def restore_fem_entities(entities, snapshots: list[dict]) -> None:
    """Apply `snapshot_fem_entities` output to a fresh scene's FEM entities (same spawn order)."""
    i = 0
    for e in entities:
        if type(e).__name__ != "FEMEntity":
            continue
        if i >= len(snapshots):
            break
        snap = snapshots[i]
        i += 1
        e.set_position(snap["pos"])
        e.set_velocity(snap["vel"])


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
    "PLATE_WALL_HEIGHT": PLATE_WALL_HEIGHT,
    "DROP_HEIGHT": DROP_HEIGHT,
    "DROP_SPREAD": DROP_SPREAD,
    # Genesis FEM explicit integration is unstable at ~1e8 Pa with typical dt; implicit is recommended.
    "FEM_USE_IMPLICIT": True,
    "FEM_NEWTON_ITERATIONS": 4,
    "FEM_NEWTON_ITERATIONS_PRECISION": 8,
    "ANALYTICAL_MODE": False,  # No-op with rigid bodies (E > 1e8); enable only for FEM (E ≤ 1e8)
    "ANALYTICAL_FALLING_DT": AnalyticalFallingTuning.dt,
    "ANALYTICAL_FALLING_SUBSTEPS": AnalyticalFallingTuning.substeps,
    "ANALYTICAL_PRECISION_DT": AnalyticalPrecisionTuning.dt,
    "ANALYTICAL_PRECISION_SUBSTEPS": AnalyticalPrecisionTuning.substeps,
    "ANALYTICAL_VEL_THRESHOLD": 0.1,
    "GRAVITY": GRAVITY,
    "DT": DT,
    # Use ThroughputSimTuning.substeps (8) as the default; the module-level SUBSTEPS=4 is
    # kept for the standalone CLI but the server should use the tuning-class value.
    "SUBSTEPS": ThroughputSimTuning.substeps,
    "SIM_DURATION": SIM_DURATION,
    "SETTLE_THRESHOLD": SETTLE_THRESHOLD,
    "CONTACT_EXTRACT_FALLING_EVERY": CONTACT_EXTRACT_FALLING_EVERY,
    "CONTACT_DEPTH_TOL": CONTACT_DEPTH_TOL,
    "OUTPUT_DIR": OUTPUT_DIR,
    "SAVE_HDF5": SAVE_HDF5,
    "SAVE_CSV": SAVE_CSV,
    "STRESS_SIGMA": 0.12,  # radians — Hertzian angular falloff width (tight ~16° FWHM keeps stress local to contact site)
    "STRESS_FLOOR": 0.05,  # clamp post-normalization noise below this to 0.0 (kills ghost gradients on non-touching particles)
    # When True: rebuild scene repeatedly — simulate k particles, snapshot FEM state, add one more at the drop height.
    # Disables analytical handoff in the server (stages conflict with mid-run scene rebuilds).
    "SEQUENTIAL_DROP": True,
    # Max simulated time per staging step (s). None → max(SIM_DURATION / N_PARTICLES, 0.25).
    "SEQUENTIAL_STAGE_DURATION": None,
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
    # Clamp to FEM band only when staying in FEM/MPM (E ≤ 1e8). Larger E selects rigid particles in spawn_particles.
    if cfg["YOUNGS_MODULUS"] <= 1e8 and cfg["YOUNGS_MODULUS"] > cfg["FEM_JAMMING_E_MAX"]:
        cfg["YOUNGS_MODULUS"] = cfg["FEM_JAMMING_E_MAX"]
    cfg["STRESS_SIGMA"] = float(cfg.get("STRESS_SIGMA", 0.12))
    cfg["STRESS_FLOOR"] = float(cfg.get("STRESS_FLOOR", 0.05))
    cfg["POISSON_RATIO"] = float(cfg["POISSON_RATIO"])
    cfg["PARTICLE_RESTITUTION"] = float(cfg["PARTICLE_RESTITUTION"])
    cfg["ENV_RESTITUTION"] = float(cfg["ENV_RESTITUTION"])
    cfg["CYLINDER_DIAMETER"] = float(cfg["CYLINDER_DIAMETER"])
    cfg["DROP_HEIGHT"] = float(cfg["DROP_HEIGHT"])
    cfg["DROP_SPREAD"] = float(cfg["DROP_SPREAD"])
    cfg["PLATE_WALL_HEIGHT"] = float(cfg.get("PLATE_WALL_HEIGHT", PLATE_WALL_HEIGHT))
    cfg["PLATE_SIZE"] = float(cfg.get("PLATE_SIZE", PLATE_SIZE))
    cfg["WALL_THICKNESS"] = float(cfg.get("WALL_THICKNESS", WALL_THICKNESS))
    cfg["FEM_NEWTON_ITERATIONS"] = int(cfg.get("FEM_NEWTON_ITERATIONS", ThroughputFEMTuning.n_newton_iterations))
    cfg["FEM_NEWTON_ITERATIONS_PRECISION"] = int(cfg.get("FEM_NEWTON_ITERATIONS_PRECISION", AnalyticalPrecisionTuning.n_newton_iterations))
    cfg["ANALYTICAL_MODE"] = bool(cfg["ANALYTICAL_MODE"])
    cfg["ANALYTICAL_FALLING_DT"] = float(cfg.get("ANALYTICAL_FALLING_DT", AnalyticalFallingTuning.dt))
    cfg["ANALYTICAL_FALLING_SUBSTEPS"] = max(1, int(cfg.get("ANALYTICAL_FALLING_SUBSTEPS", AnalyticalFallingTuning.substeps)))
    cfg["ANALYTICAL_PRECISION_DT"] = float(cfg.get("ANALYTICAL_PRECISION_DT", AnalyticalPrecisionTuning.dt))
    cfg["ANALYTICAL_PRECISION_SUBSTEPS"] = max(1, int(cfg.get("ANALYTICAL_PRECISION_SUBSTEPS", AnalyticalPrecisionTuning.substeps)))
    cfg["ANALYTICAL_VEL_THRESHOLD"] = float(cfg.get("ANALYTICAL_VEL_THRESHOLD", 0.1))
    cfg["DT"] = float(cfg["DT"])
    cfg["SUBSTEPS"] = int(cfg["SUBSTEPS"])
    cfg["SIM_DURATION"] = float(cfg["SIM_DURATION"])
    cfg["SETTLE_THRESHOLD"] = float(cfg["SETTLE_THRESHOLD"])
    _grav = cfg.get("GRAVITY", GRAVITY)
    if isinstance(_grav, (list, tuple)) and len(_grav) == 3:
        cfg["GRAVITY"] = tuple(float(x) for x in _grav)
    else:
        cfg["GRAVITY"] = tuple(float(x) for x in GRAVITY)
    cfg["CONTACT_EXTRACT_FALLING_EVERY"] = max(1, int(cfg.get("CONTACT_EXTRACT_FALLING_EVERY", CONTACT_EXTRACT_FALLING_EVERY)))
    cfg["SEQUENTIAL_DROP"] = bool(cfg.get("SEQUENTIAL_DROP", False))
    _ssd = cfg.get("SEQUENTIAL_STAGE_DURATION", None)
    cfg["SEQUENTIAL_STAGE_DURATION"] = None if _ssd is None else float(_ssd)
    cfg["ENVIRONMENT_TYPE"] = str(cfg["ENVIRONMENT_TYPE"]).strip().lower()
    # Jamming / packed FEM: implicit stepper is required at high E; do not allow config to disable it.
    cfg["FEM_USE_IMPLICIT"] = True
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


@dataclass
class ContactSampleCache:
    """Stores the last `extract_contacts` result when throttling `scene.get_contacts()`."""

    contacts: list = field(default_factory=list)
    primed: bool = False


def contact_extract_stride(max_vel: Optional[float], settle_threshold: float, falling_every: int) -> int:
    fe = max(1, int(falling_every))
    if max_vel is None:
        return fe
    if float(max_vel) >= float(settle_threshold):
        return fe
    return 1


def extract_contacts_resampled(
    scene,
    particle_ids,
    container_ids,
    depth_tol: float,
    *,
    sim_step: int,
    max_vel: Optional[float],
    settle_threshold: float,
    cache: ContactSampleCache,
    falling_every: Optional[int] = None,
    force_skip: bool = False,
) -> list:
    """
    Cheap when particles are still falling: only calls `get_contacts` every `falling_every` steps.
    After speeds drop below `settle_threshold`, samples every step so jammed contact metrics stay fresh.
    """
    if force_skip:
        return []
    fe = CONTACT_EXTRACT_FALLING_EVERY if falling_every is None else int(falling_every)
    stride = contact_extract_stride(max_vel, settle_threshold, fe)
    if (not cache.primed) or (int(sim_step) % stride == 0):
        cache.contacts = extract_contacts(scene, particle_ids, container_ids, depth_tol)
        cache.primed = True
    return cache.contacts


def _entity_id(e) -> int:
    # Genesis versions use either `.id` or `.idx`.
    if hasattr(e, "id"):
        return int(getattr(e, "id"))
    return int(getattr(e, "idx"))


def _tensor_to_numpy(x) -> np.ndarray:
    """Convert torch.Tensor or array-like to numpy float64 on CPU (Genesis may return CUDA tensors)."""
    if x is None:
        return np.zeros((0,), dtype=float)
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


def _vec3_from_xyz(x, y, z) -> np.ndarray:
    return np.array(
        [
            float(_tensor_to_numpy(x).reshape(-1)[0]),
            float(_tensor_to_numpy(y).reshape(-1)[0]),
            float(_tensor_to_numpy(z).reshape(-1)[0]),
        ],
        dtype=float,
    )


def _quat_tuple_xyzw(qw, qx, qy, qz) -> tuple[float, float, float, float]:
    """Genesis get_quat is (w,x,y,z); visualization uses (x,y,z,w)."""
    return (
        float(_tensor_to_numpy(qx).reshape(-1)[0]),
        float(_tensor_to_numpy(qy).reshape(-1)[0]),
        float(_tensor_to_numpy(qz).reshape(-1)[0]),
        float(_tensor_to_numpy(qw).reshape(-1)[0]),
    )


def _contact_pos(c):
    # Genesis versions use either `.pos` or `.position`.
    if hasattr(c, "pos"):
        return c.pos
    return c.position


def _as_vec3_any(x) -> np.ndarray:
    """Contact / pose data may be numpy, CUDA tensors, or (x,y,z) tuples of tensors."""
    if x is None:
        return np.zeros(3, dtype=float)
    if isinstance(x, (tuple, list)) and len(x) >= 3:
        return _vec3_from_xyz(x[0], x[1], x[2])
    v = _tensor_to_numpy(x).astype(float).ravel()
    if v.size < 3:
        v = np.pad(v, (0, 3 - int(v.size)))
    return v[:3]


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
        pos_try = _vec3_from_xyz(x, y, z)
        # If pose getters return something clearly not in world space (e.g.
        # very large magnitudes), fall back to state-based heuristics below.
        if np.isfinite(pos_try).all() and float(np.abs(pos_try).max()) <= 10.0:
            if hasattr(e, "get_quat"):
                qw, qx, qy, qz = e.get_quat()  # (w,x,y,z) -> (x,y,z,w)
                quat = _quat_tuple_xyzw(qw, qx, qy, qz)
            else:
                quat = (0.0, 0.0, 0.0, 1.0)
            return (pos_try, quat)
        # else: fall through to fallback extraction
        if hasattr(e, "get_quat"):
            qw, qx, qy, qz = e.get_quat()  # (w,x,y,z) -> (x,y,z,w)
            quat = _quat_tuple_xyzw(qw, qx, qy, qz)
        else:
            quat = (0.0, 0.0, 0.0, 1.0)
        # Keep the existing return for completeness; most callers will hit
        # the magnitude guard above.
        return (pos_try, quat)

    # Fallback: try entity state (post scene.build()) and compute centroid.
    try:
        if hasattr(e, "get_state"):
            st = e.get_state()
            candidate_positions: list[np.ndarray] = []
            for attr in ("pos", "x", "p"):
                if not hasattr(st, attr):
                    continue
                arr = _tensor_to_numpy(getattr(st, attr))
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
                if float(np.abs(best_pos).max()) > 500.0 and hasattr(e, "get_pos"):
                    try:
                        x, y, z = e.get_pos()
                        pos_try = _vec3_from_xyz(x, y, z)
                        if np.isfinite(pos_try).all() and float(np.abs(pos_try).max()) <= 500.0:
                            if hasattr(e, "get_quat"):
                                qw, qx, qy, qz = e.get_quat()
                                quat = _quat_tuple_xyzw(qw, qx, qy, qz)
                            else:
                                quat = (0.0, 0.0, 0.0, 1.0)
                            return (pos_try, quat)
                    except Exception:
                        pass
                return (best_pos, (0.0, 0.0, 0.0, 1.0))
    except Exception:
        pass

    # FEM/MPM: state tensors may use different field names across Genesis versions.
    # If centroid extraction failed, try rigid-style getters so the viewer still receives poses.
    if force_state_pose and hasattr(e, "get_pos"):
        try:
            x, y, z = e.get_pos()
            pos_try = _vec3_from_xyz(x, y, z)
            if np.isfinite(pos_try).all():
                if hasattr(e, "get_quat"):
                    qw, qx, qy, qz = e.get_quat()
                    quat = _quat_tuple_xyzw(qw, qx, qy, qz)
                else:
                    quat = (0.0, 0.0, 0.0, 1.0)
                return (pos_try, quat)
        except Exception:
            pass

    return (np.zeros((3,), dtype=float), (0.0, 0.0, 0.0, 1.0))


@contextmanager
def _suppress_gs_manual_pose_warnings():
    """Avoid flooding logs when correcting FEM positions after `scene.step()`."""
    lg = getattr(gs, "logger", None)
    if lg is None or not hasattr(lg, "setLevel"):
        yield
        return
    prev = getattr(lg, "level", logging.WARNING)
    try:
        lg.setLevel(logging.ERROR)
        yield
    finally:
        lg.setLevel(prev)


def _obb_world_corners(pos: np.ndarray, quat_xyzw: tuple[float, float, float, float], half_ext: np.ndarray) -> np.ndarray:
    q = np.array([quat_xyzw[0], quat_xyzw[1], quat_xyzw[2], quat_xyzw[3]], dtype=float)
    r = Rotation.from_quat(q).as_matrix()
    corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                local = np.array([sx * half_ext[0], sy * half_ext[1], sz * half_ext[2]], dtype=float)
                corners.append(pos + r @ local)
    return np.array(corners, dtype=float)


def enforce_container_bounds(entities, physics_mesh, cfg: dict) -> None:
    """
    Genesis rigid contacts and FEM–rigid coupling are velocity-based and can miss penetration.
    After each step, project particle geometry back into the analytical container (floor + rim/cylinder).
    """
    if not entities or physics_mesh is None:
        return
    try:
        bounds = physics_mesh.bounds[1] - physics_mesh.bounds[0]
    except Exception:
        return
    half_ext = np.asarray(bounds, dtype=float) * 0.5
    max_h = float(np.max(half_ext))
    if not math.isfinite(max_h) or max_h <= 0.0:
        return
    # eps must be at least 5% of the particle half-extent so it remains meaningful
    # at any scale (the old 1e-5 m floor was negligible for µm-scale particles where
    # max_h ≈ 300 µm, giving an eps that is only 3% of the floor value).
    eps = max(5e-2 * max_h, 1e-9)

    env = str(cfg.get("ENVIRONMENT_TYPE", ENVIRONMENT_TYPE)).strip().lower()
    t = float(cfg.get("WALL_THICKNESS", WALL_THICKNESS))
    surface_y = t
    if env == "plate":
        rim_h = float(cfg.get("PLATE_WALL_HEIGHT", PLATE_WALL_HEIGHT))
        top_y = t + rim_h
        s = float(cfg.get("PLATE_SIZE", PLATE_SIZE))
        half_s = 0.5 * s
        for e in entities:
            name = type(e).__name__
            if name == "FEMEntity":
                try:
                    st = e.get_state()
                    pos = _tensor_to_numpy(st.pos).astype(float)
                    if pos.ndim == 3:
                        pos = pos[0]
                    if pos.ndim != 2 or pos.shape[-1] != 3:
                        continue
                    pos = np.ascontiguousarray(pos)
                    pos[:, 1] = np.maximum(pos[:, 1], surface_y + eps)
                    if rim_h > 1e-9:
                        in_rim = (pos[:, 1] >= surface_y - eps) & (pos[:, 1] <= top_y + eps)
                        if np.any(in_rim):
                            pos[in_rim, 0] = np.clip(pos[in_rim, 0], -half_s + eps, half_s - eps)
                            pos[in_rim, 2] = np.clip(pos[in_rim, 2], -half_s + eps, half_s - eps)
                    with _suppress_gs_manual_pose_warnings():
                        e.set_position(pos)
                except Exception:
                    pass
            elif name == "RigidEntity":
                try:
                    p0, quat = _entity_pose(e)
                    pos = np.asarray(p0, dtype=float).copy()
                    for _ in range(8):
                        corners = _obb_world_corners(pos, quat, half_ext)
                        min_y = float(np.min(corners[:, 1]))
                        moved = False
                        if min_y < surface_y + eps:
                            pos[1] += (surface_y + eps) - min_y
                            moved = True
                            corners = _obb_world_corners(pos, quat, half_ext)
                        if rim_h > 1e-9:
                            in_rim = (corners[:, 1] >= surface_y - eps) & (corners[:, 1] <= top_y + eps)
                            if np.any(in_rim):
                                cr = corners[in_rim]
                                dx = 0.0
                                dz = 0.0
                                mx = float(np.max(cr[:, 0]))
                                mn = float(np.min(cr[:, 0]))
                                mz = float(np.max(cr[:, 2]))
                                mnz = float(np.min(cr[:, 2]))
                                if mx > half_s - eps:
                                    dx = (half_s - eps) - mx
                                elif mn < -half_s + eps:
                                    dx = (-half_s + eps) - mn
                                if mz > half_s - eps:
                                    dz = (half_s - eps) - mz
                                elif mnz < -half_s + eps:
                                    dz = (-half_s + eps) - mnz
                                if abs(dx) > 1e-12 or abs(dz) > 1e-12:
                                    pos[0] += dx
                                    pos[2] += dz
                                    moved = True
                        if not moved:
                            break
                    e.set_pos(pos, zero_velocity=False)
                except Exception:
                    pass
        return

    if env == "cylinder":
        r_inner = float(cfg.get("CYLINDER_DIAMETER", CYLINDER_DIAMETER)) * 0.5
        cyl_h = float(cfg.get("CYLINDER_HEIGHT", CYLINDER_HEIGHT))
        top_y = t + cyl_h
        for e in entities:
            name = type(e).__name__
            if name == "FEMEntity":
                try:
                    st = e.get_state()
                    pos = _tensor_to_numpy(st.pos).astype(float)
                    if pos.ndim == 3:
                        pos = pos[0]
                    if pos.ndim != 2 or pos.shape[-1] != 3:
                        continue
                    pos = np.ascontiguousarray(pos)
                    pos[:, 1] = np.maximum(pos[:, 1], surface_y + eps)
                    in_rim = (pos[:, 1] >= surface_y - eps) & (pos[:, 1] <= top_y + eps)
                    if np.any(in_rim):
                        xz = pos[in_rim, [0, 2]]
                        r = np.hypot(xz[:, 0], xz[:, 1])
                        mask = r > r_inner - eps
                        if np.any(mask):
                            idx = np.where(in_rim)[0][mask]
                            for i in idx:
                                x, z = float(pos[i, 0]), float(pos[i, 2])
                                rv = math.hypot(x, z)
                                if rv > 1e-12:
                                    sc = (r_inner - eps) / rv
                                    pos[i, 0] *= sc
                                    pos[i, 2] *= sc
                    with _suppress_gs_manual_pose_warnings():
                        e.set_position(pos)
                except Exception:
                    pass
            elif name == "RigidEntity":
                try:
                    p0, quat = _entity_pose(e)
                    pos = np.asarray(p0, dtype=float).copy()
                    for _ in range(8):
                        corners = _obb_world_corners(pos, quat, half_ext)
                        min_y = float(np.min(corners[:, 1]))
                        moved = False
                        if min_y < surface_y + eps:
                            pos[1] += (surface_y + eps) - min_y
                            moved = True
                            corners = _obb_world_corners(pos, quat, half_ext)
                        in_rim = (corners[:, 1] >= surface_y - eps) & (corners[:, 1] <= top_y + eps)
                        if np.any(in_rim):
                            cr = corners[in_rim]
                            xy = cr[:, [0, 2]]
                            r = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)
                            j = int(np.argmax(r))
                            rmax = float(r[j])
                            if rmax > r_inner - eps:
                                c = cr[j]
                                xv, zv = float(c[0]), float(c[2])
                                rv = math.hypot(xv, zv)
                                if rv > 1e-12:
                                    dr = rmax - (r_inner - eps)
                                    pos[0] -= (xv / rv) * dr
                                    pos[2] -= (zv / rv) * dr
                                    moved = True
                        if not moved:
                            break
                    e.set_pos(pos, zero_velocity=False)
                except Exception:
                    pass


def enforce_particle_separation(
    entities, physics_mesh, _cfg: dict, *, original_mesh=None
) -> None:
    """
    Post-step centroid-based depenetration sweep.

    Detects particle pairs whose centroids are closer than the computed minimum
    separation distance and pushes them apart with a Jacobi-style half-correction
    along the separation axis.

    The threshold is derived from the actual particle geometry (``original_mesh``)
    when available: the 10th-percentile vertex-to-centroid distance approximates the
    particle's inner (concave) radius, and ``1.7 ×`` that value is the centroid
    distance below which two particles MUST be genuinely penetrating.  This is
    tighter than the old ``char × 0.6`` heuristic which fired falsely for star
    particles whose concave faces sit naturally close to a neighbour's centroid.

    Only corrects severe overlaps so it does not fight the constraint solver during
    normal settling contact.  Both RigidEntity and FEMEntity are supported.
    """
    if not entities or physics_mesh is None:
        return
    try:
        extents = physics_mesh.bounds[1] - physics_mesh.bounds[0]
    except Exception:
        return
    char = float(max(float(extents[0]), float(extents[1]), float(extents[2]), 1e-9))

    # Compute threshold from the ORIGINAL mesh geometry (the actual star shape)
    # rather than the bounding-box extent.  For a 6-point star the 10th-percentile
    # vertex distance from the centroid approximates the inner (concave) radius r_in.
    # Two centroids closer than 1.7 × r_in are definitively interpenetrating.
    min_sep = char * 0.4  # fallback if original_mesh is unavailable
    if original_mesh is not None:
        try:
            ov = np.asarray(original_mesh.vertices, dtype=float)
            if ov.shape[0] > 0:
                oc = np.asarray(original_mesh.centroid, dtype=float)
                dists = np.linalg.norm(ov - oc, axis=1)
                r_in = float(np.percentile(dists, 10))
                if r_in > 1e-9:
                    min_sep = r_in * 1.7
        except Exception:
            pass
    min_sep_sq = min_sep * min_sep

    poses: list[tuple[Any, np.ndarray]] = []
    for e in entities:
        ename = type(e).__name__
        if ename not in ("RigidEntity", "FEMEntity"):
            continue
        try:
            p, _ = _entity_pose(e)
            if np.isfinite(p).all():
                poses.append((e, np.asarray(p, dtype=float).copy()))
        except Exception:
            continue

    if len(poses) < 2:
        return

    corrections = [np.zeros(3, dtype=float) for _ in poses]
    made_correction = False

    for i in range(len(poses)):
        for j in range(i + 1, len(poses)):
            delta = poses[i][1] - poses[j][1]
            dist_sq = float(np.dot(delta, delta))
            if dist_sq >= min_sep_sq or dist_sq < 1e-18:
                continue
            dist = math.sqrt(dist_sq)
            push = (min_sep - dist) * 0.5
            axis = delta / dist
            corrections[i] += axis * push
            corrections[j] -= axis * push
            made_correction = True

    if not made_correction:
        return

    for idx, (e, p) in enumerate(poses):
        corr = corrections[idx]
        if float(np.linalg.norm(corr)) < 1e-12:
            continue
        new_pos = p + corr
        try:
            ename = type(e).__name__
            if ename == "RigidEntity":
                e.set_pos(new_pos, zero_velocity=False)
            elif ename == "FEMEntity":
                st = e.get_state()
                vpos = _tensor_to_numpy(st.pos).astype(float)
                if vpos.ndim == 3:
                    vpos = vpos[0]
                if vpos.ndim == 2 and vpos.shape[-1] == 3:
                    vpos = np.ascontiguousarray(vpos + corr.reshape(1, 3))
                    with _suppress_gs_manual_pose_warnings():
                        e.set_position(vpos)
        except Exception:
            pass


def _coacd_proxy_path(filepath: str, scale: float) -> str:
    """Return a deterministic path for the cached compound OBJ collision proxy."""
    base, _ = os.path.splitext(filepath)
    scale_tag = f"{scale:.6g}".replace(".", "p").replace("-", "n")
    return f"{base}_coacd_proxy_s{scale_tag}.obj"


def _write_coacd_compound_obj(parts: list, filepath: str) -> None:
    """Write CoACD convex parts as a multi-object OBJ (one ``o PartN`` per hull).

    Genesis reads each ``o`` sub-mesh as a separate convex hull in a compound
    collision proxy when ``convexify=False`` — giving a faithful multi-hull shape
    instead of one bloated global convex hull.

    OBJ face indices are 1-based and global across the whole file, so we track
    the running vertex offset as we write each part.
    """
    lines = ["# Compound collision proxy — CoACD convex decomposition"]
    vert_offset = 0
    for idx, part in enumerate(parts):
        hull = part.convex_hull  # ensure each part is strictly convex
        lines.append(f"o Part{idx}")
        for v in hull.vertices:
            lines.append(f"v {v[0]:.8g} {v[1]:.8g} {v[2]:.8g}")
        for f in hull.faces:
            a, b, c = int(f[0]) + 1 + vert_offset, int(f[1]) + 1 + vert_offset, int(f[2]) + 1 + vert_offset
            lines.append(f"f {a} {b} {c}")
        vert_offset += len(hull.vertices)
    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


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

    # ── Physics normalisation ─────────────────────────────────────────────────
    # If the mesh is tiny after applying `scale` (e.g. a 600 µm particle at
    # scale=1e-6 → char ≈ 0.666 mm) Genesis will reject it with "Combined mass
    # is less than EPS" and CoACD will fail with "Weights sum to zero".  Scale
    # the mesh up to PHYSICS_NORM_TARGET so physics runs in a stable range.
    # All output positions must be divided by physics_norm before display.
    raw_extents = original_mesh.bounds[1] - original_mesh.bounds[0]
    char_after_scale = float(max(raw_extents[0], raw_extents[1], raw_extents[2], 1e-9))
    physics_norm: float = 1.0
    if char_after_scale < PHYSICS_NORM_THRESHOLD:
        physics_norm = float(PHYSICS_NORM_TARGET) / char_after_scale
        original_mesh.apply_scale(physics_norm)
        print(
            f"Physics normalisation: char={char_after_scale:.4g} m → "
            f"{char_after_scale * physics_norm:.4g} m  (×{physics_norm:.4g})"
        )

    parts = []
    try:
        parts = coacd.run_coacd(original_mesh, max_convex_hull=32)
    except Exception:
        parts = []

    if not parts:
        parts = [original_mesh.convex_hull]

    physics_mesh = trimesh.util.concatenate(parts)

    # Write (or reuse) the compound OBJ collision proxy.  Each CoACD convex hull
    # becomes a separate ``o PartN`` sub-object so Genesis can build a faithful
    # multi-hull collision shape instead of one bloated global convex hull.
    # The proxy vertices are in physics metres (scale × physics_norm already
    # applied) so Genesis must load it with scale=1.0.
    proxy_path = _coacd_proxy_path(filepath, scale * physics_norm)
    try:
        _write_coacd_compound_obj(parts, proxy_path)
        print(f"Wrote CoACD compound proxy ({len(parts)} parts) → {proxy_path}")
    except Exception as exc:
        print(f"Warning: could not write CoACD proxy to {proxy_path!r}: {exc}; falling back to convexify=True")
        proxy_path = None  # caller will fall back to original file + convexify=True

    volume = float(physics_mesh.volume) if physics_mesh.is_watertight else float(physics_mesh.convex_hull.volume)
    print(
        f"{filepath} | verts={len(physics_mesh.vertices)} | faces={len(physics_mesh.faces)} | "
        f"extents={physics_mesh.extents} | volume={volume:.6g} | parts={len(parts)}"
    )
    return (physics_mesh, original_mesh, proxy_path, physics_norm)


def _rigid_material(friction: float, restitution: float, rho: Optional[float] = None):
    """Genesis versions disagree on `restitution` vs `coup_restitution`; support both.

    `rho` is only relevant for rigid-body mass / inertia. FEM/MPM paths pass density
    directly into those material constructors.
    """
    try:
        if rho is None:
            return gs.materials.Rigid(friction=friction, restitution=restitution)
        return gs.materials.Rigid(rho=float(rho), friction=friction, restitution=restitution)
    except TypeError:
        if rho is None:
            return gs.materials.Rigid(friction=friction, coup_restitution=restitution)
        return gs.materials.Rigid(rho=float(rho), friction=friction, coup_restitution=restitution)


def create_environment(scene, kind, plate_size=0.6, cyl_diameter=0.20,
cyl_height=0.30, cyl_segments=32,
wall_thickness=WALL_THICKNESS, plate_wall_height=PLATE_WALL_HEIGHT,
env_restitution: float = ENV_RESTITUTION) -> tuple[set, dict]:
    container_ids = set()
    mat = _rigid_material(0.55, float(env_restitution))

    if kind == "plate":
        t = float(wall_thickness)
        s = float(plate_size)
        h_rim = float(plate_wall_height)
        plate = scene.add_entity(
            gs.morphs.Box(
                size=(s, t, s),
                pos=(0, t / 2, 0),
                fixed=True,
            ),
            material=mat,
        )
        container_ids.add(plate)
        if h_rim > 1e-6:
            y_c = t + h_rim / 2.0
            span = s + 2.0 * t
            for sign in (1.0, -1.0):
                w_n = scene.add_entity(
                    gs.morphs.Box(
                        size=(span, h_rim, t),
                        pos=(0, y_c, sign * (s / 2 + t / 2)),
                        fixed=True,
                    ),
                    material=mat,
                )
                container_ids.add(w_n)
            for sign in (1.0, -1.0):
                w_e = scene.add_entity(
                    gs.morphs.Box(
                        size=(t, h_rim, span),
                        pos=(sign * (s / 2 + t / 2), y_c, 0),
                        fixed=True,
                    ),
                    material=mat,
                )
                container_ids.add(w_e)
            top_y = t + h_rim
        else:
            top_y = t
        env_info = {
            "surface_y": t,
            "top_y": top_y,
            "spread_radius": s / 2,
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


def spawn_particles(
    scene,
    physics_mesh,
    n,
    env_info,
    drop_height,
    drop_spread,
    E,
    nu,
    rho,
    particle_file=PARTICLE_FILE,
    scale_factor=SCALE_FACTOR,
    particle_restitution: float = PARTICLE_RESTITUTION,
    e_fem_max: float = FEM_JAMMING_E_MAX,
    prior_fem_snapshots: Optional[list[dict]] = None,
    coacd_proxy_file: Optional[str] = None,
    physics_norm: float = 1.0,
) -> list:
    E_in = float(E)
    if E_in > 1e8:
        print(
            "Genesis particle solver: Rigid (friction=0.4, restitution=0.0); "
            f"YOUNGS_MODULUS={E_in:.6g} Pa > 1e8 (FEM/MPM path skipped)"
        )
        material = _rigid_material(0.4, 0.0, rho=rho)
    else:
        E = E_in
        if E > float(e_fem_max):
            E = float(e_fem_max)
        if E > 1e3:
            try:
                material = gs.materials.FEM(E=E, nu=nu, rho=rho, use_implicit_solver=True)
            except TypeError:
                try:
                    material = gs.materials.FEM(E=E, nu=nu, rho=rho)
                except TypeError:
                    try:
                        material = gs.materials.FEM.Elastic(E=E, nu=nu, rho=rho, use_implicit_solver=True)
                    except TypeError:
                        material = gs.materials.FEM.Elastic(E=E, nu=nu, rho=rho)
        else:
            try:
                material = gs.materials.MPM(E=E, nu=nu, rho=rho)
            except TypeError:
                material = gs.materials.MPM.Elastic(E=E, nu=nu, rho=rho)

    extents = physics_mesh.bounds[1] - physics_mesh.bounds[0]
    spawn_y0 = float(env_info["top_y"] + drop_height)
    spread = float(drop_spread)
    char = float(max(float(extents[0]), float(extents[1]), float(extents[2]), 1e-9))
    stack_gap = max(char * 1.5, 1e-4)

    prior = prior_fem_snapshots or []
    if prior and len(prior) != n - 1:
        raise ValueError(f"prior_fem_snapshots must have length n-1 ({n - 1}), got {len(prior)}")

    def _centroid_from_snapshot(snap: dict) -> tuple[float, float, float]:
        p = snap["pos"]
        arr = np.asarray(p, dtype=float)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim != 2 or arr.shape[-1] != 3:
            return (0.0, float(spawn_y0), 0.0)
        c = arr.mean(axis=0)
        return (float(c[0]), float(c[1]), float(c[2]))

    # Choose the collision file and Genesis scale:
    #   Proxy path: vertices are already in physics metres (display_scale ×
    #     physics_norm applied in load_particle_mesh) → Genesis scale = 1.0.
    #   Fallback path: original OBJ vertices are in source units (e.g. µm) →
    #     Genesis scale = display_scale × physics_norm to reach physics metres.
    if coacd_proxy_file is not None:
        collision_file = coacd_proxy_file
        use_convexify = False
        collision_scale = 1.0
    else:
        collision_file = particle_file
        use_convexify = True
        collision_scale = float(scale_factor) * float(physics_norm)

    entities = []
    if spread <= 1e-9:
        # Single column above the plate: stack along +Y so bodies do not share one point (that breaks FEM contact).
        for i in range(n):
            if i < len(prior):
                x, y, z = _centroid_from_snapshot(prior[i])
                quat = gs.utils.geom.R_to_quat(np.eye(3, dtype=float))
            else:
                x, z = 0.0, 0.0
                # Stagger each new particle above the previous; overlap at one (x,z) caused tunneling / blow-ups.
                y = spawn_y0 + float(i) * stack_gap
                R = trimesh.transformations.random_rotation_matrix()[:3, :3]
                quat = gs.utils.geom.R_to_quat(R)
            ent = scene.add_entity(
                gs.morphs.Mesh(
                    file=collision_file,
                    scale=collision_scale,
                    pos=(x, y, z),
                    quat=quat,
                    convexify=use_convexify,
                    collision=True,
                    visualization=False,
                ),
                material=material,
            )
            entities.append(ent)
        print(
            f"Spawned {len(entities)} particles (column at x=z=0, Δy={stack_gap:.4g} m between centers) | "
            f"drop_height={drop_height} | mesh_char={char:.6g} | y0={spawn_y0:.6g}"
            + (" | sequential_restore" if prior else "")
        )
    else:
        if "spread_radius" in env_info:
            r_max = float(env_info["spread_radius"]) * spread
        elif "inner_radius" in env_info:
            r_max = float(env_info["inner_radius"]) * spread
        else:
            r_max = 0.15 * spread
        r_max = max(r_max, 1e-6)

        # Vogel disk on the horizontal plane — loose pack that settles into contacts.
        golden = math.pi * (3.0 - math.sqrt(5.0))
        # Minimum spawn separation: 1.5× largest extent, matching the column stack_gap.
        # Accounts for random rotations where a concave tip can reach char/2 beyond the centroid.
        min_sep_spawn = max(char * 1.5, 1e-4)
        min_sep_spawn_sq = min_sep_spawn * min_sep_spawn
        placed_positions: list[tuple[float, float, float]] = []

        for i in range(n):
            if i < len(prior):
                x, y, z = _centroid_from_snapshot(prior[i])
                quat = gs.utils.geom.R_to_quat(np.eye(3, dtype=float))
            else:
                ri = r_max * math.sqrt((i + 0.5) / max(n, 1))
                th = i * golden
                x = ri * math.cos(th)
                z = ri * math.sin(th)
                y = spawn_y0
                # Ensure the new spawn position is at least min_sep_spawn away from every
                # already-placed particle. If XZ proximity forces overlap, push Y upward.
                for px, py, pz in placed_positions:
                    xz_sq = (x - px) ** 2 + (z - pz) ** 2
                    if xz_sq < min_sep_spawn_sq:
                        dy_needed = math.sqrt(max(min_sep_spawn_sq - xz_sq, 0.0))
                        y = max(y, py + dy_needed)
                R = trimesh.transformations.random_rotation_matrix()[:3, :3]
                quat = gs.utils.geom.R_to_quat(R)
            placed_positions.append((x, y, z))
            ent = scene.add_entity(
                gs.morphs.Mesh(
                    file=collision_file,
                    scale=collision_scale,
                    pos=(x, y, z),
                    quat=quat,
                    convexify=use_convexify,
                    collision=True,
                    visualization=False,
                ),
                material=material,
            )
            entities.append(ent)

        print(
            f"Spawned {len(entities)} particles (Vogel disk, r≤{r_max:.4g} m) | drop_height={drop_height} | "
            f"mesh_char={char:.6g} | y={spawn_y0:.6g}"
            + (" | sequential_restore" if prior else "")
        )
    return entities


def run_simulation(scene, entities, dt, substeps, duration, settle_threshold, *, update_visualizer: bool = False) -> int:
    total_steps = int(duration / dt)
    for step in range(total_steps):
        scene.step(update_visualizer=update_visualizer)
        if step % 60 == 0:
            max_vel = compute_max_velocity(entities) if entities else None
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


def compute_total_kinetic_energy(entities, particle_mass_kg: float) -> float:
    """
    Total translational kinetic energy (J) summed over particles.

    FEM / MPM: if entity state exposes per-node velocities (N,3), uses uniform nodal mass M/N.
    Otherwise falls back to rigid-style get_vel with full particle mass M.
    """
    m_part = float(particle_mass_kg)
    if m_part <= 0.0 or not entities:
        return 0.0
    total = 0.0
    for e in entities:
        used = False
        if hasattr(e, "get_state"):
            st = e.get_state()
            if hasattr(st, "vel"):
                v = _tensor_to_numpy(st.vel).astype(float)
                if v.size == 0:
                    continue
                if v.ndim == 3:
                    v = v[0]
                if v.ndim == 2 and v.shape[-1] == 3 and v.shape[0] > 0:
                    nv = int(v.shape[0])
                    mn = m_part / max(nv, 1)
                    s2 = np.sum(v * v, axis=-1)
                    total += float(0.5 * mn * np.sum(s2))
                    used = True
                elif v.ndim == 1 and v.size >= 3:
                    spd = float(np.linalg.norm(v[:3]))
                    total += 0.5 * m_part * spd * spd
                    used = True
        if not used and hasattr(e, "get_vel"):
            total += _safe_ke(e, m_part)
    return float(total)


def _safe_ke(e, mass_kg: float) -> float:
    """Translational KE from rigid-style velocity; Genesis rigid vs FEM APIs differ."""
    try:
        vel = e.get_vel()
        v = _as_vec3_any(vel)
        speed_sq = float(np.dot(v, v))
        return 0.5 * max(mass_kg, 1e-6) * speed_sq
    except Exception:
        return 0.0


def container_surface_area_m2(environment_type: str, cfg: dict) -> float:
    """
    Inner container surface area (m²) for system pressure: sum(|F_contact|) / area → Pa.

    Plate: horizontal floor + inner rim (four sides). Cylinder: inner bottom disk + inner cylindrical wall.
    """
    k = str(environment_type).strip().lower()
    if k == "plate":
        s = float(cfg.get("PLATE_SIZE", PLATE_SIZE))
        h = float(cfg.get("PLATE_WALL_HEIGHT", PLATE_WALL_HEIGHT))
        floor = s * s
        rim = 4.0 * s * max(h, 0.0)
        return max(floor + rim, 1e-18)
    if k == "cylinder":
        d = float(cfg.get("CYLINDER_DIAMETER", CYLINDER_DIAMETER))
        h = float(cfg.get("CYLINDER_HEIGHT", CYLINDER_HEIGHT))
        r = d * 0.5
        return max(math.pi * r * r + 2.0 * math.pi * r * h, 1e-18)
    return 1.0


def _pp_contact_strong_for_z(c: NormalizedContact, f_min: float) -> bool:
    if not c.is_particle_particle:
        return False
    if c.force is None:
        return False
    return abs(float(c.force)) > float(f_min)


def _pp_contact_for_z_graph(c: NormalizedContact, depth_tol: float = CONTACT_DEPTH_TOL) -> bool:
    """P–P link for coordination Z: depth only (contacts list is already resampled; threshold matches extraction)."""
    if not c.is_particle_particle:
        return False
    return abs(float(c.depth)) > float(depth_tol)


def calculate_live_metrics(
    contacts,
    particle_ids,
    entities,
    *,
    particle_mass_kg: float,
    surface_area_m2: Optional[float] = None,
    depth_tol: float = CONTACT_DEPTH_TOL,
) -> dict:
    """
    Lightweight metrics for high-frequency WebSocket updates (jamming / rattlers / energy).

    ``n_rattlers`` counts particles with no particle–particle contacts (isolated in the PP graph),
    matching ``compute_metrics``'s ``n_isolated_particles``.
    """
    m = compute_metrics(contacts, particle_ids, container_surface_area_m2=surface_area_m2, depth_tol=depth_tol)
    ke = compute_total_kinetic_energy(entities, particle_mass_kg)
    return {
        "Z": float(m.get("Z", 0.0)),
        "n_rattlers": int(m.get("n_isolated_particles", 0)),
        "kinetic_energy": float(ke),
        "system_pressure": float(m.get("system_pressure", 0.0)),
    }


def compute_max_velocity(entities) -> Optional[float]:
    speeds: list[float] = []
    for e in entities:
        if hasattr(e, "get_vel"):
            v = _tensor_to_numpy(e.get_vel())
            s = float(np.linalg.norm(v))
            if math.isfinite(s):
                speeds.append(s)
            continue
        if hasattr(e, "get_state"):
            st = e.get_state()
            if hasattr(st, "vel"):
                v = _tensor_to_numpy(st.vel)
                if v.size == 0:
                    continue
                if v.ndim == 1:
                    s = float(np.linalg.norm(v))
                else:
                    s = float(np.nanmax(np.linalg.norm(v, axis=-1)))
                if math.isfinite(s):
                    speeds.append(s)
            continue
    if not speeds:
        return None
    return max(speeds)


def compute_fem_vertex_force_stress(
    scene, entities, *, subsample_frac: float = 0.2
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], float]:
    """
    Per-vertex "stress proxy" from FEM nodal forces: ||F_i|| on a random nodal subsample.

    Only a fraction of vertices per FEM body are sampled each frame (default 20%) to limit
    CPU/GPU sync and payload size. Norms are normalized to [0, 1] as ||F_i|| / G_max using
    G_max = max sampled ||F|| across all FEM bodies (torch, on the force tensor's device).

    Returns (entity_id -> (local_vertex_indices int32, normalized float32 array)), and 1.0
    for fem_norm_global_max (values are already scaled).
    """
    fs = getattr(scene.sim, "fem_solver", None)
    if fs is None or not getattr(fs, "is_active", False):
        return {}, 1.0
    node_forces = fs.get_forces()
    if node_forces is None:
        return {}, 1.0
    if hasattr(node_forces, "detach"):
        F = node_forces.detach()
    else:
        F = torch.from_numpy(np.asarray(node_forces, dtype=np.float32))
    if F.ndim != 3 or int(F.shape[0]) < 1:
        return {}, 1.0
    forces_b = F[0]
    if forces_b.ndim != 2 or int(forces_b.shape[1]) != 3:
        return {}, 1.0
    device = forces_b.device
    dtype = forces_b.dtype
    sampled_blocks: list[torch.Tensor] = []
    meta: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    frac = float(subsample_frac)
    if not math.isfinite(frac) or frac <= 0.0:
        frac = 0.2

    for e in entities:
        if type(e).__name__ != "FEMEntity":
            continue
        if not hasattr(e, "v_start") or not hasattr(e, "n_vertices"):
            continue
        vs = int(e.v_start)
        nv = int(e.n_vertices)
        if nv < 1 or vs + nv > int(forces_b.shape[0]):
            continue
        slab = forces_b[vs : vs + nv, :]
        k = max(1, int(nv * frac))
        pick = torch.randperm(nv, device=device)[:k]
        f_k = slab[pick, :]
        norms_k = torch.linalg.norm(f_k, dim=1)
        norms_k = torch.nan_to_num(norms_k, nan=0.0, posinf=0.0, neginf=0.0).to(dtype)
        sampled_blocks.append(norms_k)
        meta.append((int(_entity_id(e)), pick.to(dtype=torch.int64), norms_k))

    if not sampled_blocks:
        return {}, 1.0
    all_norms = torch.cat(sampled_blocks)
    gmax = torch.max(all_norms)
    if not torch.isfinite(gmax) or float(gmax.item()) <= 1e-18:
        gmax_t = torch.tensor(1.0, device=device, dtype=dtype)
    else:
        gmax_t = torch.clamp(gmax, min=1e-18)

    per_entity: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for eid, pick_i, norms_k in meta:
        scaled = (norms_k / gmax_t).clamp(0.0, 1.0)
        per_entity[eid] = (
            pick_i.cpu().numpy().astype(np.int32, copy=False),
            scaled.cpu().numpy().astype(np.float32, copy=False),
        )
    return per_entity, 1.0


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
    fem_vertex_norms: Optional[dict[int, tuple[np.ndarray, np.ndarray] | np.ndarray]] = None,
    fem_norm_global_max: float = 1.0,
) -> list[dict]:
    out: list[dict] = []
    smap = stress_map or {}
    gmax = float(fem_norm_global_max)
    if not math.isfinite(gmax) or gmax <= 1e-18:
        gmax = 1.0
    for e in entities:
        pos, (qx, qy, qz, qw) = _entity_pose(e)
        if not (np.isfinite(pos).all() and all(np.isfinite([qx, qy, qz, qw]))):
            pos = np.zeros((3,), dtype=float)
            qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
        eid = int(_entity_id(e))
        contact_s = float(smap.get(eid, 0.0))
        vertex_intensities: Optional[list[float]] = None
        vertex_stress_indices: Optional[list[int]] = None
        stress_intensity = contact_s
        if fem_vertex_norms and eid in fem_vertex_norms:
            entry = fem_vertex_norms[eid]
            if isinstance(entry, tuple) and len(entry) == 2:
                idx_a, raw = entry
                idx_a = np.asarray(idx_a, dtype=np.int64)
                raw = np.nan_to_num(np.asarray(raw, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
                arr = np.clip(raw / gmax, 0.0, 1.0).astype(np.float32, copy=False)
                stress_intensity = float(np.mean(arr)) if arr.size else contact_s
                vertex_stress_indices = idx_a.astype(np.int64, copy=False).tolist()
                vertex_intensities = arr.tolist()
            else:
                raw = np.nan_to_num(np.asarray(entry, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
                arr = np.clip(raw / gmax, 0.0, 1.0).astype(np.float32, copy=False)
                stress_intensity = float(np.mean(arr)) if arr.size else contact_s
                vertex_intensities = arr.tolist()
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
        if vertex_intensities is not None:
            row["vertex_intensities"] = vertex_intensities
        if vertex_stress_indices is not None:
            row["vertex_stress_indices"] = vertex_stress_indices
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
    update_visualizer: bool = False,
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

        scene.step(update_visualizer=update_visualizer)
        current_t = float((step + 1) * dt)

        if (step % log_every == 0) or (step == total_steps - 1):
            max_vel = compute_max_velocity(entities) if entities else None

            line = f"t={current_t:.2f}s  max_vel={(max_vel if max_vel is not None else 0.0):.5f} m/s  step={step + 1}/{total_steps}"
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
                position=_as_vec3_any(_contact_pos(c)),
                normal=_as_vec3_any(c.normal),
                depth=float(c.depth),
                force=force,
                contact_area=area,
            )
        )
    return out


def extract_contacts_geometric(
    entities: list,
    particle_ids: set,
    container_entities: list,
    original_mesh,
    depth_tol: float = 1e-5,
) -> list:
    if not entities or original_mesh is None:
        return []
    extents = original_mesh.bounds[1] - original_mesh.bounds[0]
    r = float(np.linalg.norm(extents)) / 2.0
    positions = {}
    for e in entities:
        eid = _entity_id(e)
        if eid not in particle_ids:
            continue
        pos, _ = _entity_pose(e)
        positions[eid] = pos.copy()
    container_ids = {_entity_id(e) for e in container_entities}
    contacts = []
    pids = list(positions.keys())
    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            a, b = pids[i], pids[j]
            pa, pb = positions[a], positions[b]
            dist = float(np.linalg.norm(pa - pb))
            touch_dist = r * 2.0
            if dist < touch_dist:
                depth = touch_dist - dist
                if depth < depth_tol:
                    continue
                normal = (pb - pa) / max(dist, 1e-9)
                contact_pos = (pa + pb) / 2.0
                contacts.append(NormalizedContact(
                    entity_a=a, entity_b=b,
                    is_particle_particle=True, is_particle_container=False,
                    position=contact_pos, normal=normal,
                    depth=depth,
                    force=min(float(depth) * 1e4, 1.0),
                    contact_area=None,
                ))
    for e in container_entities:
        ceid = _entity_id(e)
        pos_c, _ = _entity_pose(e)
        floor_y = float(pos_c[1])
        for pid, ppos in positions.items():
            depth = (floor_y + r) - float(ppos[1])
            if depth > depth_tol:
                contacts.append(NormalizedContact(
                    entity_a=pid, entity_b=ceid,
                    is_particle_particle=False, is_particle_container=True,
                    position=np.array([ppos[0], floor_y, ppos[2]]),
                    normal=np.array([0.0, 1.0, 0.0]),
                    depth=depth,
                    force=min(float(depth) * 1e4, 1.0),
                    contact_area=None,
                ))
    return contacts


def compute_metrics(contacts, particle_ids, *, container_surface_area_m2: Optional[float] = None, z_force_min_n: float = Z_CONTACT_FORCE_MIN_N, depth_tol: float = CONTACT_DEPTH_TOL) -> dict:
    pp = [c for c in contacts if c.is_particle_particle]
    pc = [c for c in contacts if c.is_particle_container]
    f_min = float(z_force_min_n)

    G = nx.Graph()
    for pid in particle_ids:
        G.add_node(int(pid))
    for c in pp:
        if not _pp_contact_for_z_graph(c, depth_tol):
            continue
        G.add_edge(
            int(c.entity_a),
            int(c.entity_b),
            depth=float(c.depth),
            force=c.force,
            area=c.contact_area,
        )

    n = len(particle_ids) if particle_ids else 0
    pp_strong_n = sum(1 for c in pp if _pp_contact_strong_for_z(c, f_min))
    pp_z_n = sum(1 for c in pp if _pp_contact_for_z_graph(c, depth_tol))
    Z = (2 * pp_z_n / n) if n else 0.0
    contact_counts = dict(G.degree())
    n_isolated = sum(1 for pid in particle_ids if contact_counts.get(int(pid), 0) == 0)
    n_container_touching = len({int(c.entity_a) for c in pc if int(c.entity_a) in particle_ids}.union(
        {int(c.entity_b) for c in pc if int(c.entity_b) in particle_ids}
    ))

    contact_points = [_as_vec3_any(c.position) for c in contacts]
    contact_normals = [_as_vec3_any(c.normal) for c in contacts]
    contact_depths = [float(c.depth) for c in contacts]
    contact_forces = [float(c.force) for c in contacts if c.force is not None]
    contact_areas = [float(c.contact_area) for c in contacts if c.contact_area is not None]

    total_contact_force_sum = 0.0
    for c in contacts:
        if c.force is None:
            continue
        total_contact_force_sum += abs(float(c.force))
    area = float(container_surface_area_m2) if container_surface_area_m2 is not None else 0.0
    system_pressure = (total_contact_force_sum / area) if (area > 0.0 and math.isfinite(area)) else 0.0

    return {
        "total_pp_contacts": len(pp),
        "total_pp_contacts_strong": pp_strong_n,
        "total_pc_contacts": len(pc),
        "avg_contacts_per_particle": Z,
        "Z": Z,
        "Z_force_threshold_N": f_min,
        "n_isolated_particles": n_isolated,
        "n_container_touching": n_container_touching,
        "total_contact_force_sum": float(total_contact_force_sum),
        "container_surface_area_m2": float(area) if area > 0.0 else None,
        "system_pressure": float(system_pressure),
        "contact_points": contact_points,
        "contact_normals": contact_normals,
        "contact_depths": contact_depths,
        "contact_forces": contact_forces,
        "contact_areas": contact_areas,
        "contact_graph": G,
        "contact_graph_dict": nx.to_dict_of_dicts(G),
        "contact_counts_per_particle": contact_counts,
    }


def verify_contact_transform(entity, contact_point_world: np.ndarray, mesh_vertices: np.ndarray) -> None:
    """
    Diagnostic: print nearest-vertex distance after transforming a world-space contact point
    into the entity's local frame.  If nearest dist >> bounding_radius, the quaternion
    convention is wrong (world_verts are landing far from the contact site).

    Call this temporarily to confirm Bug 2 (touching particles staying blue):
        for c in contacts:
            if int(c.entity_a) == some_eid or int(c.entity_b) == some_eid:
                verify_contact_transform(entity, c.position, original_mesh.vertices)
                break
    """
    pos, quat_xyzw = _entity_pose(entity)
    qx, qy, qz, qw = quat_xyzw
    R = Rotation.from_quat(np.array([qx, qy, qz, qw], dtype=np.float64))
    # Transform contact point from world to local frame.
    local_pt = R.inv().apply(np.asarray(contact_point_world, dtype=np.float64) - pos)
    dists = np.linalg.norm(np.asarray(mesh_vertices, dtype=np.float64) - local_pt, axis=1)
    extents = np.asarray(mesh_vertices).max(axis=0) - np.asarray(mesh_vertices).min(axis=0)
    bounding_r = float(np.linalg.norm(extents)) / 2.0
    nearest = float(dists.min())
    status = "OK" if nearest <= bounding_r * 1.5 else "QUAT MISMATCH — contact outside mesh!"
    print(
        f"verify_contact_transform | entity={_entity_id(entity)} | "
        f"nearest_vertex_dist={nearest:.4f}m | bounding_r={bounding_r:.4f}m | {status}"
    )


def _accumulate_stress_for_contacts(
    contacts: list,
    eid: int,
    world_verts: np.ndarray,
    n_world: np.ndarray,
    n_verts: int,
    denom: float,
) -> np.ndarray:
    """Accumulate Gaussian stress contributions for one particle from a filtered contact list."""
    stress = np.zeros(n_verts, dtype=np.float64)
    for c in contacts:
        p_w = np.asarray(c.position, dtype=np.float64).ravel()[:3]
        if c.force is not None:
            fm = abs(float(c.force))
        else:
            fm = float(c.depth) * 1e6
        if not math.isfinite(fm):
            fm = 0.0
        vec = p_w.reshape(1, 3) - world_verts
        dist = np.linalg.norm(vec, axis=1)
        dist = np.maximum(dist, 1e-6)
        vec_n = vec / dist.reshape(-1, 1)
        cos_t = np.clip(np.sum(vec_n * n_world, axis=1), -1.0, 1.0)
        angle = np.arccos(cos_t)
        stress += fm * np.exp(-(angle**2) / denom)
    return stress


def _normalize_stress_map(raw: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Normalize a {eid: stress_array} dict to [0, 1] by global max. Returns zero arrays unchanged."""
    global_max = max((float(np.max(arr)) for arr in raw.values() if arr.size), default=0.0)
    if global_max <= 0.0:
        return {eid: np.zeros_like(arr) for eid, arr in raw.items()}
    inv = 1.0 / global_max
    return {eid: arr * inv for eid, arr in raw.items()}


def compute_vertex_stress(
    entities: list,
    contacts: list,
    original_mesh,
    particle_ids: set,
    sigma: float = 0.12,
    stress_floor: float = 0.05,
) -> dict[int, list[float]]:
    """
    Per-vertex Hertzian-style stress proxy from contact positions and forces.

    PP and PC contacts are normalized independently then blended (PC at 40% weight)
    so particle-particle contacts remain visible even when container forces dominate.
    A hard floor clamps post-normalization noise to exactly 0 on non-contact regions.
    """
    if original_mesh is None or len(original_mesh.vertices) == 0:
        return {}

    verts_local = np.asarray(original_mesh.vertices, dtype=np.float64)
    n_verts = int(verts_local.shape[0])
    vn = np.asarray(original_mesh.vertex_normals, dtype=np.float64)
    if vn.shape != (n_verts, 3):
        m = original_mesh.copy()
        vn = np.asarray(m.vertex_normals, dtype=np.float64)
    if vn.shape != (n_verts, 3):
        vn = np.zeros((n_verts, 3), dtype=np.float64)
        vn[:, 1] = 1.0

    sig = float(sigma)
    if not math.isfinite(sig) or sig <= 0.0:
        sig = 0.12
    denom = 2.0 * sig**2

    floor = float(stress_floor)
    if not math.isfinite(floor) or floor < 0.0:
        floor = 0.05

    pp_raw: dict[int, np.ndarray] = {}
    pc_raw: dict[int, np.ndarray] = {}

    for e in entities:
        eid = _entity_id(e)
        if eid not in particle_ids:
            continue
        pos, quat_xyzw = _entity_pose(e)
        qx, qy, qz, qw = quat_xyzw
        R = Rotation.from_quat(np.array([qx, qy, qz, qw], dtype=np.float64)).as_matrix()
        world_verts = (R @ verts_local.T).T + pos.reshape(1, 3)
        n_world = (R @ vn.T).T
        norms = np.linalg.norm(n_world, axis=1, keepdims=True)
        n_world = n_world / np.maximum(norms, 1e-12)

        pp_rel = [
            c for c in contacts
            if (int(c.entity_a) == eid or int(c.entity_b) == eid)
            and bool(c.is_particle_particle)
        ]
        pc_rel = [
            c for c in contacts
            if (int(c.entity_a) == eid or int(c.entity_b) == eid)
            and bool(c.is_particle_container)
        ]

        pp_raw[eid] = _accumulate_stress_for_contacts(pp_rel, eid, world_verts, n_world, n_verts, denom)
        pc_raw[eid] = _accumulate_stress_for_contacts(pc_rel, eid, world_verts, n_world, n_verts, denom)

    pp_norm = _normalize_stress_map(pp_raw)
    pc_norm = _normalize_stress_map(pc_raw)

    out: dict[int, list[float]] = {}
    all_eids = set(pp_norm) | set(pc_norm)

    if not all_eids:
        contact_counts: dict[int, int] = {}
        for c in contacts:
            contact_counts[int(c.entity_a)] = contact_counts.get(int(c.entity_a), 0) + 1
            contact_counts[int(c.entity_b)] = contact_counts.get(int(c.entity_b), 0) + 1
        max_count = max(contact_counts.values()) if contact_counts else 0
        for eid in particle_ids:
            t = float(contact_counts.get(int(eid), 0)) / max_count if max_count > 0 else 0.0
            out[int(eid)] = [t] * n_verts
        return out

    for eid in all_eids:
        pp_arr = pp_norm.get(eid, np.zeros(n_verts, dtype=np.float64))
        pc_arr = pc_norm.get(eid, np.zeros(n_verts, dtype=np.float64))
        blended = np.maximum(pp_arr, pc_arr * 0.4)
        blended = np.where(blended < floor, 0.0, blended)
        out[int(eid)] = blended.tolist()
    return out


def contact_efficiency(metrics, entities, mesh) -> float:
    return metrics["total_pp_contacts"] / (mesh.volume * len(entities))


def weighted_contact_efficiency(metrics, entities, mesh) -> float:
    total_force = sum(f for f in metrics["contact_forces"] if f)
    return total_force / (mesh.volume * len(entities))


def export_results(entities, metrics, output_dir, save_hdf5, save_csv, vertex_stress: Optional[dict[int, list[float]]] = None):
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
                "force": float(data["force"])
                if (data.get("force") is not None and math.isfinite(float(data["force"])))
                else float(data.get("depth", 0.0)) * 1e5,
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

    if save_csv and vertex_stress:
        vs_path = os.path.join(output_dir, "vertex_stress.csv")
        vs_rows: list[dict[str, float | int]] = []
        for eid, vals in vertex_stress.items():
            for vi, s in enumerate(vals):
                vs_rows.append({"particle_id": int(eid), "vertex_index": int(vi), "stress": float(s)})
        if vs_rows:
            pd.DataFrame(vs_rows).to_csv(vs_path, index=False)
            print(f"Wrote CSV: {vs_path}")

    if save_hdf5:
        h5_path = os.path.join(output_dir, "results.h5")
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("particle_positions", data=df_particles[["x", "y", "z"]].to_numpy(dtype=float))
            f.create_dataset("contact_points", data=pts)
            f.create_dataset("contact_normals", data=nrm)
            f.attrs["Z"] = float(metrics["Z"])
            f.attrs["total_pp"] = int(metrics["total_pp_contacts"])
            f.attrs["total_pp_strong"] = int(metrics.get("total_pp_contacts_strong", 0))
            f.attrs["total_pc"] = int(metrics["total_pc_contacts"])
            f.attrs["n_isolated"] = int(metrics["n_isolated_particles"])
            f.attrs["n_container_touch"] = int(metrics["n_container_touching"])
            f.attrs["system_pressure"] = float(metrics.get("system_pressure", 0.0))
            if vertex_stress is not None:
                grp = f.create_group("vertex_stress")
                for eid, vals in vertex_stress.items():
                    grp.create_dataset(str(eid), data=np.array(vals, dtype=np.float32))
        print(f"Wrote HDF5: {h5_path}")

    print("── Contact Analysis Summary ─────────────────────────────────")
    print(f"Particles simulated:               {len(entities)}")
    print(f"Total particle-particle contacts:  {metrics['total_pp_contacts']}")
    print(f"Strong PP (|F|>{metrics.get('Z_force_threshold_N', Z_CONTACT_FORCE_MIN_N):.0e} N): {metrics.get('total_pp_contacts_strong', 0)}")
    print(f"Total particle-container contacts: {metrics['total_pc_contacts']}")
    print(f"Avg contacts per particle (Z):     {metrics['Z']:.3f}  (depth-filtered PP graph, |depth|>{CONTACT_DEPTH_TOL})")
    print(f"Isolated particles (Z=0):          {metrics['n_isolated_particles']}")
    print(f"Particles touching container:      {metrics['n_container_touching']}")
    print(f"System pressure (Σ|F|/A):          {float(metrics.get('system_pressure', 0.0)):.4f} Pa")
    if metrics["contact_forces"]:
        forces = np.asarray(metrics["contact_forces"], dtype=float)
        print(f"Mean contact force:                {forces.mean():.4f} N")
        print(f"Max  contact force:                {forces.max():.4f} N")
    if metrics["contact_areas"]:
        areas = np.asarray(metrics["contact_areas"], dtype=float)
        print(f"Mean contact area:                 {(areas.mean() * 1e6):.4f} mm²")
    print("────────────────────────────────────────────────────────────")


def export_settled_obj(entities, original_mesh) -> str:
    """
    Export all settled particles as a single OBJ string.

    Each particle is written as a named group (o particle_0 … o particle_N).
    Vertex positions and normals are transformed to world space using each
    particle's settled pose.  Face indices use a cumulative 1-based offset so
    the file can be imported as a single mesh or split per group.

    Units: metres, Y-up coordinate system.
    """
    lines = [
        "# Granular jamming simulation – settled particle geometry",
        "# Coordinate system: Y-up, metres",
        "# Groups: one named group per particle (particle_0, particle_1, …)",
        "",
    ]

    verts_base = np.asarray(original_mesh.vertices, dtype=float)
    faces_base = np.asarray(original_mesh.faces, dtype=int)

    has_normals = hasattr(original_mesh, "vertex_normals") and original_mesh.vertex_normals is not None
    normals_base = np.asarray(original_mesh.vertex_normals, dtype=float) if has_normals else np.zeros_like(verts_base)

    v_offset = 0
    for i, e in enumerate(entities):
        pos, (qx, qy, qz, qw) = _entity_pose(e)
        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        world_verts = (R @ verts_base.T).T + pos
        world_normals = (R @ normals_base.T).T if has_normals else normals_base

        lines.append(f"o particle_{i}")
        for vx, vy, vz in world_verts:
            lines.append(f"v {vx:.8f} {vy:.8f} {vz:.8f}")
        if has_normals:
            for nx, ny, nz in world_normals:
                lines.append(f"vn {nx:.8f} {ny:.8f} {nz:.8f}")
        for face in faces_base:
            i0 = face[0] + 1 + v_offset
            i1 = face[1] + 1 + v_offset
            i2 = face[2] + 1 + v_offset
            if has_normals:
                lines.append(f"f {i0}//{i0} {i1}//{i1} {i2}//{i2}")
            else:
                lines.append(f"f {i0} {i1} {i2}")
        v_offset += len(verts_base)
        lines.append("")

    return "\n".join(lines)


def export_contact_network_obj(entities, contact_graph_links: list[dict]) -> str:
    """
    Export the PP contact network as OBJ line segments.

    Each vertex is a particle centre; each 'l' line connects two contacting
    particle centres.  Import as a separate layer in Rhino or Blender.

    Units: metres, Y-up coordinate system.
    """
    lines = [
        "# Granular jamming simulation – particle–particle contact network",
        "# Coordinate system: Y-up, metres",
        "# l edges connect centres of contacting particles",
        "",
        "o contact_network",
    ]

    # Map entity id → 1-based OBJ vertex index
    id_to_idx: dict[int, int] = {}
    for idx, e in enumerate(entities, start=1):
        eid = _entity_id(e)
        pos, _ = _entity_pose(e)
        lines.append(f"v {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}")
        id_to_idx[eid] = idx

    lines.append("")
    for link in contact_graph_links:
        a = int(link.get("source", -1))
        b = int(link.get("target", -1))
        if a in id_to_idx and b in id_to_idx:
            lines.append(f"l {id_to_idx[a]} {id_to_idx[b]}")

    return "\n".join(lines)


def export_summary_json(entities, metrics: dict, output_dir: str, original_mesh=None) -> dict:
    """
    Build and persist a summary.json containing scalar simulation metrics.

    Returns the dict so the caller can also stream it directly.
    """
    n = len(entities)
    total_pp = int(metrics.get("total_pp_contacts", 0))
    total_pc = int(metrics.get("total_pc_contacts", 0))

    vol_m3 = 0.0
    if original_mesh is not None:
        try:
            vol_m3 = float(original_mesh.volume) if original_mesh.is_watertight else float(original_mesh.convex_hull.volume)
        except Exception:
            vol_m3 = 0.0

    contact_eff = 0.0
    if vol_m3 > 0 and n > 0:
        contact_eff = total_pp / (vol_m3 * n)

    summary = {
        "n_particles": n,
        "Z": float(metrics.get("Z", 0.0)),
        "total_pp": total_pp,
        "total_pc": total_pc,
        "n_isolated": int(metrics.get("n_isolated_particles", 0)),
        "n_container_touch": int(metrics.get("n_container_touching", 0)),
        "system_pressure": float(metrics.get("system_pressure", 0.0)),
        "contact_efficiency": float(contact_eff),
        "total_particle_volume": float(vol_m3 * n),
        "single_particle_volume_m3": float(vol_m3),
    }

    os.makedirs(output_dir, exist_ok=True)
    import json
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    return summary


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
        px, py, pz = e.get_pos()
        pos = _vec3_from_xyz(px, py, pz)
        qw, qx, qy, qz = e.get_quat()
        qx, qy, qz, qw = _quat_tuple_xyzw(qw, qx, qy, qz)
        c = int(counts.get(_entity_id(e), 0))
        t = c / max_c

        poly = base_poly.copy(deep=True)
        rot = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        verts = (rot @ poly.points.T).T + pos
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

    sim_options = make_sim_options(
        gs,
        {"DT": args.dt, "SUBSTEPS": args.substeps, "GRAVITY": GRAVITY},
    )
    rigid_options = make_rigid_options(gs)
    fem_options = make_fem_options(gs, {"FEM_NEWTON_ITERATIONS": int(DEFAULT_CONFIG.get("FEM_NEWTON_ITERATIONS", 4))})

    # Avoid building the visualizer unless explicitly requested.
    # This prevents the viewer from throttling the run (e.g., ~0.1 FPS on CPU).
    scene = gs.Scene(
        sim_options=sim_options,
        rigid_options=rigid_options,
        fem_options=fem_options,
        show_viewer=bool(args.show_viewer),
    )

    physics_mesh, original_mesh, coacd_proxy_file, physics_norm = load_particle_mesh(PARTICLE_FILE, SCALE_FACTOR)
    container_ids, env_info = create_environment(
        scene,
        ENVIRONMENT_TYPE,
        PLATE_SIZE,
        CYLINDER_DIAMETER,
        CYLINDER_HEIGHT,
        CYLINDER_SEGMENTS,
        WALL_THICKNESS,
        PLATE_WALL_HEIGHT,
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
        coacd_proxy_file=coacd_proxy_file,
        physics_norm=physics_norm,
    )
    particle_ids = {_entity_id(e) for e in entities}
    scene.build()
    run_simulation(
        scene,
        entities,
        args.dt,
        args.substeps,
        args.duration,
        SETTLE_THRESHOLD,
        update_visualizer=bool(args.show_viewer),
    )
    contacts = extract_contacts(scene, particle_ids, container_ids, CONTACT_DEPTH_TOL)
    metrics = compute_metrics(
        contacts,
        particle_ids,
        container_surface_area_m2=container_surface_area_m2(ENVIRONMENT_TYPE, DEFAULT_CONFIG),
    )
    vertex_stress = compute_vertex_stress(
        entities,
        contacts,
        original_mesh,
        particle_ids,
        sigma=float(DEFAULT_CONFIG.get("STRESS_SIGMA", 0.12)),
        stress_floor=float(DEFAULT_CONFIG.get("STRESS_FLOOR", 0.05)),
    )
    export_results(
        entities,
        metrics,
        OUTPUT_DIR,
        save_hdf5=(SAVE_HDF5 and (not args.no_hdf5)),
        save_csv=(SAVE_CSV and (not args.no_csv)),
        vertex_stress=vertex_stress,
    )
    if args.show_viewer:
        visualize_results(entities, metrics, original_mesh)


if __name__ == "__main__":
    main()
