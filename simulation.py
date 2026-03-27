import os, math
import itertools
import argparse
import logging
import sys
import numpy as np
import torch
import pandas as pd
import h5py
import trimesh
import networkx as nx
from scipy.spatial.transform import Rotation
from dataclasses import dataclass
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
# Keep CPU as the project default for stable/reproducible runs.
BACKEND = "cpu"
try:
    N_ENVS = max(1, int(os.environ.get("GENESIS_N_ENVS", "16")))
except ValueError:
    N_ENVS = 16

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


def _kwarg_not_supported(exc: Exception, kw: str) -> bool:
    msg = str(exc)
    return ("unexpected keyword argument" in msg) and (kw in msg)


def init_genesis_compat(gs_mod, backend, n_envs: Optional[int] = None):
    """Initialize Genesis, using n_envs only when supported by this version."""
    if n_envs is None:
        return gs_mod.init(backend=backend)
    try:
        return gs_mod.init(backend=backend, n_envs=int(n_envs))
    except TypeError as exc:
        if _kwarg_not_supported(exc, "n_envs"):
            return gs_mod.init(backend=backend)
        raise


def create_scene_compat(gs_mod, *, n_envs: Optional[int] = None, **scene_kwargs):
    """Create a Scene, using n_envs only when supported by this version."""
    if n_envs is None:
        return gs_mod.Scene(**scene_kwargs)
    try:
        return gs_mod.Scene(n_envs=int(n_envs), **scene_kwargs)
    except TypeError as exc:
        if _kwarg_not_supported(exc, "n_envs"):
            return gs_mod.Scene(**scene_kwargs)
        raise

# ── STEP 2: Parameter block ────────────────────────────────────────────────────  # parameters section
# ── Input ──────────────────────────────────────────────────────────────────  # input settings
PARTICLE_FILE        = "Star600M.obj"   # OBJ or STL path
N_PARTICLES          = 50               # number of particle copies to drop
SCALE_FACTOR         = 1.0             # 0.001 converts mm mesh → metres
# ── Material ───────────────────────────────────────────────────────────────  # material settings
# YOUNGS_MODULUS: used directly by MPM elastoplastic particles.
YOUNGS_MODULUS       = 200_000_000.0  # Pa
POISSON_RATIO        = 0.45            # 0.5 = fully incompressible; use 0.45+ to limit volume loss
DENSITY              = 1200            # kg/m³
# Rigid-body restitution (Genesis maps this to internal coupling restitution).
# Non-zero values can trigger a Genesis WARNING and may reduce stability; 0 = inelastic.
PARTICLE_RESTITUTION = 0.0             # particle–contact bounciness (was 0.2)
ENV_RESTITUTION      = 0.0             # floor/walls (was 0.05)
# ── Environment ────────────────────────────────────────────────────────────  # environment settings
ENVIRONMENT_TYPE     = "plate"      # Fixed: always plate + syringe overlay
PLATE_SIZE           = 0.25            # square plate side length (m)
CYLINDER_DIAMETER    = 0.20            # inner diameter (m)
CYLINDER_HEIGHT      = 0.30            # wall height (m)
CYLINDER_SEGMENTS    = 32              # wall facets — use 24+ to avoid gaps
WALL_THICKNESS       = 0.02            # m — plate slab thickness (too thin + coarse dt → FEM tunneling)
# Syringe geometry (ENVIRONMENT_TYPE="syringe")
SYRINGE_BARREL_DIAMETER = 0.20         # m — large tube inner diameter
SYRINGE_BARREL_LENGTH = 0.30           # m — large tube inner length
SYRINGE_NEEDLE_DIAMETER = 0.04         # m — outlet hole / needle inner diameter
SYRINGE_NEEDLE_LENGTH = 0.20           # m — small tube inner length below the barrel
SYRINGE_WALL_THICKNESS = 0.003         # m — syringe tube wall thickness (barrel + needle)
SYRINGE_BOTTOM_THICKNESS = 0.003       # m — annulus slab thickness at barrel/needle junction
SYRINGE_PLATE_GAP = 0.01               # m — visual/placement gap between needle tip and plate top
SYRINGE_SEGMENTS = 32                  # wall facets for barrel/needle/hole rings
# When True, do not add a cap collider at the needle outlet so particles can flow through.
SYRINGE_OPEN_TIP = True
# Rim height for ENVIRONMENT_TYPE="plate" — keeps particles on the plate (0 = flat open plate).
PLATE_WALL_HEIGHT    = 0.15            # m — vertical walls along the square perimeter
# ── Drop ───────────────────────────────────────────────────────────────────  # drop settings
DROP_HEIGHT          = 0.15            # overridden at runtime to 0.5 * SYRINGE_BARREL_LENGTH
# 0 = stack all particles in a vertical column at (0, ·, 0); >0 = Vogel disk on XZ up to this fraction of spread radius
DROP_SPREAD          = 0.5
# ── Gravity ────────────────────────────────────────────────────────────────  # gravity settings
GRAVITY              = (0, -9.81, 0)   # Y is up; change to (0,-1.62,0) for Moon
# ── Simulation ─────────────────────────────────────────────────────────────  # simulation settings
# MPM defaults for this container scale.
DT                   = 4e-3           # s — outer step (with SUBSTEPS)
SUBSTEPS             = 10             # inner substeps per dt
SIM_DURATION         = 10.0            # max simulated time (s)
SETTLE_THRESHOLD     = 1e-3            # m/s — stop early when all particles slow
# ── Runtime / performance ───────────────────────────────────────────────────  # runtime settings
# Genesis’s viewer can dominate runtime on CPU (the FPS log you saw is from it).
# Keep it off by default so simulation runs as fast as possible.
SHOW_VIEWER          = False
# ── Contact analysis ───────────────────────────────────────────────────────  # contact analysis settings
CONTACT_DEPTH_TOL    = 5e-5            # min penetration depth to count as contact
STRESS_FLOOR         = 0.05            # normalized intensities below this are clamped to 0 (kills noise on non-touching particles)
# STRESS_SIGMA: fraction of mesh characteristic radius used as local-space Gaussian width.
# 0.30 = 30 % of half-diagonal.  Increase to widen blobs, decrease to sharpen them.
# (Old meaning: radians for angular falloff — now unused.)
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

    dt: float = 4e-3
    substeps: int = 10


def make_sim_options(gs_mod, cfg: dict):
    """Build `gs.options.SimOptions` from a runtime config dict."""
    g = getattr(gs_mod, "options", gs_mod)
    return g.SimOptions(
        dt=float(cfg["DT"]),
        substeps=int(cfg["SUBSTEPS"]),
        gravity=cfg.get("GRAVITY", (0, -9.81, 0)),
    )


def make_fem_options(gs_mod, cfg: Optional[dict] = None):
    """Build `gs.options.FEMOptions` from a runtime config dict."""
    g = getattr(gs_mod, "options", gs_mod)
    cfg = cfg or {}
    return g.FEMOptions(
        use_implicit_solver=True,
        n_newton_iterations=int(cfg.get("FEM_NEWTON_ITERATIONS", 4)),
    )


def make_mpm_options(gs_mod, _cfg: Optional[dict] = None):
    """Build an MPM domain large enough for the current runtime configuration."""
    g = getattr(gs_mod, "options", gs_mod)
    cfg = _cfg or {}

    plate_size = float(cfg.get("PLATE_SIZE", PLATE_SIZE))
    cyl_d = float(cfg.get("CYLINDER_DIAMETER", CYLINDER_DIAMETER))
    syringe_d = float(cfg.get("SYRINGE_BARREL_DIAMETER", SYRINGE_BARREL_DIAMETER))
    syringe_h = float(cfg.get("SYRINGE_BARREL_LENGTH", SYRINGE_BARREL_LENGTH))
    needle_h = float(cfg.get("SYRINGE_NEEDLE_LENGTH", SYRINGE_NEEDLE_LENGTH))
    gap = float(cfg.get("SYRINGE_PLATE_GAP", SYRINGE_PLATE_GAP))
    drop_h = float(cfg.get("DROP_HEIGHT", DROP_HEIGHT))
    wall_t = float(cfg.get("WALL_THICKNESS", WALL_THICKNESS))
    plate_wall_h = float(cfg.get("PLATE_WALL_HEIGHT", PLATE_WALL_HEIGHT))

    radial_extent = max(0.5 * plate_size, 0.5 * cyl_d, 0.5 * syringe_d, 0.2)
    lateral_margin = max(0.15 * radial_extent, 0.02)
    xz_half_span = radial_extent + lateral_margin

    # Mirror create_environment("plate") geometry exactly:
    #   syringe_lift_y = wall_t + gap - (junction_y_local - needle_h)
    #                  ≈ wall_t + gap + needle_h   (junction_y_local = t_bottom/2 ≈ 0)
    #   syringe_top_y  = syringe_lift_y + junction_y_local + barrel_h
    #                  ≈ wall_t + gap + needle_h + syringe_h
    #   env_info["top_y"] = max(wall_t + plate_wall_h, syringe_top_y)
    syringe_top_y = wall_t + gap + needle_h + syringe_h
    plate_surface_y = wall_t + plate_wall_h
    actual_top_y = max(plate_surface_y, syringe_top_y)
    # Needle bottom sits at wall_t + gap (the syringe is lifted so the needle tip is at wall_t+gap).
    syringe_bottom_y = wall_t + gap
    spawn_y0 = actual_top_y + drop_h

    lower_y = min(-0.05, syringe_bottom_y - max(0.05, 0.25 * syringe_h))
    upper_y = max(actual_top_y + max(0.1, 0.5 * syringe_h), spawn_y0 + max(0.1, 0.5 * syringe_h))

    # Genesis MPM uses an internal "safety padding" that makes the effective solver boundary
    # slightly tighter than the requested domain. If we place particles near the top of the
    # domain (e.g. DROP_HEIGHT above the syringe), they can be rejected at add_entity() with:
    # "Entity has particles outside solver boundary".
    #
    # Add explicit headroom (and a bit of floor) so spawns + random rotations stay inside even
    # after Genesis' internal padding. The effective solver boundary can shrink by O(0.05m)
    # for typical grid densities; therefore the minimum margin must exceed that shrink.
    user_margin_y = float(cfg.get("MPM_DOMAIN_MARGIN_Y", cfg.get("MPM_DOMAIN_HEADROOM_Y", 0.0)))
    # Add extra headroom specifically for the number of particles being stacked
    stack_buffer = float(cfg.get("N_PARTICLES", 50)) * 0.01
    y_margin = max(
        user_margin_y,
        0.25,  # Increased base 0.15 to 0.25
        0.5 * drop_h,
        stack_buffer,
        0.25 * syringe_h,
        0.25 * plate_size,
    )
    upper_y = float(upper_y) + float(y_margin)
    # Extra epsilon headroom to survive Genesis' internal boundary shrink + float32 rounding.
    # Genesis' effective boundary can be noticeably tighter than requested; keep this generous.
    upper_y = float(upper_y) + float(cfg.get("MPM_DOMAIN_EPS_Y", 0.12))
    lower_y = float(lower_y) - float(0.5 * y_margin)

    grid_density = max(32, int(cfg.get("MPM_GRID_DENSITY", 64)))
    return g.MPMOptions(
        lower_bound=(-xz_half_span, lower_y, -xz_half_span),
        upper_bound=(xz_half_span, upper_y, xz_half_span),
        # Genesis >=0.4 uses scalar grid_density instead of per-axis `res`.
        grid_density=grid_density,
    )


def make_rigid_options(gs_mod, _cfg: Optional[dict] = None):
    """
    Rigid solver options for mesh particles against fixed box/cylinder containers.

    `box_box_detection` improves box–box contact; stiffer `constraint_timeconst` reduces penetration.
    Increased `iterations` (80) reduces residual penetration for dense concave multi-hull packing.
    Tighter `constraint_timeconst` (0.001) shrinks per-step penetration residual before it accumulates.
    With compound-hull collision proxies the solver sees accurate geometry, so tighter settings
    converge cleanly without instability.
    """
    g = getattr(gs_mod, "options", gs_mod)
    return g.RigidOptions(
        use_gjk_collision=True,
        box_box_detection=True,
        iterations=80,
        constraint_timeconst=0.001,
    )


DEFAULT_CONFIG = {
    "PARTICLE_FILE": PARTICLE_FILE,
    "N_PARTICLES": N_PARTICLES,
    "N_ENVS": N_ENVS,
    "SCALE_FACTOR": SCALE_FACTOR,
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
    "SYRINGE_BARREL_DIAMETER": SYRINGE_BARREL_DIAMETER,
    "SYRINGE_BARREL_LENGTH": SYRINGE_BARREL_LENGTH,
    "SYRINGE_NEEDLE_DIAMETER": SYRINGE_NEEDLE_DIAMETER,
    "SYRINGE_NEEDLE_LENGTH": SYRINGE_NEEDLE_LENGTH,
    "SYRINGE_WALL_THICKNESS": SYRINGE_WALL_THICKNESS,
    "SYRINGE_BOTTOM_THICKNESS": SYRINGE_BOTTOM_THICKNESS,
    "SYRINGE_PLATE_GAP": SYRINGE_PLATE_GAP,
    "SYRINGE_SEGMENTS": SYRINGE_SEGMENTS,
    "SYRINGE_OPEN_TIP": SYRINGE_OPEN_TIP,
    "WALL_THICKNESS": WALL_THICKNESS,
    "PLATE_WALL_HEIGHT": PLATE_WALL_HEIGHT,
    "DROP_HEIGHT": DROP_HEIGHT,
    "DROP_SPREAD": DROP_SPREAD,
    "GRAVITY": GRAVITY,
    "DT": DT,
    # Use ThroughputSimTuning defaults for both server and standalone runs.
    "SUBSTEPS": ThroughputSimTuning.substeps,
    "SIM_DURATION": SIM_DURATION,
    "SETTLE_THRESHOLD": SETTLE_THRESHOLD,
    "CONTACT_DEPTH_TOL": CONTACT_DEPTH_TOL,
    "OUTPUT_DIR": OUTPUT_DIR,
    "SAVE_HDF5": SAVE_HDF5,
    "SAVE_CSV": SAVE_CSV,
    "STRESS_SIGMA": 0.30,  # fraction of mesh char-size used as Gaussian sigma_local (0.30 = 30 % of half-diagonal)
    "STRESS_FLOOR": 0.05,  # clamp post-normalization noise below this to 0.0 (kills ghost gradients on non-touching particles)
    # Preferred Genesis backend for server runs: auto | cpu | gpu.
    "BACKEND": "auto",
    # MPM particle sampling/grid resolution. Auto-retried higher on known low-sample failures.
    "MPM_GRID_DENSITY": 64,
}


def build_runtime_config(payload: Optional[dict]) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    raw = payload or {}
    for key in cfg.keys():
        if key in raw:
            cfg[key] = raw[key]

    cfg["N_PARTICLES"] = int(cfg["N_PARTICLES"])
    cfg["N_ENVS"] = max(1, int(cfg.get("N_ENVS", N_ENVS)))
    cfg["YOUNGS_MODULUS"] = float(cfg["YOUNGS_MODULUS"])
    cfg["STRESS_SIGMA"] = float(cfg.get("STRESS_SIGMA", 0.12))
    cfg["STRESS_FLOOR"] = float(cfg.get("STRESS_FLOOR", 0.05))
    cfg["POISSON_RATIO"] = float(cfg["POISSON_RATIO"])
    cfg["PARTICLE_RESTITUTION"] = float(cfg["PARTICLE_RESTITUTION"])
    cfg["ENV_RESTITUTION"] = float(cfg["ENV_RESTITUTION"])
    cfg["MPM_GRID_DENSITY"] = max(32, int(cfg.get("MPM_GRID_DENSITY", 64)))
    cfg["CYLINDER_DIAMETER"] = float(cfg["CYLINDER_DIAMETER"])
    cfg["CYLINDER_HEIGHT"] = float(cfg.get("CYLINDER_HEIGHT", CYLINDER_HEIGHT))
    cfg["CYLINDER_SEGMENTS"] = max(16, int(cfg.get("CYLINDER_SEGMENTS", CYLINDER_SEGMENTS)))
    cfg["SYRINGE_BARREL_DIAMETER"] = float(cfg.get("SYRINGE_BARREL_DIAMETER", SYRINGE_BARREL_DIAMETER))
    cfg["SYRINGE_BARREL_LENGTH"] = float(cfg.get("SYRINGE_BARREL_LENGTH", SYRINGE_BARREL_LENGTH))
    cfg["SYRINGE_NEEDLE_DIAMETER"] = float(cfg.get("SYRINGE_NEEDLE_DIAMETER", SYRINGE_NEEDLE_DIAMETER))
    cfg["SYRINGE_NEEDLE_LENGTH"] = float(cfg.get("SYRINGE_NEEDLE_LENGTH", SYRINGE_NEEDLE_LENGTH))
    raw_wall_t = max(1e-6, float(cfg.get("SYRINGE_WALL_THICKNESS", SYRINGE_WALL_THICKNESS)))
    raw_bottom_t = max(1e-6, float(cfg.get("SYRINGE_BOTTOM_THICKNESS", SYRINGE_BOTTOM_THICKNESS)))
    # Use a geometry-scaled floor so small syringes are not forced to chunky 3 mm walls.
    syringe_scale = max(
        1e-6,
        min(
            float(cfg["SYRINGE_BARREL_DIAMETER"]),
            float(cfg["SYRINGE_NEEDLE_DIAMETER"]),
            float(cfg["SYRINGE_BARREL_LENGTH"]),
            float(cfg["SYRINGE_NEEDLE_LENGTH"]),
        ),
    )
    min_wall_t = max(5.0e-5, 0.005 * syringe_scale)
    # Bottom slab floor is deliberately thicker than the side walls: the floor carries the full
    # weight of all stacked particles and tunneling through it is the dominant failure mode.
    min_bottom_t = max(5.0e-5, 0.015 * syringe_scale)
    cfg["SYRINGE_WALL_THICKNESS"] = max(raw_wall_t, min_wall_t)
    cfg["SYRINGE_BOTTOM_THICKNESS"] = max(raw_bottom_t, min_bottom_t)
    cfg["SYRINGE_PLATE_GAP"] = max(0.0, float(cfg.get("SYRINGE_PLATE_GAP", SYRINGE_PLATE_GAP)))
    cfg["SYRINGE_SEGMENTS"] = max(8, int(cfg.get("SYRINGE_SEGMENTS", SYRINGE_SEGMENTS)))
    cfg["SYRINGE_OPEN_TIP"] = bool(cfg.get("SYRINGE_OPEN_TIP", SYRINGE_OPEN_TIP))
    # Single supported container mode: flat plate + syringe. Ignore external environment selection.
    cfg["ENVIRONMENT_TYPE"] = "plate"
    # Spawn height is defined relative to syringe top (positive is above the syringe).
    # For the syringe demo we want visible free-fall into the barrel before contact.
    cfg["DROP_HEIGHT"] = 0.5 * float(cfg["SYRINGE_BARREL_LENGTH"])
    cfg["DROP_SPREAD"] = float(cfg["DROP_SPREAD"])
    cfg["PLATE_WALL_HEIGHT"] = float(cfg.get("PLATE_WALL_HEIGHT", PLATE_WALL_HEIGHT))
    cfg["PLATE_SIZE"] = float(cfg.get("PLATE_SIZE", PLATE_SIZE))
    cfg["WALL_THICKNESS"] = float(cfg.get("WALL_THICKNESS", WALL_THICKNESS))
    cfg["DT"] = float(cfg["DT"])
    cfg["SUBSTEPS"] = int(cfg["SUBSTEPS"])
    cfg["SUBSTEPS"] = max(cfg["SUBSTEPS"], 1)
    cfg = tune_mpm_timestep(cfg)

    cfg["SIM_DURATION"] = float(cfg["SIM_DURATION"])
    cfg["SETTLE_THRESHOLD"] = float(cfg["SETTLE_THRESHOLD"])
    _grav = cfg.get("GRAVITY", GRAVITY)
    if isinstance(_grav, (list, tuple)) and len(_grav) == 3:
        cfg["GRAVITY"] = tuple(float(x) for x in _grav)
    else:
        cfg["GRAVITY"] = tuple(float(x) for x in GRAVITY)
    _backend = str(cfg.get("BACKEND", "auto")).strip().lower()
    if _backend not in ("auto", "cpu", "gpu"):
        _backend = "auto"
    cfg["BACKEND"] = _backend
    cfg["ENVIRONMENT_TYPE"] = "plate"
    return cfg


def tune_mpm_timestep(cfg: dict) -> dict:
    """
    Tune DT/SUBSTEPS for Genesis MPM stability at the configured `MPM_GRID_DENSITY`.

    Important: the server may retry `scene.build()` with a higher `MPM_GRID_DENSITY`
    than the frontend requested; in that case DT/SUBSTEPS must be re-tuned to the
    new density to prevent NaNs / "particles disappearing" in the viewer.
    """
    cfg = dict(cfg)
    cfg["DT"] = float(cfg.get("DT", DT))
    cfg["SUBSTEPS"] = max(int(cfg.get("SUBSTEPS", ThroughputSimTuning.substeps)), 1)

    # Genesis MPM is sensitive to the inner substep dt (= DT/SUBSTEPS) vs grid_density.
    # Empirically (matching Genesis warnings), suggested_dt scales ~ 1/grid_density.
    grid_density = max(32, int(cfg.get("MPM_GRID_DENSITY", 64)))
    cfg["MPM_GRID_DENSITY"] = grid_density
    base_suggested_dt_at_96 = 0.000208333  # seconds, from Genesis warning at grid_density=96
    suggested_dt = base_suggested_dt_at_96 * (96.0 / float(grid_density))
    target_substep_dt = 0.9 * suggested_dt  # safety margin

    if target_substep_dt > 0.0 and cfg["DT"] > 0.0:
        required = int(math.ceil(float(cfg["DT"]) / float(target_substep_dt)))
        # Cap to keep runtimes sane; if we hit the cap, also shrink DT to maintain stability.
        max_substeps = 80
        if required > max_substeps:
            cfg["SUBSTEPS"] = max_substeps
            cfg["DT"] = float(target_substep_dt) * float(max_substeps)
        else:
            cfg["SUBSTEPS"] = max(int(cfg["SUBSTEPS"]), required)

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


# Cache last good poses so MPM visualization/export doesn't "disappear" when
# Genesis returns transient NaNs/huge poses after the first step.
_LAST_GOOD_POSE: dict[int, tuple[np.ndarray, tuple[float, float, float, float]]] = {}


def _pose_reasonable(pos: np.ndarray) -> bool:
    if pos is None:
        return False
    pos = np.asarray(pos, dtype=float).reshape(3)
    if not np.isfinite(pos).all():
        return False
    # Physics-normalized worlds can still be larger than 10m; accept up to a generous bound.
    return float(np.abs(pos).max()) <= 1.0e4


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
    # IMPORTANT:
    # - FEM entities often lack stable rigid-style pose getters; use state centroid as a fallback.
    # - MPM entities may expose a state whose `.pos` is a *global* solver particle buffer shared
    #   across entities; using its centroid can collapse all entities to the same apparent pose.
    #   Prefer rigid-style getters for MPM when available.
    is_fem = "FEMEntity" in entity_type
    is_mpm = "MPMEntity" in entity_type
    force_state_pose = is_fem

    # Prefer explicit pose getters when available (rigid-like entities).
    if (not force_state_pose) and hasattr(e, "get_pos"):
        x, y, z = e.get_pos()
        pos_try = _vec3_from_xyz(x, y, z)
        # If pose getters return something clearly not in world space (e.g.
        # very large magnitudes), fall back to state-based heuristics below.
        if _pose_reasonable(pos_try) and float(np.abs(pos_try).max()) <= 10.0:
            if hasattr(e, "get_quat"):
                qw, qx, qy, qz = e.get_quat()  # (w,x,y,z) -> (x,y,z,w)
                quat = _quat_tuple_xyzw(qw, qx, qy, qz)
            else:
                quat = (0.0, 0.0, 0.0, 1.0)
            eid = int(_entity_id(e))
            _LAST_GOOD_POSE[eid] = (np.asarray(pos_try, dtype=float).reshape(3), quat)
            return (pos_try, quat)
        # else: fall through to fallback extraction
        # If the pose is still within a generous bound, accept it (some scenes are >10m).
        if _pose_reasonable(pos_try) and float(np.abs(pos_try).max()) <= 500.0:
            if hasattr(e, "get_quat"):
                qw, qx, qy, qz = e.get_quat()  # (w,x,y,z) -> (x,y,z,w)
                quat = _quat_tuple_xyzw(qw, qx, qy, qz)
            else:
                quat = (0.0, 0.0, 0.0, 1.0)
            eid = int(_entity_id(e))
            _LAST_GOOD_POSE[eid] = (np.asarray(pos_try, dtype=float).reshape(3), quat)
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
                        if _pose_reasonable(pos_try) and float(np.abs(pos_try).max()) <= 500.0:
                            if hasattr(e, "get_quat"):
                                qw, qx, qy, qz = e.get_quat()
                                quat = _quat_tuple_xyzw(qw, qx, qy, qz)
                            else:
                                quat = (0.0, 0.0, 0.0, 1.0)
                            eid = int(_entity_id(e))
                            _LAST_GOOD_POSE[eid] = (np.asarray(pos_try, dtype=float).reshape(3), quat)
                            return (pos_try, quat)
                    except Exception:
                        pass
                # For MPM entities, prefer rigid-style getters if present to avoid using
                # a potentially global/shared particle buffer centroid.
                if is_mpm and hasattr(e, "get_pos"):
                    try:
                        x, y, z = e.get_pos()
                        pos_try = _vec3_from_xyz(x, y, z)
                        if _pose_reasonable(pos_try) and float(np.abs(pos_try).max()) <= 500.0:
                            if hasattr(e, "get_quat"):
                                qw, qx, qy, qz = e.get_quat()
                                quat = _quat_tuple_xyzw(qw, qx, qy, qz)
                            else:
                                quat = (0.0, 0.0, 0.0, 1.0)
                            eid = int(_entity_id(e))
                            _LAST_GOOD_POSE[eid] = (np.asarray(pos_try, dtype=float).reshape(3), quat)
                            return (pos_try, quat)
                    except Exception:
                        pass
                # Cache state-centroid only when it looks reasonable; otherwise keep last good.
                eid = int(_entity_id(e))
                if _pose_reasonable(best_pos):
                    pose = (np.asarray(best_pos, dtype=float).reshape(3), (0.0, 0.0, 0.0, 1.0))
                    _LAST_GOOD_POSE[eid] = pose
                    return pose
                if eid in _LAST_GOOD_POSE:
                    return _LAST_GOOD_POSE[eid]
    except Exception:
        pass

    # FEM/MPM: state tensors may use different field names across Genesis versions.
    # If centroid extraction failed, try rigid-style getters so the viewer still receives poses.
    if force_state_pose and hasattr(e, "get_pos"):
        try:
            x, y, z = e.get_pos()
            pos_try = _vec3_from_xyz(x, y, z)
            if _pose_reasonable(pos_try):
                if hasattr(e, "get_quat"):
                    qw, qx, qy, qz = e.get_quat()
                    quat = _quat_tuple_xyzw(qw, qx, qy, qz)
                else:
                    quat = (0.0, 0.0, 0.0, 1.0)
                eid = int(_entity_id(e))
                _LAST_GOOD_POSE[eid] = (np.asarray(pos_try, dtype=float).reshape(3), quat)
                return (pos_try, quat)
        except Exception:
            pass

    eid = int(_entity_id(e))
    if eid in _LAST_GOOD_POSE:
        return _LAST_GOOD_POSE[eid]
    return (np.zeros((3,), dtype=float), (0.0, 0.0, 0.0, 1.0))


def _mpm_solver_active(scene) -> bool:
    sim = getattr(scene, "sim", None)
    solver = getattr(sim, "mpm_solver", None)
    return bool(solver is not None and getattr(solver, "is_active", False))


def _entity_mpm_state(entity):
    if not hasattr(entity, "get_state"):
        return None
    try:
        return entity.get_state()
    except Exception:
        return None


def _estimate_particle_radius_from_state(state) -> Optional[float]:
    if state is None or not hasattr(state, "pos"):
        return None
    pos = _tensor_to_numpy(state.pos)
    if pos.size == 0:
        return None
    pts = np.asarray(pos, dtype=float).reshape(-1, 3)
    if pts.shape[0] < 2:
        return None
    ctr = pts.mean(axis=0)
    radii = np.linalg.norm(pts - ctr.reshape(1, 3), axis=1)
    r = float(np.percentile(radii, 75))
    if not math.isfinite(r) or r <= 0.0:
        return None
    return r


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


def _cup_mesh_proxy_path(inner_diameter: float, wall_thickness: float, wall_height: float, segments: int) -> str:
    """Deterministic cache path for a hollow cylinder cup collision mesh."""
    root = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(root, ".cache", "containers")
    os.makedirs(cache_dir, exist_ok=True)
    key = f"cup_d{inner_diameter:.9f}_t{wall_thickness:.9f}_h{wall_height:.9f}_n{int(segments)}"
    return os.path.join(cache_dir, f"{key}.obj")


def _container_mesh_proxy_path(kind: str, key: str) -> str:
    """Deterministic cache path for generated container collision meshes."""
    root = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(root, ".cache", "containers")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{kind}_{key}.obj")


def _write_cup_obj(
    filepath: str,
    *,
    inner_radius: float,
    wall_thickness: float,
    wall_height: float,
    segments: int,
) -> None:
    """
    Write a watertight hollow-cylinder cup OBJ:
    - open top
    - closed bottom (thickness = wall_thickness)
    - true circular inner/outer walls.
    """
    n = max(12, int(segments))
    r_in = max(float(inner_radius), 1e-9)
    t = max(float(wall_thickness), 1e-9)
    h = max(float(wall_height), 1e-9)
    r_out = r_in + t
    y_bot = 0.0
    y_floor = t
    y_top = t + h

    verts: list[tuple[float, float, float]] = []

    def ring(radius: float, y: float) -> list[int]:
        idx: list[int] = []
        for i in range(n):
            a = 2.0 * math.pi * i / n
            verts.append((radius * math.cos(a), y, radius * math.sin(a)))
            idx.append(len(verts))
        return idx

    outer_bottom = ring(r_out, y_bot)
    outer_top = ring(r_out, y_top)
    inner_floor = ring(r_in, y_floor)
    inner_top = ring(r_in, y_top)
    verts.append((0.0, y_bot, 0.0))
    c_bot = len(verts)
    verts.append((0.0, y_floor, 0.0))
    c_floor = len(verts)

    faces: list[tuple[int, int, int]] = []

    def add_quad(a: int, b: int, c: int, d: int) -> None:
        faces.append((a, b, c))
        faces.append((a, c, d))

    for i in range(n):
        j = (i + 1) % n
        # Outer wall (normal outward).
        add_quad(outer_bottom[i], outer_top[i], outer_top[j], outer_bottom[j])
        # Inner wall (normal inward).
        add_quad(inner_top[i], inner_floor[i], inner_floor[j], inner_top[j])
        # Top rim annulus.
        add_quad(outer_top[i], inner_top[i], inner_top[j], outer_top[j])
        # Underside disk (normal down).
        faces.append((c_bot, outer_bottom[j], outer_bottom[i]))
        # Inner floor disk (normal up).
        faces.append((c_floor, inner_floor[i], inner_floor[j]))

    lines = ["# Hollow cylinder cup collision mesh"]
    lines += [f"v {x:.9g} {y:.9g} {z:.9g}" for (x, y, z) in verts]
    lines += [f"f {a} {b} {c}" for (a, b, c) in faces]
    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _write_open_tube_obj(
    filepath: str,
    *,
    inner_radius: float,
    wall_thickness: float,
    height: float,
    segments: int,
) -> None:
    """Write a hollow tube solid (ring with open center through-height)."""
    n = max(12, int(segments))
    r_in = max(float(inner_radius), 1e-9)
    t = max(float(wall_thickness), 1e-9)
    h = max(float(height), 1e-9)
    r_out = r_in + t
    y0 = -0.5 * h
    y1 = 0.5 * h

    verts: list[tuple[float, float, float]] = []

    def ring(radius: float, y: float) -> list[int]:
        idx: list[int] = []
        for i in range(n):
            a = 2.0 * math.pi * i / n
            verts.append((radius * math.cos(a), y, radius * math.sin(a)))
            idx.append(len(verts))
        return idx

    ob = ring(r_out, y0)
    ot = ring(r_out, y1)
    ib = ring(r_in, y0)
    it = ring(r_in, y1)

    faces: list[tuple[int, int, int]] = []

    def add_quad(a: int, b: int, c: int, d: int) -> None:
        faces.append((a, b, c))
        faces.append((a, c, d))

    for i in range(n):
        j = (i + 1) % n
        add_quad(ob[i], ot[i], ot[j], ob[j])  # outer wall
        add_quad(it[i], ib[i], ib[j], it[j])  # inner wall
        add_quad(ot[i], it[i], it[j], ot[j])  # top annulus
        add_quad(ib[i], ob[i], ob[j], ib[j])  # bottom annulus

    lines = ["# Hollow tube mesh"]
    lines += [f"v {x:.9g} {y:.9g} {z:.9g}" for (x, y, z) in verts]
    lines += [f"f {a} {b} {c}" for (a, b, c) in faces]
    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _write_annulus_slab_obj(
    filepath: str,
    *,
    inner_radius: float,
    outer_radius: float,
    thickness: float,
    segments: int,
) -> None:
    """Write a thick annulus slab (washer) mesh."""
    n = max(12, int(segments))
    r_in = max(float(inner_radius), 1e-9)
    r_out = max(float(outer_radius), r_in + 1e-9)
    t = max(float(thickness), 1e-9)
    y0 = -0.5 * t
    y1 = 0.5 * t

    verts: list[tuple[float, float, float]] = []

    def ring(radius: float, y: float) -> list[int]:
        idx: list[int] = []
        for i in range(n):
            a = 2.0 * math.pi * i / n
            verts.append((radius * math.cos(a), y, radius * math.sin(a)))
            idx.append(len(verts))
        return idx

    ob = ring(r_out, y0)
    ot = ring(r_out, y1)
    ib = ring(r_in, y0)
    it = ring(r_in, y1)

    faces: list[tuple[int, int, int]] = []

    def add_quad(a: int, b: int, c: int, d: int) -> None:
        faces.append((a, b, c))
        faces.append((a, c, d))

    for i in range(n):
        j = (i + 1) % n
        add_quad(ob[i], ot[i], ot[j], ob[j])  # outer wall
        add_quad(it[i], ib[i], ib[j], it[j])  # inner wall
        add_quad(ot[i], it[i], it[j], ot[j])  # top
        add_quad(ib[i], ob[i], ob[j], ib[j])  # bottom

    lines = ["# Annulus slab mesh"]
    lines += [f"v {x:.9g} {y:.9g} {z:.9g}" for (x, y, z) in verts]
    lines += [f"f {a} {b} {c}" for (a, b, c) in faces]
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

    volume = float(original_mesh.volume) if original_mesh.is_watertight else float(original_mesh.convex_hull.volume)
    print(
        f"{filepath} | verts={len(original_mesh.vertices)} | faces={len(original_mesh.faces)} | "
        f"extents={original_mesh.extents} | volume={volume:.6g}"
    )
    return (original_mesh, physics_norm)


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


def create_environment(
    scene,
    kind,
    plate_size=0.6,
    cyl_diameter=0.20,
    cyl_height=0.30,
    cyl_segments=32,
    wall_thickness=WALL_THICKNESS,
    plate_wall_height=PLATE_WALL_HEIGHT,
    env_restitution: float = ENV_RESTITUTION,
    syringe_barrel_diameter: float = SYRINGE_BARREL_DIAMETER,
    syringe_barrel_length: float = SYRINGE_BARREL_LENGTH,
    syringe_needle_diameter: float = SYRINGE_NEEDLE_DIAMETER,
    syringe_needle_length: float = SYRINGE_NEEDLE_LENGTH,
    syringe_wall_thickness: float = SYRINGE_WALL_THICKNESS,
    syringe_bottom_thickness: float = SYRINGE_BOTTOM_THICKNESS,
    syringe_plate_gap: float = SYRINGE_PLATE_GAP,
    syringe_segments: int = SYRINGE_SEGMENTS,
    syringe_open_tip: bool = SYRINGE_OPEN_TIP,
) -> tuple[set, dict]:
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
        # Always include syringe geometry above the plate so particles can spawn/load into it.
        t_wall = max(1e-6, float(syringe_wall_thickness))
        t_bottom = max(1e-6, float(syringe_bottom_thickness))
        barrel_d = float(syringe_barrel_diameter)
        barrel_h = float(syringe_barrel_length)
        needle_d = float(syringe_needle_diameter)
        needle_h = float(syringe_needle_length)
        seg = max(8, int(syringe_segments))
        gap = max(0.0, float(syringe_plate_gap))
        if needle_d >= barrel_d:
            raise ValueError(
                f"Syringe requires needle diameter < barrel diameter, got {needle_d} >= {barrel_d}"
            )

        barrel_r = 0.5 * barrel_d
        needle_r = 0.5 * needle_d
        key = (
            f"dB{barrel_d:.9f}_hB{barrel_h:.9f}_dN{needle_d:.9f}_hN{needle_h:.9f}_"
            f"tw{t_wall:.9f}_tb{t_bottom:.9f}_gap{gap:.9f}_n{seg}"
        )
        barrel_path = _container_mesh_proxy_path("syringe_barrel", key)
        annulus_path = _container_mesh_proxy_path("syringe_annulus", key)
        needle_path = _container_mesh_proxy_path("syringe_needle", key)
        tip_cap_path = _container_mesh_proxy_path("syringe_tip_cap", key)

        _write_open_tube_obj(
            barrel_path,
            inner_radius=barrel_r,
            wall_thickness=t_wall,
            height=barrel_h,
            segments=seg,
        )
        _write_annulus_slab_obj(
            annulus_path,
            inner_radius=needle_r,
            outer_radius=barrel_r,
            thickness=t_bottom,
            segments=seg,
        )
        _write_open_tube_obj(
            needle_path,
            inner_radius=needle_r,
            wall_thickness=t_wall,
            height=needle_h,
            segments=seg,
        )
        if not syringe_open_tip:
            # Optional outlet cap for "contained syringe" mode.
            _write_annulus_slab_obj(
                tip_cap_path,
                inner_radius=0.0,
                outer_radius=needle_r,
                thickness=t_bottom,
                segments=seg,
            )
        # Lift syringe so needle tip sits at plate_top + gap.
        junction_y_local = t_bottom * 0.5
        needle_tip_local_y = junction_y_local - needle_h
        syringe_lift_y = t + gap - needle_tip_local_y

        barrel = scene.add_entity(
            gs.morphs.Mesh(
                file=barrel_path,
                scale=1.0,
                pos=(0.0, syringe_lift_y + junction_y_local + barrel_h / 2.0, 0.0),
                fixed=True,
                convexify=False,
                collision=True,
                visualization=False,
            ),
            material=mat,
        )
        container_ids.add(barrel)

        # Barrel floor safety collider:
        # - keep a thick Box-based collider for robust contact/tunneling resistance
        # - align its top face to the true annulus top (height compensation)
        # - preserve a center opening so particles can flow into the needle bore
        safe_floor_h = max(t_bottom, barrel_r * 0.15)
        outer_half = barrel_r + t_wall
        inner_half = max(needle_r + 1e-9, 1e-6)
        annulus_top_y = syringe_lift_y + junction_y_local + t_bottom * 0.5
        annulus_box_center_y = annulus_top_y - safe_floor_h * 0.5
        if inner_half < outer_half - 1e-9:
            slab_x = outer_half - inner_half
            slab_z = outer_half - inner_half
            ring_boxes = (
                # Left / right bands
                (
                    (slab_x, safe_floor_h, 2.0 * outer_half),
                    (-inner_half - 0.5 * slab_x, annulus_box_center_y, 0.0),
                ),
                (
                    (slab_x, safe_floor_h, 2.0 * outer_half),
                    (inner_half + 0.5 * slab_x, annulus_box_center_y, 0.0),
                ),
                # Front / back bands
                (
                    (2.0 * inner_half, safe_floor_h, slab_z),
                    (0.0, annulus_box_center_y, -inner_half - 0.5 * slab_z),
                ),
                (
                    (2.0 * inner_half, safe_floor_h, slab_z),
                    (0.0, annulus_box_center_y, inner_half + 0.5 * slab_z),
                ),
            )
            for size_xyz, pos_xyz in ring_boxes:
                annulus = scene.add_entity(
                    gs.morphs.Box(
                        size=size_xyz,
                        pos=pos_xyz,
                        fixed=True,
                    ),
                    material=mat,
                )
                container_ids.add(annulus)
        else:
            annulus_box_side = 2.0 * outer_half
            annulus = scene.add_entity(
                gs.morphs.Box(
                    size=(annulus_box_side, safe_floor_h, annulus_box_side),
                    pos=(0.0, annulus_box_center_y, 0.0),
                    fixed=True,
                ),
                material=mat,
            )
            container_ids.add(annulus)

        needle = scene.add_entity(
            gs.morphs.Mesh(
                file=needle_path,
                scale=1.0,
                pos=(0.0, syringe_lift_y + junction_y_local - needle_h / 2.0, 0.0),
                fixed=True,
                convexify=False,
                collision=True,
                visualization=False,
            ),
            material=mat,
        )
        container_ids.add(needle)

        if not syringe_open_tip:
            # Needle outlet cap: same safe thickness as barrel floor.
            safe_cap_h = max(t_bottom, needle_r * 0.5)
            tip_cap_side = (needle_r + t_wall) * 2.0
            tip_box_center_y = syringe_lift_y + junction_y_local - needle_h
            tip_cap = scene.add_entity(
                gs.morphs.Box(
                    size=(tip_cap_side, safe_cap_h, tip_cap_side),
                    pos=(0.0, tip_box_center_y, 0.0),
                    fixed=True,
                ),
                material=mat,
            )
            container_ids.add(tip_cap)

        syringe_top_y = syringe_lift_y + junction_y_local + barrel_h
        env_info = {
            "surface_y": t,
            "top_y": max(top_y, syringe_top_y),
            "spread_radius": barrel_r,
        }
        return (container_ids, env_info)

    if kind == "cylinder":
        r_inner = cyl_diameter / 2.0
        cup_mesh_path = _cup_mesh_proxy_path(cyl_diameter, wall_thickness, cyl_height, cyl_segments)
        _write_cup_obj(
            cup_mesh_path,
            inner_radius=r_inner,
            wall_thickness=wall_thickness,
            wall_height=cyl_height,
            segments=cyl_segments,
        )
        cup = scene.add_entity(
            gs.morphs.Mesh(
                file=cup_mesh_path,
                scale=1.0,
                pos=(0.0, 0.0, 0.0),
                fixed=True,
                convexify=False,
                collision=True,
                visualization=False,
            ),
            material=mat,
        )
        container_ids.add(cup)

        env_info = {
            "surface_y": wall_thickness,
            "top_y": wall_thickness + cyl_height,
            "inner_radius": r_inner,
            "spread_radius": r_inner,
        }
        return (container_ids, env_info)

    if kind == "syringe":
        t_wall = max(1e-6, float(syringe_wall_thickness))
        t_bottom = max(1e-6, float(syringe_bottom_thickness))
        barrel_d = float(syringe_barrel_diameter)
        barrel_h = float(syringe_barrel_length)
        needle_d = float(syringe_needle_diameter)
        needle_h = float(syringe_needle_length)
        seg = max(8, int(syringe_segments))
        if needle_d >= barrel_d:
            raise ValueError(
                f"Syringe requires needle diameter < barrel diameter, got {needle_d} >= {barrel_d}"
            )

        barrel_r = 0.5 * barrel_d
        needle_r = 0.5 * needle_d
        key = (
            f"dB{barrel_d:.9f}_hB{barrel_h:.9f}_dN{needle_d:.9f}_hN{needle_h:.9f}_"
            f"tw{t_wall:.9f}_tb{t_bottom:.9f}_n{seg}"
        )

        barrel_path = _container_mesh_proxy_path("syringe_barrel", key)
        annulus_path = _container_mesh_proxy_path("syringe_annulus", key)
        needle_path = _container_mesh_proxy_path("syringe_needle", key)

        _write_open_tube_obj(
            barrel_path,
            inner_radius=barrel_r,
            wall_thickness=t_wall,
            height=barrel_h,
            segments=seg,
        )
        _write_annulus_slab_obj(
            annulus_path,
            inner_radius=needle_r,
            outer_radius=barrel_r,
            thickness=t_bottom,
            segments=seg,
        )
        _write_open_tube_obj(
            needle_path,
            inner_radius=needle_r,
            wall_thickness=t_wall,
            height=needle_h,
            segments=seg,
        )
        junction_y = t_bottom * 0.5

        barrel = scene.add_entity(
            gs.morphs.Mesh(
                file=barrel_path,
                scale=1.0,
                pos=(0.0, junction_y + barrel_h / 2.0, 0.0),
                fixed=True,
                convexify=False,
                collision=True,
                visualization=False,
            ),
            material=mat,
        )
        container_ids.add(barrel)
        annulus = scene.add_entity(
            gs.morphs.Mesh(
                file=annulus_path,
                scale=1.0,
                pos=(0.0, junction_y, 0.0),
                fixed=True,
                convexify=False,
                collision=True,
                visualization=False,
            ),
            material=mat,
        )
        container_ids.add(annulus)
        needle = scene.add_entity(
            gs.morphs.Mesh(
                file=needle_path,
                scale=1.0,
                pos=(0.0, junction_y - needle_h / 2.0, 0.0),
                fixed=True,
                convexify=False,
                collision=True,
                visualization=False,
            ),
            material=mat,
        )
        container_ids.add(needle)

        env_info = {
            "surface_y": t_bottom,
            "top_y": junction_y + barrel_h,
            "inner_radius": barrel_r,
            "spread_radius": barrel_r,
            "outlet_radius": needle_r,
            "needle_bottom_y": junction_y - needle_h,
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
    prior_fem_snapshots: Optional[list[dict]] = None,
    physics_norm: float = 1.0,
    mpm_upper_y: Optional[float] = None,
) -> list:
    material = gs.materials.MPM.ElastoPlastic(
        E=float(E),
        nu=float(nu),
        rho=float(rho),
    )

    extents = physics_mesh.bounds[1] - physics_mesh.bounds[0]
    spawn_y0 = float(env_info["top_y"] + drop_height)
    spread = float(drop_spread)
    char = float(max(float(extents[0]), float(extents[1]), float(extents[2]), 1e-9))
    stack_gap = max(char * 1.5, 1e-4)
    mesh_scale = float(scale_factor) * float(physics_norm)

    # If the caller knows the configured MPM domain ceiling, clamp spawns below it.
    # Genesis shrinks the effective boundary vs requested domain; keep a conservative margin.
    max_centroid_y = None
    if mpm_upper_y is not None:
        # Conservative "radius" from centroid to highest point under arbitrary rotation.
        # Use scaled characteristic size with extra padding.
        centroid_radius_y = max(0.75 * char * mesh_scale, 1e-4)
        max_centroid_y = float(mpm_upper_y) - centroid_radius_y - 0.01
        # If we're stacking a single column, ensure the whole stack fits.
        if spread <= 1e-9 and n > 1:
            spawn_y0 = min(spawn_y0, max_centroid_y - float(n - 1) * stack_gap)
        else:
            spawn_y0 = min(spawn_y0, max_centroid_y)

    # Some particle assets are authored far from the local origin. Genesis uses mesh-local
    # coordinates directly, so a non-centered mesh can be spawned outside the solver domain.
    # Compute a local centroid offset and compensate spawn pose so world-space centroids land
    # at the intended (x, y, z) positions.
    local_centroid = np.zeros(3, dtype=float)
    spawn_mesh_file = particle_file
    try:
        src_mesh = trimesh.load(particle_file, force="mesh")
        if isinstance(src_mesh, trimesh.Scene):
            geoms = [g for g in src_mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            src_mesh = trimesh.util.concatenate(geoms) if len(geoms) > 1 else (geoms[0] if geoms else None)
        if isinstance(src_mesh, trimesh.Trimesh) and len(src_mesh.vertices) > 0:
            # Genesis internals may request k=12 neighbors during MPM setup.
            # Very low-vertex meshes can produce only ~5 sampled particles and fail with:
            # "ValueError: kth(=11) out of bounds (5)". Upsample once into a cached proxy.
            if len(src_mesh.vertices) < 12 and len(src_mesh.faces) > 0:
                up = src_mesh.copy()
                for _ in range(4):
                    if len(up.vertices) >= 24:
                        break
                    try:
                        v_sub, f_sub = trimesh.remesh.subdivide(up.vertices, up.faces)
                    except Exception:
                        break
                    up = trimesh.Trimesh(vertices=v_sub, faces=f_sub, process=False)
                if len(up.vertices) >= 12 and len(up.faces) > 0:
                    try:
                        p_abs = os.path.abspath(particle_file)
                        p_mtime = os.path.getmtime(p_abs)
                        key = f"{os.path.basename(p_abs)}_v{len(up.vertices)}_f{len(up.faces)}_m{p_mtime:.6f}"
                        proxy_path = _container_mesh_proxy_path("particle_spawn_proxy", key)
                        up.export(proxy_path)
                        spawn_mesh_file = proxy_path
                    except Exception:
                        spawn_mesh_file = particle_file
            local_centroid = np.asarray(src_mesh.centroid, dtype=float)
    except Exception:
        local_centroid = np.zeros(3, dtype=float)

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

    def _spawn_pos_for_centroid(cx: float, cy: float, cz: float, R_local_to_world: np.ndarray) -> tuple[float, float, float]:
        # centroid_world = pos + R @ (scale * centroid_local)  =>  pos = desired - that offset
        offset = R_local_to_world @ (local_centroid * mesh_scale)
        return (float(cx - offset[0]), float(cy - offset[1]), float(cz - offset[2]))

    entities = []
    if spread <= 1e-9:
        # Single column above the plate: stack along +Y so bodies do not share one point (that breaks FEM contact).
        for i in range(n):
            if i < len(prior):
                x, y, z = _centroid_from_snapshot(prior[i])
                R = np.eye(3, dtype=float)
                quat = gs.utils.geom.R_to_quat(np.eye(3, dtype=float))
            else:
                x, z = 0.0, 0.0
                # Stagger each new particle above the previous; overlap at one (x,z) caused tunneling / blow-ups.
                y = spawn_y0 + float(i) * stack_gap
                R = trimesh.transformations.random_rotation_matrix()[:3, :3]
                quat = gs.utils.geom.R_to_quat(R)
            if max_centroid_y is not None:
                y = min(float(y), float(max_centroid_y))
            spawn_pos = _spawn_pos_for_centroid(x, y, z, R)
            ent = scene.add_entity(
                gs.morphs.Mesh(
                    file=spawn_mesh_file,
                    scale=mesh_scale,
                    pos=spawn_pos,
                    quat=quat,
                    collision=True,
                    visualization=False,
                ),
                material=material,
                surface=gs.surfaces.Default(vis_mode="visual"),
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
                R = np.eye(3, dtype=float)
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
            if max_centroid_y is not None:
                y = min(float(y), float(max_centroid_y))
            placed_positions.append((x, y, z))
            spawn_pos = _spawn_pos_for_centroid(x, y, z, R)
            ent = scene.add_entity(
                gs.morphs.Mesh(
                    file=spawn_mesh_file,
                    scale=mesh_scale,
                    pos=spawn_pos,
                    quat=quat,
                    collision=True,
                    visualization=False,
                ),
                material=material,
                surface=gs.surfaces.Default(vis_mode="visual"),
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
                # NaNs can appear during unstable steps; don't poison metrics.
                v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
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

    Plate: horizontal floor + inner rim (four sides).
    Cylinder: inner bottom disk + inner cylindrical wall.
    Syringe: barrel inner wall + bottom annulus + needle inner wall.
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
    if k == "syringe":
        barrel_d = float(cfg.get("SYRINGE_BARREL_DIAMETER", SYRINGE_BARREL_DIAMETER))
        barrel_h = float(cfg.get("SYRINGE_BARREL_LENGTH", SYRINGE_BARREL_LENGTH))
        needle_d = float(cfg.get("SYRINGE_NEEDLE_DIAMETER", SYRINGE_NEEDLE_DIAMETER))
        needle_h = float(cfg.get("SYRINGE_NEEDLE_LENGTH", SYRINGE_NEEDLE_LENGTH))
        r_barrel = barrel_d * 0.5
        r_needle = needle_d * 0.5
        annulus = math.pi * max(r_barrel * r_barrel - r_needle * r_needle, 0.0)
        barrel_wall = 2.0 * math.pi * r_barrel * max(barrel_h, 0.0)
        needle_wall = 2.0 * math.pi * r_needle * max(needle_h, 0.0)
        return max(annulus + barrel_wall + needle_wall, 1e-18)
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


def compute_mpm_vertex_stress(
    entities: list,
    original_mesh,
    particle_ids: set[int],
) -> dict[int, np.ndarray]:
    """
    Compute per-vertex Von Mises stress from MPM stress tensors.

    Interpolation from material points to mesh vertices uses inverse-distance
    weighting (distance^(-2)) in world space.
    """
    if original_mesh is None or len(original_mesh.vertices) == 0:
        return {}
    verts_local = np.asarray(original_mesh.vertices, dtype=np.float64)
    if verts_local.size == 0:
        return {}

    out: dict[int, np.ndarray] = {}
    for e in entities:
        eid = int(_entity_id(e))
        if eid not in particle_ids:
            continue
        st = _entity_mpm_state(e)
        if st is None or not hasattr(st, "pos") or not hasattr(st, "stress"):
            continue
        mp_pos = _tensor_to_numpy(st.pos)
        mp_stress = _tensor_to_numpy(st.stress)
        if mp_pos.size == 0 or mp_stress.size == 0:
            continue

        pts = np.asarray(mp_pos, dtype=float).reshape(-1, 3)
        S = np.asarray(mp_stress, dtype=float).reshape(-1, 3, 3)
        if pts.shape[0] != S.shape[0]:
            n = min(pts.shape[0], S.shape[0])
            if n <= 0:
                continue
            pts = pts[:n]
            S = S[:n]

        s11 = S[:, 0, 0]
        s22 = S[:, 1, 1]
        s33 = S[:, 2, 2]
        s12 = S[:, 0, 1]
        s23 = S[:, 1, 2]
        s31 = S[:, 2, 0]
        vm = np.sqrt(
            0.5
            * (
                (s11 - s22) ** 2
                + (s22 - s33) ** 2
                + (s33 - s11) ** 2
                + 6.0 * (s12**2 + s23**2 + s31**2)
            )
        )
        vm = np.nan_to_num(vm, nan=0.0, posinf=0.0, neginf=0.0)

        pos, quat_xyzw = _entity_pose(e)
        qx, qy, qz, qw = quat_xyzw
        R = Rotation.from_quat(np.array([qx, qy, qz, qw], dtype=np.float64)).as_matrix()
        verts_world = (R @ verts_local.T).T + np.asarray(pos, dtype=float).reshape(1, 3)

        d = np.linalg.norm(verts_world[:, None, :] - pts[None, :, :], axis=2)
        weights = 1.0 / np.maximum(d, 1e-6) ** 2
        denom = np.sum(weights, axis=1)
        numer = weights @ vm
        vertex_vm = np.where(denom > 0.0, numer / denom, 0.0)
        out[eid] = vertex_vm.astype(np.float64, copy=False)
    return out


def compute_particle_stress_map(contacts, particle_ids: set[int]) -> dict[int, float]:
    # Aggregate per-particle contact "intensity" from force and depth.
    # Only PP (particle–particle) contacts contribute; PC (floor/wall) contacts are excluded
    # so particles sitting on the container floor don't appear red.
    raw_scores: dict[int, float] = {int(pid): 0.0 for pid in particle_ids}
    for c in contacts:
        if not bool(c.is_particle_particle):
            continue
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
    if _mpm_solver_active(scene):
        return []
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


def extract_mpm_contacts(
    entities: list,
    particle_ids: set[int],
    *,
    characteristic_radius: Optional[float] = None,
    depth_tol: float = 1e-5,
) -> list[NormalizedContact]:
    """
    Approximate MPM particle contacts from centroid proximity and stress.

    A pair is in contact when centroid distance is below 1.5 times pair radius.
    Contact force magnitude is estimated as stress Frobenius norm times contact area.
    """
    if not entities:
        return []

    particle_entities = [e for e in entities if int(_entity_id(e)) in particle_ids]
    if len(particle_entities) < 2:
        return []

    default_radius = float(characteristic_radius) if characteristic_radius is not None else None
    if default_radius is not None and (not math.isfinite(default_radius) or default_radius <= 0.0):
        default_radius = None

    centroid_by_id: dict[int, np.ndarray] = {}
    radius_by_id: dict[int, float] = {}
    stress_norm_by_id: dict[int, float] = {}

    for e in particle_entities:
        eid = int(_entity_id(e))
        centroid, _ = _entity_pose(e)
        centroid = np.asarray(centroid, dtype=float).reshape(3)
        centroid_by_id[eid] = centroid

        st = _entity_mpm_state(e)
        r = _estimate_particle_radius_from_state(st)
        if r is None:
            r = default_radius
        if r is None:
            r = 1e-3
        radius_by_id[eid] = max(float(r), 1e-6)

        stress_mag = 0.0
        if st is not None and hasattr(st, "stress"):
            s = _tensor_to_numpy(st.stress)
            if s.size:
                s = np.asarray(s, dtype=float).reshape(-1, 3, 3)
                if hasattr(st, "pos"):
                    mp = _tensor_to_numpy(st.pos)
                    if mp.size:
                        mp = np.asarray(mp, dtype=float).reshape(-1, 3)
                        if mp.shape[0] == s.shape[0]:
                            idx = int(np.argmin(np.linalg.norm(mp - centroid.reshape(1, 3), axis=1)))
                        else:
                            idx = int(s.shape[0] // 2)
                    else:
                        idx = int(s.shape[0] // 2)
                else:
                    idx = int(s.shape[0] // 2)
                stress_mag = float(np.linalg.norm(s[idx], ord="fro"))
                if not math.isfinite(stress_mag):
                    stress_mag = 0.0
        stress_norm_by_id[eid] = max(stress_mag, 0.0)

    contacts: list[NormalizedContact] = []
    for a, b in itertools.combinations(sorted(centroid_by_id.keys()), 2):
        pa = centroid_by_id[a]
        pb = centroid_by_id[b]
        delta = pb - pa
        dist = float(np.linalg.norm(delta))
        if not math.isfinite(dist):
            continue
        r_pair = 0.5 * (radius_by_id[a] + radius_by_id[b])
        threshold = 1.5 * r_pair
        if dist >= threshold:
            continue
        depth = max(0.0, threshold - dist)
        if depth < float(depth_tol):
            continue
        normal = delta / max(dist, 1e-12)
        area = math.pi * min(radius_by_id[a], radius_by_id[b]) ** 2
        stress_pair = 0.5 * (stress_norm_by_id[a] + stress_norm_by_id[b])
        force_mag = float(stress_pair * area)
        contacts.append(
            NormalizedContact(
                entity_a=a,
                entity_b=b,
                is_particle_particle=True,
                is_particle_container=False,
                position=0.5 * (pa + pb),
                normal=normal,
                depth=depth,
                force=force_mag,
                contact_area=area,
            )
        )
    return contacts


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
    pos_world: Optional[np.ndarray] = None,
    R_inv=None,
    verts_local: Optional[np.ndarray] = None,
    sigma_local: float = 0.01,
) -> np.ndarray:
    """Accumulate per-vertex stress contributions for one particle from a contact list.

    When `pos_world`, `R_inv`, and `verts_local` are supplied (preferred path), stress is
    computed as a purely distance-based Gaussian in the particle's *local* mesh frame:

        stress_v += force * exp(-||local_v - local_contact||² / (2 * sigma_local²))

    This places the stress blob exactly at the contact site regardless of particle rotation,
    and the blob width (sigma_local ≈ 25–35 % of mesh char size) keeps it spatially tight.

    Legacy fallback (angular): uses the angle between the vertex normal and the
    vertex→contact direction — retained only when local-frame params are unavailable.
    """
    stress = np.zeros(n_verts, dtype=np.float64)
    for c in contacts:
        p_w = np.asarray(c.position, dtype=np.float64).ravel()[:3]
        if c.force is not None:
            fm = abs(float(c.force))
        else:
            fm = float(c.depth) * 1e6
        if not math.isfinite(fm) or fm < 0.0:
            fm = 0.0

        if pos_world is not None and R_inv is not None and verts_local is not None:
            # ── Primary path: local-space Euclidean distance ──────────────────
            # Transform the world-space contact point into the particle's local frame.
            # This is rotation-invariant and places the highlight at the correct face.
            local_pt = R_inv.apply(p_w - pos_world)
            dist_sq = np.sum((verts_local - local_pt) ** 2, axis=1)
            denom_d = 2.0 * sigma_local ** 2
            stress += fm * np.exp(-dist_sq / denom_d)
        else:
            # ── Legacy fallback: angular Gaussian in world space ──────────────
            vec = p_w.reshape(1, 3) - world_verts
            dist = np.linalg.norm(vec, axis=1)
            dist = np.maximum(dist, 1e-6)
            vec_n = vec / dist.reshape(-1, 1)
            cos_t = np.clip(np.sum(vec_n * n_world, axis=1), -1.0, 1.0)
            angle = np.arccos(cos_t)
            stress += fm * np.exp(-(angle**2) / denom)
    return stress


def _normalize_stress_map(raw: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Normalize all particles' stress arrays by a single global maximum.

    Global normalization ensures only the most heavily contacted vertices reach
    1.0 (red).  Per-particle normalization was inflating lightly-touched particles
    to full red because each particle's tiny local max was scaled to 1.0.
    """
    if not raw:
        return {}
    global_max = float(max(np.max(arr) for arr in raw.values() if arr.size))
    if global_max <= 0.0:
        return {eid: np.zeros_like(arr) for eid, arr in raw.items()}
    return {eid: arr / global_max for eid, arr in raw.items()}


def compute_vertex_stress(
    entities: list,
    contacts: list,
    original_mesh,
    particle_ids: set,
    sigma: float = 0.30,
    stress_floor: float = 0.05,
) -> dict[int, list[float]]:
    """Per-vertex stress map from MPM stress tensors (Von Mises + IDW interpolation)."""
    _ = contacts  # kept for backward-compatible signature
    _ = sigma
    if original_mesh is None or len(original_mesh.vertices) == 0:
        return {}
    n_verts = int(np.asarray(original_mesh.vertices).shape[0])
    floor = float(stress_floor)
    if not math.isfinite(floor) or floor < 0.0:
        floor = 0.05

    pp_norm = _normalize_stress_map(compute_mpm_vertex_stress(entities, original_mesh, particle_ids))

    out: dict[int, list[float]] = {}
    all_eids = set(pp_norm)

    if not all_eids:
        contact_counts: dict[int, int] = {}
        for c in contacts:
            if bool(c.is_particle_particle):
                contact_counts[int(c.entity_a)] = contact_counts.get(int(c.entity_a), 0) + 1
                contact_counts[int(c.entity_b)] = contact_counts.get(int(c.entity_b), 0) + 1
        max_count = max(contact_counts.values()) if contact_counts else 0
        for eid in particle_ids:
            t = float(contact_counts.get(int(eid), 0)) / max_count if max_count > 0 else 0.0
            out[int(eid)] = [t] * n_verts
        return out

    for eid in all_eids:
        blended = pp_norm.get(eid, np.zeros(n_verts, dtype=np.float64))
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
        n_envs = max(1, int(DEFAULT_CONFIG.get("N_ENVS", N_ENVS)))
        init_genesis_compat(gs, backend=backend, n_envs=n_envs)
    except Exception as e:
        if args.backend in {"auto", "gpu"}:
            print(f"[WARNING] gs.init(GPU) failed, falling back to CPU: {e}")
            init_genesis_compat(gs, backend=gs.cpu, n_envs=n_envs)
        else:
            raise

    sim_options = make_sim_options(
        gs,
        {"DT": args.dt, "SUBSTEPS": args.substeps, "GRAVITY": GRAVITY},
    )
    mpm_options = make_mpm_options(gs)
    rigid_options = make_rigid_options(gs)

    # Avoid building the visualizer unless explicitly requested.
    # This prevents the viewer from throttling the run (e.g., ~0.1 FPS on CPU).
    scene = create_scene_compat(
        gs,
        sim_options=sim_options,
        mpm_options=mpm_options,
        rigid_options=rigid_options,
        show_viewer=bool(args.show_viewer),
        n_envs=n_envs,
    )

    original_mesh, physics_norm = load_particle_mesh(PARTICLE_FILE, SCALE_FACTOR)
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
        syringe_barrel_diameter=SYRINGE_BARREL_DIAMETER,
        syringe_barrel_length=SYRINGE_BARREL_LENGTH,
        syringe_needle_diameter=SYRINGE_NEEDLE_DIAMETER,
        syringe_needle_length=SYRINGE_NEEDLE_LENGTH,
        syringe_wall_thickness=SYRINGE_WALL_THICKNESS,
        syringe_bottom_thickness=SYRINGE_BOTTOM_THICKNESS,
        syringe_plate_gap=SYRINGE_PLATE_GAP,
        syringe_segments=SYRINGE_SEGMENTS,
    )
    container_ids = {_entity_id(e) for e in container_ids}
    derived_drop_height = -0.5 * float(SYRINGE_BARREL_LENGTH)
    entities = spawn_particles(
        scene,
        original_mesh,
        args.n,
        env_info,
        derived_drop_height,
        DROP_SPREAD,
        YOUNGS_MODULUS,
        POISSON_RATIO,
        DENSITY,
        particle_restitution=PARTICLE_RESTITUTION,
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
    contacts = extract_mpm_contacts(
        entities,
        particle_ids,
        depth_tol=CONTACT_DEPTH_TOL,
    )
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