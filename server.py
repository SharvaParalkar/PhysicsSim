import io
import json
import logging
import math
import os
import sys
import traceback
import zipfile

import asyncio
import queue
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Optional

import h5py
import pandas as pd
from tqdm import tqdm
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

import simulation

logger = logging.getLogger(__name__)
RUNTIME_BACKEND: str = "cpu"


def _patch_uvicorn_h11_graceful_400() -> None:
    """
    During shutdown, stray TCP probes can trigger RemoteProtocolError → send_400_response while
    h11 is already CLOSED, raising LocalProtocolError and noisy tracebacks. Swallow that case.
    """
    try:
        import h11
        from uvicorn.protocols.http import h11_impl

        _orig = h11_impl.H11Protocol.send_400_response

        def _send_400_safe(self, msg: str) -> None:
            try:
                _orig(self, msg)
            except h11.LocalProtocolError:
                try:
                    if self.transport is not None and not self.transport.is_closing():
                        self.transport.close()
                except Exception:
                    pass

        h11_impl.H11Protocol.send_400_response = _send_400_safe  # type: ignore[method-assign]
    except Exception:
        pass


_patch_uvicorn_h11_graceful_400()


def _sanitize_floats(obj: Any) -> Any:
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_floats(v) for v in obj]
    return obj


def _use_cpu_backend(preferred_backend: str = "auto") -> bool:
    """
    Resolve backend preference.

    Priority:
      1) Explicit preferred_backend argument (`cpu` | `gpu` | `auto`)
      2) GENESIS_USE_CPU (legacy bool override)
      3) GENESIS_BACKEND (`cpu` | `gpu` | `auto`)
      4) Auto-detect: use GPU when torch reports CUDA available
    """
    pb = str(preferred_backend or "auto").strip().lower()
    if pb == "cpu":
        return True
    if pb == "gpu":
        return False
    gc = os.environ.get("GENESIS_USE_CPU", "").strip().lower()
    if gc in ("1", "true", "yes"):
        return True
    if gc in ("0", "false", "no"):
        return False
    env_backend = os.environ.get("GENESIS_BACKEND", "").strip().lower()
    if env_backend == "cpu":
        return True
    if env_backend == "gpu":
        return False
    try:
        import torch

        return not bool(torch.cuda.is_available())
    except Exception:
        return True


def _gpu_cpu_fallback_allowed() -> bool:
    return os.environ.get("GENESIS_NO_CPU_FALLBACK", "").strip().lower() not in ("1", "true", "yes")


def _prepare_cuda_on_worker_thread() -> None:
    """
    Create/bind the CUDA primary context on this thread before Genesis imports torch
    or allocates on GPU. Otherwise PyTorch/CUDA may associate the context with another
    thread and cuMemAllocAsync fails with CUDA_ERROR_INVALID_CONTEXT on the worker.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return
        torch.cuda.init()
        try:
            dev = int(os.environ.get("GENESIS_CUDA_DEVICE", "0"))
        except ValueError:
            dev = 0
        if dev < 0 or dev >= torch.cuda.device_count():
            dev = 0
        torch.cuda.set_device(dev)
        torch.cuda.synchronize()
    except Exception:
        pass


def _init_genesis_on_sim_thread(preferred_backend: str = "auto") -> None:
    """
    Initialize Genesis on the dedicated simulation thread only.

    Taichi/Quadrants LLVM state is tied to the thread that calls gs.init(); running
    Scene.build/step on another thread triggers main_thread_id assertion failures.

    GPU: call _prepare_cuda_on_worker_thread() before any simulation.gs access so
    torch/CUDA bind to this thread. Set GENESIS_USE_CPU=1 to force CPU. Set
    GENESIS_NO_CPU_FALLBACK=1 to surface GPU failures instead of falling back to CPU.
    """
    global RUNTIME_BACKEND
    use_cpu = _use_cpu_backend(preferred_backend)
    n_envs = max(1, int(getattr(simulation, "N_ENVS", 1)))
    if not use_cpu:
        _prepare_cuda_on_worker_thread()
    backend = simulation.gs.cpu if use_cpu else simulation.gs.gpu
    try:
        simulation.init_genesis_compat(simulation.gs, backend=backend, n_envs=n_envs)
        RUNTIME_BACKEND = "cpu" if use_cpu else "gpu"
        return
    except Exception as exc:
        if "already initialized" in str(exc).lower():
            return
        if use_cpu:
            raise
        if not _gpu_cpu_fallback_allowed():
            raise RuntimeError(
                "Genesis GPU init failed and GENESIS_NO_CPU_FALLBACK=1 (no CPU fallback). "
                "Fix CUDA or unset GENESIS_NO_CPU_FALLBACK."
            ) from exc
        logger.warning("Genesis GPU init failed (%s: %s); falling back to gs.cpu.", type(exc).__name__, exc)
    try:
        simulation.init_genesis_compat(simulation.gs, backend=simulation.gs.cpu, n_envs=n_envs)
        RUNTIME_BACKEND = "cpu"
    except Exception as exc2:
        if "already initialized" in str(exc2).lower():
            return
        raise exc2

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

OUTPUT_DIR = getattr(simulation, "OUTPUT_DIR", "./results")
PARTICLES_DIR = Path(__file__).resolve().parent / "Particles"
ENVIRONMENT_DIR = Path(__file__).resolve().parent / "environment"
DEFAULT_PARTICLE_NAME = "particle.obj"
LATEST_Z_HISTORY: list[dict[str, float]] = []
LATEST_MAX_VEL_HISTORY: list[dict[str, float]] = []
LATEST_RATTLERS_HISTORY: list[dict[str, float]] = []
LATEST_KE_HISTORY: list[dict[str, float]] = []
LATEST_PRESSURE_HISTORY: list[dict[str, float]] = []
LATEST_CONTACT_GRAPH_DICT: dict[str, dict[str, dict[str, float | None]]] = {}
LATEST_CONTACT_GRAPH_LINKS: list[dict[str, Any]] = []
# Populated after each completed simulation run; used by /export and /download/obj
LATEST_EXPORT_SUMMARY: dict[str, Any] = {}
LATEST_SIM_TIMESTAMP: str = ""


def _entity_id(e) -> int:
    if hasattr(e, "id"):
        return int(getattr(e, "id"))
    return int(getattr(e, "idx"))


def _list_particle_names() -> list[str]:
    if not PARTICLES_DIR.exists() or not PARTICLES_DIR.is_dir():
        return []
    return sorted(
        p.name
        for p in PARTICLES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() == ".obj"
    )


def _resolve_particle_file(particle_value: Any) -> str:
    # Accept only a plain file name from the Particles folder.
    name = str(particle_value or "").strip()
    if not name:
        name = DEFAULT_PARTICLE_NAME
    name = Path(name).name
    candidate = (PARTICLES_DIR / name).resolve()
    try:
        candidate.relative_to(PARTICLES_DIR.resolve())
    except ValueError as exc:
        raise ValueError(f"Invalid particle file: {name}") from exc
    if not candidate.exists() or not candidate.is_file():
        raise ValueError(f"Particle file not found: {name}")
    if candidate.suffix.lower() != ".obj":
        raise ValueError(f"Unsupported particle file extension: {candidate.suffix}")
    return str(candidate)


def _infer_scale_factor_for_particle_file(particle_file: str) -> float:
    """
    Heuristic unit mapping by filename convention.

    Rhino-exported assets named like `*600M*.obj` are authored in microns.
    Genesis expects metres, so we convert µm -> m via 1e-6.
    """
    name = Path(particle_file).name.lower()
    if "600m" in name:
        return 1e-6
    return 1.0


def _particle_mass_kg_from_runtime(runtime: "SimulationRuntime", rho: float) -> float:
    """Single particle mass from physics mesh volume × density; retries if first volume estimate is zero."""
    pmesh = getattr(runtime, "physics_mesh", None)
    if pmesh is None:
        return 0.0
    vol = 0.0
    try:
        vol = float(pmesh.volume) if pmesh.is_watertight else float(pmesh.convex_hull.volume)
    except Exception:
        vol = 0.0
    mass = max(rho * vol, 0.0)
    if mass <= 0.0:
        vol_fb = 0.0
        try:
            vol_fb = float(pmesh.volume)
        except Exception:
            pass
        if vol_fb <= 0.0:
            try:
                vol_fb = float(pmesh.convex_hull.volume)
            except Exception:
                pass
        mass = max(rho * vol_fb, 0.0)
    return mass


# Physical length keys that must be scaled when physics normalisation is applied.
_PHYS_LENGTH_KEYS: tuple[str, ...] = (
    "PLATE_SIZE",
    "WALL_THICKNESS",
    "PLATE_WALL_HEIGHT",
    "DROP_HEIGHT",
    "CYLINDER_DIAMETER",
    "CYLINDER_HEIGHT",
    "SYRINGE_BARREL_DIAMETER",
    "SYRINGE_BARREL_LENGTH",
    "SYRINGE_NEEDLE_DIAMETER",
    "SYRINGE_NEEDLE_LENGTH",
    "SYRINGE_WALL_THICKNESS",
    "SYRINGE_BOTTOM_THICKNESS",
    "SYRINGE_PLATE_GAP",
)


def _scale_cfg_lengths(cfg: dict, factor: float) -> dict:
    """Return a copy of *cfg* with all physical length keys multiplied by *factor*.

    Used to convert a display-scale config (e.g. PLATE_SIZE=6 mm) into a
    physics-scale config (PLATE_SIZE=6 mm × physics_norm) so that Genesis sees
    a self-consistent world at a numerically stable size.
    """
    if abs(factor - 1.0) < 1e-9:
        return dict(cfg)
    out = dict(cfg)
    for key in _PHYS_LENGTH_KEYS:
        if key in out:
            out[key] = float(out[key]) * factor
    return out


def _rescale_positions(particles: list[dict], factor: float) -> list[dict]:
    """Divide the x/y/z centroid of every particle by *factor*.

    Converts physics-scale positions back to display-scale metres so the viewer
    renders particles at the correct location inside the µm-scale container.
    Returns the original list unchanged when factor ≈ 1.
    """
    if abs(factor - 1.0) < 1e-9:
        return particles
    out: list[dict] = []
    for p in particles:
        pp = dict(p)
        pp["x"] = float(p["x"]) / factor
        pp["y"] = float(p["y"]) / factor
        pp["z"] = float(p["z"]) / factor
        out.append(pp)
    return out


class SimulationAborted(Exception):
    """User requested cancel during scene build or other cooperative checkpoints."""


class SimulationRuntime:
    """
    Persistent process state: Genesis is initialized once.
    `gs.Scene` cannot be `build()` twice — destroy and recreate the scene for each run.
    """

    def __init__(self) -> None:
        self.scene = None
        self.active_entities: list[Any] = []
        self.active_containers: list[Any] = []
        self.physics_mesh = None
        self._original_mesh = None
        self._physics_norm: float = 1.0
        self._phys_cfg: dict = {}
        self.default_particle_file = _resolve_particle_file(DEFAULT_PARTICLE_NAME)
        self._busy = asyncio.Lock()
        self.cancel_requested = False
        self.preferred_backend = "auto"
        self._gs_initialized = False
        # Single worker: all gs.* calls run on this thread (matches LLVM "main" thread).
        self._job_queue: queue.Queue[Any] = queue.Queue()
        self._worker = threading.Thread(target=self._genesis_worker_loop, name="genesis-worker", daemon=True)
        self._worker.start()

    def _genesis_worker_loop(self) -> None:
        """Owns gs.init and every Scene build/step for the process lifetime."""
        while True:
            job = self._job_queue.get()
            if job is None:
                break
            if isinstance(job, tuple) and len(job) == 3 and job[0] == "preflight":
                _, result_q, preferred_backend = job
                try:
                    if not self._gs_initialized:
                        _init_genesis_on_sim_thread(str(preferred_backend or "auto"))
                        self._gs_initialized = True
                    result_q.put((True, f"Genesis preflight OK (backend={RUNTIME_BACKEND})"))
                except Exception as exc:
                    result_q.put((False, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))
                continue
            if isinstance(job, tuple) and len(job) == 2 and job[0] == "clear":
                _, done = job
                try:
                    self.destroy_scene()
                finally:
                    done.set()
                continue
            if isinstance(job, tuple) and len(job) == 2 and job[0] == "shutdown":
                _, done = job
                try:
                    self.destroy_scene()
                except Exception:
                    pass
                try:
                    simulation.gs.destroy()
                except Exception:
                    pass
                finally:
                    done.set()
                continue
            sync_q, cfg, payload = job
            if not self._gs_initialized:
                _init_genesis_on_sim_thread(self.preferred_backend)
                self._gs_initialized = True
            _simulation_thread_main(sync_q, cfg, payload)

    def preflight_runtime(self, preferred_backend: str = "auto", timeout_s: float = 30.0) -> tuple[bool, str]:
        """
        Initialize Genesis on the worker thread before the API starts accepting requests.
        Returns (ok, message) with traceback details on failure.
        """
        result_q: queue.Queue[tuple[bool, str]] = queue.Queue(maxsize=1)
        self._job_queue.put(("preflight", result_q, preferred_backend))
        try:
            ok, msg = result_q.get(timeout=max(1.0, float(timeout_s)))
            return bool(ok), str(msg)
        except Exception as exc:
            return False, f"Timeout/error waiting for preflight: {type(exc).__name__}: {exc}"

    def destroy_scene(self) -> None:
        """Tear down the current scene (safe to call multiple times)."""
        if self.scene is not None:
            try:
                self.scene.destroy()
            except Exception:
                pass
            self.scene = None
        self.active_entities = []
        self.active_containers = []

    def clear_scene(self) -> None:
        """Alias: full destroy for API compatibility with /ws clear command."""
        self.destroy_scene()

    def build_scene(
        self,
        cfg: dict,
        on_progress: Optional[Callable[[str, float, str], None]] = None,
    ) -> tuple[set[int], set[int], list[Any]]:
        def _p(phase: str, pct: float, detail: str = "") -> None:
            if on_progress:
                on_progress(phase, pct, detail)

        # Genesis: add_entity / build only on an unbuilt scene — recreate each run.
        self.destroy_scene()
        if self.cancel_requested:
            raise SimulationAborted()
        _p("init", 0.0, "Creating scene…")
        particle_file = _resolve_particle_file(cfg.get("PARTICLE_FILE"))
        scale_factor = float(cfg.get("SCALE_FACTOR", 1.0))
        if abs(scale_factor - 1.0) < 1e-15:
            inferred = _infer_scale_factor_for_particle_file(particle_file)
            if abs(inferred - 1.0) > 1e-15:
                scale_factor = inferred
                logger.info(
                    "Auto scale applied for %s: SCALE_FACTOR=%g (filename unit hint).",
                    Path(particle_file).name,
                    scale_factor,
                )
        _p("mesh", 0.12, "Loading particle mesh…")
        self.physics_mesh, physics_norm = \
            simulation.load_particle_mesh(particle_file, scale_factor)
        self._original_mesh = self.physics_mesh
        self._physics_norm = physics_norm

        # Build physics-scale config: scale all length dimensions up by physics_norm
        # so Genesis sees a numerically stable world (e.g. particle.obj scale).
        # The original cfg is kept at display scale for the viewer / metrics.
        _phys_cfg = _scale_cfg_lengths(dict(cfg), physics_norm)
        base_grid_density = int(_phys_cfg.get("MPM_GRID_DENSITY", cfg.get("MPM_GRID_DENSITY", 64)))
        build_grid_candidates = [max(32, base_grid_density), 96, 128, 160]
        seen_gd: set[int] = set()
        build_grid_candidates = [gd for gd in build_grid_candidates if not (gd in seen_gd or seen_gd.add(gd))]
        last_exc: Optional[Exception] = None
        entities: list[Any] = []
        for attempt_i, gd in enumerate(build_grid_candidates):
            _phys_cfg["MPM_GRID_DENSITY"] = int(gd)
            # If we retry with a denser MPM grid, also retune DT/SUBSTEPS so substep_dt
            # stays below Genesis' suggested_dt for this grid_density; otherwise the solver
            # can generate NaNs and particles appear to "disappear" in the viewer.
            _phys_cfg = simulation.tune_mpm_timestep(_phys_cfg)
            self._phys_cfg = dict(_phys_cfg)
            # Use the physics-scale tuned dt/substeps. We retune DT/SUBSTEPS alongside
            # MPM grid_density retries; building SimOptions from the untuned display cfg
            # can reintroduce the Genesis substep_dt > suggested_dt instability warning.
            sim_options = simulation.make_sim_options(simulation.gs, self._phys_cfg)
            mpm_options = simulation.make_mpm_options(simulation.gs, self._phys_cfg)
            # Rigid container geometry uses the rigid solver; small dt/substeps => tiny
            # _substep_dt and a warning unless GJK is enabled (see rigid_solver.py).
            rigid_options = simulation.make_rigid_options(simulation.gs, cfg)
            # Explicit FEM at E~1e8 Pa is unstable at typical dt; Genesis recommends implicit FEM.
            fem_options = simulation.make_fem_options(simulation.gs, cfg)
            self.scene = simulation.create_scene_compat(
                simulation.gs,
                sim_options=sim_options,
                mpm_options=mpm_options,
                rigid_options=rigid_options,
                fem_options=fem_options,
                show_viewer=False,
                n_envs=max(1, int(cfg.get("N_ENVS", getattr(simulation, "N_ENVS", 1)))),
            )

            _p("environment", 0.35, f"Building container geometry (grid_density={gd})…")
            containers, env_info = simulation.create_environment(
                self.scene,
                self._phys_cfg["ENVIRONMENT_TYPE"],
                self._phys_cfg["PLATE_SIZE"],
                self._phys_cfg["CYLINDER_DIAMETER"],
                self._phys_cfg["CYLINDER_HEIGHT"],
                self._phys_cfg["CYLINDER_SEGMENTS"],
                self._phys_cfg["WALL_THICKNESS"],
                float(self._phys_cfg.get("PLATE_WALL_HEIGHT", simulation.PLATE_WALL_HEIGHT)),
                env_restitution=float(self._phys_cfg.get("ENV_RESTITUTION", 0.0)),
                syringe_barrel_diameter=float(self._phys_cfg.get("SYRINGE_BARREL_DIAMETER", simulation.SYRINGE_BARREL_DIAMETER)),
                syringe_barrel_length=float(self._phys_cfg.get("SYRINGE_BARREL_LENGTH", simulation.SYRINGE_BARREL_LENGTH)),
                syringe_needle_diameter=float(self._phys_cfg.get("SYRINGE_NEEDLE_DIAMETER", simulation.SYRINGE_NEEDLE_DIAMETER)),
                syringe_needle_length=float(self._phys_cfg.get("SYRINGE_NEEDLE_LENGTH", simulation.SYRINGE_NEEDLE_LENGTH)),
                syringe_wall_thickness=float(self._phys_cfg.get("SYRINGE_WALL_THICKNESS", simulation.SYRINGE_WALL_THICKNESS)),
                syringe_bottom_thickness=float(self._phys_cfg.get("SYRINGE_BOTTOM_THICKNESS", simulation.SYRINGE_BOTTOM_THICKNESS)),
                syringe_plate_gap=float(self._phys_cfg.get("SYRINGE_PLATE_GAP", simulation.SYRINGE_PLATE_GAP)),
                syringe_segments=int(self._phys_cfg.get("SYRINGE_SEGMENTS", simulation.SYRINGE_SEGMENTS)),
                syringe_open_tip=bool(self._phys_cfg.get("SYRINGE_OPEN_TIP", simulation.SYRINGE_OPEN_TIP)),
            )
            self.active_containers = list(containers)
            _p("spawn", 0.55, f"Spawning {int(self._phys_cfg['N_PARTICLES'])} particles…")
            # Pass the configured domain ceiling so spawn can clamp inside the effective boundary.
            mpm_upper_y = None
            try:
                ub = getattr(mpm_options, "upper_bound", None)
                if ub is not None and len(ub) >= 2:
                    mpm_upper_y = float(ub[1])
            except Exception:
                mpm_upper_y = None

            entities = simulation.spawn_particles(
                self.scene,
                self.physics_mesh,
                self._phys_cfg["N_PARTICLES"],
                env_info,
                self._phys_cfg["DROP_HEIGHT"],
                self._phys_cfg["DROP_SPREAD"],
                self._phys_cfg["YOUNGS_MODULUS"],
                self._phys_cfg["POISSON_RATIO"],
                self._phys_cfg["DENSITY"],
                particle_file=particle_file,
                scale_factor=scale_factor,
                particle_restitution=float(self._phys_cfg.get("PARTICLE_RESTITUTION", 0.0)),
                physics_norm=physics_norm,
                mpm_upper_y=mpm_upper_y,
            )
            self.active_entities = list(entities)
            if not _use_cpu_backend(str(cfg.get("BACKEND", "auto"))):
                _prepare_cuda_on_worker_thread()
            _p("build", 0.72, f"Compiling scene (Genesis / Taichi, grid_density={gd})…")
            if self.cancel_requested:
                raise SimulationAborted()
            try:
                self.scene.build()
                break
            except Exception as exc:
                last_exc = exc
                msg = f"{type(exc).__name__}: {exc}"
                is_kth_oob = ("kth(" in msg.lower()) and ("out of bounds" in msg.lower())
                if is_kth_oob and attempt_i < len(build_grid_candidates) - 1:
                    logger.warning(
                        "Scene build failed with low-sample neighbor error at grid_density=%d; retrying with denser MPM grid.",
                        gd,
                    )
                    self.destroy_scene()
                    continue
                raise
        if last_exc is not None and self.scene is None:
            raise last_exc
        _p("ready", 1.0, "Scene ready")
        container_ids = {_entity_id(e) for e in self.active_containers}
        particle_ids = {_entity_id(e) for e in self.active_entities}
        return container_ids, particle_ids, entities


RUNTIME = SimulationRuntime()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail fast: surface Genesis/runtime init errors before Uvicorn reports startup complete.
    ok, msg = await asyncio.to_thread(RUNTIME.preflight_runtime, os.environ.get("GENESIS_BACKEND", "auto"), 45.0)
    if not ok:
        logger.error("Startup preflight failed:\n%s", msg)
        raise RuntimeError(f"Server preflight failed:\n{msg}")
    logger.info("%s", msg)
    yield
    # Tear down Genesis on the same thread as gs.init() so OpenGL / Taichi contexts match.
    RUNTIME.cancel_requested = True
    done = threading.Event()
    RUNTIME._job_queue.put(("shutdown", done))
    await asyncio.to_thread(done.wait, 30.0)


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/particles")
def get_particles():
    particles = _list_particle_names()
    if DEFAULT_PARTICLE_NAME in particles:
        default_particle = DEFAULT_PARTICLE_NAME
    elif particles:
        default_particle = particles[0]
    else:
        default_particle = DEFAULT_PARTICLE_NAME
    return {
        "particles": particles,
        "default": default_particle,
    }


@app.get("/particles/{particle_name}")
def get_particle_obj(particle_name: str):
    try:
        resolved = Path(_resolve_particle_file(particle_name))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(
        content=resolved.read_bytes(),
        media_type="text/plain; charset=utf-8",
    )


def _read_csv_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    if df.empty:
        return []
    return df.where(pd.notna(df), None).to_dict(orient="records")


def _build_graph_from_contact_pairs(contact_pairs: list[dict[str, Any]]) -> tuple[dict[str, dict[str, dict[str, float | None]]], list[dict[str, Any]]]:
    graph: dict[str, dict[str, dict[str, float | None]]] = {}
    links: list[dict[str, Any]] = []
    for row in contact_pairs:
        try:
            a = int(row.get("particle_a"))
            b = int(row.get("particle_b"))
        except Exception:
            continue
        edge = {
            "depth": (float(row["depth"]) if row.get("depth") is not None else 0.0),
            "force": (float(row["force"]) if row.get("force") is not None else None),
            "area": (float(row["contact_area"]) if row.get("contact_area") is not None else None),
        }
        sa = str(a)
        sb = str(b)
        graph.setdefault(sa, {})[sb] = edge
        graph.setdefault(sb, {})[sa] = edge
        links.append({"source": a, "target": b, **edge})
    return graph, links


@app.get("/results")
def get_results():
    out_dir = Path(OUTPUT_DIR)
    return _sanitize_floats(
        {
            "particles": _read_csv_records(out_dir / "particles.csv"),
            "contact_pairs": _read_csv_records(out_dir / "contact_pairs.csv"),
            "contact_points": _read_csv_records(out_dir / "contact_points.csv"),
        }
    )


@app.get("/metrics")
def get_metrics():
    out_dir = Path(OUTPUT_DIR)
    h5_path = out_dir / "simulation.h5"
    if not h5_path.exists():
        h5_path = out_dir / "results.h5"
    out = {
        "Z": 0.0,
        "total_pp": 0,
        "total_pc": 0,
        "n_isolated": 0,
        "n_container_touch": 0,
        "system_pressure": 0.0,
        "z_history": LATEST_Z_HISTORY,
        "max_vel_history": LATEST_MAX_VEL_HISTORY,
        "rattlers_history": LATEST_RATTLERS_HISTORY,
        "kinetic_energy_history": LATEST_KE_HISTORY,
        "pressure_history": LATEST_PRESSURE_HISTORY,
        "contact_graph_dict": LATEST_CONTACT_GRAPH_DICT,
        "contact_graph_links": LATEST_CONTACT_GRAPH_LINKS,
    }
    if not h5_path.exists():
        pairs = _read_csv_records(out_dir / "contact_pairs.csv")
        graph_dict, graph_links = _build_graph_from_contact_pairs(pairs)
        out["contact_graph_dict"] = graph_dict
        out["contact_graph_links"] = graph_links
        return _sanitize_floats(out)
    try:
        with h5py.File(h5_path, "r") as f:
            attrs = f.attrs
            out["Z"] = float(attrs.get("Z", 0.0))
            out["total_pp"] = int(attrs.get("total_pp", 0))
            out["total_pc"] = int(attrs.get("total_pc", 0))
            out["n_isolated"] = int(attrs.get("n_isolated", 0))
            out["n_container_touch"] = int(attrs.get("n_container_touch", 0))
            out["system_pressure"] = float(attrs.get("system_pressure", 0.0))
            return _sanitize_floats(out)
    except Exception:
        return _sanitize_floats(out)


@app.post("/environment/save")
def save_environment_config(payload: dict[str, Any]):
    """Persist syringe/environment presets as JSON files in ./environment."""
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid payload")
    name = str(payload.get("name", "environment")).strip()
    raw_cfg = payload.get("config", {})
    if not isinstance(raw_cfg, dict):
        raise HTTPException(status_code=400, detail="'config' must be an object")

    safe = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in name).strip("_")
    if not safe:
        safe = "environment"

    unit_map = {
        "m": 1.0,
        "meter": 1.0,
        "meters": 1.0,
        "mm": 1e-3,
        "millimeter": 1e-3,
        "millimeters": 1e-3,
        "um": 1e-6,
        "µm": 1e-6,
        "micron": 1e-6,
        "microns": 1e-6,
    }
    units_obj = payload.get("units", {}) if isinstance(payload, dict) else {}
    length_unit = str(units_obj.get("length", "m")).strip().lower() if isinstance(units_obj, dict) else "m"
    length_to_m = float(unit_map.get(length_unit, 1.0))
    syringe_length_keys = (
        "SYRINGE_BARREL_DIAMETER",
        "SYRINGE_BARREL_LENGTH",
        "SYRINGE_NEEDLE_DIAMETER",
        "SYRINGE_NEEDLE_LENGTH",
        "SYRINGE_WALL_THICKNESS",
        "SYRINGE_BOTTOM_THICKNESS",
        "SYRINGE_PLATE_GAP",
    )
    cfg_raw_m = dict(raw_cfg)
    if abs(length_to_m - 1.0) > 1e-18:
        for k in syringe_length_keys:
            if k in cfg_raw_m:
                cfg_raw_m[k] = float(cfg_raw_m[k]) * length_to_m

    cfg = simulation.build_runtime_config(cfg_raw_m)
    wanted = {
        "SYRINGE_BARREL_DIAMETER": float(cfg.get("SYRINGE_BARREL_DIAMETER", simulation.SYRINGE_BARREL_DIAMETER)),
        "SYRINGE_BARREL_LENGTH": float(cfg.get("SYRINGE_BARREL_LENGTH", simulation.SYRINGE_BARREL_LENGTH)),
        "SYRINGE_NEEDLE_DIAMETER": float(cfg.get("SYRINGE_NEEDLE_DIAMETER", simulation.SYRINGE_NEEDLE_DIAMETER)),
        "SYRINGE_NEEDLE_LENGTH": float(cfg.get("SYRINGE_NEEDLE_LENGTH", simulation.SYRINGE_NEEDLE_LENGTH)),
        "SYRINGE_WALL_THICKNESS": float(cfg.get("SYRINGE_WALL_THICKNESS", simulation.SYRINGE_WALL_THICKNESS)),
        "SYRINGE_BOTTOM_THICKNESS": float(cfg.get("SYRINGE_BOTTOM_THICKNESS", simulation.SYRINGE_BOTTOM_THICKNESS)),
        "SYRINGE_PLATE_GAP": float(cfg.get("SYRINGE_PLATE_GAP", 0.0)),
        "SYRINGE_SEGMENTS": int(cfg.get("SYRINGE_SEGMENTS", simulation.SYRINGE_SEGMENTS)),
    }
    if wanted["SYRINGE_NEEDLE_DIAMETER"] >= wanted["SYRINGE_BARREL_DIAMETER"]:
        raise HTTPException(status_code=400, detail="Needle diameter must be less than barrel diameter")

    ENVIRONMENT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ENVIRONMENT_DIR / f"{safe}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "name": safe,
                "type": "syringe",
                "units": {"length": "m"},
                "config": wanted,
            },
            f,
            indent=2,
        )

    return {"ok": True, "path": str(out_path.relative_to(Path(__file__).resolve().parent))}


@app.get("/environment/load")
def load_environment_config(name: str):
    """Load a saved syringe/environment preset from ./environment by name."""
    raw_name = str(name or "").strip()
    if not raw_name:
        raise HTTPException(status_code=400, detail="Missing preset name")
    safe = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in raw_name).strip("_")
    if not safe:
        raise HTTPException(status_code=400, detail="Invalid preset name")

    in_path = ENVIRONMENT_DIR / f"{safe}.json"
    if not in_path.exists():
        raise HTTPException(status_code=404, detail=f"Preset '{safe}' not found")

    try:
        with in_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read preset: {exc}") from exc

    unit_map = {
        "m": 1.0,
        "meter": 1.0,
        "meters": 1.0,
        "mm": 1e-3,
        "millimeter": 1e-3,
        "millimeters": 1e-3,
        "um": 1e-6,
        "µm": 1e-6,
        "micron": 1e-6,
        "microns": 1e-6,
    }
    units_obj = payload.get("units", {}) if isinstance(payload, dict) else {}
    length_unit = str(units_obj.get("length", "m")).strip().lower() if isinstance(units_obj, dict) else "m"
    length_to_m = float(unit_map.get(length_unit, 1.0))
    raw_cfg = payload.get("config", {}) if isinstance(payload, dict) else {}
    if not isinstance(raw_cfg, dict):
        raise HTTPException(status_code=400, detail="Preset config is invalid")
    syringe_length_keys = (
        "SYRINGE_BARREL_DIAMETER",
        "SYRINGE_BARREL_LENGTH",
        "SYRINGE_NEEDLE_DIAMETER",
        "SYRINGE_NEEDLE_LENGTH",
        "SYRINGE_WALL_THICKNESS",
        "SYRINGE_BOTTOM_THICKNESS",
        "SYRINGE_PLATE_GAP",
    )
    cfg_raw_m = dict(raw_cfg)
    if abs(length_to_m - 1.0) > 1e-18:
        for k in syringe_length_keys:
            if k in cfg_raw_m:
                cfg_raw_m[k] = float(cfg_raw_m[k]) * length_to_m

    cfg = simulation.build_runtime_config(cfg_raw_m)
    wanted = {
        "SYRINGE_BARREL_DIAMETER": float(cfg.get("SYRINGE_BARREL_DIAMETER", simulation.SYRINGE_BARREL_DIAMETER)),
        "SYRINGE_BARREL_LENGTH": float(cfg.get("SYRINGE_BARREL_LENGTH", simulation.SYRINGE_BARREL_LENGTH)),
        "SYRINGE_NEEDLE_DIAMETER": float(cfg.get("SYRINGE_NEEDLE_DIAMETER", simulation.SYRINGE_NEEDLE_DIAMETER)),
        "SYRINGE_NEEDLE_LENGTH": float(cfg.get("SYRINGE_NEEDLE_LENGTH", simulation.SYRINGE_NEEDLE_LENGTH)),
        "SYRINGE_WALL_THICKNESS": float(cfg.get("SYRINGE_WALL_THICKNESS", simulation.SYRINGE_WALL_THICKNESS)),
        "SYRINGE_BOTTOM_THICKNESS": float(cfg.get("SYRINGE_BOTTOM_THICKNESS", simulation.SYRINGE_BOTTOM_THICKNESS)),
        "SYRINGE_PLATE_GAP": float(cfg.get("SYRINGE_PLATE_GAP", 0.0)),
        "SYRINGE_SEGMENTS": int(cfg.get("SYRINGE_SEGMENTS", simulation.SYRINGE_SEGMENTS)),
    }
    return {"ok": True, "name": safe, "units": {"length": "m"}, "config": wanted}


def _build_export_payload() -> dict[str, Any]:
    """Build the full simulation export payload (summary + enriched particles + contacts)."""
    out_dir = Path(OUTPUT_DIR)
    particles_raw = _read_csv_records(out_dir / "particles.csv")
    contact_pairs_raw = _read_csv_records(out_dir / "contact_pairs.csv")
    contact_points_raw = _read_csv_records(out_dir / "contact_points.csv")

    # Build neighbor lists from contact pairs
    neighbors: dict[int, list[int]] = {}
    for row in contact_pairs_raw:
        try:
            a, b = int(row["particle_a"]), int(row["particle_b"])
        except Exception:
            continue
        neighbors.setdefault(a, []).append(b)
        neighbors.setdefault(b, []).append(a)

    particles: list[dict[str, Any]] = []
    for p in particles_raw:
        pid = int(p.get("id", 0))
        particles.append({
            "id": pid,
            "x": p.get("x"),
            "y": p.get("y"),
            "z": p.get("z"),
            "qx": p.get("qx"),
            "qy": p.get("qy"),
            "qz": p.get("qz"),
            "qw": p.get("qw"),
            "n_contacts": p.get("n_contacts"),
            "neighbors": neighbors.get(pid, []),
        })

    # Merge contact points into contact pairs (best-effort by index)
    contacts: list[dict[str, Any]] = []
    for i, row in enumerate(contact_pairs_raw):
        entry: dict[str, Any] = {
            "particle_a": row.get("particle_a"),
            "particle_b": row.get("particle_b"),
            "depth": row.get("depth"),
            "force": row.get("force"),
            "px": None, "py": None, "pz": None,
            "nx": None, "ny": None, "nz": None,
        }
        if i < len(contact_points_raw):
            pt = contact_points_raw[i]
            entry.update(px=pt.get("x"), py=pt.get("y"), pz=pt.get("z"),
                         nx=pt.get("nx"), ny=pt.get("ny"), nz=pt.get("nz"))
        contacts.append(entry)

    summary: dict[str, Any] = dict(LATEST_EXPORT_SUMMARY)
    if not summary:
        # Fall back to reading from files if globals not yet populated
        h5_path = out_dir / "simulation.h5"
        if not h5_path.exists():
            h5_path = out_dir / "results.h5"
        try:
            if h5_path.exists():
                with h5py.File(h5_path, "r") as f:
                    summary = {
                        "Z": float(f.attrs.get("Z", 0.0)),
                        "total_pp": int(f.attrs.get("total_pp", 0)),
                        "total_pc": int(f.attrs.get("total_pc", 0)),
                        "n_isolated": int(f.attrs.get("n_isolated", 0)),
                        "n_container_touch": int(f.attrs.get("n_container_touch", 0)),
                        "system_pressure": float(f.attrs.get("system_pressure", 0.0)),
                        "contact_efficiency": 0.0,
                        "total_particle_volume": 0.0,
                        "n_particles": len(particles),
                    }
        except Exception:
            summary = {}

    summary["timestamp"] = LATEST_SIM_TIMESTAMP or ""

    return _sanitize_floats({
        "summary": summary,
        "particles": particles,
        "contacts": contacts,
    })


@app.get("/export")
def get_export():
    """Full post-simulation payload for the results overlay."""
    return _build_export_payload()


@app.get("/download/particles-csv")
def download_particles_csv():
    path = Path(OUTPUT_DIR) / "particles.csv"
    if not path.exists():
        return Response(content="", media_type="text/csv",
                        headers={"Content-Disposition": "attachment; filename=particles.csv"})
    return Response(content=path.read_bytes(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=particles.csv"})


@app.get("/download/contacts-csv")
def download_contacts_csv():
    path = Path(OUTPUT_DIR) / "contact_pairs.csv"
    if not path.exists():
        return Response(content="", media_type="text/csv",
                        headers={"Content-Disposition": "attachment; filename=contact_pairs.csv"})
    return Response(content=path.read_bytes(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=contact_pairs.csv"})


@app.get("/download/contact-points-csv")
def download_contact_points_csv():
    path = Path(OUTPUT_DIR) / "contact_points.csv"
    if not path.exists():
        return Response(content="", media_type="text/csv",
                        headers={"Content-Disposition": "attachment; filename=contact_points.csv"})
    return Response(content=path.read_bytes(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=contact_points.csv"})


@app.get("/download/summary-json")
def download_summary_json():
    path = Path(OUTPUT_DIR) / "summary.json"
    if path.exists():
        content = path.read_bytes()
    elif LATEST_EXPORT_SUMMARY:
        content = json.dumps(LATEST_EXPORT_SUMMARY, indent=2).encode("utf-8")
    else:
        content = b"{}"
    return Response(content=content, media_type="application/json",
                    headers={"Content-Disposition": "attachment; filename=summary.json"})


@app.get("/download/results-zip")
def download_results_zip():
    """Package all CSV and JSON results files into a single ZIP for Analysis.html import."""
    import datetime
    out_dir = Path(OUTPUT_DIR)
    buf = io.BytesIO()
    files_added: list[str] = []

    ts = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for ext in ("*.csv", "*.json"):
            for path in sorted(out_dir.glob(ext)):
                zf.write(path, path.name)
                files_added.append(path.name)

        # Generate and embed the full export JSON (enriched particles + contacts + summary)
        try:
            export_payload = _build_export_payload()
            export_filename = f"simulation_export_{ts}.json"
            zf.writestr(export_filename, json.dumps(export_payload, indent=2))
            files_added.append(export_filename)
        except Exception as exc:
            logger.warning("Could not build export JSON for ZIP: %s", exc)

        # Embed a manifest so Analysis.html knows what's inside
        manifest = {
            "exported_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "output_dir": str(out_dir),
            "files": files_added,
        }
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))

    if not files_added:
        return Response(
            content="No results found in output directory. Run a simulation first.",
            status_code=404,
        )

    buf.seek(0)
    return Response(
        content=buf.read(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=simulation_results_{ts}.zip"},
    )


@app.get("/download/obj")
def download_obj():
    """Return a ZIP containing settled_particles.obj, contact_network.obj, README.txt."""
    entities = list(RUNTIME.active_entities)
    original_mesh = RUNTIME._original_mesh

    if not entities or original_mesh is None:
        # Try to return whatever CSVs exist with a helpful message
        return Response(
            content="No settled simulation data available. Run a simulation first.",
            status_code=404,
        )

    try:
        particles_obj = simulation.export_settled_obj(entities, original_mesh)
    except Exception as exc:
        logger.error("export_settled_obj failed: %s", exc)
        return Response(content=f"OBJ export failed: {exc}", status_code=500)

    try:
        network_obj = simulation.export_contact_network_obj(entities, LATEST_CONTACT_GRAPH_LINKS)
    except Exception as exc:
        logger.error("export_contact_network_obj failed: %s", exc)
        network_obj = "# Contact network export failed\n"

    n = len(entities)
    readme = (
        "GRANULAR JAMMING SIMULATION — SETTLED GEOMETRY\n"
        "================================================\n\n"
        "Coordinate system : Y-up, metres\n"
        f"Particle count    : {n}\n\n"
        "Files\n"
        "-----\n"
        "settled_particles.obj\n"
        f"  {n} named mesh groups (o particle_0 … o particle_{n - 1}).\n"
        "  Each group is one particle in world-space coordinates.\n\n"
        "contact_network.obj\n"
        "  Line segments (l commands) connecting the centres of contacting\n"
        "  particle pairs.  Import as a separate layer.\n\n"
        "How to import into Rhino\n"
        "------------------------\n"
        "  1. File > Import > settled_particles.obj\n"
        "     Each particle arrives as a separate mesh object.\n"
        "  2. File > Import > contact_network.obj\n"
        "     Lines land on a new layer; use as a reference network.\n\n"
        "Units: metres.  Scale by 1000 to convert to millimetres.\n"
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("settled_particles.obj", particles_obj)
        zf.writestr("contact_network.obj", network_obj)
        zf.writestr("README.txt", readme)
    buf.seek(0)

    return Response(
        content=buf.read(),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=settled_geometry.zip"},
    )


@app.get("/vertex-stress")
def get_vertex_stress():
    h5_path = Path(OUTPUT_DIR) / "results.h5"
    if not h5_path.exists():
        return {}
    with h5py.File(h5_path, "r") as f:
        if "vertex_stress" not in f:
            return {}
        grp = f["vertex_stress"]
        return {str(eid): grp[eid][:].tolist() for eid in grp.keys()}


@app.post("/cancel")
def cancel_simulation():
    RUNTIME.cancel_requested = True
    return {"ok": True, "message": "Cancellation requested"}


def _simulation_thread_main(sync_q: "queue.Queue[Any | None]", cfg: dict[str, Any], payload: dict[str, Any]) -> None:
    """Blocking Genesis loop (runs in a worker thread). Puts dict messages on sync_q; ends with None."""
    global LATEST_Z_HISTORY, LATEST_CONTACT_GRAPH_DICT, LATEST_CONTACT_GRAPH_LINKS, LATEST_MAX_VEL_HISTORY
    global LATEST_RATTLERS_HISTORY, LATEST_KE_HISTORY, LATEST_PRESSURE_HISTORY
    global LATEST_EXPORT_SUMMARY, LATEST_SIM_TIMESTAMP
    frame_every = int(payload.get("frame_every", 30))
    frame_every = max(1, frame_every)
    live_metrics_every = max(1, int(payload.get("live_metrics_every", 10)))
    log_every = max(1, int(payload.get("log_every", 60)))
    try:
        RUNTIME.cancel_requested = False
        sync_q.put({"type": "log", "line": "Preparing persistent scene (new Scene per run)..."})

        def _on_build_progress(phase: str, pct: float, detail: str = "") -> None:
            if RUNTIME.cancel_requested:
                raise SimulationAborted()
            sync_q.put({"type": "progress", "phase": phase, "pct": pct, "detail": detail})

        cfg_run = dict(cfg)

        # NOTE: The UI already pre-scales environment dimensions (PLATE_SIZE, DROP_HEIGHT, etc.)
        # to match the selected particle mesh when Star600M.obj (or other µm-scale meshes) is
        # chosen. A previous server-side auto-scale by the mesh scale_factor (1e-6) caused
        # double-scaling (e.g. 6mm → 6nm), making the container impossibly tiny and particles
        # invisible. Environment scaling is now handled entirely by the frontend.

        duration = float(cfg["SIM_DURATION"])
        settle_threshold = float(cfg["SETTLE_THRESHOLD"])
        depth_tol = float(cfg_run["CONTACT_DEPTH_TOL"])
        LATEST_Z_HISTORY = []
        LATEST_MAX_VEL_HISTORY = []
        LATEST_RATTLERS_HISTORY = []
        LATEST_KE_HISTORY = []
        LATEST_PRESSURE_HISTORY = []
        LATEST_CONTACT_GRAPH_DICT = {}
        LATEST_CONTACT_GRAPH_LINKS = []
        surface_area_m2 = float(simulation.container_surface_area_m2(str(cfg.get("ENVIRONMENT_TYPE", "plate")), cfg))
        rho = float(cfg["DENSITY"])

        container_ids: set[int] = set()
        particle_ids: set[int] = set()
        entities: list[Any] = []
        particle_mass_kg = 0.0

        container_ids, particle_ids, entities = RUNTIME.build_scene(cfg_run, on_progress=_on_build_progress)
        particle_mass_kg = _particle_mass_kg_from_runtime(RUNTIME, rho)
        # If physics normalisation was applied, override depth_tol to physics scale
        if RUNTIME._phys_cfg:
            depth_tol = float(RUNTIME._phys_cfg.get("CONTACT_DEPTH_TOL", depth_tol))

        dt = float(cfg_run["DT"])
        # Headless: skip per-step visualizer GPU/raster updates (still built at scene.build()).
        # Otherwise each step pays full visualizer.update() cost even with show_viewer=False.
        def _step() -> None:
            RUNTIME.scene.step(update_visualizer=False)

        # Show spawn poses immediately so the UI is not blank until the first (slow) CPU step.
        sync_q.put(
            {
                "type": "frame",
                "step": -1,
                "t": 0.0,
                "max_vel": 0.0,
                "Z": 0.0,
                "particles": _rescale_positions(
                    simulation._collect_particle_transforms(entities),
                    RUNTIME._physics_norm,
                ),
            }
        )

        t_elapsed = 0.0
        last_ws_pct = -1.0
        last_live_Z = 0.0

        def _run_step_block(frame_step_idx: int, t_now: float, phase_step: int, phase_len: int) -> bool:
            """
            After a physics substep: metrics, frames, and settle checks.
            Returns True to stop the outer simulation (cancel or settled).
            """
            nonlocal entities, particle_ids, container_ids, dt, t_elapsed, last_ws_pct, particle_mass_kg, last_live_Z
            if RUNTIME.cancel_requested:
                sync_q.put({"type": "log", "line": "Simulation cancelled by user"})
                sync_q.put({"type": "cancelled"})
                sync_q.put({"type": "idle", "message": "Ready for next run"})
                return True

            max_vel = simulation.compute_max_velocity(entities)

            if duration > 0:
                sp = t_now / duration
                if sp >= 1.0:
                    sp = 1.0
                if sp - last_ws_pct >= 0.02 or frame_step_idx == 0:
                    last_ws_pct = sp
                    sync_q.put(
                        {
                            "type": "progress",
                            "phase": "simulate",
                            "pct": sp,
                            "detail": f"t={t_now:.3f}s frame={frame_step_idx} phase_step={phase_step + 1}/{phase_len}",
                        }
                    )

            need_contacts = (frame_step_idx % live_metrics_every == 0) or (
                frame_step_idx % frame_every == 0
            )
            if need_contacts:
                contacts_snapshot = simulation.extract_mpm_contacts(
                    entities,
                    particle_ids,
                    depth_tol=depth_tol,
                )

            if frame_step_idx % live_metrics_every == 0:
                lm = simulation.calculate_live_metrics(
                    contacts_snapshot,
                    particle_ids,
                    entities,
                    particle_mass_kg=particle_mass_kg,
                    surface_area_m2=surface_area_m2,
                    depth_tol=depth_tol,
                )
                last_live_Z = float(lm["Z"])
                LATEST_Z_HISTORY.append({"t": t_now, "Z": last_live_Z})
                LATEST_RATTLERS_HISTORY.append({"t": t_now, "n_rattlers": float(lm["n_rattlers"])})
                LATEST_KE_HISTORY.append({"t": t_now, "kinetic_energy": float(lm["kinetic_energy"])})
                LATEST_PRESSURE_HISTORY.append({"t": t_now, "system_pressure": float(lm["system_pressure"])})
                sync_q.put(
                    {
                        "type": "live_metrics",
                        "step": int(frame_step_idx),
                        "t": t_now,
                        "Z": last_live_Z,
                        "n_rattlers": int(lm["n_rattlers"]),
                        "kinetic_energy": float(lm["kinetic_energy"]),
                        "system_pressure": float(lm["system_pressure"]),
                    }
                )

            if frame_step_idx % frame_every == 0:
                # Live frames: per-particle scalar stress from contacts (fast dict aggregate); full
                # per-vertex Hertzian gradient replaces this when the sim completes.
                stress_map = simulation.compute_particle_stress_map(contacts_snapshot, particle_ids)
                fem_maps: dict = {}
                gmax = 1.0
                LATEST_MAX_VEL_HISTORY.append({"t": t_now, "max_vel": float(max_vel or 0.0)})
                sync_q.put(
                    {
                        "type": "frame",
                        "step": int(frame_step_idx),
                        "t": t_now,
                        "max_vel": float(max_vel or 0.0),
                        "Z": last_live_Z,
                        "particles": _rescale_positions(
                            simulation._collect_particle_transforms(
                                entities,
                                stress_map,
                                fem_vertex_norms=fem_maps,
                                fem_norm_global_max=gmax,
                            ),
                            RUNTIME._physics_norm,
                        ),
                    }
                )

            if (phase_step % log_every == 0) or (phase_step == phase_len - 1):
                sync_q.put({"type": "log", "line": f"t={t_now:.2f}s phase_step={phase_step + 1}/{phase_len}"})

            if max_vel is not None and max_vel < settle_threshold:
                sync_q.put({"type": "log", "line": f"Settled at t={t_now:.2f}s"})
                return True
            return False

        # Avoid tqdm's terminal control sequences when stderr is not a TTY (e.g. some IDE
        # captures) — they can garble logs and occasionally upset Windows consoles.
        total_steps = int(duration / dt) if dt > 0 else 0
        sim_pbar = tqdm(
            range(total_steps),
            desc="Simulation",
            unit="step",
            dynamic_ncols=True,
            mininterval=0.25,
            file=sys.stderr,
            disable=not sys.stderr.isatty(),
        )
        try:
            for step in sim_pbar:
                if RUNTIME.cancel_requested:
                    sync_q.put({"type": "log", "line": "Simulation cancelled by user"})
                    sync_q.put({"type": "cancelled"})
                    sync_q.put({"type": "idle", "message": "Ready for next run"})
                    return
                if step == 0:
                    sync_q.put(
                        {
                            "type": "log",
                            "line": "First physics step after build: Taichi/JIT + implicit FEM can take minutes on CPU; tqdm may sit at 0% until it completes.",
                        }
                    )
                _step()
                t_elapsed += dt
                t_now = t_elapsed
                if _run_step_block(step, t_now, step, total_steps):
                    break
        finally:
            sim_pbar.close()

        if RUNTIME.cancel_requested:
            return

        sync_q.put({"type": "progress", "phase": "simulate", "pct": 1.0, "detail": "Finishing…"})
        contacts = simulation.extract_mpm_contacts(
            entities,
            particle_ids,
            depth_tol=depth_tol,
        )
        if len(contacts) == 0:
            print("[INFO] falling back to geometric contact detection", flush=True)
            contacts = simulation.extract_contacts_geometric(
                entities,
                particle_ids,
                RUNTIME.active_containers,
                RUNTIME._original_mesh,
                depth_tol=depth_tol,
            )
        print(f"[DEBUG] final contact count={len(contacts)}", flush=True)
        metrics = simulation.compute_metrics(contacts, particle_ids, container_surface_area_m2=surface_area_m2, depth_tol=depth_tol)
        vertex_stress: dict[int, list[float]] = {}
        if RUNTIME._original_mesh is not None:
            vertex_stress = simulation.compute_vertex_stress(
                entities,
                contacts,
                RUNTIME._original_mesh,
                particle_ids,
                sigma=float(cfg.get("STRESS_SIGMA", 0.12)),
                stress_floor=float(cfg.get("STRESS_FLOOR", 0.05)),
            )
        LATEST_CONTACT_GRAPH_DICT = metrics.get("contact_graph_dict", {}) or {}
        LATEST_CONTACT_GRAPH_LINKS = [
            {
                "source": int(a),
                "target": int(b),
                "depth": float(data.get("depth", 0.0)),
                "force": (float(data["force"]) if data.get("force") is not None else None),
                "area": (float(data["area"]) if data.get("area") is not None else None),
            }
            for a, b, data in metrics["contact_graph"].edges(data=True)
        ]
        simulation.export_results(
            entities,
            metrics,
            cfg["OUTPUT_DIR"],
            save_hdf5=bool(cfg["SAVE_HDF5"]),
            save_csv=bool(cfg["SAVE_CSV"]),
            vertex_stress=vertex_stress,
        )
        # Build and persist summary.json; store in global for /export endpoint
        try:
            LATEST_EXPORT_SUMMARY = simulation.export_summary_json(
                entities, metrics, cfg["OUTPUT_DIR"], RUNTIME._original_mesh
            )
        except Exception as _exc_sum:
            logger.warning("export_summary_json failed: %s", _exc_sum)
            LATEST_EXPORT_SUMMARY = {}
        import datetime
        LATEST_SIM_TIMESTAMP = datetime.datetime.now().isoformat(timespec="seconds")
        mesh_vertex_count = len(RUNTIME._original_mesh.vertices) if RUNTIME._original_mesh else 0
        sync_q.put(
            _sanitize_floats(
                {
                    "type": "complete",
                    "metrics": {
                        "Z": float(metrics.get("Z", 0.0)),
                        "total_pp": int(metrics.get("total_pp_contacts", 0)),
                        "system_pressure": float(metrics.get("system_pressure", 0.0)),
                    },
                    "vertex_stress": {str(eid): vals for eid, vals in vertex_stress.items()},
                    "mesh_vertex_count": mesh_vertex_count,
                }
            )
        )
        sync_q.put({"type": "idle", "message": "Ready for next run"})
    except SimulationAborted:
        try:
            RUNTIME.destroy_scene()
        except Exception:
            pass
        sync_q.put({"type": "log", "line": "Simulation cancelled by user"})
        sync_q.put({"type": "cancelled"})
        sync_q.put({"type": "idle", "message": "Ready for next run"})
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("Simulation worker failed: %s: %s\n%s", type(exc).__name__, exc, tb)
        sync_q.put({"type": "error", "message": f"{type(exc).__name__}: {exc}", "traceback": tb})
        sync_q.put({"type": "idle", "message": "Ready after error"})
    finally:
        sync_q.put(None)


async def _run_simulation(ws: WebSocket, payload: dict[str, Any]) -> None:
    cfg = simulation.build_runtime_config(payload.get("config") if isinstance(payload, dict) else {})
    requested_backend = str(cfg.get("BACKEND", "auto")).strip().lower()
    sync_q: queue.Queue[Any | None] = queue.Queue()
    try:
        async with RUNTIME._busy:
            if not RUNTIME._gs_initialized:
                RUNTIME.preferred_backend = requested_backend
            elif requested_backend in ("cpu", "gpu") and requested_backend != RUNTIME_BACKEND:
                await ws.send_json(
                    {
                        "type": "log",
                        "line": (
                            f"Backend already initialized as {RUNTIME_BACKEND}; "
                            f"ignoring requested BACKEND={requested_backend} for this process."
                        ),
                    }
                )
            RUNTIME._job_queue.put((sync_q, cfg, payload))
            while True:
                item = await asyncio.to_thread(sync_q.get)
                if item is None:
                    break
                await ws.send_json(item)
    except WebSocketDisconnect:
        RUNTIME.cancel_requested = True
        raise


def _drain_finished_sim_task(task: Optional[asyncio.Task]) -> Optional[asyncio.Task]:
    if task is None or not task.done():
        return task
    try:
        task.result()
    except WebSocketDisconnect:
        pass
    except asyncio.CancelledError:
        pass
    except Exception:
        pass
    return None


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    await ws.send_json({"type": "ready", "message": "Send {type:'start', config:{...}}"})
    active_sim: Optional[asyncio.Task] = None
    try:
        while True:
            active_sim = _drain_finished_sim_task(active_sim)

            payload = await ws.receive_json()
            if not isinstance(payload, dict):
                await ws.send_json({"type": "error", "message": "Invalid payload"})
                continue
            cmd = str(payload.get("type", "")).strip().lower()
            if cmd == "start":
                active_sim = _drain_finished_sim_task(active_sim)
                if active_sim is not None:
                    await ws.send_json({"type": "error", "message": "Simulation already running"})
                    continue
                active_sim = asyncio.create_task(_run_simulation(ws, payload))
                continue
            if cmd == "cancel":
                RUNTIME.cancel_requested = True
                await ws.send_json({"type": "log", "line": "Cancellation requested"})
                continue
            if cmd == "clear":
                done = threading.Event()
                RUNTIME._job_queue.put(("clear", done))
                await asyncio.to_thread(done.wait)
                await ws.send_json({"type": "cleared"})
                continue
            await ws.send_json({"type": "error", "message": "Unknown command. Use start|cancel|clear"})
    except WebSocketDisconnect:
        RUNTIME.cancel_requested = True
        if active_sim is not None and not active_sim.done():
            active_sim.cancel()
        return
    except Exception as exc:
        try:
            await ws.send_json({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    # Proactor + graceful shutdown can hit asyncio assert / h11 races on Windows; selector is stabler.
    if sys.platform == "win32":
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception:
            pass

    # Pass `app` directly. Using "server:app" makes uvicorn import `server` again while
    # this file already ran as __main__, duplicating RUNTIME / genesis-worker / gs.init.
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level=str(os.environ.get("UVICORN_LOG_LEVEL", "info")).lower(),
    )

