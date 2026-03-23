import io
import json
import logging
import math
import os
import sys
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


def _use_cpu_backend() -> bool:
    """Match CLI: GENESIS_USE_CPU=1|0 overrides; else use simulation.BACKEND (e.g. cpu vs auto)."""
    env = os.environ.get("GENESIS_USE_CPU", "").strip().lower()
    if env in ("1", "true", "yes"):
        return True
    if env in ("0", "false", "no"):
        return False
    return getattr(simulation, "BACKEND", "auto").strip().lower() == "cpu"


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


def _init_genesis_on_sim_thread() -> None:
    """
    Initialize Genesis on the dedicated simulation thread only.

    Taichi/Quadrants LLVM state is tied to the thread that calls gs.init(); running
    Scene.build/step on another thread triggers main_thread_id assertion failures.

    GPU: call _prepare_cuda_on_worker_thread() before any simulation.gs access so
    torch/CUDA bind to this thread. Set GENESIS_USE_CPU=1 to force CPU. Set
    GENESIS_NO_CPU_FALLBACK=1 to surface GPU failures instead of falling back to CPU.
    """
    use_cpu = _use_cpu_backend()
    if not use_cpu:
        _prepare_cuda_on_worker_thread()
    backend = simulation.gs.cpu if use_cpu else simulation.gs.gpu
    try:
        simulation.gs.init(backend=backend)
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
        simulation.gs.init(backend=simulation.gs.cpu)
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
        self.default_particle_file = _resolve_particle_file(DEFAULT_PARTICLE_NAME)
        self._busy = asyncio.Lock()
        self.cancel_requested = False
        # Single worker: all gs.* calls run on this thread (matches LLVM "main" thread).
        self._job_queue: queue.Queue[Any] = queue.Queue()
        self._worker = threading.Thread(target=self._genesis_worker_loop, name="genesis-worker", daemon=True)
        self._worker.start()

    def _genesis_worker_loop(self) -> None:
        """Owns gs.init and every Scene build/step for the process lifetime."""
        _init_genesis_on_sim_thread()
        while True:
            job = self._job_queue.get()
            if job is None:
                break
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
            _simulation_thread_main(sync_q, cfg, payload)

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
        *,
        prior_fem_snapshots: Optional[list[dict]] = None,
    ) -> tuple[set[int], set[int], list[Any]]:
        def _p(phase: str, pct: float, detail: str = "") -> None:
            if on_progress:
                on_progress(phase, pct, detail)

        # Genesis: add_entity / build only on an unbuilt scene — recreate each run.
        self.destroy_scene()
        if self.cancel_requested:
            raise SimulationAborted()
        _p("init", 0.0, "Creating scene…")
        sim_options = simulation.make_sim_options(simulation.gs, cfg)
        # Rigid container geometry uses the rigid solver; small dt/substeps => tiny
        # _substep_dt and a warning unless GJK is enabled (see rigid_solver.py).
        rigid_options = simulation.make_rigid_options(simulation.gs, cfg)
        # Explicit FEM at E~1e8 Pa is unstable at typical dt; Genesis recommends implicit FEM.
        fem_options = simulation.make_fem_options(simulation.gs, cfg)
        self.scene = simulation.gs.Scene(
            sim_options=sim_options,
            rigid_options=rigid_options,
            fem_options=fem_options,
            show_viewer=False,
        )
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
        self.physics_mesh, self._original_mesh = simulation.load_particle_mesh(particle_file, scale_factor)
        _p("environment", 0.35, "Building container geometry…")
        containers, env_info = simulation.create_environment(
            self.scene,
            cfg["ENVIRONMENT_TYPE"],
            cfg["PLATE_SIZE"],
            cfg["CYLINDER_DIAMETER"],
            cfg["CYLINDER_HEIGHT"],
            cfg["CYLINDER_SEGMENTS"],
            cfg["WALL_THICKNESS"],
            float(cfg.get("PLATE_WALL_HEIGHT", simulation.PLATE_WALL_HEIGHT)),
            env_restitution=float(cfg.get("ENV_RESTITUTION", 0.0)),
        )
        self.active_containers = list(containers)
        _p("spawn", 0.55, f"Spawning {int(cfg['N_PARTICLES'])} particles…")
        entities = simulation.spawn_particles(
            self.scene,
            self.physics_mesh,
            cfg["N_PARTICLES"],
            env_info,
            cfg["DROP_HEIGHT"],
            cfg["DROP_SPREAD"],
            cfg["YOUNGS_MODULUS"],
            cfg["POISSON_RATIO"],
            cfg["DENSITY"],
            particle_file=particle_file,
            scale_factor=scale_factor,
            particle_restitution=float(cfg.get("PARTICLE_RESTITUTION", 0.0)),
            e_fem_max=float(cfg["FEM_JAMMING_E_MAX"]),
            prior_fem_snapshots=prior_fem_snapshots,
        )
        self.active_entities = list(entities)
        if not _use_cpu_backend():
            _prepare_cuda_on_worker_thread()
        _p("build", 0.72, "Compiling scene (Genesis / Taichi)…")
        if self.cancel_requested:
            raise SimulationAborted()
        self.scene.build()
        if prior_fem_snapshots:
            simulation.restore_fem_entities(self.active_entities, prior_fem_snapshots)
        _p("ready", 1.0, "Scene ready")
        container_ids = {_entity_id(e) for e in self.active_containers}
        particle_ids = {_entity_id(e) for e in self.active_entities}
        return container_ids, particle_ids, entities


RUNTIME = SimulationRuntime()


@asynccontextmanager
async def lifespan(app: FastAPI):
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


@app.get("/export")
def get_export():
    """Full post-simulation payload for the results overlay."""
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
        sequential = bool(cfg.get("SEQUENTIAL_DROP"))
        analytical = bool(cfg.get("ANALYTICAL_MODE"))
        if sequential:
            # Use cfg DT/SUBSTEPS — the old “analytical falling” coarse step caused FEM tunneling through thin plates.
            analytical = False
        elif analytical:
            cfg_run["DT"] = float(cfg["ANALYTICAL_FALLING_DT"])
            cfg_run["SUBSTEPS"] = int(cfg["ANALYTICAL_FALLING_SUBSTEPS"])
            cfg_run["FEM_NEWTON_ITERATIONS"] = int(cfg.get("FEM_NEWTON_ITERATIONS", 4))

        duration = float(cfg["SIM_DURATION"])
        settle_threshold = float(cfg["SETTLE_THRESHOLD"])
        depth_tol = float(cfg["CONTACT_DEPTH_TOL"])
        vel_threshold = float(cfg.get("ANALYTICAL_VEL_THRESHOLD", 0.1))
        LATEST_Z_HISTORY = []
        LATEST_MAX_VEL_HISTORY = []
        LATEST_RATTLERS_HISTORY = []
        LATEST_KE_HISTORY = []
        LATEST_PRESSURE_HISTORY = []
        LATEST_CONTACT_GRAPH_DICT = {}
        LATEST_CONTACT_GRAPH_LINKS = []
        surface_area_m2 = float(simulation.container_surface_area_m2(str(cfg.get("ENVIRONMENT_TYPE", "plate")), cfg))
        falling_contact_stride = int(cfg.get("CONTACT_EXTRACT_FALLING_EVERY", simulation.CONTACT_EXTRACT_FALLING_EVERY))
        rho = float(cfg["DENSITY"])

        container_ids: set[int] = set()
        particle_ids: set[int] = set()
        entities: list[Any] = []
        contact_cache = simulation.ContactSampleCache()
        particle_mass_kg = 0.0

        if not sequential:
            container_ids, particle_ids, entities = RUNTIME.build_scene(cfg_run, on_progress=_on_build_progress)
            particle_mass_kg = _particle_mass_kg_from_runtime(RUNTIME, rho)

        dt = float(cfg_run["DT"])

        # Headless: skip per-step visualizer GPU/raster updates (still built at scene.build()).
        # Otherwise each step pays full visualizer.update() cost even with show_viewer=False.
        def _step() -> None:
            RUNTIME.scene.step(update_visualizer=False)
            if RUNTIME.active_entities and RUNTIME.physics_mesh is not None:
                simulation.enforce_container_bounds(RUNTIME.active_entities, RUNTIME.physics_mesh, cfg_run)

        if not sequential:
            # Show spawn poses immediately so the UI is not blank until the first (slow) CPU step.
            sync_q.put(
                {
                    "type": "frame",
                    "step": -1,
                    "t": 0.0,
                    "max_vel": 0.0,
                    "Z": 0.0,
                    "particles": simulation._collect_particle_transforms(entities),
                }
            )

        precision_phase = not analytical
        t_elapsed = 0.0
        last_ws_pct = -1.0
        last_live_Z = 0.0

        def _force_skip_contacts(max_vel: Optional[float]) -> bool:
            if not analytical or precision_phase:
                return False
            return max_vel is None or float(max_vel) >= vel_threshold

        def _run_step_block(frame_step_idx: int, t_now: float, phase_step: int, phase_len: int) -> bool:
            """
            After a physics substep: metrics, frames, settle / analytical handoff.
            Returns True to stop the outer simulation (cancel, settle, or completed precision phase).
            """
            nonlocal entities, particle_ids, container_ids, contact_cache, dt, precision_phase, t_elapsed, last_ws_pct, particle_mass_kg, last_live_Z
            if RUNTIME.cancel_requested:
                sync_q.put({"type": "log", "line": "Simulation cancelled by user"})
                sync_q.put({"type": "cancelled"})
                sync_q.put({"type": "idle", "message": "Ready for next run"})
                return True

            max_vel = simulation.compute_max_velocity(entities)
            skip = _force_skip_contacts(max_vel)

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
                contacts_snapshot = simulation.extract_contacts_resampled(
                    RUNTIME.scene,
                    particle_ids,
                    container_ids,
                    depth_tol,
                    sim_step=frame_step_idx,
                    max_vel=max_vel,
                    settle_threshold=settle_threshold,
                    falling_every=falling_contact_stride,
                    cache=contact_cache,
                    force_skip=skip,
                )

            if frame_step_idx % live_metrics_every == 0:
                lm = simulation.calculate_live_metrics(
                    contacts_snapshot,
                    particle_ids,
                    entities,
                    particle_mass_kg=particle_mass_kg,
                    surface_area_m2=surface_area_m2,
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
                        "particles": simulation._collect_particle_transforms(
                            entities,
                            stress_map,
                            fem_vertex_norms=fem_maps,
                            fem_norm_global_max=gmax,
                        ),
                    }
                )

            if (phase_step % log_every == 0) or (phase_step == phase_len - 1):
                sync_q.put({"type": "log", "line": f"t={t_now:.2f}s phase_step={phase_step + 1}/{phase_len}"})

            if max_vel is not None and max_vel < settle_threshold:
                sync_q.put({"type": "log", "line": f"Settled at t={t_now:.2f}s"})
                return True

            if (
                analytical
                and (not precision_phase)
                and max_vel is not None
                and float(max_vel) < vel_threshold
            ):
                snaps = simulation.snapshot_fem_entities(entities)
                remaining = duration - t_elapsed
                if snaps and remaining > 1e-9:
                    sync_q.put(
                        {
                            "type": "log",
                            "line": "Analytical mode: max_vel < {:.2g} m/s — rebuilding at 500 Hz / 16 substeps…".format(
                                vel_threshold
                            ),
                        }
                    )
                    cfg2 = dict(cfg)
                    cfg2["DT"] = float(cfg["ANALYTICAL_PRECISION_DT"])
                    cfg2["SUBSTEPS"] = int(cfg["ANALYTICAL_PRECISION_SUBSTEPS"])
                    cfg2["FEM_NEWTON_ITERATIONS"] = int(cfg.get("FEM_NEWTON_ITERATIONS_PRECISION", 8))
                    container_ids, particle_ids, entities = RUNTIME.build_scene(cfg2, on_progress=_on_build_progress)
                    particle_mass_kg = _particle_mass_kg_from_runtime(RUNTIME, rho)
                    try:
                        simulation.restore_fem_entities(entities, snaps)
                    except Exception as exc:
                        sync_q.put({"type": "log", "line": f"[warn] FEM state restore failed ({exc}); continuing from spawn."})
                    contact_cache = simulation.ContactSampleCache()
                    dt = float(cfg2["DT"])
                    precision_phase = True
                    phase2_steps = max(0, int(remaining / dt))
                    phase2_frame0 = frame_step_idx + 1
                    phase2_pbar = tqdm(
                        range(phase2_steps),
                        desc="Simulation (precision)",
                        unit="step",
                        dynamic_ncols=True,
                        mininterval=0.25,
                        file=sys.stderr,
                        disable=not sys.stderr.isatty(),
                    )
                    try:
                        for step2 in phase2_pbar:
                            _step()
                            t_elapsed += dt
                            t_now = t_elapsed
                            if _run_step_block(phase2_frame0 + step2, t_now, step2, phase2_steps):
                                return True
                    finally:
                        phase2_pbar.close()
                    return True
                if not snaps:
                    sync_q.put(
                        {
                            "type": "log",
                            "line": "[warn] Analytical precision handoff skipped (no FEMEntity snapshots; e.g. MPM mode).",
                        }
                    )
            return False

        if sequential:
            n_total = max(1, int(cfg["N_PARTICLES"]))
            _ssd = cfg.get("SEQUENTIAL_STAGE_DURATION")
            stage_cap_default = float(_ssd) if _ssd is not None else max(duration / max(n_total, 1), 0.25)
            total_budget = duration
            accrued: list[dict] = []
            global_frame_idx = 0
            sync_q.put(
                {
                    "type": "log",
                    "line": "SEQUENTIAL_DROP: rebuild scene per particle; prior bodies restored from FEM snapshot.",
                }
            )
            for k in range(1, n_total + 1):
                if RUNTIME.cancel_requested:
                    sync_q.put({"type": "log", "line": "Simulation cancelled by user"})
                    sync_q.put({"type": "cancelled"})
                    sync_q.put({"type": "idle", "message": "Ready for next run"})
                    return
                if total_budget <= 0.0:
                    break
                cfg_k = dict(cfg_run)
                cfg_k["N_PARTICLES"] = k
                sync_q.put({"type": "log", "line": f"Sequential stage {k}/{n_total}: building with {k} particle(s)…"})
                prior = accrued if accrued else None
                container_ids, particle_ids, entities = RUNTIME.build_scene(
                    cfg_k,
                    on_progress=_on_build_progress,
                    prior_fem_snapshots=prior,
                )
                if particle_mass_kg <= 0.0:
                    particle_mass_kg = _particle_mass_kg_from_runtime(RUNTIME, rho)
                contact_cache = simulation.ContactSampleCache()
                dt = float(cfg_k["DT"])
                sync_q.put(
                    {
                        "type": "frame",
                        "step": -1,
                        "t": float(t_elapsed),
                        "max_vel": 0.0,
                        "Z": 0.0,
                        "particles": simulation._collect_particle_transforms(entities),
                    }
                )
                stage_budget = min(stage_cap_default, total_budget)
                steps_stage = int(stage_budget / dt) if dt > 0 else 0
                last_ws_pct = -1.0
                sim_pbar = tqdm(
                    range(steps_stage),
                    desc=f"Simulation stage {k}/{n_total}",
                    unit="step",
                    dynamic_ncols=True,
                    mininterval=0.25,
                    file=sys.stderr,
                    disable=not sys.stderr.isatty(),
                )
                stage_time_used = 0.0
                try:
                    for step in sim_pbar:
                        if RUNTIME.cancel_requested:
                            sync_q.put({"type": "log", "line": "Simulation cancelled by user"})
                            sync_q.put({"type": "cancelled"})
                            sync_q.put({"type": "idle", "message": "Ready for next run"})
                            return
                        if k == 1 and step == 0:
                            sync_q.put(
                                {
                                    "type": "log",
                                    "line": "First physics step after build: Taichi/JIT + implicit FEM can take minutes on CPU; tqdm may sit at 0% until it completes.",
                                }
                            )
                        _step()
                        t_elapsed += dt
                        t_now = t_elapsed
                        stage_time_used += dt
                        if _run_step_block(global_frame_idx, t_now, step, steps_stage):
                            break
                        global_frame_idx += 1
                finally:
                    sim_pbar.close()
                total_budget -= stage_time_used
                accrued = simulation.snapshot_fem_entities(entities)
                if k < n_total and len(accrued) != k:
                    sync_q.put(
                        {
                            "type": "log",
                            "line": "[warn] SEQUENTIAL_DROP requires FEM particles (snapshot count != k). Stopping staged drops.",
                        }
                    )
                    break
        else:
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
        contacts = simulation.extract_contacts(RUNTIME.scene, particle_ids, container_ids, depth_tol)
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
        metrics = simulation.compute_metrics(contacts, particle_ids, container_surface_area_m2=surface_area_m2)
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
        sync_q.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        sync_q.put({"type": "idle", "message": "Ready after error"})
    finally:
        sync_q.put(None)


async def _run_simulation(ws: WebSocket, payload: dict[str, Any]) -> None:
    cfg = simulation.build_runtime_config(payload.get("config") if isinstance(payload, dict) else {})
    sync_q: queue.Queue[Any | None] = queue.Queue()
    try:
        async with RUNTIME._busy:
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
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

