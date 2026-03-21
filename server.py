import os
import sys

# Optional: hide CUDA from PyTorch before import (CPU-only server / debugging).
if os.environ.get("GENESIS_USE_CPU", "").strip().lower() in ("1", "true", "yes"):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import asyncio
import queue
import threading
from pathlib import Path
from typing import Any

import h5py
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import simulation


def _init_genesis_on_sim_thread() -> None:
    """
    Initialize Genesis on the dedicated simulation thread only.

    Taichi/Quadrants LLVM state is tied to the thread that calls gs.init(); running
    Scene.build/step on another thread triggers main_thread_id assertion failures.
    """
    try:
        simulation.gs.init(backend=simulation.gs.gpu)
        return
    except Exception as exc:
        if "already initialized" in str(exc).lower():
            return
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
LATEST_Z_HISTORY: list[dict[str, float]] = []
LATEST_MAX_VEL_HISTORY: list[dict[str, float]] = []
LATEST_CONTACT_GRAPH_DICT: dict[str, dict[str, dict[str, float | None]]] = {}
LATEST_CONTACT_GRAPH_LINKS: list[dict[str, Any]] = []


def _entity_id(e) -> int:
    if hasattr(e, "id"):
        return int(getattr(e, "id"))
    return int(getattr(e, "idx"))


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
        self.default_particle_file = str(Path(__file__).resolve().parent / "particle.obj")
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

    def build_scene(self, cfg: dict) -> tuple[set[int], set[int], list[Any]]:
        # Genesis: add_entity / build only on an unbuilt scene — recreate each run.
        self.destroy_scene()
        sim_options = simulation.gs.options.SimOptions(
            dt=float(cfg["DT"]),
            substeps=int(cfg["SUBSTEPS"]),
            gravity=cfg.get("GRAVITY", (0, -9.81, 0)),
        )
        # Rigid container geometry uses the rigid solver; small dt/substeps => tiny
        # _substep_dt and a warning unless GJK is enabled (see rigid_solver.py).
        rigid_options = simulation.gs.options.RigidOptions(use_gjk_collision=True)
        self.scene = simulation.gs.Scene(
            sim_options=sim_options,
            rigid_options=rigid_options,
            show_viewer=False,
        )
        particle_file = str(cfg.get("PARTICLE_FILE") or self.default_particle_file)
        scale_factor = float(cfg.get("SCALE_FACTOR", 1.0))
        self.physics_mesh, _ = simulation.load_particle_mesh(particle_file, scale_factor)
        containers, env_info = simulation.create_environment(
            self.scene,
            cfg["ENVIRONMENT_TYPE"],
            cfg["PLATE_SIZE"],
            cfg["CYLINDER_DIAMETER"],
            cfg["CYLINDER_HEIGHT"],
            cfg["CYLINDER_SEGMENTS"],
            cfg["WALL_THICKNESS"],
            env_restitution=float(cfg.get("ENV_RESTITUTION", 0.0)),
        )
        self.active_containers = list(containers)
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
        )
        self.active_entities = list(entities)
        self.scene.build()
        container_ids = {_entity_id(e) for e in self.active_containers}
        particle_ids = {_entity_id(e) for e in self.active_entities}
        return container_ids, particle_ids, entities


RUNTIME = SimulationRuntime()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    return {
        "particles": _read_csv_records(out_dir / "particles.csv"),
        "contact_pairs": _read_csv_records(out_dir / "contact_pairs.csv"),
        "contact_points": _read_csv_records(out_dir / "contact_points.csv"),
    }


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
        "z_history": LATEST_Z_HISTORY,
        "max_vel_history": LATEST_MAX_VEL_HISTORY,
        "contact_graph_dict": LATEST_CONTACT_GRAPH_DICT,
        "contact_graph_links": LATEST_CONTACT_GRAPH_LINKS,
    }
    if not h5_path.exists():
        pairs = _read_csv_records(out_dir / "contact_pairs.csv")
        graph_dict, graph_links = _build_graph_from_contact_pairs(pairs)
        out["contact_graph_dict"] = graph_dict
        out["contact_graph_links"] = graph_links
        return out
    try:
        with h5py.File(h5_path, "r") as f:
            attrs = f.attrs
            out["Z"] = float(attrs.get("Z", 0.0))
            out["total_pp"] = int(attrs.get("total_pp", 0))
            out["total_pc"] = int(attrs.get("total_pc", 0))
            out["n_isolated"] = int(attrs.get("n_isolated", 0))
            out["n_container_touch"] = int(attrs.get("n_container_touch", 0))
            return out
    except Exception:
        return out


@app.post("/cancel")
def cancel_simulation():
    RUNTIME.cancel_requested = True
    return {"ok": True, "message": "Cancellation requested"}


def _simulation_thread_main(sync_q: "queue.Queue[Any | None]", cfg: dict[str, Any], payload: dict[str, Any]) -> None:
    """Blocking Genesis loop (runs in a worker thread). Puts dict messages on sync_q; ends with None."""
    global LATEST_Z_HISTORY, LATEST_CONTACT_GRAPH_DICT, LATEST_CONTACT_GRAPH_LINKS, LATEST_MAX_VEL_HISTORY
    frame_every = int(payload.get("frame_every", 2))
    frame_every = max(1, frame_every)
    log_every = max(1, int(payload.get("log_every", 60)))
    try:
        RUNTIME.cancel_requested = False
        sync_q.put({"type": "log", "line": "Preparing persistent scene (new Scene per run)..."})
        container_ids, particle_ids, entities = RUNTIME.build_scene(cfg)
        dt = float(cfg["DT"])
        duration = float(cfg["SIM_DURATION"])
        settle_threshold = float(cfg["SETTLE_THRESHOLD"])
        depth_tol = float(cfg["CONTACT_DEPTH_TOL"])
        total_steps = int(duration / dt)
        LATEST_Z_HISTORY = []
        LATEST_MAX_VEL_HISTORY = []
        LATEST_CONTACT_GRAPH_DICT = {}
        LATEST_CONTACT_GRAPH_LINKS = []

        for step in range(total_steps):
            if RUNTIME.cancel_requested:
                sync_q.put({"type": "log", "line": "Simulation cancelled by user"})
                sync_q.put({"type": "cancelled"})
                sync_q.put({"type": "idle", "message": "Ready for next run"})
                return

            if step % frame_every == 0:
                contacts_live = simulation.extract_contacts(RUNTIME.scene, particle_ids, container_ids, depth_tol)
                stress_map = simulation.compute_particle_stress_map(contacts_live, particle_ids)
                metrics_live = simulation.compute_metrics(contacts_live, particle_ids)
                max_vel = simulation.compute_max_velocity(entities)
                fem_maps, gmax = simulation.compute_fem_vertex_force_stress(RUNTIME.scene, entities)
                t_now = float(step * dt)
                LATEST_Z_HISTORY.append({"t": t_now, "Z": float(metrics_live.get("Z", 0.0))})
                LATEST_MAX_VEL_HISTORY.append({"t": t_now, "max_vel": float(max_vel or 0.0)})
                sync_q.put(
                    {
                        "type": "frame",
                        "step": int(step),
                        "t": t_now,
                        "max_vel": float(max_vel or 0.0),
                        "Z": float(metrics_live.get("Z", 0.0)),
                        "particles": simulation._collect_particle_transforms(
                            entities,
                            stress_map,
                            fem_vertex_norms=fem_maps,
                            fem_norm_global_max=gmax,
                        ),
                    }
                )

            RUNTIME.scene.step()

            if (step % log_every == 0) or (step == total_steps - 1):
                sync_q.put({"type": "log", "line": f"t={step * dt:.2f}s step={step}/{total_steps}"})

            max_vel = simulation.compute_max_velocity(entities)
            if max_vel is not None and max_vel < settle_threshold:
                sync_q.put({"type": "log", "line": f"Settled at t={step * dt:.2f}s"})
                break

        contacts = simulation.extract_contacts(RUNTIME.scene, particle_ids, container_ids, depth_tol)
        metrics = simulation.compute_metrics(contacts, particle_ids)
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
        )
        sync_q.put({"type": "complete", "metrics": {"Z": float(metrics.get("Z", 0.0)), "total_pp": int(metrics.get("total_pp_contacts", 0))}})
        sync_q.put({"type": "idle", "message": "Ready for next run"})
    except Exception as exc:
        sync_q.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        sync_q.put({"type": "idle", "message": "Ready after error"})
    finally:
        sync_q.put(None)


async def _run_simulation(ws: WebSocket, payload: dict[str, Any]) -> None:
    cfg = simulation.build_runtime_config(payload.get("config") if isinstance(payload, dict) else {})
    sync_q: queue.Queue[Any | None] = queue.Queue()
    async with RUNTIME._busy:
        RUNTIME._job_queue.put((sync_q, cfg, payload))
        while True:
            item = await asyncio.to_thread(sync_q.get)
            if item is None:
                break
            await ws.send_json(item)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    await ws.send_json({"type": "ready", "message": "Send {type:'start', config:{...}}"})
    try:
        while True:
            payload = await ws.receive_json()
            cmd = str(payload.get("type", "")).strip().lower() if isinstance(payload, dict) else ""
            if cmd == "start":
                await _run_simulation(ws, payload)
                continue
            if cmd == "clear":
                done = threading.Event()
                RUNTIME._job_queue.put(("clear", done))
                await asyncio.to_thread(done.wait)
                await ws.send_json({"type": "cleared"})
                continue
            await ws.send_json({"type": "error", "message": "Unknown command. Use start|clear"})
    except WebSocketDisconnect:
        return
    except Exception as exc:
        try:
            await ws.send_json({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)

