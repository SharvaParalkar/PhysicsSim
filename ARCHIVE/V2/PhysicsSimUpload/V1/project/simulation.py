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


# ── Input ──────────────────────────────────────────────────────────
PARTICLE_FILE        = "particle.obj"  # Path to the input particle mesh (OBJ/STL); e.g. "assets/particle.stl" if you want to keep meshes organized.
N_PARTICLES          = 100             # Number of particle instances to spawn; e.g. 300 for denser packings or 20 for quick smoke tests.
SCALE_FACTOR         = 1.0             # Uniform scale applied to the input mesh; e.g. 0.001 to convert a millimeter-modeled mesh into meters.

# ── Material ───────────────────────────────────────────────────────
# YOUNGS_MODULUS is the single dial for rigidness.
# The correct internal solver is selected automatically from this value.
#
#   Steel / ceramic:   E = 200_000_000  (200 MPa)  → Rigid solver
#   Hard plastic:      E =  50_000_000  (50 MPa)   → Rigid solver
#   Hard rubber:       E =   1_000_000  (1 MPa)    → FEM solver
#   Soft silicone:     E =      10_000  (10 kPa)   → FEM solver
#   Gel / putty:       E =       1_000  (1 kPa)    → MPM solver
YOUNGS_MODULUS       = 1_000_000       # Elastic modulus in Pa (stiffness dial); e.g. 50_000_000 for hard plastic-like behavior or 10_000 for very soft silicone-like behavior.
POISSON_RATIO        = 0.45            # Lateral compressibility (0.5 ≈ incompressible rubber); e.g. 0.30 for more compressible plastics/foams.
DENSITY              = 1200            # Material density in kg/m³; e.g. 7800 for steel or ~1000 for water-like polymers.

# ── Environment ────────────────────────────────────────────────────
ENVIRONMENT_TYPE     = "cylinder"      # Container type ("cylinder" confined packing or "plate" unconfined pile); e.g. "plate" for open pile tests.
PLATE_SIZE           = 0.6             # Plate side length in meters (used when ENVIRONMENT_TYPE="plate"); e.g. 1.0 for a larger catch area.
CYLINDER_DIAMETER    = 0.20            # Cylinder inner diameter in meters (used when ENVIRONMENT_TYPE="cylinder"); e.g. 0.30 for more capacity.
CYLINDER_HEIGHT      = 0.30            # Cylinder wall height in meters; e.g. 0.50 for taller containment.
CYLINDER_SEGMENTS    = 32              # Number of wall facets approximating the cylinder; e.g. 64 for smoother walls or 24 to speed up geometry.
WALL_THICKNESS       = 0.008           # Wall thickness in meters; e.g. 0.012 for more robust collision thickness or 0.004 for thinner walls.

# ── Drop configuration ─────────────────────────────────────────────
DROP_HEIGHT          = 0.15            # Drop height above the container top edge (or plate surface) in meters; e.g. 0.30 for higher impact energy.
DROP_SPREAD          = 0.85            # Fraction of container inner radius/half-size used for initial XZ spread; e.g. 0.60 to start more centralized and reduce early wall contacts.

# ── Gravity ────────────────────────────────────────────────────────
GRAVITY              = (0, -9.81, 0)   # Gravity vector in m/s² (Y is up, so negative Y falls); e.g. (0, -3.72, 0) for Mars-like gravity.

# ── Simulation ─────────────────────────────────────────────────────
DT                   = 1 / 240         # Simulation timestep (seconds); e.g. 1/120 for faster runs or 1/480 for improved stability with stiff contacts.
SUBSTEPS             = 4               # Solver substeps per timestep; e.g. 8 for stiffer systems or 2 for faster but less stable runs.
SIM_DURATION         = 5.0             # Total simulated time in seconds; e.g. 10.0 to allow long settling or 2.0 for quick iterations.
SETTLE_THRESHOLD     = 1e-4            # Velocity magnitude (m/s) below which the pile is considered settled; e.g. 5e-4 for earlier termination or 1e-5 for stricter settling.

# ── Contact analysis ───────────────────────────────────────────────
CONTACT_SAMPLE_EVERY = 10              # Sample/compute contacts every N simulation steps; e.g. 1 for per-step contact tracking or 30 to reduce analysis cost.
CONTACT_DEPTH_TOL    = 1e-5            # Minimum penetration depth (m) to count as a real contact; e.g. 5e-5 to filter more noise or 1e-6 to be more sensitive.

# ── Output ─────────────────────────────────────────────────────────
OUTPUT_DIR           = "./results"     # Output directory for all artifacts; e.g. "./runs/run_001/results" to keep runs separated.
SAVE_HDF5            = True            # Whether to save an HDF5 replay/summary file; e.g. False if you only want lightweight CSV outputs.
SAVE_CSV             = True            # Whether to save CSV exports (particles/contacts); e.g. False if you only want HDF5 for downstream processing.

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


def load_particle_mesh(filepath: str, scale: float = 1.0) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    # Resolve relative paths from this script's directory (so running from other CWDs works).
    resolved_path = filepath
    if not os.path.isabs(resolved_path):
        resolved_path = os.path.join(os.path.dirname(__file__), resolved_path)

    ext = resolved_path.rsplit(".", 1)[-1].lower()
    if ext not in ("obj", "stl"):
        raise ValueError(f"Unsupported mesh format '.{ext}'. Expected OBJ or STL.")

    # If the file exists but is empty/near-empty, trimesh will return an empty mesh. Provide a useful fallback.
    if os.path.exists(resolved_path) and os.path.getsize(resolved_path) <= 8:
        print(f"[WARN] Mesh file '{filepath}' is empty ({os.path.getsize(resolved_path)} bytes). Using a generated sphere mesh instead.")
        original_mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.01)
    else:
        original_mesh = trimesh.load(resolved_path, force="mesh")

    if original_mesh.is_empty:
        raise ValueError(
            f"Mesh loaded from '{filepath}' is empty. Resolved path: '{resolved_path}'. "
            "Ensure the OBJ/STL contains vertices/faces (not just an empty file)."
        )

    original_mesh.apply_scale(scale)
    original_mesh.apply_translation(-original_mesh.centroid)

    # CoACD expects its own Mesh wrapper (vertices + triangle indices).
    if getattr(original_mesh, "faces", None) is None or len(original_mesh.faces) == 0:
        raise ValueError(
            f"Mesh loaded from '{filepath}' contains no faces. Resolved path: '{resolved_path}'."
        )

    # Ensure triangles (CoACD assumes Nx3 indices).
    if np.asarray(original_mesh.faces).shape[1] != 3:
        original_mesh = original_mesh.triangulate()

    coacd_mesh = coacd.Mesh(
        vertices=np.asarray(original_mesh.vertices, dtype=np.float64),
        indices=np.asarray(original_mesh.faces, dtype=np.int32),
    )
    parts = coacd.run_coacd(coacd_mesh, max_convex_hull=32)

    part_meshes = [
        trimesh.Trimesh(vertices=v, faces=f, process=False) for (v, f) in parts
    ]
    physics_mesh = trimesh.util.concatenate(part_meshes) if part_meshes else original_mesh.copy()

    extents = original_mesh.extents
    extents_str = " ".join(f"{v:.5f}" for v in extents)

    print(f"Loaded: {filepath}")
    print(f"  Resolved path: {resolved_path}")
    print(f"  Vertices : {len(original_mesh.vertices)}")
    print(f"  Faces    : {len(original_mesh.faces)}")
    print(f"  Extents  : {extents_str} m")
    print(f"  Volume   : {original_mesh.volume:.6f} m³")
    print(f"  Convex parts after decomposition: {len(parts)}")

    return physics_mesh, original_mesh


def create_environment(
    scene: gs.Scene,
    kind: str,
    plate_size: float     = 0.6,
    cyl_diameter: float   = 0.20,
    cyl_height: float     = 0.30,
    cyl_segments: int     = 32,
    wall_thickness: float = 0.008,
) -> tuple[set, dict]:
    mat = gs.materials.Rigid(friction=0.55, coup_restitution=0.05)
    container_ids: set[int] = set()

    if kind == "plate":
        t = wall_thickness
        plate = scene.add_entity(
            gs.morphs.Box(size=(plate_size, t, plate_size), pos=(0, t / 2, 0), fixed=True),
            material=mat,
        )
        container_ids.add(plate.idx)
        env_info = {
            "surface_y": wall_thickness,
            "top_y": wall_thickness,
            "spread_radius": plate_size / 2 * 0.9,
        }

    elif kind == "cylinder":
        r_inner = cyl_diameter / 2
        t = wall_thickness

        bottom = scene.add_entity(
            gs.morphs.Cylinder(radius=(r_inner + t), height=t, pos=(0, t / 2, 0), fixed=True),
            material=mat,
        )
        container_ids.add(bottom.idx)

        angle_step = 2 * math.pi / cyl_segments
        chord_width = 2 * (cyl_diameter / 2) * math.sin(angle_step / 2)

        for i in range(cyl_segments):
            angle = i * angle_step
            cx = (cyl_diameter / 2 + wall_thickness / 2) * math.cos(angle)
            cz = (cyl_diameter / 2 + wall_thickness / 2) * math.sin(angle)
            cy = wall_thickness + cyl_height / 2

            panel = scene.add_entity(
                gs.morphs.Box(
                    size=(chord_width, cyl_height, wall_thickness),
                    pos=(cx, cy, cz),
                    quat=gs.utils.geom.euler_to_quat((0, math.degrees(angle), 0)),
                    fixed=True,
                ),
                material=mat,
            )
            container_ids.add(panel.idx)

        env_info = {
            "surface_y": wall_thickness,
            "top_y": wall_thickness + cyl_height,
            "inner_radius": cyl_diameter / 2,
            "spread_radius": (cyl_diameter / 2) * 0.85,
        }

    else:
        raise ValueError(f"Unknown environment kind '{kind}'. Valid options are 'plate' or 'cylinder'.")

    return container_ids, env_info


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
    if E > 1e8:
        material = gs.materials.Rigid(friction=0.4, coup_restitution=0.2)
    elif E > 1e3:
        material = gs.materials.FEM(E=E, nu=nu, rho=rho)
    else:
        material = gs.materials.MPM(E=E, nu=nu, rho=rho)

    extents = physics_mesh.bounds[1] - physics_mesh.bounds[0]
    p_radius = float(np.linalg.norm(extents)) / 2.0
    spacing = p_radius * 2.4

    spawn_r = env_info["spread_radius"] * drop_spread
    spawn_y0 = env_info["top_y"] + drop_height
    cols = int(math.ceil(math.sqrt(n)))

    entities = []

    for i in range(n):
        row, col = divmod(i, cols)
        x = (col - cols / 2 + 0.5) * spacing
        z = (row - cols / 2 + 0.5) * spacing

        dist = math.sqrt(x**2 + z**2)
        if dist > spawn_r:
            s = spawn_r / dist * np.random.uniform(0.7, 1.0)
            x *= s
            z *= s

        x += np.random.uniform(-p_radius * 0.2, p_radius * 0.2)
        z += np.random.uniform(-p_radius * 0.2, p_radius * 0.2)

        y = spawn_y0 + (i % cols) * p_radius * 0.3

        quat = gs.utils.mat_to_quat(
            trimesh.transformations.random_rotation_matrix()[:3, :3]
        )

        entity = scene.add_entity(
            gs.morphs.Mesh(mesh=physics_mesh, pos=(x, y, z), quat=quat),
            material=material,
        )
        entities.append(entity)

    print(f"Spawned {n} particles")
    print(f"  Drop height above container top: {drop_height} m")
    print(f"  Spawn Y range: {spawn_y0} – {spawn_y0 + p_radius} m")

    return entities


def run_simulation(
    scene: gs.Scene,
    entities: list,
    dt: float,
    substeps: int,
    duration: float,
    settle_threshold: float,
) -> int:
    total_steps = int(duration / dt)

    for step in range(total_steps):
        scene.step()

        if step % 60 == 0:
            velocities = [float(np.linalg.norm(e.get_vel())) for e in entities]
            max_vel = max(velocities) if velocities else 0.0

            print(
                f"t={step * dt:.2f}s  max_vel={max_vel:.5f} m/s  step={step}/{total_steps}",
                end="\r",
            )

            if max_vel < settle_threshold:
                print()
                print(f"Settled at t={step * dt:.3f}s  (step {step})")
                return step

    print()
    print(f"Reached max duration ({duration}s) without settling")
    return total_steps


def extract_contacts(
    scene: gs.Scene,
    particle_ids: set,
    container_ids: set,
    depth_tol: float = 1e-5,
) -> list[NormalizedContact]:
    raw = scene.get_contacts()
    results: list[NormalizedContact] = []

    for c in raw:
        if abs(c.depth) < depth_tol:
            continue

        a = c.entity_a.idx
        b = c.entity_b.idx

        is_pp = (a in particle_ids) and (b in particle_ids)
        is_pc = (
            (a in particle_ids and b in container_ids)
            or (b in particle_ids and a in container_ids)
        )

        if not (is_pp or is_pc):
            continue

        force = float(c.force) if hasattr(c, "force") else None
        area = float(c.area) if hasattr(c, "area") else None

        results.append(
            NormalizedContact(
                entity_a=a,
                entity_b=b,
                is_particle_particle=is_pp,
                is_particle_container=is_pc,
                position=np.array(c.pos),
                normal=np.array(c.normal),
                depth=float(c.depth),
                force=force,
                contact_area=area,
            )
        )

    return results


def compute_metrics(
    contacts: list[NormalizedContact],
    particle_ids: set,
) -> dict:
    pp = [c for c in contacts if c.is_particle_particle]
    pc = [c for c in contacts if c.is_particle_container]
    n = len(particle_ids)

    G = nx.Graph()
    G.add_nodes_from(particle_ids)
    for c in pp:
        G.add_edge(
            c.entity_a,
            c.entity_b,
            depth=c.depth,
            force=c.force,
            area=c.contact_area,
        )

    contact_counts = dict(G.degree())
    isolated = [pid for pid, deg in contact_counts.items() if deg == 0]
    container_touching = {
        (c.entity_a if c.entity_a in particle_ids else c.entity_b) for c in pc
    }

    return {
        "total_pp_contacts": len(pp),
        "total_pc_contacts": len(pc),
        "avg_contacts_per_particle": 2 * len(pp) / n if n else 0.0,
        "Z": 2 * len(pp) / n if n else 0.0,
        "n_isolated_particles": len(isolated),
        "n_container_touching": len(container_touching),
        "contact_points": [c.position for c in pp],
        "contact_normals": [c.normal for c in pp],
        "contact_depths": [c.depth for c in pp],
        "contact_forces": [c.force for c in pp if c.force is not None],
        "contact_areas": [
            c.contact_area for c in pp if c.contact_area is not None
        ],
        "contact_graph": G,
        "contact_graph_dict": {pid: list(G.neighbors(pid)) for pid in particle_ids},
        "contact_counts_per_particle": contact_counts,
    }


def contact_efficiency(
    metrics: dict,
    entities: list,
    mesh: trimesh.Trimesh,
) -> float:
    volume = mesh.volume * len(entities)
    return metrics["total_pp_contacts"] / volume


def weighted_contact_efficiency(
    metrics: dict,
    entities: list,
    mesh: trimesh.Trimesh,
) -> float:
    total_force = sum(f for f in metrics["contact_forces"] if f)
    volume = mesh.volume * len(entities)
    return total_force / volume


def export_results(
    entities: list,
    metrics: dict,
    output_dir: str,
    save_hdf5: bool,
    save_csv: bool,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    counts = metrics["contact_counts_per_particle"]
    for e in entities:
        pos = e.get_pos()
        quat = e.get_quat()
        rows.append(
            {
                "id": e.idx,
                "x": pos[0],
                "y": pos[1],
                "z": pos[2],
                "qx": quat[0],
                "qy": quat[1],
                "qz": quat[2],
                "qw": quat[3],
                "n_contacts": counts.get(e.idx, 0),
            }
        )
    df_particles = pd.DataFrame(rows)

    G = metrics["contact_graph"]
    df_contacts = pd.DataFrame(
        [
            {
                "particle_a": a,
                "particle_b": b,
                "depth": data.get("depth"),
                "force": data.get("force"),
                "contact_area": data.get("area"),
            }
            for a, b, data in G.edges(data=True)
        ]
    )

    pts = metrics["contact_points"]
    nrms = metrics["contact_normals"]
    df_points = pd.DataFrame(
        [
            {"x": p[0], "y": p[1], "z": p[2], "nx": n[0], "ny": n[1], "nz": n[2]}
            for p, n in zip(pts, nrms)
        ]
    )

    if save_csv:
        df_particles.to_csv(os.path.join(output_dir, "particles.csv"), index=False)
        df_contacts.to_csv(os.path.join(output_dir, "contact_pairs.csv"), index=False)
        df_points.to_csv(os.path.join(output_dir, "contact_points.csv"), index=False)
        print(f"CSV files written to {output_dir}/")

    if save_hdf5:
        cp_array = np.array(pts) if len(pts) > 0 else np.empty((0, 3))
        cn_array = np.array(nrms) if len(nrms) > 0 else np.empty((0, 3))
        with h5py.File(os.path.join(output_dir, "simulation.h5"), "w") as f:
            f.create_dataset("particle_positions", data=df_particles[["x", "y", "z"]].values)
            f.create_dataset("contact_points", data=cp_array)
            f.create_dataset("contact_normals", data=cn_array)
            f.attrs["Z"] = metrics["Z"]
            f.attrs["total_pp"] = metrics["total_pp_contacts"]
            f.attrs["total_pc"] = metrics["total_pc_contacts"]
            f.attrs["n_isolated"] = metrics["n_isolated_particles"]
            f.attrs["n_container_touch"] = metrics["n_container_touching"]
        print(f"HDF5 replay written to {output_dir}/simulation.h5")

    print("── Contact Analysis Summary ────────────────────────────────")
    print(f"  Particles simulated:               {len(entities)}")
    print(f"  Total particle-particle contacts:  {metrics['total_pp_contacts']}")
    print(f"  Total particle-container contacts: {metrics['total_pc_contacts']}")
    print(f"  Avg contacts per particle (Z):     {metrics['Z']:.3f}")
    print(f"  Isolated particles (Z=0):          {metrics['n_isolated_particles']}")
    print(f"  Particles touching container:      {metrics['n_container_touching']}")

    if metrics["contact_forces"]:
        forces = np.array(metrics["contact_forces"], dtype=float)
        print(f"  Mean contact force:                {forces.mean():.4f} N")
        print(f"  Max  contact force:                {forces.max():.4f} N")

    if metrics["contact_areas"]:
        areas = np.array(metrics["contact_areas"], dtype=float)
        print(f"  Mean contact area:                 {areas.mean() * 1e6:.4f} mm²")

    print("───────────────────────────────────────────────────────────")


def visualize_results(
    entities: list,
    metrics: dict,
    original_mesh: trimesh.Trimesh,
) -> None:
    counts = metrics["contact_counts_per_particle"]
    max_c = max(counts.values()) if counts and max(counts.values()) > 0 else 1

    # Build a base PyVista mesh from the original trimesh geometry
    faces = original_mesh.faces
    faces_pv = np.hstack(
        [np.full((faces.shape[0], 1), 3, dtype=np.int64), faces.astype(np.int64)]
    ).ravel()
    base_mesh = pv.PolyData(original_mesh.vertices, faces_pv)

    plotter = pv.Plotter()

    for e in entities:
        pos = e.get_pos()
        quat = e.get_quat()
        c = counts.get(e.idx, 0)
        t = c / max_c

        # Copy base mesh and apply transform from quaternion + translation
        mesh = base_mesh.copy()
        T = trimesh.transformations.quaternion_matrix(quat)
        T[:3, 3] = pos
        mesh.transform(T)

        color = (t, 0.2, 1.0 - t)
        plotter.add_mesh(mesh, color=color)

    # Add contact points as small yellow spheres
    for pt in metrics["contact_points"]:
        sphere = pv.Sphere(radius=0.003, center=pt)
        plotter.add_mesh(sphere, color=(1.0, 1.0, 0.0))

    plotter.add_axes()
    plotter.show(window_size=(1280, 800), title="Particle Contact Analysis")


def main() -> None:
    gs.init(backend=gs.cpu)
    scene = gs.Scene(sim_options=gs.options.SimOptions(gravity=GRAVITY, dt=DT, substeps=SUBSTEPS))

    physics_mesh, original_mesh = load_particle_mesh(PARTICLE_FILE, SCALE_FACTOR)

    container_ids, env_info = create_environment(
        scene,
        kind=ENVIRONMENT_TYPE,
        plate_size=PLATE_SIZE,
        cyl_diameter=CYLINDER_DIAMETER,
        cyl_height=CYLINDER_HEIGHT,
        cyl_segments=CYLINDER_SEGMENTS,
        wall_thickness=WALL_THICKNESS,
    )

    entities = spawn_particles(
        scene=scene,
        physics_mesh=physics_mesh,
        n=N_PARTICLES,
        env_info=env_info,
        drop_height=DROP_HEIGHT,
        drop_spread=DROP_SPREAD,
        E=YOUNGS_MODULUS,
        nu=POISSON_RATIO,
        rho=DENSITY,
    )

    particle_ids = {e.idx for e in entities}

    scene.build()

    run_simulation(
        scene=scene,
        entities=entities,
        dt=DT,
        substeps=SUBSTEPS,
        duration=SIM_DURATION,
        settle_threshold=SETTLE_THRESHOLD,
    )

    contacts = extract_contacts(
        scene=scene,
        particle_ids=particle_ids,
        container_ids=container_ids,
        depth_tol=CONTACT_DEPTH_TOL,
    )
    metrics = compute_metrics(contacts, particle_ids)

    export_results(
        entities=entities,
        metrics=metrics,
        output_dir=OUTPUT_DIR,
        save_hdf5=SAVE_HDF5,
        save_csv=SAVE_CSV,
    )
    visualize_results(entities, metrics, original_mesh)


if __name__ == "__main__":
    main()
