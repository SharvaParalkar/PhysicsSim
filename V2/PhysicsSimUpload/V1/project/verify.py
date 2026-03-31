import os
import math
import traceback
import importlib
from typing import Callable, Tuple, List, Optional


CheckFunc = Callable[[], Tuple[bool, Optional[str]]]


TOTAL_PASSED = 0
TOTAL_FAILED = 0
FAILED_CHECK_IDS: List[str] = []


def _print_section(title: str) -> None:
    print(title)


def run_check(check_id: str, description: str, func: CheckFunc) -> None:
    global TOTAL_PASSED, TOTAL_FAILED

    try:
        passed, message = func()
    except Exception as exc:  # noqa: BLE001
        passed = False
        tb = traceback.format_exc()
        message = f"Exception: {exc!r}\n{tb}"

    if passed:
        TOTAL_PASSED += 1
        print(f"[PASS] {check_id}  {description}")
    else:
        TOTAL_FAILED += 1
        FAILED_CHECK_IDS.append(check_id)
        print(f"[FAIL] {check_id}  {description}")
        if message:
            first, *rest = message.splitlines()
            print(f"       \u2192 {first}")
            for line in rest:
                print(f"         {line}")


def section_1() -> None:
    _print_section("── Section 1: Imports & dependencies ───────────────────────────")

    required_modules = [
        "numpy",
        "pandas",
        "scipy",
        "networkx",
        "h5py",
        "trimesh",
        "coacd",
        "genesis",
        "open3d",
    ]

    def make_import_check(mod_name: str) -> CheckFunc:
        def _check() -> Tuple[bool, Optional[str]]:
            try:
                importlib.import_module(mod_name)
                return True, None
            except Exception as exc:  # noqa: BLE001
                return False, f"Could not import {mod_name}: {exc!r}"

        return _check

    for idx, name in enumerate(required_modules, start=1):
        cid = f"1.1.{idx}"
        desc = f"Core dependency importable: {name}"
        run_check(cid, desc, make_import_check(name))

    def check_12() -> Tuple[bool, Optional[str]]:
        try:
            import simulation  # type: ignore[import-not-found]

            _ = simulation  # noqa: F841
            return True, None
        except (ImportError, SyntaxError) as exc:
            return False, f"simulation import failed: {exc!r}"

    run_check("1.2", "simulation.py importable", check_12)

    def check_13() -> Tuple[bool, Optional[str]]:
        try:
            import simulation  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            return False, f"simulation not importable: {exc!r}"

        required_names = [
            "PARTICLE_FILE",
            "N_PARTICLES",
            "SCALE_FACTOR",
            "YOUNGS_MODULUS",
            "POISSON_RATIO",
            "DENSITY",
            "ENVIRONMENT_TYPE",
            "PLATE_SIZE",
            "CYLINDER_DIAMETER",
            "CYLINDER_HEIGHT",
            "CYLINDER_SEGMENTS",
            "WALL_THICKNESS",
            "DROP_HEIGHT",
            "DROP_SPREAD",
            "GRAVITY",
            "DT",
            "SUBSTEPS",
            "SIM_DURATION",
            "SETTLE_THRESHOLD",
            "CONTACT_SAMPLE_EVERY",
            "CONTACT_DEPTH_TOL",
            "OUTPUT_DIR",
            "SAVE_HDF5",
            "SAVE_CSV",
        ]
        missing = [n for n in required_names if not hasattr(simulation, n)]
        if missing:
            return False, f"Missing module-level attributes: {', '.join(missing)}"
        return True, None

    run_check("1.3", "All required parameter names present", check_13)

    def check_14() -> Tuple[bool, Optional[str]]:
        try:
            import simulation  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            return False, f"simulation not importable: {exc!r}"

        errors: List[str] = []

        def add(cond: bool, msg: str) -> None:
            if not cond:
                errors.append(msg)

        add(isinstance(simulation.PARTICLE_FILE, str), "PARTICLE_FILE must be str")
        add(isinstance(simulation.N_PARTICLES, int) and simulation.N_PARTICLES > 0, "N_PARTICLES must be int > 0")
        add(isinstance(simulation.SCALE_FACTOR, (int, float)) and simulation.SCALE_FACTOR > 0, "SCALE_FACTOR must be float/int > 0")
        add(isinstance(simulation.YOUNGS_MODULUS, (int, float)) and simulation.YOUNGS_MODULUS > 0, "YOUNGS_MODULUS must be float/int > 0")
        add(isinstance(simulation.POISSON_RATIO, float) and 0 < simulation.POISSON_RATIO < 0.5, "POISSON_RATIO must be float and 0 < value < 0.5")
        add(isinstance(simulation.DENSITY, (int, float)) and simulation.DENSITY > 0, "DENSITY must be float/int > 0")
        add(isinstance(simulation.ENVIRONMENT_TYPE, str) and simulation.ENVIRONMENT_TYPE in ("cylinder", "plate"), "ENVIRONMENT_TYPE must be 'cylinder' or 'plate'")
        add(isinstance(simulation.CYLINDER_SEGMENTS, int) and simulation.CYLINDER_SEGMENTS >= 24, "CYLINDER_SEGMENTS must be int >= 24")
        add(isinstance(simulation.DT, float) and simulation.DT < 0.1, "DT must be float and < 0.1")
        add(isinstance(simulation.SUBSTEPS, int) and simulation.SUBSTEPS >= 1, "SUBSTEPS must be int >= 1")
        add(isinstance(simulation.SIM_DURATION, (int, float)) and simulation.SIM_DURATION > 0, "SIM_DURATION must be float > 0")
        add(isinstance(simulation.SETTLE_THRESHOLD, (int, float)) and simulation.SETTLE_THRESHOLD > 0, "SETTLE_THRESHOLD must be float > 0")
        add(isinstance(simulation.GRAVITY, tuple) and len(simulation.GRAVITY) == 3, "GRAVITY must be tuple of len 3")
        add(isinstance(simulation.SAVE_HDF5, bool), "SAVE_HDF5 must be bool")
        add(isinstance(simulation.SAVE_CSV, bool), "SAVE_CSV must be bool")

        if errors:
            return False, "; ".join(errors)
        return True, None

    run_check("1.4", "Parameter types correct", check_14)

    def check_15() -> Tuple[bool, Optional[str]]:
        try:
            import simulation  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            return False, f"simulation not importable: {exc!r}"

        required_funcs = [
            "load_particle_mesh",
            "create_environment",
            "spawn_particles",
            "run_simulation",
            "extract_contacts",
            "compute_metrics",
            "export_results",
            "visualize_results",
            "main",
            "contact_efficiency",
            "weighted_contact_efficiency",
        ]
        missing = [name for name in required_funcs if not callable(getattr(simulation, name, None))]
        if missing:
            return False, f"Missing or non-callable functions: {', '.join(missing)}"
        return True, None

    run_check("1.5", "All required functions present", check_15)

    def check_16() -> Tuple[bool, Optional[str]]:
        try:
            from simulation import NormalizedContact  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            return False, f"Could not import NormalizedContact: {exc!r}"

        required_fields = [
            "entity_a",
            "entity_b",
            "is_particle_particle",
            "is_particle_container",
            "position",
            "normal",
            "depth",
            "force",
            "contact_area",
        ]
        missing = [f for f in required_fields if f not in getattr(NormalizedContact, "__annotations__", {})]
        if missing:
            return False, f"Missing dataclass fields: {', '.join(missing)}"
        return True, None

    run_check("1.6", "NormalizedContact dataclass present and has correct fields", check_16)


def section_2() -> None:
    _print_section("── Section 2: Mesh loading ─────────────────────────────────────")

    try:
        import numpy as np  # type: ignore[import-not-found]
        import trimesh  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        msg = f"trimesh / numpy not available: {exc!r}"

        def make_fail_check(_: str) -> CheckFunc:
            def _fail() -> Tuple[bool, Optional[str]]:
                return False, msg

            return _fail

        checks = [
            ("2.1", "OBJ file loads without error"),
            ("2.2", "STL file loads without error"),
            ("2.3", "Scale factor is applied correctly"),
            ("2.4", "Mesh is centered at origin after load"),
            ("2.5", "Unsupported format raises ValueError"),
            ("2.6", "Empty/nonexistent file raises ValueError"),
            ("2.7", "Physics mesh is not empty after decomposition"),
        ]
        for cid, desc in checks:
            run_check(cid, desc, make_fail_check(cid))
        return

    try:
        import simulation  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        msg = f"simulation not importable: {exc!r}"

        def make_fail_check(_: str) -> CheckFunc:
            def _fail() -> Tuple[bool, Optional[str]]:
                return False, msg

            return _fail

        checks = [
            ("2.1", "OBJ file loads without error"),
            ("2.2", "STL file loads without error"),
            ("2.3", "Scale factor is applied correctly"),
            ("2.4", "Mesh is centered at origin after load"),
            ("2.5", "Unsupported format raises ValueError"),
            ("2.6", "Empty/nonexistent file raises ValueError"),
            ("2.7", "Physics mesh is not empty after decomposition"),
        ]
        for cid, desc in checks:
            run_check(cid, desc, make_fail_check(cid))
        return

    # Reusable test mesh
    test_mesh = trimesh.creation.box(extents=[0.02, 0.02, 0.02])
    test_dir = "/tmp"
    os.makedirs(test_dir, exist_ok=True)
    test_mesh_path = "/tmp/verify_particle.obj"
    test_mesh_stl_path = "/tmp/verify_particle.stl"
    test_mesh.export(test_mesh_path)
    test_mesh.export(test_mesh_stl_path)

    def check_21() -> Tuple[bool, Optional[str]]:
        physics_mesh, original_mesh = simulation.load_particle_mesh(test_mesh_path, scale=1.0)
        if not isinstance(physics_mesh, trimesh.Trimesh) or not isinstance(original_mesh, trimesh.Trimesh):
            return False, "Return value must be a tuple of two trimesh.Trimesh objects"
        return True, None

    run_check("2.1", "OBJ file loads without error", check_21)

    def check_22() -> Tuple[bool, Optional[str]]:
        physics_mesh, original_mesh = simulation.load_particle_mesh(test_mesh_stl_path, scale=1.0)
        if not isinstance(physics_mesh, trimesh.Trimesh) or not isinstance(original_mesh, trimesh.Trimesh):
            return False, "Return value must be a tuple of two trimesh.Trimesh objects"
        return True, None

    run_check("2.2", "STL file loads without error", check_22)

    def check_23() -> Tuple[bool, Optional[str]]:
        _, mesh1 = simulation.load_particle_mesh(test_mesh_path, scale=1.0)
        _, mesh2 = simulation.load_particle_mesh(test_mesh_path, scale=2.0)
        ratio = mesh2.extents / mesh1.extents
        if not np.all((ratio >= 1.98) & (ratio <= 2.02)):
            return False, f"Expected ratio in [1.98, 2.02], got {ratio}"
        return True, None

    run_check("2.3", "Scale factor is applied correctly", check_23)

    def check_24() -> Tuple[bool, Optional[str]]:
        _, mesh = simulation.load_particle_mesh(test_mesh_path, scale=1.0)
        centroid = mesh.centroid
        if not np.allclose(centroid, np.zeros(3), atol=1e-6):
            return False, f"Centroid not at origin: {centroid}"
        return True, None

    run_check("2.4", "Mesh is centered at origin after load", check_24)

    def check_25() -> Tuple[bool, Optional[str]]:
        path = "/tmp/verify_particle.ply"
        test_mesh.export(path)
        try:
            simulation.load_particle_mesh(path, scale=1.0)
        except ValueError:
            return True, None
        except Exception as exc:  # noqa: BLE001
            return False, f"Expected ValueError, got {type(exc).__name__}: {exc!r}"
        else:
            return False, "Expected ValueError for unsupported '.ply' format, but no exception was raised"

    run_check("2.5", "Unsupported format raises ValueError", check_25)

    def check_26() -> Tuple[bool, Optional[str]]:
        path = "/tmp/does_not_exist.obj"
        try:
            simulation.load_particle_mesh(path, scale=1.0)
        except (ValueError, FileNotFoundError):
            return True, None
        except Exception as exc:  # noqa: BLE001
            return False, f"Expected ValueError or FileNotFoundError, got {type(exc).__name__}: {exc!r}"
        else:
            return False, "Expected an exception for nonexistent file, but none was raised"

    run_check("2.6", "Empty/nonexistent file raises ValueError", check_26)

    def check_27() -> Tuple[bool, Optional[str]]:
        physics_mesh, _ = simulation.load_particle_mesh(test_mesh_path, scale=1.0)
        if len(physics_mesh.vertices) <= 0 or len(physics_mesh.faces) <= 0:
            return False, "physics_mesh has empty vertices or faces"
        return True, None

    run_check("2.7", "Physics mesh is not empty after decomposition", check_27)


def _init_genesis() -> Tuple[bool, Optional[str]]:
    try:
        import genesis as gs  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        return False, f"genesis not importable: {exc!r}"

    try:
        gs.init(backend=gs.cpu)
    except Exception as exc:  # noqa: BLE001
        return False, f"gs.init failed: {exc!r}"
    return True, None


def section_3() -> None:
    _print_section("── Section 3: Environment construction ────────────────────────")

    ok, msg = _init_genesis()
    if not ok:
        def make_fail_check(_: str) -> CheckFunc:
            def _fail() -> Tuple[bool, Optional[str]]:
                return False, msg

            return _fail

        checks = [
            ("3.1", "Cylinder environment returns correct env_info keys"),
            ("3.2", "Cylinder env_info values are numerically correct"),
            ("3.3", "Cylinder container_ids is non-empty and contains only ints"),
            ("3.4", "Plate environment returns correct env_info keys"),
            ("3.5", "Plate env_info values are numerically correct"),
            ("3.6", "Invalid environment type raises ValueError"),
            ("3.7", "scene.build() succeeds after environment construction"),
        ]
        for cid, desc in checks:
            run_check(cid, desc, make_fail_check(cid))
        return

    import genesis as gs  # type: ignore[import-not-found]  # noqa: E401
    try:
        import simulation  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        msg = f"simulation not importable: {exc!r}"

        def make_fail_check(_: str) -> CheckFunc:
            def _fail() -> Tuple[bool, Optional[str]]:
                return False, msg

            return _fail

        checks = [
            ("3.1", "Cylinder environment returns correct env_info keys"),
            ("3.2", "Cylinder env_info values are numerically correct"),
            ("3.3", "Cylinder container_ids is non-empty and contains only ints"),
            ("3.4", "Plate environment returns correct env_info keys"),
            ("3.5", "Plate env_info values are numerically correct"),
            ("3.6", "Invalid environment type raises ValueError"),
            ("3.7", "scene.build() succeeds after environment construction"),
        ]
        for cid, desc in checks:
            run_check(cid, desc, make_fail_check(cid))
        return

    def _new_scene() -> "gs.Scene":
        return gs.Scene(gravity=simulation.GRAVITY, dt=simulation.DT, substeps=simulation.SUBSTEPS)

    def check_31() -> Tuple[bool, Optional[str]]:
        scene = _new_scene()
        container_ids, env_info = simulation.create_environment(
            scene,
            kind="cylinder",
            plate_size=simulation.PLATE_SIZE,
            cyl_diameter=simulation.CYLINDER_DIAMETER,
            cyl_height=simulation.CYLINDER_HEIGHT,
            cyl_segments=simulation.CYLINDER_SEGMENTS,
            wall_thickness=simulation.WALL_THICKNESS,
        )
        required_keys = {"surface_y", "top_y", "inner_radius", "spread_radius"}
        missing = [k for k in required_keys if k not in env_info]
        if missing:
            return False, f"Missing keys in env_info: {', '.join(missing)}"
        if not container_ids:
            return False, "container_ids is empty"
        return True, None

    run_check("3.1", "Cylinder environment returns correct env_info keys", check_31)

    def check_32() -> Tuple[bool, Optional[str]]:
        scene = _new_scene()
        _, env_info = simulation.create_environment(
            scene,
            kind="cylinder",
            plate_size=simulation.PLATE_SIZE,
            cyl_diameter=simulation.CYLINDER_DIAMETER,
            cyl_height=simulation.CYLINDER_HEIGHT,
            cyl_segments=simulation.CYLINDER_SEGMENTS,
            wall_thickness=simulation.WALL_THICKNESS,
        )
        expected_surface_y = simulation.WALL_THICKNESS
        expected_top_y = simulation.WALL_THICKNESS + simulation.CYLINDER_HEIGHT
        expected_inner_radius = simulation.CYLINDER_DIAMETER / 2
        expected_spread_radius = expected_inner_radius * 0.85

        def near(a: float, b: float) -> bool:
            return abs(a - b) < 1e-9

        errors = []
        if not near(env_info["surface_y"], expected_surface_y):
            errors.append(f"surface_y expected {expected_surface_y}, got {env_info['surface_y']}")
        if not near(env_info["top_y"], expected_top_y):
            errors.append(f"top_y expected {expected_top_y}, got {env_info['top_y']}")
        if not near(env_info["inner_radius"], expected_inner_radius):
            errors.append(f"inner_radius expected {expected_inner_radius}, got {env_info['inner_radius']}")
        if not near(env_info["spread_radius"], expected_spread_radius):
            errors.append(f"spread_radius expected {expected_spread_radius}, got {env_info['spread_radius']}")

        if errors:
            return False, "; ".join(errors)
        return True, None

    run_check("3.2", "Cylinder env_info values are numerically correct", check_32)

    def check_33() -> Tuple[bool, Optional[str]]:
        scene = _new_scene()
        container_ids, _ = simulation.create_environment(
            scene,
            kind="cylinder",
            plate_size=simulation.PLATE_SIZE,
            cyl_diameter=simulation.CYLINDER_DIAMETER,
            cyl_height=simulation.CYLINDER_HEIGHT,
            cyl_segments=simulation.CYLINDER_SEGMENTS,
            wall_thickness=simulation.WALL_THICKNESS,
        )
        if len(container_ids) != simulation.CYLINDER_SEGMENTS + 1:
            return False, f"Expected {simulation.CYLINDER_SEGMENTS + 1} container_ids, got {len(container_ids)}"
        if not all(isinstance(i, int) for i in container_ids):
            return False, "container_ids contains non-integer values"
        return True, None

    run_check("3.3", "Cylinder container_ids is non-empty and contains only ints", check_33)

    def check_34() -> Tuple[bool, Optional[str]]:
        scene = _new_scene()
        container_ids, env_info = simulation.create_environment(
            scene,
            kind="plate",
            plate_size=simulation.PLATE_SIZE,
            cyl_diameter=simulation.CYLINDER_DIAMETER,
            cyl_height=simulation.CYLINDER_HEIGHT,
            cyl_segments=simulation.CYLINDER_SEGMENTS,
            wall_thickness=simulation.WALL_THICKNESS,
        )
        required_keys = {"surface_y", "top_y", "spread_radius"}
        missing = [k for k in required_keys if k not in env_info]
        if missing:
            return False, f"Missing keys in env_info: {', '.join(missing)}"
        if not container_ids:
            return False, "container_ids is empty"
        return True, None

    run_check("3.4", "Plate environment returns correct env_info keys", check_34)

    def check_35() -> Tuple[bool, Optional[str]]:
        scene = _new_scene()
        _, env_info = simulation.create_environment(
            scene,
            kind="plate",
            plate_size=simulation.PLATE_SIZE,
            cyl_diameter=simulation.CYLINDER_DIAMETER,
            cyl_height=simulation.CYLINDER_HEIGHT,
            cyl_segments=simulation.CYLINDER_SEGMENTS,
            wall_thickness=simulation.WALL_THICKNESS,
        )
        expected_surface_y = simulation.WALL_THICKNESS
        expected_top_y = simulation.WALL_THICKNESS
        expected_spread_radius = simulation.PLATE_SIZE / 2 * 0.9

        def near(a: float, b: float) -> bool:
            return abs(a - b) < 1e-9

        errors = []
        if not near(env_info["surface_y"], expected_surface_y):
            errors.append(f"surface_y expected {expected_surface_y}, got {env_info['surface_y']}")
        if not near(env_info["top_y"], expected_top_y):
            errors.append(f"top_y expected {expected_top_y}, got {env_info['top_y']}")
        if not near(env_info["spread_radius"], expected_spread_radius):
            errors.append(f"spread_radius expected {expected_spread_radius}, got {env_info['spread_radius']}")

        if errors:
            return False, "; ".join(errors)
        return True, None

    run_check("3.5", "Plate env_info values are numerically correct", check_35)

    def check_36() -> Tuple[bool, Optional[str]]:
        scene = _new_scene()
        try:
            simulation.create_environment(
                scene,
                kind="cone",
                plate_size=simulation.PLATE_SIZE,
                cyl_diameter=simulation.CYLINDER_DIAMETER,
                cyl_height=simulation.CYLINDER_HEIGHT,
                cyl_segments=simulation.CYLINDER_SEGMENTS,
                wall_thickness=simulation.WALL_THICKNESS,
            )
        except ValueError:
            return True, None
        except Exception as exc:  # noqa: BLE001
            return False, f"Expected ValueError, got {type(exc).__name__}: {exc!r}"
        else:
            return False, "Expected ValueError for invalid environment type 'cone', but no exception was raised"

    run_check("3.6", "Invalid environment type raises ValueError", check_36)

    def check_37() -> Tuple[bool, Optional[str]]:
        scene = _new_scene()
        simulation.create_environment(
            scene,
            kind="cylinder",
            plate_size=simulation.PLATE_SIZE,
            cyl_diameter=simulation.CYLINDER_DIAMETER,
            cyl_height=simulation.CYLINDER_HEIGHT,
            cyl_segments=simulation.CYLINDER_SEGMENTS,
            wall_thickness=simulation.WALL_THICKNESS,
        )
        scene.build()
        return True, None

    run_check("3.7", "scene.build() succeeds after environment construction", check_37)


def section_4() -> None:
    _print_section("── Section 4: Particle spawning ───────────────────────────────")

    ok, msg = _init_genesis()
    if not ok:
        def make_fail_check(_: str) -> CheckFunc:
            def _fail() -> Tuple[bool, Optional[str]]:
                return False, msg

            return _fail

        checks = [
            ("4.1", "Spawn returns correct entity count"),
            ("4.2", "All spawn Y positions are above container top"),
            ("4.3", "All spawn XZ positions are within spread radius"),
            ("4.4", "Rigid material selected when E > 1e8"),
            ("4.5", "FEM material selected when 1e3 < E <= 1e8"),
            ("4.6", "MPM material selected when E <= 1e3"),
            ("4.7", "scene.build() succeeds after spawning"),
        ]
        for cid, desc in checks:
            run_check(cid, desc, make_fail_check(cid))
        return

    try:
        import numpy as np  # type: ignore[import-not-found]
        import trimesh  # type: ignore[import-not-found]
        import genesis as gs  # type: ignore[import-not-found]
        import simulation  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        msg = f"Required modules not importable for spawning tests: {exc!r}"

        def make_fail_check(_: str) -> CheckFunc:
            def _fail() -> Tuple[bool, Optional[str]]:
                return False, msg

            return _fail

        checks = [
            ("4.1", "Spawn returns correct entity count"),
            ("4.2", "All spawn Y positions are above container top"),
            ("4.3", "All spawn XZ positions are within spread radius"),
            ("4.4", "Rigid material selected when E > 1e8"),
            ("4.5", "FEM material selected when 1e3 < E <= 1e8"),
            ("4.6", "MPM material selected when E <= 1e3"),
            ("4.7", "scene.build() succeeds after spawning"),
        ]
        for cid, desc in checks:
            run_check(cid, desc, make_fail_check(cid))
        return

    # Reuse test mesh from section 2 logic
    test_mesh = trimesh.creation.box(extents=[0.02, 0.02, 0.02])
    os.makedirs("/tmp", exist_ok=True)
    mesh_path = "/tmp/verify_spawn_particle.obj"
    test_mesh.export(mesh_path)
    physics_mesh, _ = simulation.load_particle_mesh(mesh_path, scale=1.0)

    n_test = 10
    E_rigid = 200_000_000
    E_fem = 1_000_000
    E_mpm = 500

    def _build_env_scene() -> Tuple["gs.Scene", dict, set, List]:
        scene = gs.Scene(gravity=simulation.GRAVITY, dt=simulation.DT, substeps=simulation.SUBSTEPS)
        container_ids, env_info = simulation.create_environment(
            scene,
            kind="cylinder",
            plate_size=simulation.PLATE_SIZE,
            cyl_diameter=simulation.CYLINDER_DIAMETER,
            cyl_height=simulation.CYLINDER_HEIGHT,
            cyl_segments=simulation.CYLINDER_SEGMENTS,
            wall_thickness=simulation.WALL_THICKNESS,
        )
        entities = simulation.spawn_particles(
            scene=scene,
            physics_mesh=physics_mesh,
            n=n_test,
            env_info=env_info,
            drop_height=simulation.DROP_HEIGHT,
            drop_spread=simulation.DROP_SPREAD,
            E=simulation.YOUNGS_MODULUS,
            nu=simulation.POISSON_RATIO,
            rho=simulation.DENSITY,
        )
        return scene, env_info, container_ids, entities

    def check_41() -> Tuple[bool, Optional[str]]:
        _, _, _, entities = _build_env_scene()
        if len(entities) != n_test:
            return False, f"Expected {n_test} entities, got {len(entities)}"
        return True, None

    run_check("4.1", "Spawn returns correct entity count", check_41)

    def check_42() -> Tuple[bool, Optional[str]]:
        _, env_info, _, entities = _build_env_scene()
        top_y = env_info["top_y"]
        positions = [e.get_pos() for e in entities]
        below = [p for p in positions if p[1] < top_y]
        if below:
            return False, f"{len(below)} particles spawned below container top (top_y={top_y})"
        return True, None

    run_check("4.2", "All spawn Y positions are above container top", check_42)

    def check_43() -> Tuple[bool, Optional[str]]:
        _, env_info, _, entities = _build_env_scene()
        limit = env_info["spread_radius"] * simulation.DROP_SPREAD * 1.05
        violations = 0
        for e in entities:
            x, y, z = e.get_pos()
            _ = y
            r = math.sqrt(x * x + z * z)
            if r > limit:
                violations += 1
        if violations:
            return False, f"{violations} particles spawned outside allowed radius {limit}"
        return True, None

    run_check("4.3", "All spawn XZ positions are within spread radius", check_43)

    def _spawn_one(E: float) -> "gs.Entity":
        scene = gs.Scene(gravity=simulation.GRAVITY, dt=simulation.DT, substeps=simulation.SUBSTEPS)
        _, env_info = simulation.create_environment(
            scene,
            kind="cylinder",
            plate_size=simulation.PLATE_SIZE,
            cyl_diameter=simulation.CYLINDER_DIAMETER,
            cyl_height=simulation.CYLINDER_HEIGHT,
            cyl_segments=simulation.CYLINDER_SEGMENTS,
            wall_thickness=simulation.WALL_THICKNESS,
        )
        entities = simulation.spawn_particles(
            scene=scene,
            physics_mesh=physics_mesh,
            n=1,
            env_info=env_info,
            drop_height=simulation.DROP_HEIGHT,
            drop_spread=simulation.DROP_SPREAD,
            E=E,
            nu=simulation.POISSON_RATIO,
            rho=simulation.DENSITY,
        )
        return entities[0]

    def check_44() -> Tuple[bool, Optional[str]]:
        e = _spawn_one(E_rigid)
        if not isinstance(e.material, gs.materials.Rigid):
            return False, f"Expected gs.materials.Rigid, got {type(e.material).__name__}"
        return True, None

    run_check("4.4", "Rigid material selected when E > 1e8", check_44)

    def check_45() -> Tuple[bool, Optional[str]]:
        e = _spawn_one(E_fem)
        if not isinstance(e.material, gs.materials.FEM):
            return False, f"Expected gs.materials.FEM, got {type(e.material).__name__}"
        return True, None

    run_check("4.5", "FEM material selected when 1e3 < E <= 1e8", check_45)

    def check_46() -> Tuple[bool, Optional[str]]:
        e = _spawn_one(E_mpm)
        if not isinstance(e.material, gs.materials.MPM):
            return False, f"Expected gs.materials.MPM, got {type(e.material).__name__}"
        return True, None

    run_check("4.6", "MPM material selected when E <= 1e3", check_46)

    def check_47() -> Tuple[bool, Optional[str]]:
        scene, _, _, entities = _build_env_scene()
        scene.build()
        _ = entities
        return True, None

    run_check("4.7", "scene.build() succeeds after spawning", check_47)


def _build_full_pipeline(
    n_particles: int,
    sim_duration: float,
) -> Tuple[object, List[object], set, dict, object, object]:
    import numpy as np  # type: ignore[import-not-found]
    import genesis as gs  # type: ignore[import-not-found]
    import trimesh  # type: ignore[import-not-found]
    import simulation  # type: ignore[import-not-found]

    _ = np, trimesh  # just to ensure imports

    gs.init(backend=gs.cpu)
    scene = gs.Scene(gravity=simulation.GRAVITY, dt=simulation.DT, substeps=simulation.SUBSTEPS)

    physics_mesh, original_mesh = simulation.load_particle_mesh(simulation.PARTICLE_FILE, simulation.SCALE_FACTOR)
    container_ids, env_info = simulation.create_environment(
        scene,
        kind=simulation.ENVIRONMENT_TYPE,
        plate_size=simulation.PLATE_SIZE,
        cyl_diameter=simulation.CYLINDER_DIAMETER,
        cyl_height=simulation.CYLINDER_HEIGHT,
        cyl_segments=simulation.CYLINDER_SEGMENTS,
        wall_thickness=simulation.WALL_THICKNESS,
    )
    entities = simulation.spawn_particles(
        scene=scene,
        physics_mesh=physics_mesh,
        n=n_particles,
        env_info=env_info,
        drop_height=simulation.DROP_HEIGHT,
        drop_spread=simulation.DROP_SPREAD,
        E=simulation.YOUNGS_MODULUS,
        nu=simulation.POISSON_RATIO,
        rho=simulation.DENSITY,
    )
    scene.build()
    simulation.run_simulation(
        scene=scene,
        entities=entities,
        dt=simulation.DT,
        substeps=simulation.SUBSTEPS,
        duration=sim_duration,
        settle_threshold=simulation.SETTLE_THRESHOLD,
    )
    return scene, entities, container_ids, env_info, physics_mesh, original_mesh


def section_5() -> None:
    _print_section("── Section 5: Simulation loop ─────────────────────────────────")

    try:
        import simulation  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
        import genesis as gs  # type: ignore[import-not-found]
        _ = np, gs
    except Exception as exc:  # noqa: BLE001
        msg = f"Required modules not importable for simulation tests: {exc!r}"

        def make_fail_check(_: str) -> CheckFunc:
            def _fail() -> Tuple[bool, Optional[str]]:
                return False, msg

            return _fail

        checks = [
            ("5.1", "run_simulation() returns an int"),
            ("5.2", "Return value is within valid range"),
            ("5.3", "Simulation does not raise on normal execution"),
            ("5.4", "Particles have moved from spawn positions after simulation"),
        ]
        for cid, desc in checks:
            run_check(cid, desc, make_fail_check(cid))
        return

    def check_51() -> Tuple[bool, Optional[str]]:
        scene, entities, _, _, _, _ = _build_full_pipeline(5, 0.5)
        steps = simulation.run_simulation(
            scene=scene,
            entities=entities,
            dt=simulation.DT,
            substeps=simulation.SUBSTEPS,
            duration=0.5,
            settle_threshold=simulation.SETTLE_THRESHOLD,
        )
        if not isinstance(steps, int):
            return False, f"Expected int, got {type(steps).__name__}"
        return True, None

    run_check("5.1", "run_simulation() returns an int", check_51)

    def check_52() -> Tuple[bool, Optional[str]]:
        scene, entities, _, _, _, _ = _build_full_pipeline(5, 0.5)
        steps = simulation.run_simulation(
            scene=scene,
            entities=entities,
            dt=simulation.DT,
            substeps=simulation.SUBSTEPS,
            duration=0.5,
            settle_threshold=simulation.SETTLE_THRESHOLD,
        )
        max_steps = int(0.5 / simulation.DT)
        if not (0 <= steps <= max_steps):
            return False, f"Step count {steps} not in [0, {max_steps}]"
        return True, None

    run_check("5.2", "Return value is within valid range", check_52)

    def check_53() -> Tuple[bool, Optional[str]]:
        _build_full_pipeline(5, 0.5)
        return True, None

    run_check("5.3", "Simulation does not raise on normal execution", check_53)

    def check_54() -> Tuple[bool, Optional[str]]:
        import numpy as np  # type: ignore[import-not-found]

        scene, entities, _, _, _, _ = _build_full_pipeline(5, 0.5)
        spawn_y = [e.get_pos()[1] for e in entities]
        simulation.run_simulation(
            scene=scene,
            entities=entities,
            dt=simulation.DT,
            substeps=simulation.SUBSTEPS,
            duration=0.5,
            settle_threshold=simulation.SETTLE_THRESHOLD,
        )
        moved = 0
        for e, y0 in zip(entities, spawn_y):
            y1 = e.get_pos()[1]
            if abs(float(y1) - float(y0)) > 1e-4:
                moved += 1
        if moved < int(0.8 * len(entities)):
            return False, f"Only {moved}/{len(entities)} particles moved more than 1e-4 m in Y"
        _ = np
        return True, None

    run_check("5.4", "Particles have moved from spawn positions after simulation", check_54)


def section_6() -> None:
    _print_section("── Section 6: Contact extraction & metrics ───────────────────")

    try:
        import numpy as np  # type: ignore[import-not-found]
        import networkx as nx  # type: ignore[import-not-found]
        import simulation  # type: ignore[import-not-found]
        _ = np, nx
    except Exception as exc:  # noqa: BLE001
        msg = f"Required modules not importable for contact tests: {exc!r}"

        def make_fail_check(_: str) -> CheckFunc:
            def _fail() -> Tuple[bool, Optional[str]]:
                return False, msg

            return _fail

        checks = [
            ("6.1", "extract_contacts() returns a list"),
            ("6.2", "All returned items are NormalizedContact instances"),
            ("6.3", "No contact has depth below CONTACT_DEPTH_TOL"),
            ("6.4", "Every contact is classified as pp or pc but not both"),
            ("6.5", "Contact entity IDs are known particles or container"),
            ("6.6", "compute_metrics() returns all required keys"),
            ("6.7", "Z equals avg_contacts_per_particle"),
            ("6.8", "Z is numerically consistent with pp contact count"),
            ("6.9", "contact_graph has correct node count"),
            ("6.10", "contact_graph edge count matches total_pp_contacts"),
            ("6.11", "contact_points and contact_normals have equal length"),
            ("6.12", "All contact normals are unit vectors"),
            ("6.13", "n_isolated_particles + particles with contacts == N"),
        ]
        for cid, desc in checks:
            run_check(cid, desc, make_fail_check(cid))
        return

    # Full simulation with N_PARTICLES=20 and SIM_DURATION=2.0
    N_PARTICLES_TEST = 20
    SIM_DURATION_TEST = 2.0

    scene, entities, container_ids, _, _, _ = _build_full_pipeline(N_PARTICLES_TEST, SIM_DURATION_TEST)
    particle_ids = {e.id for e in entities}
    contacts = simulation.extract_contacts(
        scene=scene,
        particle_ids=particle_ids,
        container_ids=container_ids,
        depth_tol=simulation.CONTACT_DEPTH_TOL,
    )
    metrics = simulation.compute_metrics(contacts, particle_ids)

    def check_61() -> Tuple[bool, Optional[str]]:
        if not isinstance(contacts, list):
            return False, f"Expected list, got {type(contacts).__name__}"
        return True, None

    run_check("6.1", "extract_contacts() returns a list", check_61)

    def check_62() -> Tuple[bool, Optional[str]]:
        from simulation import NormalizedContact  # type: ignore[import-not-found]

        if contacts and not all(isinstance(c, NormalizedContact) for c in contacts):
            first_bad = next(c for c in contacts if not isinstance(c, NormalizedContact))
            return False, f"Non-NormalizedContact in list: {type(first_bad).__name__}"
        return True, None

    run_check("6.2", "All returned items are NormalizedContact instances", check_62)

    def check_63() -> Tuple[bool, Optional[str]]:
        tol = simulation.CONTACT_DEPTH_TOL
        violations = [c for c in contacts if abs(c.depth) < tol]
        if violations:
            return False, f"{len(violations)} contacts have |depth| < CONTACT_DEPTH_TOL ({tol})"
        return True, None

    run_check("6.3", "No contact has depth below CONTACT_DEPTH_TOL", check_63)

    def check_64() -> Tuple[bool, Optional[str]]:
        bad = 0
        for c in contacts:
            if (c.is_particle_particle and c.is_particle_container) or (
                not c.is_particle_particle and not c.is_particle_container
            ):
                bad += 1
        if bad:
            return False, f"{bad} contacts have invalid pp/pc classification"
        return True, None

    run_check("6.4", "Every contact is classified as pp or pc but not both", check_64)

    def check_65() -> Tuple[bool, Optional[str]]:
        unknown = set()
        known = particle_ids | container_ids
        for c in contacts:
            if c.entity_a not in known:
                unknown.add(c.entity_a)
            if c.entity_b not in known:
                unknown.add(c.entity_b)
        if unknown:
            return False, f"Unknown entity IDs in contacts: {sorted(unknown)}"
        return True, None

    run_check("6.5", "Contact entity IDs are known particles or container", check_65)

    def check_66() -> Tuple[bool, Optional[str]]:
        required = [
            "total_pp_contacts",
            "total_pc_contacts",
            "avg_contacts_per_particle",
            "Z",
            "n_isolated_particles",
            "n_container_touching",
            "contact_points",
            "contact_normals",
            "contact_depths",
            "contact_forces",
            "contact_areas",
            "contact_graph",
            "contact_graph_dict",
            "contact_counts_per_particle",
        ]
        missing = [k for k in required if k not in metrics]
        if missing:
            return False, f"Missing metrics keys: {', '.join(missing)}"
        return True, None

    run_check("6.6", "compute_metrics() returns all required keys", check_66)

    def check_67() -> Tuple[bool, Optional[str]]:
        if metrics["Z"] != metrics["avg_contacts_per_particle"]:
            return False, f"Z={metrics['Z']} != avg_contacts_per_particle={metrics['avg_contacts_per_particle']}"
        return True, None

    run_check("6.7", "Z equals avg_contacts_per_particle", check_67)

    def check_68() -> Tuple[bool, Optional[str]]:
        Z = metrics["Z"]
        expected = (2 * metrics["total_pp_contacts"]) / N_PARTICLES_TEST if N_PARTICLES_TEST else 0.0
        if abs(Z - expected) >= 1e-9:
            return False, f"Z={Z} inconsistent with 2*pp/{N_PARTICLES_TEST}={expected}"
        return True, None

    run_check("6.8", "Z is numerically consistent with pp contact count", check_68)

    def check_69() -> Tuple[bool, Optional[str]]:
        G = metrics["contact_graph"]
        if len(G.nodes) != N_PARTICLES_TEST:
            return False, f"Expected {N_PARTICLES_TEST} nodes, got {len(G.nodes)}"
        return True, None

    run_check("6.9", "contact_graph has correct node count", check_69)

    def check_610() -> Tuple[bool, Optional[str]]:
        G = metrics["contact_graph"]
        if len(G.edges) != metrics["total_pp_contacts"]:
            return False, f"Expected {metrics['total_pp_contacts']} edges, got {len(G.edges)}"
        return True, None

    run_check("6.10", "contact_graph edge count matches total_pp_contacts", check_610)

    def check_611() -> Tuple[bool, Optional[str]]:
        if len(metrics["contact_points"]) != len(metrics["contact_normals"]):
            return False, f"points={len(metrics['contact_points'])}, normals={len(metrics['contact_normals'])}"
        return True, None

    run_check("6.11", "contact_points and contact_normals have equal length", check_611)

    def check_612() -> Tuple[bool, Optional[str]]:
        normals = metrics["contact_normals"]
        bad = 0
        for n in normals:
            norm = float(np.linalg.norm(n))
            if abs(norm - 1.0) >= 1e-5:
                bad += 1
        if bad:
            return False, f"{bad} / {len(normals)} normals are not unit length"
        return True, None

    run_check("6.12", "All contact normals are unit vectors", check_612)

    def check_613() -> Tuple[bool, Optional[str]]:
        counts = metrics["contact_counts_per_particle"]
        n_isolated = metrics["n_isolated_particles"]
        n_with_contacts = sum(1 for d in counts.values() if d > 0)
        if n_isolated + n_with_contacts != N_PARTICLES_TEST:
            return False, f"n_isolated + n_with_contacts = {n_isolated + n_with_contacts}, expected {N_PARTICLES_TEST}"
        return True, None

    run_check("6.13", "n_isolated_particles + particles with contacts == N", check_613)


def section_7() -> None:
    _print_section("── Section 7: Export ──────────────────────────────────────────")

    try:
        import pandas as pd  # type: ignore[import-not-found]
        import h5py  # type: ignore[import-not-found]
        import simulation  # type: ignore[import-not-found]
        _ = pd, h5py
    except Exception as exc:  # noqa: BLE001
        msg = f"Required modules not importable for export tests: {exc!r}"

        def make_fail_check(_: str) -> CheckFunc:
            def _fail() -> Tuple[bool, Optional[str]]:
                return False, msg

            return _fail

        checks = [
            ("7.1", "Output directory is created"),
            ("7.2", "All three CSV files exist"),
            ("7.3", "particles.csv has correct columns"),
            ("7.4", "contact_pairs.csv has correct columns"),
            ("7.5", "contact_points.csv has correct columns"),
            ("7.6", "HDF5 file exists and has correct datasets"),
            ("7.7", "HDF5 attributes are present and correct type"),
            ("7.8", "HDF5 particle_positions shape matches N_PARTICLES"),
            ("7.9", "CSV particle positions match HDF5 positions"),
        ]
        for cid, desc in checks:
            run_check(cid, desc, make_fail_check(cid))
        return

    N_PARTICLES_TEST = 20
    SIM_DURATION_TEST = 2.0

    scene, entities, container_ids, _, _, _ = _build_full_pipeline(N_PARTICLES_TEST, SIM_DURATION_TEST)
    particle_ids = {e.id for e in entities}
    contacts = simulation.extract_contacts(
        scene=scene,
        particle_ids=particle_ids,
        container_ids=container_ids,
        depth_tol=simulation.CONTACT_DEPTH_TOL,
    )
    metrics = simulation.compute_metrics(contacts, particle_ids)

    out_dir = "/tmp/verify_results"
    os.makedirs(out_dir, exist_ok=True)
    simulation.export_results(
        entities=entities,
        metrics=metrics,
        output_dir=out_dir,
        save_hdf5=True,
        save_csv=True,
    )

    def check_71() -> Tuple[bool, Optional[str]]:
        if not os.path.isdir(out_dir):
            return False, f"Output directory {out_dir} was not created"
        return True, None

    run_check("7.1", "Output directory is created", check_71)

    def check_72() -> Tuple[bool, Optional[str]]:
        files = [
            os.path.join(out_dir, "particles.csv"),
            os.path.join(out_dir, "contact_pairs.csv"),
            os.path.join(out_dir, "contact_points.csv"),
        ]
        missing = [f for f in files if not os.path.isfile(f) or os.path.getsize(f) <= 0]
        if missing:
            return False, f"Missing or empty CSV files: {', '.join(missing)}"
        return True, None

    run_check("7.2", "All three CSV files exist", check_72)

    def check_73() -> Tuple[bool, Optional[str]]:
        import pandas as pd  # type: ignore[import-not-found]

        path = os.path.join(out_dir, "particles.csv")
        df = pd.read_csv(path)
        required_cols = ["id", "x", "y", "z", "qx", "qy", "qz", "qw", "n_contacts"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return False, f"Missing columns in particles.csv: {', '.join(missing)}"
        if len(df) != N_PARTICLES_TEST:
            return False, f"Expected {N_PARTICLES_TEST} rows, got {len(df)}"
        return True, None

    run_check("7.3", "particles.csv has correct columns", check_73)

    def check_74() -> Tuple[bool, Optional[str]]:
        import pandas as pd  # type: ignore[import-not-found]

        path = os.path.join(out_dir, "contact_pairs.csv")
        df = pd.read_csv(path)
        required_cols = ["particle_a", "particle_b", "depth"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return False, f"Missing columns in contact_pairs.csv: {', '.join(missing)}"
        expected_rows = metrics["total_pp_contacts"]
        if len(df) != expected_rows:
            return False, f"Expected {expected_rows} rows, got {len(df)}"
        return True, None

    run_check("7.4", "contact_pairs.csv has correct columns", check_74)

    def check_75() -> Tuple[bool, Optional[str]]:
        import pandas as pd  # type: ignore[import-not-found]

        path = os.path.join(out_dir, "contact_points.csv")
        df = pd.read_csv(path)
        required_cols = ["x", "y", "z", "nx", "ny", "nz"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return False, f"Missing columns in contact_points.csv: {', '.join(missing)}"
        expected_rows = len(metrics["contact_points"])
        if len(df) != expected_rows:
            return False, f"Expected {expected_rows} rows, got {len(df)}"
        return True, None

    run_check("7.5", "contact_points.csv has correct columns", check_75)

    def check_76() -> Tuple[bool, Optional[str]]:
        import h5py  # type: ignore[import-not-found]

        path = os.path.join(out_dir, "simulation.h5")
        if not os.path.isfile(path):
            return False, f"HDF5 file {path} does not exist"
        with h5py.File(path, "r") as f:
            required = ["particle_positions", "contact_points", "contact_normals"]
            missing = [d for d in required if d not in f.keys()]
            if missing:
                return False, f"Missing datasets in HDF5: {', '.join(missing)}"
        return True, None

    run_check("7.6", "HDF5 file exists and has correct datasets", check_76)

    def check_77() -> Tuple[bool, Optional[str]]:
        import h5py  # type: ignore[import-not-found]

        path = os.path.join(out_dir, "simulation.h5")
        with h5py.File(path, "r") as f:
            required = ["Z", "total_pp", "total_pc", "n_isolated", "n_container_touch"]
            missing = [a for a in required if a not in f.attrs]
            if missing:
                return False, f"Missing attributes in HDF5: {', '.join(missing)}"
            for a in required:
                if not isinstance(f.attrs[a], (int, float)):
                    return False, f"Attribute {a} must be numeric, got {type(f.attrs[a]).__name__}"
        return True, None

    run_check("7.7", "HDF5 attributes are present and correct type", check_77)

    def check_78() -> Tuple[bool, Optional[str]]:
        import h5py  # type: ignore[import-not-found]

        path = os.path.join(out_dir, "simulation.h5")
        with h5py.File(path, "r") as f:
            shape = f["particle_positions"].shape
            if shape != (N_PARTICLES_TEST, 3):
                return False, f"particle_positions shape {shape}, expected {(N_PARTICLES_TEST, 3)}"
        return True, None

    run_check("7.8", "HDF5 particle_positions shape matches N_PARTICLES", check_78)

    def check_79() -> Tuple[bool, Optional[str]]:
        import pandas as pd  # type: ignore[import-not-found]
        import h5py  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]

        csv_path = os.path.join(out_dir, "particles.csv")
        h5_path = os.path.join(out_dir, "simulation.h5")
        df = pd.read_csv(csv_path)
        xyz_csv = df[["x", "y", "z"]].values
        with h5py.File(h5_path, "r") as f:
            xyz_h5 = f["particle_positions"][:]
        if xyz_csv.shape != xyz_h5.shape:
            return False, f"CSV positions shape {xyz_csv.shape}, HDF5 shape {xyz_h5.shape}"
        diff = np.max(np.abs(xyz_csv - xyz_h5))
        if diff > 1e-6:
            return False, f"Max discrepancy between CSV and HDF5 positions: {diff}"
        return True, None

    run_check("7.9", "CSV particle positions match HDF5 positions", check_79)


def section_8() -> None:
    _print_section("── Section 8: End-to-end smoke test ──────────────────────────")

    try:
        import pandas as pd  # type: ignore[import-not-found]
        import h5py  # type: ignore[import-not-found]
        import simulation  # type: ignore[import-not-found]
        _ = pd, h5py
    except Exception as exc:  # noqa: BLE001
        msg = f"Required modules not importable for end-to-end tests: {exc!r}"

        def make_fail_check(_: str) -> CheckFunc:
            def _fail() -> Tuple[bool, Optional[str]]:
                return False, msg

            return _fail

        checks = [
            ("8.1", "main() completes without raising any exception"),
            ("8.2", "Output files exist after main()"),
            ("8.3", "Particle count in output matches N_PARTICLES"),
            ("8.4", "Z value is non-negative and physically plausible"),
        ]
        for cid, desc in checks:
            run_check(cid, desc, make_fail_check(cid))
        return

    out_dir = "/tmp/verify_e2e"

    def _run_main_patched() -> None:
        import genesis as gs  # type: ignore[import-not-found]

        simulation.visualize_results = lambda *a, **kw: None  # type: ignore[assignment]
        simulation.N_PARTICLES = 5  # type: ignore[assignment]
        simulation.SIM_DURATION = 1.0  # type: ignore[assignment]
        simulation.OUTPUT_DIR = out_dir  # type: ignore[assignment]
        simulation.SAVE_HDF5 = True  # type: ignore[assignment]
        simulation.SAVE_CSV = True  # type: ignore[assignment]

        gs.init(backend=gs.cpu)
        simulation.main()

    def check_81() -> Tuple[bool, Optional[str]]:
        try:
            _run_main_patched()
            return True, None
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            return False, f"Exception during main(): {exc!r}\n{tb}"

    run_check("8.1", "main() completes without raising any exception", check_81)

    def check_82() -> Tuple[bool, Optional[str]]:
        files = [
            os.path.join(out_dir, "particles.csv"),
            os.path.join(out_dir, "contact_pairs.csv"),
            os.path.join(out_dir, "contact_points.csv"),
            os.path.join(out_dir, "simulation.h5"),
        ]
        missing = [f for f in files if not os.path.isfile(f) or os.path.getsize(f) <= 0]
        if missing:
            return False, f"Missing or empty output files after main(): {', '.join(missing)}"
        return True, None

    run_check("8.2", "Output files exist after main()", check_82)

    def check_83() -> Tuple[bool, Optional[str]]:
        import pandas as pd  # type: ignore[import-not-found]

        path = os.path.join(out_dir, "particles.csv")
        df = pd.read_csv(path)
        if len(df) != 5:
            return False, f"Expected 5 rows in particles.csv, got {len(df)}"
        return True, None

    run_check("8.3", "Particle count in output matches N_PARTICLES", check_83)

    def check_84() -> Tuple[bool, Optional[str]]:
        import h5py  # type: ignore[import-not-found]

        path = os.path.join(out_dir, "simulation.h5")
        with h5py.File(path, "r") as f:
            Z = float(f.attrs.get("Z", -1.0))
        if not (0.0 <= Z <= 26.0):
            return False, f"Z={Z} not in [0, 26]"
        return True, None

    run_check("8.4", "Z value is non-negative and physically plausible", check_84)


def main() -> None:
    section_1()
    section_2()
    section_3()
    section_4()
    section_5()
    section_6()
    section_7()
    section_8()

    print("════════════════════════════════════════════════════════════")
    print(f"Verification complete: {TOTAL_PASSED} passed, {TOTAL_FAILED} failed")
    if FAILED_CHECK_IDS:
        joined = ", ".join(FAILED_CHECK_IDS)
        print(f"Failed checks: {joined}")
    else:
        print("Failed checks: none")
    print("════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()

