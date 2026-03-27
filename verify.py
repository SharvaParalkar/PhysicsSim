"""
verify_simulation_fixes.py

Verification script for the 5 MPM migration fixes applied to simulation.py and server.py.
Run this AFTER applying the fixes and BEFORE running a full simulation.

Usage:
    python verify_simulation_fixes.py
    python verify_simulation_fixes.py --simulation-path ./simulation.py
    python verify_simulation_fixes.py --server-path ./server.py
    python verify_simulation_fixes.py --run-genesis  # also attempts a live Genesis smoke test

Each check prints PASS, FAIL, or WARN with a short explanation.
Exit code 0 = all required checks passed. Exit code 1 = one or more FAIL.
"""

import ast
import importlib.util
import inspect
import os
import sys
import argparse
import traceback
from pathlib import Path
from typing import Optional

# ── Colour helpers (disabled on Windows if no ANSI support) ──────────────────
_USE_COLOR = sys.platform != "win32" or os.environ.get("TERM") == "xterm"

def _green(s):  return f"\033[92m{s}\033[0m" if _USE_COLOR else s
def _red(s):    return f"\033[91m{s}\033[0m" if _USE_COLOR else s
def _yellow(s): return f"\033[93m{s}\033[0m" if _USE_COLOR else s
def _bold(s):   return f"\033[1m{s}\033[0m"  if _USE_COLOR else s

PASS  = _green("PASS")
FAIL  = _red("FAIL")
WARN  = _yellow("WARN")

results: list[tuple[str, str, str]] = []  # (check_name, status, message)


def record(name: str, status: str, message: str):
    results.append((name, status, message))
    label = {"PASS": PASS, "FAIL": FAIL, "WARN": WARN}.get(status, status)
    print(f"  [{label}] {name}: {message}")


# ── AST helpers ───────────────────────────────────────────────────────────────

def load_source(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return None


def parse_ast(source: str) -> Optional[ast.Module]:
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def get_function_names(tree: ast.Module) -> set[str]:
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}


def get_class_names(tree: ast.Module) -> set[str]:
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}


def get_all_names(tree: ast.Module) -> set[str]:
    """All Name nodes referenced anywhere in the AST."""
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def get_string_literals(tree: ast.Module) -> list[str]:
    return [node.s for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.s, str)]


def function_body_source(source: str, func_name: str) -> Optional[str]:
    """Extract raw source lines for a named function."""
    lines = source.splitlines()
    in_func = False
    func_lines = []
    indent = None
    for line in lines:
        if not in_func:
            stripped = line.lstrip()
            if stripped.startswith(f"def {func_name}(") or stripped.startswith(f"async def {func_name}("):
                in_func = True
                indent = len(line) - len(line.lstrip())
                func_lines.append(line)
        else:
            current_indent = len(line) - len(line.lstrip())
            if line.strip() == "" or current_indent > indent:
                func_lines.append(line)
            else:
                break
    return "\n".join(func_lines) if func_lines else None


def source_contains(source: str, *substrings: str) -> bool:
    return any(s in source for s in substrings)


def source_contains_all(source: str, *substrings: str) -> bool:
    return all(s in source for s in substrings)


# ═════════════════════════════════════════════════════════════════════════════
# CHECK GROUP 1 — simulation.py structural checks
# ═════════════════════════════════════════════════════════════════════════════

def check_simulation_file(sim_path: Path):
    print(_bold("\n── Group 1: simulation.py structural checks ─────────────────"))

    source = load_source(sim_path)
    if source is None:
        record("sim_file_readable", "FAIL", f"Cannot read {sim_path}")
        return
    record("sim_file_readable", "PASS", f"Read {sim_path} ({len(source)} chars)")

    tree = parse_ast(source)
    if tree is None:
        record("sim_file_parses", "FAIL", "File has a SyntaxError — fix before running")
        return
    record("sim_file_parses", "PASS", "No syntax errors")

    funcs = get_function_names(tree)

    # ── FIX 1: MPM material, no Rigid/FEM branching ──────────────────────────
    print(_bold("\n  Fix 1 — MPM material solver"))

    if "enforce_container_bounds" in funcs:
        record("fix1_no_enforce_bounds", "FAIL",
               "enforce_container_bounds() still present — should be deleted (Fix 3)")
    else:
        record("fix1_no_enforce_bounds", "PASS", "enforce_container_bounds() removed")

    if "enforce_particle_separation" in funcs:
        record("fix1_no_enforce_separation", "FAIL",
               "enforce_particle_separation() still present — should be deleted (Fix 3)")
    else:
        record("fix1_no_enforce_separation", "PASS", "enforce_particle_separation() removed")

    # MPM.Elastoplastic must appear in spawn_particles or material setup
    if "MPM.Elastoplastic" in source or "MPM" in source:
        record("fix1_mpm_material_present", "PASS", "MPM material reference found")
    else:
        record("fix1_mpm_material_present", "FAIL",
               "No MPM material reference found — spawn_particles() likely still uses Rigid/FEM only")

    # Old E > 1e8 rigid branch detection
    if "E_in > 1e8" in source or "E_in>1e8" in source:
        record("fix1_no_rigid_branch", "FAIL",
               "E_in > 1e8 Rigid branch still present — MPM should replace all solver branching")
    else:
        record("fix1_no_rigid_branch", "PASS", "E > 1e8 Rigid branch removed")

    if "FEM_JAMMING_E_MAX" in source:
        record("fix1_no_fem_jamming_emax", "WARN",
               "FEM_JAMMING_E_MAX still referenced — safe to remove once FEM path is gone")
    else:
        record("fix1_no_fem_jamming_emax", "PASS", "FEM_JAMMING_E_MAX removed")

    # ── FIX 2: MPMOptions present, analytical mode removed ───────────────────
    print(_bold("\n  Fix 2 — MPMOptions + removal of analytical mode"))

    if "MPMOptions" in source or "mpm_options" in source:
        record("fix2_mpm_options_present", "PASS", "MPMOptions found in scene setup")
    else:
        record("fix2_mpm_options_present", "FAIL",
               "MPMOptions not found — scene has no MPM grid domain configured")

    if "lower_bound" in source and "upper_bound" in source:
        record("fix2_mpm_bounds_set", "PASS", "MPM lower_bound/upper_bound found")
    else:
        record("fix2_mpm_bounds_set", "WARN",
               "MPM lower_bound/upper_bound not found — grid domain may be using defaults")

    if "ANALYTICAL_MODE" in source:
        record("fix2_no_analytical_mode", "WARN",
               "ANALYTICAL_MODE still present — should be removed once MPM is the sole solver")
    else:
        record("fix2_no_analytical_mode", "PASS", "ANALYTICAL_MODE removed")

    analytical_keys = ["ANALYTICAL_FALLING_DT", "ANALYTICAL_PRECISION_DT",
                       "ANALYTICAL_VEL_THRESHOLD", "ANALYTICAL_FALLING_SUBSTEPS"]
    remaining = [k for k in analytical_keys if k in source]
    if remaining:
        record("fix2_no_analytical_keys", "WARN",
               f"Analytical config keys still present: {remaining}")
    else:
        record("fix2_no_analytical_keys", "PASS", "All analytical config keys removed")

    if "snapshot_fem_entities" in funcs:
        record("fix2_no_fem_snapshot", "WARN",
               "snapshot_fem_entities() still present — only needed for analytical FEM handoff")
    else:
        record("fix2_no_fem_snapshot", "PASS", "snapshot_fem_entities() removed")

    if "restore_fem_entities" in funcs:
        record("fix2_no_fem_restore", "WARN",
               "restore_fem_entities() still present — only needed for analytical FEM handoff")
    else:
        record("fix2_no_fem_restore", "PASS", "restore_fem_entities() removed")

    # ── FIX 3: Container bounds + separation enforcement removed ─────────────
    print(_bold("\n  Fix 3 — enforce_container_bounds / enforce_particle_separation"))

    # Already checked above in Fix 1 section; just confirm call sites are gone
    if "enforce_container_bounds(" in source:
        record("fix3_no_bounds_callsite", "FAIL",
               "enforce_container_bounds() is still being called — remove all call sites")
    else:
        record("fix3_no_bounds_callsite", "PASS", "No enforce_container_bounds() call sites")

    if "enforce_particle_separation(" in source:
        record("fix3_no_separation_callsite", "FAIL",
               "enforce_particle_separation() is still being called — remove all call sites")
    else:
        record("fix3_no_separation_callsite", "PASS", "No enforce_particle_separation() call sites")

    # ── FIX 4: CoACD pipeline removed ────────────────────────────────────────
    print(_bold("\n  Fix 4 — CoACD pipeline"))

    if "import coacd" in source or "coacd.run_coacd" in source:
        record("fix4_no_coacd_import", "FAIL",
               "coacd import or coacd.run_coacd() still present — CoACD pipeline must be removed")
    else:
        record("fix4_no_coacd_import", "PASS", "coacd import removed")

    if "_coacd_proxy_path" in funcs:
        record("fix4_no_coacd_proxy_fn", "FAIL",
               "_coacd_proxy_path() still present — delete this function")
    else:
        record("fix4_no_coacd_proxy_fn", "PASS", "_coacd_proxy_path() removed")

    if "_write_coacd_compound_obj" in funcs:
        record("fix4_no_write_coacd_fn", "FAIL",
               "_write_coacd_compound_obj() still present — delete this function")
    else:
        record("fix4_no_write_coacd_fn", "PASS", "_write_coacd_compound_obj() removed")

    if "coacd_proxy_file" in source:
        record("fix4_no_proxy_file_param", "WARN",
               "coacd_proxy_file parameter still referenced — should be removed from spawn_particles()")
    else:
        record("fix4_no_proxy_file_param", "PASS", "coacd_proxy_file parameter removed")

    if "convexify=True" in source or "convexify=False" in source:
        record("fix4_no_convexify", "WARN",
               "convexify= still set on Mesh morph — MPM does not use collision proxies, safe to remove")
    else:
        record("fix4_no_convexify", "PASS", "convexify parameter removed from Mesh morphs")

    # vis_mode visual should be present for MPM mesh skinning
    if "vis_mode" in source and "visual" in source:
        record("fix4_vis_mode_visual", "PASS", "vis_mode='visual' found for MPM mesh skinning")
    else:
        record("fix4_vis_mode_visual", "WARN",
               "vis_mode='visual' not found — MPM particles may not render skinned to your mesh shape")

    # ── FIX 5: MPM stress contacts ────────────────────────────────────────────
    print(_bold("\n  Fix 5 — MPM stress field / contact extraction"))

    if "extract_mpm_contacts" in funcs:
        record("fix5_mpm_contacts_fn", "PASS", "extract_mpm_contacts() function found")
    else:
        record("fix5_mpm_contacts_fn", "FAIL",
               "extract_mpm_contacts() not found — MPM contact extraction function must be added")

    if "ContactSampleCache" in source:
        record("fix5_no_contact_cache", "WARN",
               "ContactSampleCache still present — no longer needed with MPM geometric contacts")
    else:
        record("fix5_no_contact_cache", "PASS", "ContactSampleCache removed")

    if "contact_extract_stride" in funcs:
        record("fix5_no_stride_fn", "WARN",
               "contact_extract_stride() still present — only needed for throttling scene.get_contacts()")
    else:
        record("fix5_no_stride_fn", "PASS", "contact_extract_stride() removed")

    if "extract_contacts_resampled" in funcs:
        record("fix5_no_resampled_fn", "WARN",
               "extract_contacts_resampled() still present — replace with extract_mpm_contacts()")
    else:
        record("fix5_no_resampled_fn", "PASS", "extract_contacts_resampled() removed")

    # MPM stress field usage
    if ".stress" in source or "get_state().stress" in source:
        record("fix5_stress_field_read", "PASS", "MPM stress field (.stress) is being read")
    else:
        record("fix5_stress_field_read", "WARN",
               ".stress field not read anywhere — MPM stress-based visualization may not be wired up")

    if "von_mises" in source.lower() or "von mises" in source.lower() or "frobenius" in source.lower():
        record("fix5_stress_metric", "PASS", "Von Mises / Frobenius stress metric found")
    else:
        record("fix5_stress_metric", "WARN",
               "No Von Mises/Frobenius computation found — stress scalar for force chains may be missing")

    # compute_fem_vertex_force_stress should be replaced
    if "compute_fem_vertex_force_stress" in funcs:
        record("fix5_no_fem_vertex_stress", "WARN",
               "compute_fem_vertex_force_stress() still present — replace with MPM stress field reader")
    else:
        record("fix5_no_fem_vertex_stress", "PASS", "compute_fem_vertex_force_stress() removed")

    # Core metrics pipeline must still be intact
    print(_bold("\n  Core metrics pipeline (must survive all fixes)"))

    for fn in ["compute_metrics", "calculate_live_metrics", "compute_max_velocity",
               "export_results", "build_runtime_config"]:
        if fn in funcs:
            record(f"core_fn_{fn}", "PASS", f"{fn}() present")
        else:
            record(f"core_fn_{fn}", "FAIL", f"{fn}() missing — this function must not be deleted")

    # NormalizedContact dataclass must survive
    classes = get_class_names(tree)
    if "NormalizedContact" in classes:
        record("core_normalized_contact", "PASS", "NormalizedContact dataclass present")
    else:
        record("core_normalized_contact", "FAIL",
               "NormalizedContact dataclass missing — compute_metrics() depends on it")


# ═════════════════════════════════════════════════════════════════════════════
# CHECK GROUP 2 — server.py structural checks
# ═════════════════════════════════════════════════════════════════════════════

def check_server_file(server_path: Path):
    print(_bold("\n── Group 2: server.py structural checks ──────────────────────"))

    source = load_source(server_path)
    if source is None:
        record("server_file_readable", "FAIL", f"Cannot read {server_path}")
        return
    record("server_file_readable", "PASS", f"Read {server_path} ({len(source)} chars)")

    tree = parse_ast(source)
    if tree is None:
        record("server_file_parses", "FAIL", "File has a SyntaxError — fix before running")
        return
    record("server_file_parses", "PASS", "No syntax errors")

    funcs = get_function_names(tree)

    # Genesis init must still be present
    if "_init_genesis_on_sim_thread" in funcs or "init_genesis_compat" in source:
        record("server_genesis_init", "PASS", "Genesis initialization function present")
    else:
        record("server_genesis_init", "FAIL",
               "Genesis initialization missing from server.py — simulation thread will fail to start")

    # ANALYTICAL_MODE references should be gone from server.py too
    if "ANALYTICAL_MODE" in source:
        record("server_no_analytical_mode", "WARN",
               "ANALYTICAL_MODE still referenced in server.py — remove along with simulation.py changes")
    else:
        record("server_no_analytical_mode", "PASS", "ANALYTICAL_MODE not referenced in server.py")

    # Scene rebuild / FEM snapshot logic in server
    if "snapshot_fem_entities" in source:
        record("server_no_fem_snapshot", "WARN",
               "snapshot_fem_entities call still in server.py — remove with analytical mode cleanup")
    else:
        record("server_no_fem_snapshot", "PASS", "No snapshot_fem_entities call in server.py")

    if "restore_fem_entities" in source:
        record("server_no_fem_restore", "WARN",
               "restore_fem_entities call still in server.py — remove with analytical mode cleanup")
    else:
        record("server_no_fem_restore", "PASS", "No restore_fem_entities call in server.py")

    # WebSocket route must exist
    if "websocket" in source.lower() or "WebSocket" in source:
        record("server_websocket_route", "PASS", "WebSocket route present")
    else:
        record("server_websocket_route", "FAIL",
               "No WebSocket route found in server.py — frontend will have no live data feed")

    # simulation import
    if "import simulation" in source or "from simulation import" in source:
        record("server_imports_simulation", "PASS", "simulation module imported")
    else:
        record("server_imports_simulation", "FAIL",
               "simulation module not imported — server cannot run any physics")

    # CoACD references should be gone from server too
    if "coacd" in source.lower():
        record("server_no_coacd", "WARN",
               "coacd still referenced in server.py — should be removed with Fix 4")
    else:
        record("server_no_coacd", "PASS", "No coacd references in server.py")

    # enforce_container_bounds call sites
    if "enforce_container_bounds" in source:
        record("server_no_enforce_bounds", "FAIL",
               "enforce_container_bounds() called in server.py — remove this call site (Fix 3)")
    else:
        record("server_no_enforce_bounds", "PASS", "No enforce_container_bounds call in server.py")


# ═════════════════════════════════════════════════════════════════════════════
# CHECK GROUP 3 — dependency checks
# ═════════════════════════════════════════════════════════════════════════════

def check_dependencies():
    print(_bold("\n── Group 3: dependency checks ────────────────────────────────"))

    # coacd should NOT be importable (or if it is, it's just not used — warn not fail)
    try:
        import coacd
        record("dep_coacd_not_needed", "WARN",
               "coacd is still installed — safe to uninstall: pip uninstall coacd")
    except ImportError:
        record("dep_coacd_not_needed", "PASS", "coacd not installed (correct for MPM path)")

    # genesis must be importable
    try:
        import genesis as gs
        record("dep_genesis_importable", "PASS", f"genesis imported successfully")
    except ImportError as exc:
        record("dep_genesis_importable", "FAIL", f"genesis not importable: {exc}")
        return  # no point checking sub-features

    # MPM materials must exist
    try:
        _ = gs.materials.MPM
        record("dep_genesis_mpm_available", "PASS", "gs.materials.MPM available")
    except AttributeError:
        record("dep_genesis_mpm_available", "FAIL",
               "gs.materials.MPM not found — Genesis version may be too old; update genesis")

    try:
        _ = gs.materials.MPM.ElastoPlastic
        record("dep_genesis_mpm_elastoplastic", "PASS", "gs.materials.MPM.ElastoPlastic available")
    except AttributeError:
        record("dep_genesis_mpm_elastoplastic", "FAIL",
               "gs.materials.MPM.ElastoPlastic not found — check Genesis version compatibility")

    # MPMOptions must exist
    try:
        opts = getattr(gs.options, "MPMOptions", None) or getattr(gs, "MPMOptions", None)
        if opts is not None:
            record("dep_genesis_mpm_options", "PASS", "gs.options.MPMOptions available")
        else:
            record("dep_genesis_mpm_options", "FAIL",
                   "MPMOptions not found in gs.options — scene MPM grid cannot be configured")
    except AttributeError:
        record("dep_genesis_mpm_options", "FAIL",
               "gs.options not available — Genesis import may be incomplete")

    # Core non-negotiable deps
    for mod_name in ["trimesh", "numpy", "torch", "networkx", "scipy", "h5py", "pandas", "fastapi"]:
        try:
            __import__(mod_name)
            record(f"dep_{mod_name}", "PASS", f"{mod_name} importable")
        except ImportError:
            record(f"dep_{mod_name}", "FAIL", f"{mod_name} not installed — required for simulation")


# ═════════════════════════════════════════════════════════════════════════════
# CHECK GROUP 4 — live Genesis smoke test (optional, --run-genesis flag)
# ═════════════════════════════════════════════════════════════════════════════

def run_genesis_smoke_test():
    print(_bold("\n── Group 4: live Genesis MPM smoke test ──────────────────────"))
    print("  (This initializes Genesis and runs 5 MPM steps — takes ~10-30s on CPU)")

    try:
        import genesis as gs
        import numpy as np
    except ImportError as exc:
        record("smoke_import", "FAIL", f"Cannot import genesis or numpy: {exc}")
        return

    try:
        gs.init(backend=gs.cpu)
        record("smoke_genesis_init", "PASS", "gs.init(backend=cpu) succeeded")
    except Exception as exc:
        if "already initialized" in str(exc).lower():
            record("smoke_genesis_init", "PASS", "Genesis already initialized (ok)")
        else:
            record("smoke_genesis_init", "FAIL", f"gs.init() failed: {exc}")
            return

    try:
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=4e-3, substeps=10, gravity=(0, -9.81, 0)),
            mpm_options=gs.options.MPMOptions(
                lower_bound=(-0.2, -0.05, -0.2),
                upper_bound=(0.2, 0.6, 0.2),
            ),
        )
        record("smoke_scene_created", "PASS", "Scene with MPMOptions created")
    except Exception as exc:
        record("smoke_scene_created", "FAIL", f"Scene creation failed: {exc}")
        return

    try:
        material = gs.materials.MPM.ElastoPlastic(
            E=10_000,
            nu=0.49,
            rho=1200,
        )
        record("smoke_mpm_material", "PASS", "MPM.ElastoPlastic material created")
    except Exception as exc:
        record("smoke_mpm_material", "FAIL", f"MPM.ElastoPlastic creation failed: {exc}")
        return

    try:
        _ = scene.add_entity(
            gs.morphs.Sphere(radius=0.02, pos=(0, 0.1, 0)),
            material=material,
        )
        record("smoke_entity_added", "PASS", "MPM sphere entity added to scene")
    except Exception as exc:
        record("smoke_entity_added", "FAIL", f"add_entity(MPM) failed: {exc}")
        return

    try:
        _ = scene.add_entity(
            gs.morphs.Box(size=(0.3, 0.02, 0.3), pos=(0, 0, 0), fixed=True),
            material=gs.materials.Rigid(friction=0.5),
        )
        record("smoke_rigid_floor", "PASS", "Rigid floor entity added")
    except Exception as exc:
        record("smoke_rigid_floor", "WARN", f"Rigid floor add failed (non-fatal): {exc}")

    try:
        scene.build()
        record("smoke_scene_build", "PASS", "scene.build() succeeded")
    except Exception as exc:
        record("smoke_scene_build", "FAIL", f"scene.build() failed: {exc}")
        return

    try:
        for i in range(5):
            scene.step()
        record("smoke_scene_step", "PASS", "5 MPM simulation steps completed")
    except Exception as exc:
        record("smoke_scene_step", "FAIL", f"scene.step() failed on step {i}: {exc}")
        return

    try:
        # Verify we can read MPM state
        entities = [e for e in scene.entities if type(e).__name__ != "RigidEntity"]
        if entities:
            st = entities[0].get_state()
            if hasattr(st, "pos"):
                pos = st.pos
                record("smoke_state_read", "PASS", f"MPM particle state readable, pos shape={getattr(pos, 'shape', 'unknown')}")
            else:
                record("smoke_state_read", "WARN", "MPM state has no .pos attribute — stress/position reads may differ by Genesis version")
        else:
            record("smoke_state_read", "WARN", "No non-rigid entities found after build — check entity type names")
    except Exception as exc:
        record("smoke_state_read", "WARN", f"State read check failed (non-fatal): {exc}")

    try:
        # Verify stress field readable (key for Fix 5)
        if entities and hasattr(entities[0].get_state(), "stress"):
            record("smoke_stress_readable", "PASS", "MPM .stress field readable from entity state")
        else:
            record("smoke_stress_readable", "WARN",
                   ".stress not found on entity state — Fix 5 stress visualization needs alternate field name; check gs version")
    except Exception as exc:
        record("smoke_stress_readable", "WARN", f"Stress field check failed: {exc}")


# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

def print_summary():
    print(_bold("\n═══════════════════════════════════════════════════════════════"))
    print(_bold("SUMMARY"))
    print(_bold("═══════════════════════════════════════════════════════════════"))

    fails  = [(n, m) for n, s, m in results if s == "FAIL"]
    warns  = [(n, m) for n, s, m in results if s == "WARN"]
    passes = [(n, m) for n, s, m in results if s == "PASS"]

    print(f"  {_green(f'{len(passes)} passed')}   {_yellow(f'{len(warns)} warnings')}   {_red(f'{len(fails)} failed')}\n")

    if fails:
        print(_red("  FAILED checks (must fix before simulation will run correctly):"))
        for name, msg in fails:
            print(f"    • {name}: {msg}")

    if warns:
        print(_yellow("\n  WARNINGS (simulation may run but results may be incorrect or sub-optimal):"))
        for name, msg in warns:
            print(f"    • {name}: {msg}")

    if not fails:
        print(_green("\n  All required checks passed. Safe to run the simulation."))
    else:
        print(_red(f"\n  {len(fails)} required check(s) failed. Fix the items above before running."))

    print()
    return len(fails)


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Verify the 5 MPM migration fixes applied to simulation.py / server.py"
    )
    parser.add_argument("--simulation-path", default="simulation.py",
                        help="Path to simulation.py (default: ./simulation.py)")
    parser.add_argument("--server-path", default="server.py",
                        help="Path to server.py (default: ./server.py)")
    parser.add_argument("--run-genesis", action="store_true",
                        help="Also run a live Genesis MPM smoke test (slower, requires Genesis installed)")
    parser.add_argument("--skip-server", action="store_true",
                        help="Skip server.py checks")
    parser.add_argument("--skip-deps", action="store_true",
                        help="Skip dependency import checks")
    args = parser.parse_args()

    sim_path    = Path(args.simulation_path)
    server_path = Path(args.server_path)

    print(_bold("═══════════════════════════════════════════════════════════════"))
    print(_bold("  verify_simulation_fixes.py — MPM migration verification"))
    print(_bold("═══════════════════════════════════════════════════════════════"))
    print(f"  simulation.py : {sim_path.resolve()}")
    print(f"  server.py     : {server_path.resolve()}")
    print(f"  live Genesis  : {'yes (--run-genesis)' if args.run_genesis else 'no  (pass --run-genesis to enable)'}")

    check_simulation_file(sim_path)

    if not args.skip_server:
        check_server_file(server_path)

    if not args.skip_deps:
        check_dependencies()

    if args.run_genesis:
        run_genesis_smoke_test()

    n_fails = print_summary()
    sys.exit(0 if n_fails == 0 else 1)


if __name__ == "__main__":
    main()