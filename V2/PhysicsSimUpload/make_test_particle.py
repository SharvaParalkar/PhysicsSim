"""Generate a smoothed box OBJ (particle.obj) for testing the simulation."""

import numpy as np
import trimesh
import trimesh.smoothing


def make_test_particle():
    """Create a Laplacian-smoothed box and export as particle.obj."""
    # Slightly non-cubic dimensions give more interesting packing behaviour
    mesh = trimesh.creation.box(extents=[0.020, 0.015, 0.025])

    # One pass of Laplacian smoothing rounds the sharp edges slightly
    mesh = trimesh.smoothing.filter_laplacian(mesh, lamb=0.5, iterations=1)

    output = "particle.obj"
    mesh.export(output)

    print(f"Saved '{output}'")
    print(f"  Vertices : {len(mesh.vertices)}")
    print(f"  Faces    : {len(mesh.faces)}")
    print(f"  Extents  : {np.round(mesh.bounding_box.extents, 4)} m")
    if mesh.is_watertight:
        print(f"  Volume   : {mesh.volume:.4e} m³")
    else:
        print("  Note: mesh not watertight (expected for basic box smoothing)")


if __name__ == "__main__":
    make_test_particle()
