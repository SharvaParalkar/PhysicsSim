import json
import argparse
from pathlib import Path

def compress_particle_geometry(input_path, output_path, grid_size=0.05):
    with open(input_path, 'r') as f:
        data = json.load(f)

    def simplify_mesh(verts, indices, voxel_size):
        """Groups vertices into a grid (voxels) to reduce count."""
        unique_verts = []
        vert_map = {} # Maps old index to new index
        grid = {}     # Maps grid coordinate to new vertex index

        for i in range(len(verts) // 3):
            x, y, z = verts[3*i], verts[3*i+1], verts[3*i+2]
            # Create a grid key (voxel)
            key = (int(x / voxel_size), int(y / voxel_size), int(z / voxel_size))
            
            if key not in grid:
                grid[key] = len(unique_verts) // 3
                unique_verts.extend([round(x, 4), round(y, 4), round(z, 4)])
            
            vert_map[i] = grid[key]

        # Remap indices and remove degenerate faces/tets
        new_indices = []
        step = 4 if len(indices) % 4 == 0 and "tet" in str(indices) else 3
        for i in range(0, len(indices), step):
            group = [vert_map[idx] for idx in indices[i:i+step]]
            # Only add if the tetrahedron/triangle hasn't collapsed to a point/line
            if len(set(group)) == step:
                new_indices.extend(group)
        
        return unique_verts, new_indices

    # Simplify Physics Mesh (Tetrahedra)
    if "tet" in data:
        print(f"Original Tets: {len(data['tet']['tetIds']) // 4}")
        v, i = simplify_mesh(data['tet']['verts'], data['tet']['tetIds'], grid_size)
        data['tet']['verts'] = v
        data['tet']['tetIds'] = i
        print(f"Compressed Tets: {len(i) // 4}")

    # Simplify Visual Mesh (Triangles)
    if "vis" in data:
        print(f"Original Vis Verts: {len(data['vis']['verts']) // 3}")
        v, i = simplify_mesh(data['vis']['verts'], data['vis']['triIds'], grid_size)
        data['vis']['verts'] = v
        data['vis']['triIds'] = i
        # Note: Skinning is invalidated by heavy re-topology; 
        # for best results, regenerate skinning using your previous script.
        data['vis']['skinning'] = [] 
        print(f"Compressed Vis Verts: {len(v) // 3}")

    with open(output_path, 'w') as f:
        json.dump(data, f, separators=(',', ':'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Heavy JSON")
    parser.add_argument("output", help="Light JSON")
    parser.add_argument("--grid", type=float, default=0.08, help="Size of collapse grid (higher = lighter)")
    args = parser.parse_args()
    
    compress_particle_geometry(args.input, args.output, args.grid)