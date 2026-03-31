import math
from pathlib import Path

def generate_precise_star(filename, points=5, outer=0.332, inner=0.129, thickness=0.6):
    verts = []
    faces = []
    
    # 1. Generate the 2D star shape (XY Plane)
    # We use 2 * points to account for tips and valleys
    star_2d = []
    for i in range(2 * points):
        angle = (i * math.pi) / points
        r = outer if i % 2 == 0 else inner
        star_2d.append((r * math.cos(angle), r * math.sin(angle)))

    # 2. Extrude to 3D (Z axis)
    z_half = thickness / 2.0
    
    # Bottom ring (Indices 0 to 9)
    for x, y in star_2d:
        verts.append((x, y, -z_half))
    
    # Top ring (Indices 10 to 19)
    for x, y in star_2d:
        verts.append((x, y, z_half))
        
    # Center points for caps (Indices 20 and 21)
    verts.append((0, 0, -z_half)) # Bottom center
    verts.append((0, 0, z_half))  # Top center

    # 3. Create Faces (1-based indexing for OBJ)
    n = 2 * points
    
    # Side Walls
    for i in range(n):
        b0 = i + 1
        b1 = ((i + 1) % n) + 1
        t0 = i + 1 + n
        t1 = ((i + 1) % n) + 1 + n
        faces.append((b0, b1, t1))
        faces.append((b0, t1, t0))

    # Bottom Cap (Clockwise for outward normal)
    bc = 2 * n + 1
    for i in range(n):
        faces.append((bc, ((i + 1) % n) + 1, i + 1))

    # Top Cap (Counter-clockwise for outward normal)
    tc = 2 * n + 2
    for i in range(n):
        faces.append((tc, i + 1 + n, ((i + 1) % n) + 1 + n))

    # 4. Write to File
    with open(filename, 'w') as f:
        f.write(f"# Precise Star Particle: Height {thickness}mm\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")

    print(f"File saved to: {filename}")
    print(f"Geometry: {len(verts)} vertices, {len(faces)} triangles.")

if __name__ == "__main__":
    generate_precise_star("my_star_particle.obj")