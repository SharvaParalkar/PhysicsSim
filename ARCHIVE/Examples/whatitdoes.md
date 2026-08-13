## Examples folder: what each demo does

This repo’s `Examples/` folder contains a set of self-contained HTML demos (mostly based on Matthias Müller’s *Ten Minute Physics*) that visualize different physics simulation techniques. Each file runs directly in the browser and includes its own rendering + interaction code.

---

## `Examples/10-softBodies.html` — Soft body (tetrahedral XPBD) demo

This demo simulates a deformable “soft body” built from a **tetrahedral mesh** (a volumetric mesh made of tets) and renders it as a shaded surface extracted from the tet surface triangles. The solver is constraint-based (XPBD-style): it enforces **edge-length constraints** (to resist stretching) and **volume constraints** (to resist collapsing), then integrates under gravity across multiple substeps per frame.

Interaction-wise, you can **Run/Stop**, **Restart**, and **Squash** the soft bodies (it forcibly flattens them to a plane in Y, so you can see the solver recover). **Bodies++** spawns additional copies at random positions so you can stress test stability/performance. The **Compliance** slider adjusts how “soft” the body is (higher compliance = easier to deform). You can also **grab and drag** the soft body with the pointer: a particle is pinned temporarily (mass set to 0) and moved with the cursor to demonstrate direct manipulation.

---

## `Examples/11-hashing.html` — Spatial hashing + particle collisions

This demo visualizes **broad-phase neighbor search** using a 3D **spatial hash grid**. A set of same-radius spheres move within world bounds and collide. Each frame, particles are inserted into a hash table keyed by integer cell coordinates (cell size tied to particle diameter), and queries gather candidates from nearby cells. This avoids \(O(n^2)\) all-pairs checks and makes collision detection scale much better with particle count.

The UI includes **Run/Restart**, a **Show collisions** toggle (to highlight colliding pairs/contacts), and readouts for **particle count** and **ms per frame** so you can see the cost of the simulation. Conceptually, it’s a compact “why spatial hashing matters” demo: same collision response, but with a faster neighbor lookup than brute force.

---

## `Examples/12-softBodySkinning.html` — Soft body driving a high-res render mesh (tet skinning)

This demo combines a volumetric soft-body simulation (tet mesh) with a separate **visual surface mesh** that is “skinned” to the simulation. The core idea: for each visual vertex, it finds a containing (or nearest appropriate) tet and stores **barycentric coordinates** (“skinning info”). During simulation, the visual vertex position is recomputed each frame as a barycentric blend of the tet’s four simulated particle positions. This gives you a deforming, smooth-looking mesh without having to simulate a dense volume.

Controls include **Run/Stop**, **Restart**, **Squash**, a **Show tets** toggle (draws the tet edges as a line mesh), and a **Compliance** slider that changes deformability. Like the soft body demo, it supports **pointer grabbing** to pin/move a particle. The page also reports counts for **tets, tris, and verts**, emphasizing the typical workflow: low-ish tet count for physics, higher triangle/vertex count for visuals.

---

## `Examples/17-fluidSim.html` — 2D Eulerian grid fluid (incompressible “smoke”)

This is a 2D **Eulerian** fluid solver on a grid, rendered on a `<canvas>`. It simulates velocity (u/v) and a scalar “smoke/dye” field, applying advection and then solving for **incompressibility** by iteratively adjusting pressure (a projection step). The “Overrelax” toggle switches the pressure solve between standard relaxation and **over-relaxation** (faster convergence / fewer iterations for similar visual quality), making it a practical demo of iterative Poisson solving behavior.

The demo provides four presets: **Tank** (gravity + closed boundaries, often visualized as pressure), **Wind Tunnel** and **Hires Tunnel** (inflow with an obstacle producing vortex shedding; the hires option increases resolution/iterations), and **Paint** (interactive dye/velocity injection). Visualization toggles include **Streamlines**, **Velocities**, **Pressure**, and **Smoke**; mouse/touch dragging is used to place/move an obstacle or “paint” into the field depending on the selected scene.

---

## `Examples/22-rigidBodies.html` — Rigid bodies with distance constraints (chain/mobile)

This demo is a simple rigid-body system driven primarily by **distance constraints** rather than full contact manifolds. Bodies (boxes and spheres) are integrated under gravity, then constraint solvers correct positions to satisfy distances, and velocities are updated from the corrected motion. This makes it a clear, minimal illustration of constraint-based rigid-body dynamics (similar spirit to position-based methods), including how substepping affects stability.

There are two scenes selectable from the UI: **Crib mobile** (a hanging, branching structure with bars and spheres linked by unilateral distance constraints, behaving like a mobile) and **Chain** (a chain of boxes linked by distance constraints; it also displays per-body mass text). You can adjust the **time step size** and restart to see stability/behavior changes. Pointer raycasting is used so you can **grab and drag** bodies via a temporary “drag constraint.”

---

## `Examples/25-joints.html` — Joint types (hinge/servo/motor/etc.) + interactive control

This demo is focused on **joint constraints** between rigid bodies (e.g., ball/hinge-like angular limits, prismatic/cylindrical motion limits, distance joints, motors/servos). It includes infrastructure to load a scene description (intended to be JSON files such as `basicJoints.json`, `steering.json`, and `pendulum.json`) and then build rigid bodies + joints from that data. Each joint solves positional/angular corrections per substep, optionally with compliance (softness) and damping.

The UI offers **Start/Restart**, a **Toggle View** (switches between a “simulation view” and more visual/illustrative joint gizmos), and a scene selector (**Basic Joints**, **Steering**, **Pendulums**). There’s also a touch-friendly on-screen control pad; its 2D control vector is applied to relevant joints (e.g., motor velocity, servo target angle, or target distance), so you can interactively steer/drive mechanisms and immediately see how different joint types constrain motion.

