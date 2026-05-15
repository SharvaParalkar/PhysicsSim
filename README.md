# PhysicsSim

*[sharvaparalkar.com/simulation](https://sharvaparalkar.com/simulation)**

Browser-based simulation of deformable gel particles in syringes and custom containers. The current version (**V3**) runs entirely in the client using an **XPBD** (extended position-based dynamics) solver, with real-time 3D rendering and interactive controls.

Earlier iterations (**V1** and **V2**) used **[Genesis](https://github.com/Genesis-Embodied-AI/Genesis)** on the Python side—FEM/rigid-body physics with a separate viewer (Three.js + WebSocket in V2). That stack was powerful but heavier to run, harder to share, and tied simulation to a backend process. **V3** moves the solver into JavaScript so the same physics runs in any modern browser without installing Genesis, CUDA, or a Python server. The XPBD approach (tetrahedral soft bodies, spatial-hash contacts, constraint substeps) proved more practical for this use case: stable deformation, interactive tuning, and straightforward deployment as static HTML.

## What V3 simulates

- **Soft gel particles** — tetrahedral meshes with XPBD **edge** and **volume** constraints (patterns from Matthias Müller’s [*Ten Minute Physics*](https://www.youtube.com/channel/UCTG_vrRdKYfrpqCv_WV4eyA)).
- **Particle–particle contact** — spatial-hash broadphase and centroid-level XPBD corrections.
- **Containers** — analytic syringe (barrel, taper, needle) or an invisible box volume filled from a watertight OBJ.
- **Extrusion & cohesion** — bonds and contact parameters for filament-like behavior through the needle.
- **Analysis & export** — stress coloring, porosity metrics, OBJ/PLY/CSV export, pore void meshes.

Bundled particle shapes live in `V3/assets/` (see `manifest.json`). You can load custom particles from JSON produced by the mesh preprocessing scripts.

## Quick start (local)

Serve the `V3` folder over HTTP (required for loading JSON assets; `file://` may block fetches):

```bash
python -m http.server --directory V3 8000
```

Then open:

- Main demo: [http://localhost:8000/syringeGelParticles.html](http://localhost:8000/syringeGelParticles.html)
- Deformation sandbox: [http://localhost:8000/particleDeformTester.html](http://localhost:8000/particleDeformTester.html)

Use **Run** / **Restart** in the toolbar; adjust syringe dimensions, compliance, substeps, and contact settings in the sidebar.

### Custom particle from OBJ

1. Convert a watertight OBJ to tet JSON (from repo root):

   ```bash
   python V3/uploads/preprocess.py path/to/mesh.obj path/to/particle.json
   ```

2. In the demo, use **Load particle JSON**, then **Restart**.

Optional pipeline helpers: `V3/finalprocess.py`, `V3/uploads/create.py`, and tools under `V3/assets/tools/`.

Default parameters are in `V3/syringeGelParticles.config.json`.

## Repository layout

| Path | Description |
|------|-------------|
| **`V3/`** | **Current version** — browser XPBD sim (`syringeGelParticles.html`, `particleDeformTester.html`), assets, preprocessing scripts |
| `V2/` | Genesis backend (`simulation.py`, `server.py`) + React/Three.js frontend; see `V2/README.md` for setup |
| `V1/` | Earlier Genesis + frontend prototype |
| `Examples/` | Standalone *Ten Minute Physics* HTML demos (soft bodies, hashing, fluids, etc.); see `Examples/whatitdoes.md` |
| `WEBSITEUPLOAD/` | Static build + Cloudflare Worker for hosting (e.g. `/simulation` on a custom domain) |

## Web deployment

`WEBSITEUPLOAD/` contains the production static bundle and `worker.js`, which serves assets and maps paths like `/simulation/*` for custom domains. Configure via `wrangler.jsonc` and deploy with [Wrangler](https://developers.cloudflare.com/workers/wrangler/).

## Evolution at a glance

```
V1 / V2                          V3
────────────────────────────     ────────────────────────────
Genesis (Python)                 XPBD solver (JavaScript)
FEM / rigid coupling             Tet soft bodies + contacts
Backend + viewer                 Single-page, in-browser
GPU/CPU env, WebSocket           Static HTTP server only
```

V1 and V2 remain useful references for Genesis-based batch runs and validation; **V3** is the recommended entry point for interactive simulation and sharing.

## Credits

V3 builds on constraint-based soft-body and broadphase ideas from [*Ten Minute Physics*](https://www.matthiasMueller.info/tenMinutePhysics) (Matthias Müller). MIT-licensed patterns are noted in the HTML sources.
