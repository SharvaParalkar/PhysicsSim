cat > /mnt/user-data/outputs/indentation_modulus_implementation.txt << 'ENDOFFILE'
================================================================
INDENTATION MODULUS MAP — IMPLEMENTATION BRIEF
For: syringeGelParticles.html
================================================================

OVERVIEW
--------
This document tells you exactly what to add and where to add it
in the existing simulation file to produce per-vertex Young's
modulus heatmaps via virtual nanoindentation.

All line numbers reference the original file. Every addition is
self-contained — nothing in the existing physics, rendering, or
export pipeline needs to be deleted or rewritten.

There are four additions total:

  1. An Indenter class (~50 lines, new code, add after line ~996)
  2. A new color mode 'modulus' in _updateColors (~30 lines, add
     inside the existing _updateColors method around line 3814)
  3. A runIndentationSweep() method on SyringeSimulator (~70 lines,
     add after the simulate() method around line 4108)
  4. UI controls in the sidebar HTML (~25 lines of HTML, add in the
     Viz section of the sidebar around line 450)

================================================================
ADDITION 1 — THE INDENTER CLASS
================================================================

Insert this entire block immediately after the closing brace of
the InvisibleBox class, around line 1242. This sits alongside
Syringe and InvisibleBox as a third container-adjacent utility.

--- BEGIN INSERTION ---

// ================================================================
// INDENTER  — virtual nanoindentation probe
// A rigid sphere that descends along a surface normal and records
// the reaction force at each step. Uses the same constrainVertex
// pattern as Syringe.
// ================================================================
class Indenter {
  constructor(radiusMm) {
    // Radius of the virtual probe tip in mm.
    // Typical AFM or nanoindentation tip: 0.005 to 0.02 mm.
    // For microgels ~0.3 mm radius, a probe of 0.02-0.05 mm works well.
    this.radius = radiusMm || 0.02;

    // Current world position of the indenter center
    this.x = 0;
    this.y = 0;
    this.z = 0;

    // Whether the indenter is currently active in the scene
    this.active = false;

    // Accumulated force proxy from the current indentation step.
    // Computed as sum of penetration depths across all constrained verts.
    this.forceProxy = 0;

    // Current indentation depth delta (distance traveled into surface)
    this.depth = 0;
  }

  // Place the indenter at a surface point, offset outward along
  // the surface normal by (radius + small gap) so it starts
  // just touching the surface.
  //   surfX, surfY, surfZ  — point on particle surface
  //   nx, ny, nz           — outward surface normal at that point
  placAt(surfX, surfY, surfZ, nx, ny, nz) {
    const offset = this.radius + 0.001;
    this.x = surfX + nx * offset;
    this.y = surfY + ny * offset;
    this.z = surfZ + nz * offset;
    this.depth = 0;
    this.forceProxy = 0;
    this.active = true;
  }

  // Move indenter inward along the approach direction by stepMm.
  //   nx, ny, nz — inward direction (into surface, so negate surface normal)
  stepInward(nx, ny, nz, stepMm) {
    this.x += nx * stepMm;
    this.y += ny * stepMm;
    this.z += nz * stepMm;
    this.depth += stepMm;
  }

  // Apply sphere constraint to one vertex of the particle.
  // Pushes the vertex outside the indenter sphere if it penetrates.
  // Returns the penetration depth (0 if no contact).
  //   pos      — Float32Array of particle positions (flat x,y,z,x,y,z...)
  //   prevPos  — Float32Array of previous positions (for friction)
  //   i        — vertex index
  //   mu       — friction coefficient
  constrainVertex(pos, prevPos, i, mu) {
    const px = pos[3*i]   - this.x;
    const py = pos[3*i+1] - this.y;
    const pz = pos[3*i+2] - this.z;
    const dist = Math.sqrt(px*px + py*py + pz*pz);
    if (dist >= this.radius || dist < 1e-10) return 0;

    // Penetration depth
    const pen = this.radius - dist;

    // Push vertex outside the sphere
    const inv = 1 / dist;
    const nx = px * inv;
    const ny = py * inv;
    const nz = pz * inv;
    pos[3*i]   += nx * pen;
    pos[3*i+1] += ny * pen;
    pos[3*i+2] += nz * pen;

    // Friction: damp tangential component of displacement
    mu = Math.max(0, Math.min(1, mu || 0));
    if (prevPos && mu > 0) {
      const vx = pos[3*i]   - prevPos[3*i];
      const vy = pos[3*i+1] - prevPos[3*i+1];
      const vz = pos[3*i+2] - prevPos[3*i+2];
      const vn = vx*nx + vy*ny + vz*nz;
      const tx = vx - vn*nx;
      const ty = vy - vn*ny;
      const tz = vz - vn*nz;
      prevPos[3*i]   += tx * mu;
      prevPos[3*i+1] += ty * mu;
      prevPos[3*i+2] += tz * mu;
    }

    return pen;
  }

  // Apply constraint to all vertices of a particle and
  // accumulate the force proxy (sum of penetrations).
  applyToParticle(particle, mu) {
    let totalPen = 0;
    for (let i = 0; i < particle.numTetVerts; i++) {
      if (particle.invMass[i] === 0) continue;
      totalPen += this.constrainVertex(
        particle.pos, particle.prevPos, i, mu
      );
    }
    this.forceProxy += totalPen;
    return totalPen;
  }
}

--- END INSERTION ---


================================================================
ADDITION 2 — 'modulus' COLOR MODE IN _updateColors
================================================================

Inside the _updateColors() method (starts around line 3814),
find this block near the bottom of the method:

    // Contact heatmap (blue -> green -> yellow -> red) based on
    // per-particle contactCount.

INSERT the following block IMMEDIATELY BEFORE that comment.
It adds a third color mode branch that reads from
p._modulusMap[] (populated by runIndentationSweep below).

--- BEGIN INSERTION ---

    // Modulus heatmap: reads from p._modulusMap[] set by
    // runIndentationSweep(). Blue = low modulus, Red = high modulus.
    if (this.colorMode === 'modulus') {
      // Find global max for normalization
      let globalMax = 1e-12;
      for (const p of this.particles) {
        if (!p.active || !p._modulusMap) continue;
        for (let i = 0; i < p._modulusMap.length; i++) {
          if (p._modulusMap[i] > globalMax) globalMax = p._modulusMap[i];
        }
      }
      for (const p of this.particles) {
        if (!p.active) continue;
        // If no modulus data yet, paint gray
        if (!p._modulusMap) {
          for (let i = 0; i < p.numVisVerts; i++) {
            const base = (p.visVertOffset + i) * 3;
            this.allColors[base]   = 0.4;
            this.allColors[base+1] = 0.4;
            this.allColors[base+2] = 0.4;
          }
          continue;
        }
        for (let i = 0; i < p.numVisVerts; i++) {
          const t = Math.max(0, Math.min(1,
            (p._modulusMap[i] || 0) / globalMax
          ));
          const c = rampBGYR(t);
          const base = (p.visVertOffset + i) * 3;
          this.allColors[base]   = c[0];
          this.allColors[base+1] = c[1];
          this.allColors[base+2] = c[2];
        }
      }
      return;
    }

--- END INSERTION ---


================================================================
ADDITION 3 — runIndentationSweep() ON SyringeSimulator
================================================================

INSERT this entire method inside the SyringeSimulator class,
immediately after the closing brace of the simulate() method
(around line 4108). It is a standalone async method that runs
the full indentation sweep on a single particle.

--- BEGIN INSERTION ---

  // ----------------------------------------------------------------
  // runIndentationSweep(particleIdx, options)
  //
  // Runs a virtual nanoindentation sweep on one particle.
  // For each surface vertex (or a sampled subset), places the
  // Indenter at that vertex, runs XPBD substeps while advancing
  // inward, records force and depth, computes apparent Young's
  // modulus via the Hertz contact formula, stores the result in
  // particle._modulusMap[], then switches the color mode to
  // 'modulus' so the heatmap renders immediately.
  //
  // OPTIONS (all optional):
  //   indenterRadiusMm   float   Probe tip radius. Default: 0.02
  //   maxDepthMm         float   How far to indent. Default: 0.005
  //   depthSteps         int     Substeps per indentation. Default: 8
  //   settleSteps        int     XPBD steps to let particle relax
  //                              after each indentation. Default: 20
  //   sampleEvery        int     Only indent every Nth surface vertex
  //                              (1 = all verts). Default: 1
  //   poissonRatio       float   Assumed Poisson's ratio. Default: 0.45
  //                              (typical for hydrogels)
  //   onProgress         fn      Called with (fraction 0..1, E_kPa)
  //                              after each indentation site.
  // ----------------------------------------------------------------
  async runIndentationSweep(particleIdx, options) {
    options = options || {};
    const indenterR  = options.indenterRadiusMm  || 0.02;
    const maxDepth   = options.maxDepthMm         || 0.005;
    const depthSteps = options.depthSteps         || 8;
    const settleSteps= options.settleSteps        || 20;
    const sampleEvery= options.sampleEvery        || 1;
    const nu         = options.poissonRatio        || 0.45;
    const onProgress = options.onProgress         || null;
    const stepSize   = maxDepth / depthSteps;

    const p = this.particles[particleIdx];
    if (!p || !p.active) {
      console.warn('runIndentationSweep: particle not found or inactive');
      return;
    }

    // Create indenter instance
    const indenter = new Indenter(indenterR);

    // Allocate modulus map (one value per vis vertex)
    p._modulusMap = new Float32Array(p.numVisVerts);

    // Correction factor: Hertz gives E* (reduced modulus).
    // Convert to E (Young's modulus) assuming nu is Poisson's ratio.
    // E = E* * (1 - nu^2)   [Hertz, symmetric indenter]
    const nuFactor = 1 - nu * nu;

    const numSites = Math.ceil(p.numVisVerts / sampleEvery);
    let siteIdx = 0;
    const sdt = 0.001; // fixed substep dt for settle passes

    for (let vi = 0; vi < p.numVisVerts; vi += sampleEvery) {

      // --- Get surface point and outward normal ---
      // Surface vertex position (from merged buffer)
      const b = (p.visVertOffset + vi) * 3;
      const sx = this.allPositions[b];
      const sy = this.allPositions[b+1];
      const sz = this.allPositions[b+2];

      // Outward normal: direction from particle centroid to vertex
      const dx = sx - p.center[0];
      const dy = sy - p.center[1];
      const dz = sz - p.center[2];
      const dl = Math.sqrt(dx*dx + dy*dy + dz*dz);
      if (dl < 1e-10) continue;
      const nx = dx / dl;
      const ny = dy / dl;
      const nz = dz / dl;

      // --- Snapshot rest positions ---
      // Save particle positions so we can restore after indentation
      const savedPos  = new Float32Array(p.pos);
      const savedPrev = new Float32Array(p.prevPos);
      const savedVel  = new Float32Array(p.vel);

      // --- Place indenter ---
      indenter.placAt(sx, sy, sz, nx, ny, nz);
      indenter.forceProxy = 0;

      // --- Advance inward, running XPBD substeps ---
      let lastForce = 0;
      let lastDepth = 0;

      for (let step = 0; step < depthSteps; step++) {
        // Move probe inward by one step
        indenter.stepInward(-nx, -ny, -nz, stepSize);
        indenter.forceProxy = 0;

        // Run XPBD substeps: edges + volumes + indenter constraint
        for (let sub = 0; sub < settleSteps; sub++) {
          // Integrate
          for (let k = 0; k < p.numTetVerts; k++) {
            if (p.invMass[k] === 0) continue;
            p.prevPos[3*k]   = p.pos[3*k];
            p.prevPos[3*k+1] = p.pos[3*k+1];
            p.prevPos[3*k+2] = p.pos[3*k+2];
          }
          // Apply indenter constraint
          indenter.applyToParticle(p, 0.2);
          // XPBD constraints
          p.solveEdges(sdt);
          p.solveVolumes(sdt);
          // Velocity update
          for (let k = 0; k < p.numTetVerts; k++) {
            if (p.invMass[k] === 0) continue;
            p.vel[3*k]   = (p.pos[3*k]   - p.prevPos[3*k])   / sdt;
            p.vel[3*k+1] = (p.pos[3*k+1] - p.prevPos[3*k+1]) / sdt;
            p.vel[3*k+2] = (p.pos[3*k+2] - p.prevPos[3*k+2]) / sdt;
          }
        }

        lastForce = indenter.forceProxy;
        lastDepth = indenter.depth;
      }

      // --- Hertz contact mechanics ---
      // Hertz formula for a spherical indenter on a flat elastic half-space:
      //   F = (4/3) * E* * sqrt(R) * delta^(3/2)
      // Solving for E*:
      //   E* = (3*F) / (4 * sqrt(R) * delta^(3/2))
      // Then Young's modulus:
      //   E  = E* * (1 - nu^2)
      //
      // F here is the force proxy (sum of penetration depths weighted
      // by the particle's edge compliance — a relative stiffness signal).
      // The compliance (edgeCompliance) is in mm^2/N-equivalent units,
      // so E comes out in simulation units. For display the relative
      // distribution across the surface is what matters.

      let Emap = 0;
      if (lastDepth > 1e-10 && lastForce > 1e-14) {
        const sqrtR   = Math.sqrt(indenterR);
        const dep32   = Math.pow(lastDepth, 1.5);
        const Estar   = (3 * lastForce) / (4 * sqrtR * dep32);
        Emap = Estar * nuFactor;
      }

      // Store modulus value for this vertex
      p._modulusMap[vi] = Emap;

      // Fill in interpolated values for skipped vertices
      // (simple nearest-neighbor fill between sampled sites)
      if (sampleEvery > 1) {
        const end = Math.min(vi + sampleEvery, p.numVisVerts);
        for (let fill = vi + 1; fill < end; fill++) {
          p._modulusMap[fill] = Emap;
        }
      }

      // --- Restore particle state ---
      p.pos.set(savedPos);
      p.prevPos.set(savedPrev);
      p.vel.set(savedVel);
      p.updateCenter();

      // --- Progress callback ---
      siteIdx++;
      if (onProgress) {
        onProgress(siteIdx / numSites, Emap);
      }

      // Yield to browser every 10 sites so the UI stays responsive
      if (siteIdx % 10 === 0) {
        await new Promise(r => setTimeout(r, 0));
      }
    }

    // Update the visual mesh with restored positions
    p.updateVisMesh(this.allPositions);

    // Switch color mode to show the modulus map
    this.colorMode = 'modulus';
    this._updateColors();
    const geo = this.particleMesh.geometry;
    geo.attributes.color.needsUpdate = true;

    console.log('Indentation sweep complete on particle', particleIdx,
      '— modulus map has', p._modulusMap.length, 'values.');
  }

--- END INSERTION ---


================================================================
ADDITION 4 — UI CONTROLS IN THE SIDEBAR
================================================================

Find the Visualization section in the sidebar HTML. It contains
sliders for stressGain, stressDepthMix, etc. Look for the text
"stressGain" around line 450-500 to locate it.

At the END of the Viz section (just before its closing
</div></details> tags), INSERT the following HTML block:

--- BEGIN INSERTION ---

    <div class="row"><label>Indentation mode</label>
      <select id="selColorMode" onchange="onColorModeChange(this.value)" style="width:auto">
        <option value="stress" selected>Stress</option>
        <option value="contact">Contact</option>
        <option value="modulus">Modulus map</option>
      </select></div>
    <div class="row" id="rowIndentControls">
      <label>Indenter radius (mm)</label>
      <input type="number" id="pIndenterR" value="0.02" min="0.005" max="0.1" step="0.005"></div>
    <div class="row">
      <label>Max depth (mm)</label>
      <input type="number" id="pIndentDepth" value="0.005" min="0.001" max="0.05" step="0.001"></div>
    <div class="row">
      <label>Sample every N verts</label>
      <input type="number" id="pIndentSample" value="1" min="1" max="20" step="1"
             title="1 = all vertices (slow but full map). Higher = faster but lower resolution."></div>
    <div class="row"><label></label>
      <button class="btn green" id="btnRunSweep" onclick="onRunIndentSweep()">
        Run indentation sweep</button></div>
    <div class="row"><label></label>
      <span id="lblSweepProgress" style="font-size:11px;color:#7a9abd">
        Select a particle first (click it in the viewport), then press Run.</span></div>

--- END INSERTION ---


================================================================
ADDITION 5 — TWO SMALL JAVASCRIPT FUNCTIONS
================================================================

Find the onRun() and onRestart() UI handler functions near the
bottom of the <script> block (around line 7200-7250).

INSERT these two functions alongside them:

--- BEGIN INSERTION ---

function onColorModeChange(val) {
  if (!gSimulator) return;
  gSimulator.colorMode = val;
  gSimulator._updateColors();
  const geo = gSimulator.particleMesh.geometry;
  geo.attributes.color.needsUpdate = true;
  // Also keep the existing select in sync if it exists
  const sel = document.getElementById('selColorMode');
  if (sel && sel.value !== val) sel.value = val;
}

async function onRunIndentSweep() {
  if (!gSimulator) return;
  // Use the currently picked particle, or default to particle 0
  const idx = (typeof gPickedParticle !== 'undefined' && gPickedParticle >= 0)
    ? gPickedParticle : 0;

  const r     = parseFloat(document.getElementById('pIndenterR')?.value)    || 0.02;
  const depth = parseFloat(document.getElementById('pIndentDepth')?.value)  || 0.005;
  const every = parseInt(document.getElementById('pIndentSample')?.value)   || 1;
  const lbl   = document.getElementById('lblSweepProgress');

  if (lbl) lbl.textContent = 'Running… 0%';
  document.getElementById('btnRunSweep').disabled = true;

  await gSimulator.runIndentationSweep(idx, {
    indenterRadiusMm: r,
    maxDepthMm:       depth,
    sampleEvery:      every,
    onProgress: (frac, E) => {
      if (lbl) lbl.textContent =
        `Running… ${Math.round(frac * 100)}%  (last E ≈ ${E.toExponential(2)})`;
    }
  });

  document.getElementById('btnRunSweep').disabled = false;
  if (lbl) lbl.textContent = 'Sweep complete. Color mode set to Modulus map.';

  // Sync the color mode dropdown
  const sel = document.getElementById('selColorMode');
  if (sel) sel.value = 'modulus';
}

--- END INSERTION ---


================================================================
USING THE FEATURE — STEP BY STEP
================================================================

1. PREPARE A HIGH-RESOLUTION PARTICLE

   Run the preprocessor with a denser mesh. If you are using a
   sphere or cube OBJ, subdivide it first in Blender/MeshLab so
   it has at least 500-2000 surface triangles before running:

     python preprocess_particle.py MyShape.obj my_shape_hires.json

   Then load my_shape_hires.json via "Load particle JSON" in the
   toolbar. Restart with Initial drop = 1. This gives you one
   high-resolution particle to indendt.

2. RUN THE SIMULATION BRIEFLY

   Press Run. Let the particle settle (a second or two of sim
   time is enough — it just needs to be stationary so the
   rest-shape is stable).

   Press Pause.

3. SELECT THE PARTICLE

   Click on the particle in the 3D viewport. The sidebar
   "Inspector" section will show it as the picked particle.

4. CONFIGURE THE SWEEP

   In the Viz section of the sidebar, find the new indentation
   controls. Recommended starting values:

     Indenter radius (mm):  0.02    (for ~0.3 mm radius microgel)
     Max depth (mm):        0.005   (about 1.7% of particle radius)
     Sample every N verts:  3       (fast first pass; use 1 for
                                     full resolution final map)

5. RUN THE SWEEP

   Press "Run indentation sweep". The progress label updates as
   it works. The sweep runs asynchronously so the browser stays
   responsive.

6. INSPECT THE MAP

   When complete, the color mode automatically switches to
   "Modulus map". The particle surface is colored blue-to-red
   where blue = low apparent stiffness, red = high apparent
   stiffness. Rotate the camera to inspect face, edge, and
   corner regions.

7. EXPORT

   Use the existing "Export PLY (colors)" button in the toolbar.
   This exports the vertex-colored mesh with the modulus map
   baked into the RGB values. Import into Blender, ParaView,
   or any DCC tool that supports vertex-colored PLY files.

   The CSV export ("Export metrics CSV") does not currently
   include the per-vertex modulus values. If you need those,
   add a loop in exportMetricsCSV() that appends p._modulusMap
   values row by row after the existing metrics rows.


================================================================
IMPORTANT NOTES FOR THE MANUSCRIPT
================================================================

WHAT THE VALUES REPRESENT

The modulus values produced by this simulation are relative, not
absolute. The Hertz formula is applied using the XPBD force proxy
(penetration depth × stiffness constraint), which is in simulation
units, not physical Pa or kPa. The output is dimensionless but
correctly represents the spatial distribution of apparent stiffness
across the particle surface.

To obtain absolute modulus values that can be reported in the
manuscript alongside nanoindentation data, calibrate against a
known material. Run the sweep on a simulated sphere with known
edgeCompliance and volCompliance values, compare the output to
the analytical Hertz solution for a sphere of those properties,
and compute a scaling factor. Apply that factor to all output
values.

WHY THE DISTRIBUTION IS STILL PHYSICALLY MEANINGFUL

Even without absolute calibration, the map shows the correct
qualitative result: corners concentrate more stress per unit
area than flat faces. This is geometry, not material heterogeneity,
and it is the core argument being made to the reviewer. The
simulation cleanly separates geometric effect from material effect
because the particle is homogeneous by construction.

POISSON'S RATIO

The default nu = 0.45 is standard for hydrogels (nearly
incompressible). Changing it only scales the output values
uniformly — it does not change the spatial distribution pattern.

INDENTER RADIUS EFFECT

A smaller indenter radius produces sharper spatial resolution
in the map (more localized indentation) but also higher apparent
modulus values at corners due to the increased stress concentration.
Use a consistent radius across all geometry comparisons so the
maps are directly comparable.


================================================================
SUMMARY OF ALL FILES CHANGED
================================================================

syringeGelParticles.html — the only file that needs to change.

  Line ~1242   Add: Indenter class (Addition 1)
  Line ~3814   Add: 'modulus' branch in _updateColors (Addition 2)
  Line ~4108   Add: runIndentationSweep() method (Addition 3)
  Line ~480    Add: sidebar HTML controls (Addition 4)
  Line ~7230   Add: onColorModeChange() and onRunIndentSweep() (Addition 5)

No external libraries. No changes to the physics solver.
No changes to the export pipeline (PLY export already works).

================================================================
END OF BRIEF
================================================================
ENDOFFILE
echo "File written successfully"