export interface ParticleData {
  id: number
  x: number
  y: number
  z: number
  qx: number
  qy: number
  qz: number
  qw: number
  n_contacts: number
}

export interface ContactPair {
  particle_a: number
  particle_b: number
  depth: number
  force: number | null
  contact_area: number | null
}

export interface ContactPoint {
  x: number
  y: number
  z: number
  nx: number
  ny: number
  nz: number
}

export interface SimMetrics {
  Z: number
  total_pp: number
  total_pc: number
  n_isolated: number
  n_container_touch: number
  /** Σ|F_contact| / container inner surface area (Pa), from live `compute_metrics`. */
  system_pressure?: number
}

export interface ZHistoryPoint {
  t: number
  Z: number
}

export interface ContactGraphLink {
  source: number
  target: number
  depth: number
  force: number | null
  area: number | null
}

export interface MetricsResponse extends SimMetrics {
  z_history: ZHistoryPoint[]
  max_vel_history?: Array<{ t: number; max_vel: number }>
  rattlers_history?: Array<{ t: number; n_rattlers: number }>
  kinetic_energy_history?: Array<{ t: number; kinetic_energy: number }>
  pressure_history?: Array<{ t: number; system_pressure: number }>
  contact_graph_dict: Record<string, Record<string, { depth?: number; force?: number | null; area?: number | null }>>
  contact_graph_links: ContactGraphLink[]
}

export interface WsFrameParticle {
  id: number
  x: number
  y: number
  z: number
  qx: number
  qy: number
  qz: number
  qw: number
  stress_intensity?: number
  /** Normalized [0,1] stress intensity per mesh vertex (FEM nodal-force magnitude); same order/length as render mesh vertices when available. */
  vertex_intensities?: number[]
  /** When set (same length as vertex_intensities), sparse FEM sample: values apply at these local mesh vertex indices. */
  vertex_stress_indices?: number[]
}

/** Latest physics frame from the WebSocket stream (used for 3D + temporal interpolation). */
export interface LivePhysicsFrame {
  /** Simulation step when present; do not rely on this alone for deduplication (spawn uses -1, payloads may omit). */
  step: number
  t: number
  particles: WsFrameParticle[]
  /** Increments on every `frame` message so the viewer always ingests new poses. */
  serial: number
}

export interface SimulationConfig {
  N_PARTICLES: number
  YOUNGS_MODULUS: number
  POISSON_RATIO: number
  DT: number
  SUBSTEPS: number
  /** When true: large dt while falling, no contact extraction until max_vel &lt; threshold, then rebuild at 500 Hz / 16 substeps. */
  ANALYTICAL_MODE?: boolean
  /** Rebuild scene per particle: simulate k bodies, snapshot FEM, add the next at drop height (FEM only; disables analytical handoff). */
  SEQUENTIAL_DROP?: boolean
  /** Max simulated seconds per staging step; omit for max(SIM_DURATION / N_PARTICLES, 0.25). */
  SEQUENTIAL_STAGE_DURATION?: number | null
  SIM_DURATION: number
  ENVIRONMENT_TYPE: 'plate' | 'cylinder'
  CYLINDER_DIAMETER: number
  DROP_HEIGHT: number
  /** 0 = vertical stack at origin; &gt;0 = Vogel-disk spread (fraction of container radius) */
  DROP_SPREAD: number
  /** Matches simulation `PLATE_SIZE` / `WALL_THICKNESS` / `PLATE_WALL_HEIGHT` for 3D preview */
  PLATE_SIZE: number
  WALL_THICKNESS: number
  /** Rim height (m) — plate mode only; keeps particles on the plate in the solver. */
  PLATE_WALL_HEIGHT: number
  /** Radians — Hertzian angular falloff for post-process vertex stress (backend STRESS_SIGMA). */
  STRESS_SIGMA?: number
}

export type WsLog = { type: 'log'; line: string }
export type WsFrame = { type: 'frame'; step: number; t: number; Z?: number; max_vel?: number; particles: WsFrameParticle[] }
export type WsComplete = {
  type: 'complete'
  metrics?: { Z?: number; total_pp?: number; system_pressure?: number }
  vertex_stress?: Record<string, number[]>
  mesh_vertex_count?: number
}
export type WsError = { type: 'error'; message: string }
export type WsCancelled = { type: 'cancelled' }
export type WsProgress = { type: 'progress'; phase: string; pct: number; detail?: string }
export type WsLiveMetrics = {
  type: 'live_metrics'
  step: number
  t: number
  Z: number
  n_rattlers: number
  kinetic_energy: number
  /** Pa — use with low KE / Z plateau to spot modular jamming. */
  system_pressure: number
}
export type WsMsg = WsLog | WsFrame | WsComplete | WsError | WsCancelled | WsProgress | WsLiveMetrics

