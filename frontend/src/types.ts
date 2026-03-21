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
  /** Up to 8 normalized [0,1] per-vertex values (FEM nodal-force proxy); matches particle mesh vertex count (padded). */
  vertex_stress?: number[]
}

export interface SimulationConfig {
  N_PARTICLES: number
  YOUNGS_MODULUS: number
  POISSON_RATIO: number
  DT: number
  SUBSTEPS: number
  SIM_DURATION: number
  ENVIRONMENT_TYPE: 'plate' | 'cylinder'
  CYLINDER_DIAMETER: number
  DROP_HEIGHT: number
}

export type WsLog = { type: 'log'; line: string }
export type WsFrame = { type: 'frame'; step: number; t: number; Z?: number; max_vel?: number; particles: WsFrameParticle[] }
export type WsComplete = { type: 'complete'; metrics?: { Z?: number; total_pp?: number } }
export type WsError = { type: 'error'; message: string }
export type WsCancelled = { type: 'cancelled' }
export type WsMsg = WsLog | WsFrame | WsComplete | WsError | WsCancelled

