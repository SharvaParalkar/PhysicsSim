export interface ParticleData {
  id: number;
  x: number;
  y: number;
  z: number;
  qx: number;
  qy: number;
  qz: number;
  qw: number;
  n_contacts: number;
}

export interface ContactPair {
  particle_a: number;
  particle_b: number;
  depth: number;
  force: number;
  contact_area: number;
}

export interface ContactPoint {
  x: number;
  y: number;
  z: number;
  nx: number;
  ny: number;
  nz: number;
}

export interface SimMetrics {
  Z: number;
  total_pp: number;
  total_pc: number;
  n_isolated: number;
  n_container_touch: number;
}

export interface SimulationResults {
  particles: ParticleData[];
  contactPairs: ContactPair[];
  contactPoints: ContactPoint[];
}

