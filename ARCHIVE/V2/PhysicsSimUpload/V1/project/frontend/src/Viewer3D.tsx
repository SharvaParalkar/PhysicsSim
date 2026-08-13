import React, { useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import type { ParticleData, ContactPoint, ContactPair } from "./types";

// ── Constants ───────────────────────────────────────────────────────────────

/** Sphere radius used to visually represent each particle (placeholder for .obj). */
const VISUAL_PARTICLE_RADIUS = 0.012;

/**
 * Container dimensions match the simulation script defaults so the boundary
 * lines up correctly with the settled pile.
 *   CYLINDER_DIAMETER = 0.20  →  radius = 0.10
 *   CYLINDER_HEIGHT   = 0.30
 *   PLATE_SIZE        = 0.60
 */
const CYLINDER_RADIUS = 0.10;
const CYLINDER_HEIGHT = 0.30;
const BOX_SIZE = 0.60;
const BOX_HEIGHT = 0.30;

// ── Types ───────────────────────────────────────────────────────────────────

export type EnvironmentType = "cylinder" | "plate";

type Viewer3DProps = {
  particles: ParticleData[];
  contactPairs: ContactPair[];
  contactPoints: ContactPoint[];
  environmentType?: EnvironmentType;
};

// ── Container ───────────────────────────────────────────────────────────────

type ContainerProps = { environmentType: EnvironmentType };

/**
 * Renders the simulation boundary as a wireframe + faint-fill overlay so
 * particles inside remain visible.  The geometry is chosen based on the
 * environment type set in the simulation script.
 *
 * Cylinder: open-ended, rendered with BackSide fill + wireframe front.
 * Box/Plate: floor slab + open wireframe box for the walls.
 */
const Container: React.FC<ContainerProps> = ({ environmentType }) => {
  if (environmentType === "cylinder") {
    return (
      // Centre the cylinder so its base sits at y = 0
      <group position={[0, CYLINDER_HEIGHT / 2, 0]}>
        {/* Inner ghost fill – visible from outside the rim */}
        <mesh>
          <cylinderGeometry args={[CYLINDER_RADIUS, CYLINDER_RADIUS, CYLINDER_HEIGHT, 48, 1, true]} />
          <meshStandardMaterial
            color="#9ca3af"
            transparent
            opacity={0.06}
            side={THREE.BackSide}
            depthWrite={false}
          />
        </mesh>

        {/* Wireframe rim */}
        <mesh>
          <cylinderGeometry args={[CYLINDER_RADIUS, CYLINDER_RADIUS, CYLINDER_HEIGHT, 48, 1, true]} />
          <meshStandardMaterial color="#6b7280" wireframe transparent opacity={0.45} />
        </mesh>
      </group>
    );
  }

  // "plate" → open-top box
  return (
    <group>
      {/* Thin floor slab */}
      <mesh position={[0, -0.002, 0]}>
        <boxGeometry args={[BOX_SIZE, 0.004, BOX_SIZE]} />
        <meshStandardMaterial color="#9ca3af" transparent opacity={0.18} />
      </mesh>

      {/* Wireframe walls */}
      <mesh position={[0, BOX_HEIGHT / 2, 0]}>
        <boxGeometry args={[BOX_SIZE, BOX_HEIGHT, BOX_SIZE]} />
        <meshStandardMaterial color="#6b7280" wireframe transparent opacity={0.30} />
      </mesh>
    </group>
  );
};

// ── Particle ─────────────────────────────────────────────────────────────────

type ParticleMeshProps = {
  particle: ParticleData;
  maxContacts: number;
};

/**
 * Single particle rendered as a sphere (SphereGeometry placeholder for the
 * real .obj).  Colour is interpolated blue → red over [0, maxContacts] using
 * HSL: hue 0.66 (blue) at zero contacts, hue 0.0 (red) at max contacts.
 */
const ParticleMesh: React.FC<ParticleMeshProps> = ({ particle, maxContacts }) => {
  const contactRatio = maxContacts > 0 ? particle.n_contacts / maxContacts : 0;

  const color = useMemo(
    () => new THREE.Color().setHSL((1 - contactRatio) * 0.66, 0.85, 0.52),
    [contactRatio],
  );

  // THREE.Quaternion(x, y, z, w) matches our qx/qy/qz/qw data fields.
  const quaternion = useMemo(
    () => new THREE.Quaternion(particle.qx, particle.qy, particle.qz, particle.qw),
    [particle.qx, particle.qy, particle.qz, particle.qw],
  );

  return (
    <mesh
      position={[particle.x, particle.y, particle.z]}
      quaternion={quaternion}
    >
      {/* SphereGeometry (formerly SphereBufferGeometry, merged in r125) */}
      <sphereGeometry args={[VISUAL_PARTICLE_RADIUS, 16, 12]} />
      <meshStandardMaterial color={color} roughness={0.4} metalness={0.15} />
    </mesh>
  );
};

// ── Contact point ────────────────────────────────────────────────────────────

type ContactPointMeshProps = { contactPoint: ContactPoint };

/** Small yellow emissive sphere marking a particle–particle contact location. */
const ContactPointMesh: React.FC<ContactPointMeshProps> = ({ contactPoint }) => (
  <mesh position={[contactPoint.x, contactPoint.y, contactPoint.z]}>
    <sphereGeometry args={[0.002, 8, 6]} />
    <meshStandardMaterial color="#facc15" emissive="#facc15" emissiveIntensity={0.7} />
  </mesh>
);

// ── HUD overlays ─────────────────────────────────────────────────────────────

const ContactCountLegend: React.FC = () => (
  <div
    style={{
      position: "absolute",
      bottom: 16,
      right: 16,
      background: "rgba(15, 15, 26, 0.85)",
      borderRadius: 8,
      border: "1px solid #2d2d3f",
      padding: "10px 14px",
      pointerEvents: "none",
      display: "flex",
      flexDirection: "column",
      gap: 6,
    }}
  >
    <span style={{ fontSize: 11, color: "#94a3b8", fontWeight: 600, letterSpacing: "0.05em" }}>
      Contact count
    </span>
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div
        style={{
          width: 80,
          height: 10,
          borderRadius: 4,
          background: "linear-gradient(to right, #2563eb, #dc2626)",
        }}
      />
      <span style={{ fontSize: 10, color: "#64748b", whiteSpace: "nowrap" }}>0 → max</span>
    </div>
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <div
        style={{
          width: 10,
          height: 10,
          borderRadius: "50%",
          background: "#facc15",
          flexShrink: 0,
        }}
      />
      <span style={{ fontSize: 10, color: "#64748b" }}>Contact point</span>
    </div>
  </div>
);

const EmptyStateOverlay: React.FC = () => (
  <div
    style={{
      position: "absolute",
      bottom: 20,
      left: "50%",
      transform: "translateX(-50%)",
      color: "#4b5563",
      fontSize: 13,
      pointerEvents: "none",
      whiteSpace: "nowrap",
    }}
  >
    No simulation data — press &ldquo;Start Simulation&rdquo; to run
  </div>
);

// ── Main export ───────────────────────────────────────────────────────────────

/**
 * 3-D viewer that renders simulation output inside a @react-three/fiber Canvas.
 *
 * Props
 * ─────
 * particles       – array of settled particle states (position, quaternion, contact count)
 * contactPairs    – raw pair data (not rendered here, kept for future use)
 * contactPoints   – 3-D locations of particle–particle contacts → yellow dots
 * environmentType – "cylinder" (default) | "plate"  controls which container
 *                   boundary geometry is shown
 *
 * Scene elements
 * ──────────────
 * · Container  – semi-transparent + wireframe Cylinder or Box
 * · Particles  – coloured spheres (blue = few contacts, red = many)
 * · Contacts   – yellow spheres at each contact location (r = 0.002 m)
 * · GridHelper – ground-plane reference grid
 * · OrbitControls – mouse tumble / pan / zoom
 */
export const Viewer3D: React.FC<Viewer3DProps> = ({
  particles,
  contactPairs,
  contactPoints,
  environmentType = "cylinder",
}) => {
  const maxContacts = useMemo(
    () => Math.max(...particles.map((p) => p.n_contacts), 1),
    [particles],
  );

  const hasData = particles.length > 0;

  return (
    <div style={{ width: "100%", height: "100%", position: "relative", background: "#0a0a14" }}>
      <Canvas camera={{ position: [0.3, 0.3, 0.45], fov: 50 }}>
        {/* Lighting */}
        <ambientLight intensity={0.35} />
        <directionalLight position={[1, 2, 1.5]} intensity={1.1} castShadow />
        <pointLight position={[-0.5, 0.5, -0.5]} intensity={0.4} color="#a78bfa" />

        {/* Navigation */}
        <OrbitControls makeDefault target={[0, 0.12, 0]} />

        {/* Ground-plane spatial reference */}
        <gridHelper args={[2, 40, "#3d3d5a", "#252538"]} position={[0, 0, 0]} />

        {/* Container boundary */}
        <Container environmentType={environmentType} />

        {/* Simulation geometry */}
        {hasData ? (
          <>
            {particles.map((particle) => (
              <ParticleMesh
                key={particle.id}
                particle={particle}
                maxContacts={maxContacts}
              />
            ))}

            {contactPoints.map((cp, i) => (
              // index is stable – contact points carry no persistent id
              <ContactPointMesh key={i} contactPoint={cp} />
            ))}
          </>
        ) : (
          // Placeholder so the viewer looks intentional before any run
          <mesh position={[0, 0.08, 0]}>
            <sphereGeometry args={[0.06, 32, 24]} />
            <meshStandardMaterial color="#7c3aed" wireframe opacity={0.5} transparent />
          </mesh>
        )}
      </Canvas>

      {hasData && <ContactCountLegend />}
      {!hasData && <EmptyStateOverlay />}
    </div>
  );
};
