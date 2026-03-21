import { Canvas, useFrame, useLoader } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js'
import type { WsFrameParticle } from './types'

const STRESS_SLOTS = 8

function createStressMaterial() {
  const mat = new THREE.MeshStandardMaterial({
    color: '#ffffff',
    vertexColors: false,
    roughness: 0.42,
    metalness: 0.05,
    emissive: new THREE.Color('#15304f'),
    emissiveIntensity: 0.08,
  })
  mat.onBeforeCompile = (shader) => {
    shader.vertexShader = `
      attribute vec4 stressA;
      attribute vec4 stressB;
      varying float vStress;
      ${shader.vertexShader}
    `.replace(
      '#include <begin_vertex>',
      `#include <begin_vertex>
       int sid = int(mod(float(gl_VertexID), 8.0));
       vStress = sid == 0 ? stressA.x
         : sid == 1 ? stressA.y
         : sid == 2 ? stressA.z
         : sid == 3 ? stressA.w
         : sid == 4 ? stressB.x
         : sid == 5 ? stressB.y
         : sid == 6 ? stressB.z
         : stressB.w;
      `,
    )
    shader.fragmentShader = `
      varying float vStress;
      ${shader.fragmentShader}
    `.replace(
      '#include <color_fragment>',
      `#include <color_fragment>
       vec3 cLow = vec3(0.0, 0.28, 0.78);
       vec3 cHigh = vec3(0.95, 0.12, 0.08);
       diffuseColor.rgb *= mix(cLow, cHigh, clamp(vStress, 0.0, 1.0));
      `,
    )
  }
  return mat
}

function InstancedParticles(props: { count: number; radius: number; liveTransformsRef: React.MutableRefObject<WsFrameParticle[]> }) {
  const { count, radius, liveTransformsRef } = props
  const meshRef = useRef<THREE.InstancedMesh | null>(null)
  const obj = useLoader(OBJLoader, '/particle.obj')
  const { geometry, stressA, stressB } = useMemo(() => {
    let source: THREE.BufferGeometry | null = null
    obj.traverse((child) => {
      if (source) return
      const mesh = child as THREE.Mesh
      if (mesh?.isMesh && mesh.geometry) source = mesh.geometry
    })

    const geom = (source ? source.clone() : new THREE.SphereGeometry(radius, 18, 18)) as THREE.BufferGeometry
    geom.computeVertexNormals()

    const a = new THREE.InstancedBufferAttribute(new Float32Array(count * 4), 4)
    const b = new THREE.InstancedBufferAttribute(new Float32Array(count * 4), 4)
    a.setUsage(THREE.DynamicDrawUsage)
    b.setUsage(THREE.DynamicDrawUsage)
    geom.setAttribute('stressA', a)
    geom.setAttribute('stressB', b)

    return { geometry: geom, stressA: a, stressB: b }
  }, [obj, radius, count])

  const material = useMemo(() => createStressMaterial(), [])
  const dummy = useMemo(() => new THREE.Object3D(), [])

  useFrame(() => {
    const mesh = meshRef.current
    if (!mesh) return

    const live = liveTransformsRef.current
    if (!live?.length) return

    const arrA = stressA.array as Float32Array
    const arrB = stressB.array as Float32Array
    const n = Math.min(count, live.length)
    for (let i = 0; i < n; i++) {
      const p = live[i]
      dummy.position.set(p.x, p.y, p.z)
      dummy.quaternion.set(p.qx, p.qy, p.qz, p.qw).normalize()
      dummy.updateMatrix()
      mesh.setMatrixAt(i, dummy.matrix)

      const vs = p.vertex_stress
      const fallback = Math.max(0, Math.min(1, p.stress_intensity ?? 0))
      if (vs && vs.length >= STRESS_SLOTS) {
        for (let j = 0; j < 4; j++) arrA[i * 4 + j] = vs[j] ?? 0
        for (let j = 0; j < 4; j++) arrB[i * 4 + j] = vs[j + 4] ?? 0
      } else {
        for (let j = 0; j < 4; j++) arrA[i * 4 + j] = fallback
        for (let j = 0; j < 4; j++) arrB[i * 4 + j] = fallback
      }
    }
    mesh.instanceMatrix.needsUpdate = true
    stressA.needsUpdate = true
    stressB.needsUpdate = true
  })

  useEffect(() => {
    return () => {
      geometry.dispose()
      material.dispose()
    }
  }, [geometry, material])

  return <instancedMesh ref={meshRef} args={[geometry, material, count]} />
}

function Scene(props: { count: number; liveTransformsRef: React.MutableRefObject<WsFrameParticle[]> }) {
  const { count, liveTransformsRef } = props
  return (
    <>
      <color attach="background" args={['#dfeaf7']} />
      <hemisphereLight args={['#f0f7ff', '#9cb4d1', 0.8]} />
      <ambientLight intensity={0.65} />
      <directionalLight position={[1.6, 2.6, 1.2]} intensity={1.25} />
      <OrbitControls makeDefault enableDamping dampingFactor={0.08} />
      <InstancedParticles count={count} radius={0.02} liveTransformsRef={liveTransformsRef} />
    </>
  )
}

type Viewer3DProps = {
  particleCount: number
  liveTransformsRef: React.MutableRefObject<WsFrameParticle[]>
}

/** 3D view only — WebSocket lives in App (shared socket survives React Strict Mode). */
export default function Viewer3D(props: Viewer3DProps) {
  const { particleCount, liveTransformsRef } = props
  return (
    <Canvas style={{ width: '100%', height: '100%' }} camera={{ position: [0.26, 0.25, 0.26], fov: 55 }}>
      <Scene count={particleCount} liveTransformsRef={liveTransformsRef} />
    </Canvas>
  )
}
