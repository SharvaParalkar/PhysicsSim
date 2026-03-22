import { Canvas, useFrame, useLoader } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { useCallback, useEffect, useMemo, useRef, useState, type MutableRefObject } from 'react'
import * as THREE from 'three'
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js'
import type { LivePhysicsFrame, WsFrameParticle } from './types'

const _m = new THREE.Matrix4()
const _qA = new THREE.Quaternion()
const _qB = new THREE.Quaternion()
const _qOut = new THREE.Quaternion()
const _posA = new THREE.Vector3()
const _posB = new THREE.Vector3()
const _scale = new THREE.Vector3(1, 1, 1)

function withSettledVertexStress(
  p: WsFrameParticle | undefined,
  settled: Record<string, number[]> | null,
): WsFrameParticle | undefined {
  if (!p || !settled) return p
  const arr = settled[String(p.id)]
  if (!arr?.length) return p
  return { ...p, vertex_intensities: arr }
}

function cloneParticles(ps: WsFrameParticle[]): WsFrameParticle[] {
  const out: WsFrameParticle[] = new Array(ps.length)
  for (let i = 0; i < ps.length; i++) {
    const p = ps[i]!
    out[i] = {
      ...p,
      vertex_intensities: p.vertex_intensities ? p.vertex_intensities.slice() : undefined,
      vertex_stress_indices: p.vertex_stress_indices ? p.vertex_stress_indices.slice() : undefined,
    }
  }
  return out
}

function lerpParticleTransform(a: WsFrameParticle, b: WsFrameParticle, alpha: number, target: WsFrameParticle) {
  const t = alpha
  target.id = b.id
  _posA.set(a.x, a.y, a.z)
  _posB.set(b.x, b.y, b.z)
  target.x = THREE.MathUtils.lerp(_posA.x, _posB.x, t)
  target.y = THREE.MathUtils.lerp(_posA.y, _posB.y, t)
  target.z = THREE.MathUtils.lerp(_posA.z, _posB.z, t)
  _qA.set(a.qx, a.qy, a.qz, a.qw).normalize()
  _qB.set(b.qx, b.qy, b.qz, b.qw).normalize()
  THREE.Quaternion.slerpQuaternions(_qA, _qB, t, _qOut)
  target.qx = _qOut.x
  target.qy = _qOut.y
  target.qz = _qOut.z
  target.qw = _qOut.w
  const sa = a.stress_intensity ?? 0
  const sb = b.stress_intensity ?? 0
  target.stress_intensity = THREE.MathUtils.lerp(sa, sb, t)
}

/** Writes lerped per-vertex stress into `row` (length ≥ verts) without allocating. */
function fillStressRowLerp(
  row: Float32Array,
  verts: number,
  pa: WsFrameParticle | undefined,
  pb: WsFrameParticle | undefined,
  alpha: number,
) {
  const va = pa?.vertex_intensities
  const vb = pb?.vertex_intensities
  const fa = pa ? Math.max(0, Math.min(1, pa.stress_intensity ?? 0)) : 0
  const fb = pb ? Math.max(0, Math.min(1, pb.stress_intensity ?? 0)) : 0
  const fMix = THREE.MathUtils.lerp(fa, fb, alpha)
  const sparseA = (pa?.vertex_stress_indices?.length ?? 0) > 0
  const sparseB = (pb?.vertex_stress_indices?.length ?? 0) > 0
  if (sparseA || sparseB) {
    row.fill(fMix)
    return
  }
  if (va && vb && va.length > 0 && vb.length > 0) {
    const m = Math.min(verts, va.length, vb.length)
    for (let k = 0; k < m; k++) row[k] = THREE.MathUtils.lerp(va[k]!, vb[k]!, alpha)
    const tail = THREE.MathUtils.lerp(va[m - 1]!, vb[m - 1]!, alpha)
    for (let k = m; k < verts; k++) row[k] = tail
  } else if (vb && vb.length > 0) {
    const m = Math.min(verts, vb.length)
    for (let k = 0; k < m; k++) row[k] = THREE.MathUtils.lerp(fa, vb[k]!, alpha)
    const tail = row[m - 1]!
    for (let k = m; k < verts; k++) row[k] = tail
  } else if (va && va.length > 0) {
    const m = Math.min(verts, va.length)
    for (let k = 0; k < m; k++) row[k] = THREE.MathUtils.lerp(va[k]!, fb, alpha)
    const tail = row[m - 1]!
    for (let k = m; k < verts; k++) row[k] = tail
  } else {
    row.fill(fMix)
  }
}

/** Placeholder MeshStandardMaterial on InstancedMesh until OBJ + stress layout are ready. */
function createBootstrapMaterial() {
  return new THREE.MeshStandardMaterial({
    color: '#c8d6e8',
    roughness: 0.45,
    metalness: 0.06,
    emissive: new THREE.Color('#15304f'),
    emissiveIntensity: 0.06,
  })
}

function glslReadInstanceStress(nMat4: number): string {
  const pickFromM = /* glsl */ `
    int col = k2 / 4;
    int row = k2 - col * 4;
    if (col == 0) return M[0][row];
    if (col == 1) return M[1][row];
    if (col == 2) return M[2][row];
    return M[3][row];
`
  if (nMat4 === 1) {
    return /* glsl */ `
      float readInstanceStress(int k) {
        int k2 = k;
        mat4 M = instanceStress0;
        ${pickFromM}
      }
    `
  }
  return /* glsl */ `
    float readInstanceStress(int k) {
      mat4 M;
      int k2;
      if (k < 16) {
        M = instanceStress0;
        k2 = k;
      } else {
        M = instanceStress1;
        k2 = k - 16;
      }
      ${pickFromM}
    }
  `
}

function createBufferStressShaderMaterial(nMat4: 1 | 2): THREE.ShaderMaterial {
  const attrs =
    nMat4 === 1
      ? /* glsl */ `attribute mat4 instanceStress0;`
      : /* glsl */ `
      attribute mat4 instanceStress0;
      attribute mat4 instanceStress1;
    `
  return new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL1,
    uniforms: {
      uLightDir: { value: new THREE.Vector3(0.45, 0.85, 0.35).normalize() },
      uAmbient: { value: 0.38 },
      uDiffuse: { value: 0.62 },
    },
    // Do not redeclare position/normal/instanceMatrix/uniforms — Three prepends them (ShaderMaterial).
    vertexShader: /* glsl */ `
      attribute float vertexStressIndex;
      ${attrs}

      varying vec3 vNormalW;
      varying vec3 vAlbedo;

      ${glslReadInstanceStress(nMat4)}

      void main() {
        int vk = int(vertexStressIndex + 0.5);
        float raw = readInstanceStress(vk);
        float t = clamp(raw, 0.0, 1.0);
        vAlbedo = vec3(t, 1.0 - abs(t - 0.5) * 2.0, 1.0 - t);

        mat4 worldMat = modelMatrix * instanceMatrix;
        vNormalW = normalize(mat3(worldMat) * normal);
        gl_Position = projectionMatrix * modelViewMatrix * instanceMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: /* glsl */ `
      varying vec3 vNormalW;
      varying vec3 vAlbedo;
      uniform vec3 uLightDir;
      uniform float uAmbient;
      uniform float uDiffuse;

      void main() {
        float ndl = max(dot(normalize(vNormalW), normalize(uLightDir)), 0.0);
        vec3 col = vAlbedo * (uAmbient + uDiffuse * ndl);
        gl_FragColor = vec4(col, 1.0);
      }
    `,
    instancing: true,
  })
}

/** Fallback when mesh has too many vertices for packed mat4 instanced attributes (WebGL attrib limits). */
function createTextureStressShaderMaterial(stressTex: THREE.DataTexture, texW: number, texH: number): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL1,
    uniforms: {
      uVertexStress: { value: stressTex },
      uVertexStressSize: { value: new THREE.Vector2(texW, texH) },
      uLightDir: { value: new THREE.Vector3(0.45, 0.85, 0.35).normalize() },
      uAmbient: { value: 0.38 },
      uDiffuse: { value: 0.62 },
    },
    vertexShader: /* glsl */ `
      attribute float vertexStressIndex;

      uniform sampler2D uVertexStress;
      uniform vec2 uVertexStressSize;

      varying vec3 vNormalW;
      varying vec3 vAlbedo;

      void main() {
        vec2 suv = vec2(
          (vertexStressIndex + 0.5) / uVertexStressSize.x,
          (float(gl_InstanceID) + 0.5) / uVertexStressSize.y
        );
        float raw = texture2D(uVertexStress, suv).r;
        float t = clamp(raw, 0.0, 1.0);
        vAlbedo = vec3(t, 1.0 - abs(t - 0.5) * 2.0, 1.0 - t);

        mat4 worldMat = modelMatrix * instanceMatrix;
        vNormalW = normalize(mat3(worldMat) * normal);
        gl_Position = projectionMatrix * modelViewMatrix * instanceMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: /* glsl */ `
      varying vec3 vNormalW;
      varying vec3 vAlbedo;
      uniform vec3 uLightDir;
      uniform float uAmbient;
      uniform float uDiffuse;

      void main() {
        float ndl = max(dot(normalize(vNormalW), normalize(uLightDir)), 0.0);
        vec3 col = vAlbedo * (uAmbient + uDiffuse * ndl);
        gl_FragColor = vec4(col, 1.0);
      }
    `,
    instancing: true,
  })
}

/** Pack per-vertex intensities into column-major mat4 blocks (16 floats each) for instanced attributes. */
function packStressRowToMat4s(row: Float32Array, verts: number, dest: Float32Array, instIndex: number, nMat4: number) {
  const stride = nMat4 * 16
  const base = instIndex * stride
  dest.fill(0, base, base + stride)
  for (let k = 0; k < verts; k++) {
    const col = k >> 2
    const r = k & 3
    dest[base + col * 4 + r] = row[k]!
  }
}

function fillStressRow(
  row: Float32Array,
  verts: number,
  p: WsFrameParticle | undefined,
  fallback: number,
) {
  if (p?.vertex_intensities && p.vertex_intensities.length > 0) {
    const data = p.vertex_intensities
    const idxs = p.vertex_stress_indices
    if (idxs && idxs.length === data.length) {
      row.fill(fallback)
      for (let j = 0; j < idxs.length; j++) {
        const ix = idxs[j]!
        if (ix >= 0 && ix < verts) row[ix] = data[j]!
      }
    } else {
      const m = Math.min(verts, data.length)
      for (let k = 0; k < m; k++) row[k] = data[k]!
      const tail = data[m - 1] ?? fallback
      for (let k = m; k < verts; k++) row[k] = tail
    }
  } else {
    const f = p ? Math.max(0, Math.min(1, p.stress_intensity ?? fallback)) : fallback
    row.fill(f)
  }
}

type Snapshot = { wallS: number; simT: number; particles: WsFrameParticle[] }

function InstancedFemParticles(props: {
  count: number
  radius: number
  liveFrameRef: MutableRefObject<LivePhysicsFrame>
  simRunId: number
  settledVertexStress: Record<string, number[]> | null
  onMeshVertexCount?: (verts: number, sphereFallback?: boolean) => void
}) {
  const { count, radius, liveFrameRef, simRunId, settledVertexStress, onMeshVertexCount } = props
  const meshRef = useRef<THREE.InstancedMesh | null>(null)
  const obj = useLoader(OBJLoader, '/particle.obj')

  const MAX_STRESS_MAT4 = 2
  const FLOATS_PER_INSTANCE_MAT4 = 16

  const {
    geometry,
    verts,
    sphereFallback,
    stressPath,
    stressInterleaved,
    stressNMat4,
    stressScratchRow,
    stressTexW,
    stressData,
    stressTexture,
  } = useMemo(() => {
    let sphereFallback = false
    const base: THREE.BufferGeometry = (() => {
      let g: THREE.BufferGeometry | undefined
      obj.traverse((child) => {
        if (g) return
        const mesh = child as THREE.Mesh
        if (mesh?.isMesh && mesh.geometry) g = mesh.geometry
      })
      if (!g) {
        sphereFallback = true
        console.warn(
          '[Viewer3D] particle.obj not found in /public — falling back to sphere. Vertex stress colors will not align correctly.',
        )
        g = new THREE.SphereGeometry(radius, 5, 4)
      }
      return g.clone()
    })()
    base.computeVertexNormals()
    const vCount = base.getAttribute('position').count
    const idx = new Float32Array(vCount)
    for (let i = 0; i < vCount; i++) idx[i] = i
    base.setAttribute('vertexStressIndex', new THREE.BufferAttribute(idx, 1))

    const matsNeeded = Math.ceil(vCount / FLOATS_PER_INSTANCE_MAT4)
    const useBuffer = matsNeeded <= MAX_STRESS_MAT4
    const nMat4 = useBuffer ? matsNeeded : 0
    const stressScratchRow = new Float32Array(vCount)

    let stressInterleaved: THREE.InstancedInterleavedBuffer | null = null
    if (useBuffer && nMat4 > 0) {
      const stride = nMat4 * FLOATS_PER_INSTANCE_MAT4
      const arr = new Float32Array(Math.max(1, count) * stride)
      stressInterleaved = new THREE.InstancedInterleavedBuffer(arr, stride, 1)
      stressInterleaved.setUsage(THREE.DynamicDrawUsage)
      base.setAttribute('instanceStress0', new THREE.InterleavedBufferAttribute(stressInterleaved, 16, 0))
      if (nMat4 > 1) {
        base.setAttribute('instanceStress1', new THREE.InterleavedBufferAttribute(stressInterleaved, 16, 16))
      }
    }

    const tw = vCount
    const th = Math.max(1, count)
    let stressData: Float32Array | null = null
    let stressTexture: THREE.DataTexture | null = null
    if (!useBuffer) {
      stressData = new Float32Array(tw * th)
      stressTexture = new THREE.DataTexture(stressData, tw, th, THREE.RedFormat, THREE.FloatType)
      stressTexture.minFilter = THREE.NearestFilter
      stressTexture.magFilter = THREE.NearestFilter
      stressTexture.wrapS = THREE.ClampToEdgeWrapping
      stressTexture.wrapT = THREE.ClampToEdgeWrapping
      stressTexture.needsUpdate = true
    }

    return {
      geometry: base,
      verts: vCount,
      sphereFallback,
      stressPath: useBuffer ? ('buffer' as const) : ('texture' as const),
      stressInterleaved,
      stressNMat4: nMat4,
      stressScratchRow,
      stressTexW: tw,
      stressData,
      stressTexture,
    }
  }, [obj, radius, count])

  useEffect(() => {
    onMeshVertexCount?.(verts, sphereFallback)
  }, [verts, sphereFallback, onMeshVertexCount])

  const bootstrapMat = useMemo(() => createBootstrapMaterial(), [])
  const shaderMat = useMemo(() => {
    if (stressPath === 'buffer' && (stressNMat4 === 1 || stressNMat4 === 2)) {
      return createBufferStressShaderMaterial(stressNMat4)
    }
    if (!stressTexture) {
      throw new Error('Viewer3D: texture stress path missing DataTexture')
    }
    return createTextureStressShaderMaterial(stressTexture, stressTexW, count)
  }, [stressPath, stressNMat4, stressTexture, stressTexW, count])

  const [useShader, setUseShader] = useState(false)
  useEffect(() => {
    setUseShader(false)
  }, [geometry, simRunId])

  useEffect(() => {
    const t = window.setTimeout(() => setUseShader(true), 0)
    return () => window.clearTimeout(t)
  }, [geometry, simRunId])

  const snapA = useRef<Snapshot | null>(null)
  const snapB = useRef<Snapshot | null>(null)
  const lastConsumedSerial = useRef<number | null>(null)
  const lerpScratch = useRef<WsFrameParticle[]>([])

  useEffect(() => {
    snapA.current = null
    snapB.current = null
    lastConsumedSerial.current = null
  }, [simRunId])

  useEffect(() => {
    lastConsumedSerial.current = null
  }, [settledVertexStress])

  useFrame(({ clock }) => {
    const mesh = meshRef.current
    if (!mesh) return

    const frame = liveFrameRef.current
    const serial = frame.serial
    if (lastConsumedSerial.current !== serial) {
      lastConsumedSerial.current = serial
      const wallS = clock.elapsedTime
      const simT = typeof frame.t === 'number' ? frame.t : 0
      const cloned = cloneParticles(frame.particles)
      if (!snapB.current) {
        snapB.current = { wallS, simT, particles: cloned }
        snapA.current = { wallS, simT, particles: cloneParticles(cloned) }
      } else {
        snapA.current = snapB.current
        snapB.current = { wallS, simT, particles: cloned }
      }
    }

    const a = snapA.current
    const b = snapB.current
    let alpha = 1
    if (a && b && a !== b && b.wallS > a.wallS) {
      const dtWall = b.wallS - a.wallS
      if (dtWall > 1e-6) {
        alpha = THREE.MathUtils.clamp((clock.elapsedTime - a.wallS) / dtWall, 0, 1)
      }
    }

    const particlesA = a?.particles
    const particlesB = b?.particles
    if (!particlesB?.length) {
      mesh.count = 0
      return
    }

    let scratch = lerpScratch.current
    if (scratch.length < count) {
      scratch = new Array(count)
      for (let i = 0; i < count; i++) scratch[i] = { ...EMPTY_PARTICLE }
      lerpScratch.current = scratch
    }

    mesh.count = count

    for (let i = 0; i < count; i++) {
      const pa = particlesA?.[i]
      const pb = particlesB[i]
      const tgt = scratch[i]!
      if (pa && pb && alpha < 1) {
        lerpParticleTransform(pa, pb, alpha, tgt)
      } else if (pb) {
        tgt.id = pb.id
        tgt.x = pb.x
        tgt.y = pb.y
        tgt.z = pb.z
        tgt.qx = pb.qx
        tgt.qy = pb.qy
        tgt.qz = pb.qz
        tgt.qw = pb.qw
        tgt.stress_intensity = pb.stress_intensity
      } else {
        tgt.x = 0
        tgt.y = -1e6
        tgt.z = 0
        tgt.qx = 0
        tgt.qy = 0
        tgt.qz = 0
        tgt.qw = 1
        tgt.stress_intensity = 0
      }

      _m.compose(
        _posA.set(tgt.x, tgt.y, tgt.z),
        _qOut.set(tgt.qx, tgt.qy, tgt.qz, tgt.qw).normalize(),
        _scale,
      )
      mesh.setMatrixAt(i, _m)

      const row = stressScratchRow
      if (pa && pb && alpha < 1) {
        fillStressRowLerp(
          row,
          verts,
          withSettledVertexStress(pa, settledVertexStress),
          withSettledVertexStress(pb, settledVertexStress),
          alpha,
        )
      } else {
        const fb = Math.max(0, Math.min(1, tgt.stress_intensity ?? 0))
        fillStressRow(row, verts, withSettledVertexStress(pb, settledVertexStress), fb)
      }

      if (stressPath === 'buffer' && stressInterleaved && stressNMat4 > 0) {
        packStressRowToMat4s(row, verts, stressInterleaved.array as Float32Array, i, stressNMat4)
      } else if (stressData) {
        const rowOffset = i * stressTexW
        stressData.set(row.subarray(0, stressTexW), rowOffset)
      }
    }

    mesh.instanceMatrix.needsUpdate = true
    if (stressPath === 'buffer' && stressInterleaved) {
      stressInterleaved.needsUpdate = true
    } else {
      stressTexture.needsUpdate = true
    }
  })

  useEffect(() => {
    return () => {
      geometry.dispose()
      bootstrapMat.dispose()
      shaderMat.dispose()
      stressTexture?.dispose()
    }
  }, [geometry, bootstrapMat, shaderMat, stressTexture])

  return (
    <instancedMesh
      key={useShader ? 'stress-glsl' : 'bootstrap-std'}
      ref={meshRef}
      args={[geometry, useShader ? shaderMat : bootstrapMat, count]}
      frustumCulled={false}
    />
  )
}

const EMPTY_PARTICLE: WsFrameParticle = {
  id: -1,
  x: 0,
  y: 0,
  z: 0,
  qx: 0,
  qy: 0,
  qz: 0,
  qw: 1,
}

function ShallowBasket(props: {
  diameter: number
  wallThickness: number
  height: number
  segments: number
  transparent: boolean
}) {
  const { diameter, wallThickness, height, segments, transparent } = props
  const rInner = diameter / 2
  const py = wallThickness / 2
  const wallY = wallThickness + height / 2
  const angleStep = (2 * Math.PI) / segments
  const chordW = diameter * Math.sin(angleStep / 2)

  const wallMat = useMemo(() => {
    if (transparent) {
      return new THREE.MeshPhysicalMaterial({
        color: '#9db4d4',
        roughness: 0.35,
        metalness: 0.05,
        transparent: true,
        opacity: 0.22,
        depthWrite: false,
        side: THREE.DoubleSide,
        transmission: 0.15,
        thickness: 0.02,
        clearcoat: 0.2,
      })
    }
    return new THREE.MeshStandardMaterial({ color: '#b8c4d4', roughness: 0.55, metalness: 0.12 })
  }, [transparent])

  const bottomMat = wallMat

  useEffect(() => {
    return () => {
      wallMat.dispose()
    }
  }, [wallMat])

  const panels = useMemo(() => {
    const items: { pos: [number, number, number]; rotY: number }[] = []
    for (let i = 0; i < segments; i++) {
      const angle = (2 * Math.PI * i) / segments
      const cx = (diameter / 2 + wallThickness / 2) * Math.cos(angle)
      const cz = (diameter / 2 + wallThickness / 2) * Math.sin(angle)
      items.push({ pos: [cx, wallY, cz], rotY: angle })
    }
    return items
  }, [diameter, wallThickness, wallY, segments])

  return (
    <group>
      <mesh position={[0, py, 0]} receiveShadow material={bottomMat}>
        <cylinderGeometry args={[rInner + wallThickness, rInner + wallThickness, wallThickness, segments]} />
      </mesh>
      {panels.map((p, i) => (
        <mesh key={i} position={p.pos} rotation={[0, p.rotY, 0]} receiveShadow material={wallMat}>
          <boxGeometry args={[chordW, height, wallThickness]} />
        </mesh>
      ))}
    </group>
  )
}

function Scene(props: {
  count: number
  liveFrameRef: MutableRefObject<LivePhysicsFrame>
  simRunId: number
  settledVertexStress: Record<string, number[]> | null
  onMeshVertexCount?: (verts: number, sphereFallback?: boolean) => void
  environmentType: 'plate' | 'cylinder'
  plateSize: number
  wallThickness: number
  plateWallHeight: number
  cylinderDiameter: number
  cylinderHeight: number
  cylinderSegments: number
  transparentContainer: boolean
}) {
  const {
    count,
    liveFrameRef,
    simRunId,
    settledVertexStress,
    onMeshVertexCount,
    environmentType,
    plateSize,
    wallThickness,
    plateWallHeight,
    cylinderDiameter,
    cylinderHeight,
    cylinderSegments,
    transparentContainer,
  } = props
  const py = wallThickness / 2
  const s = plateSize
  const t = wallThickness
  const h = plateWallHeight
  const span = s + 2 * t
  const yWall = t + h / 2

  const plateBaseMat = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: '#b8c4d4',
        roughness: 0.55,
        metalness: 0.12,
      }),
    [],
  )

  /** Rim walls: nearly invisible glass (geometry stays for depth / collisions in scene). */
  const plateWallMat = useMemo(
    () =>
      new THREE.MeshPhysicalMaterial({
        color: '#9eb6d4',
        roughness: 0.2,
        metalness: 0,
        transparent: true,
        opacity: 0.04,
        depthWrite: false,
        side: THREE.DoubleSide,
        transmission: 0.98,
        thickness: 0.12,
        ior: 1.45,
        clearcoat: 0,
      }),
    [],
  )

  useEffect(() => {
    return () => {
      plateBaseMat.dispose()
      plateWallMat.dispose()
    }
  }, [plateBaseMat, plateWallMat])

  return (
    <>
      <color attach="background" args={['#dfeaf7']} />
      <hemisphereLight args={['#f0f7ff', '#9cb4d1', 0.8]} />
      <ambientLight intensity={0.65} />
      <directionalLight position={[1.6, 2.6, 1.2]} intensity={1.25} />
      <OrbitControls makeDefault enableDamping dampingFactor={0.08} />
      {environmentType === 'plate' ? (
        <group>
          <mesh position={[0, py, 0]} receiveShadow material={plateBaseMat}>
            <boxGeometry args={[plateSize, wallThickness, plateSize]} />
          </mesh>
          {h > 1e-6 ? (
            <>
              <mesh position={[0, yWall, s / 2 + t / 2]} receiveShadow material={plateWallMat}>
                <boxGeometry args={[span, h, t]} />
              </mesh>
              <mesh position={[0, yWall, -(s / 2 + t / 2)]} receiveShadow material={plateWallMat}>
                <boxGeometry args={[span, h, t]} />
              </mesh>
              <mesh position={[s / 2 + t / 2, yWall, 0]} receiveShadow material={plateWallMat}>
                <boxGeometry args={[t, h, span]} />
              </mesh>
              <mesh position={[-(s / 2 + t / 2), yWall, 0]} receiveShadow material={plateWallMat}>
                <boxGeometry args={[t, h, span]} />
              </mesh>
            </>
          ) : null}
        </group>
      ) : (
        <ShallowBasket
          diameter={cylinderDiameter}
          wallThickness={wallThickness}
          height={cylinderHeight}
          segments={cylinderSegments}
          transparent={transparentContainer}
        />
      )}
      <InstancedFemParticles
        count={count}
        radius={0.02}
        liveFrameRef={liveFrameRef}
        simRunId={simRunId}
        settledVertexStress={settledVertexStress}
        onMeshVertexCount={onMeshVertexCount}
      />
    </>
  )
}

type Viewer3DProps = {
  particleCount: number
  liveFrameRef: MutableRefObject<LivePhysicsFrame>
  simRunId: number
  settledVertexStress: Record<string, number[]> | null
  onMeshVertexCount?: (verts: number, sphereFallback?: boolean) => void
  environmentType: 'plate' | 'cylinder'
  plateSize: number
  wallThickness: number
  plateWallHeight: number
  cylinderDiameter: number
  cylinderHeight: number
  cylinderSegments: number
  transparentContainer: boolean
}

/** 3D view only — WebSocket lives in App (shared socket survives React Strict Mode). */
export default function Viewer3D(props: Viewer3DProps) {
  const {
    particleCount,
    liveFrameRef,
    simRunId,
    settledVertexStress,
    onMeshVertexCount,
    environmentType,
    plateSize,
    wallThickness,
    plateWallHeight,
    cylinderDiameter,
    cylinderHeight,
    cylinderSegments,
    transparentContainer,
  } = props
  const [meshVerts, setMeshVerts] = useState<number | null>(null)
  const [sphereFallbackMesh, setSphereFallbackMesh] = useState(false)
  useEffect(() => {
    setMeshVerts(null)
    setSphereFallbackMesh(false)
  }, [simRunId])
  const reportMeshVerts = useCallback(
    (verts: number, sphereFallback?: boolean) => {
      setMeshVerts(verts)
      setSphereFallbackMesh(!!sphereFallback)
      onMeshVertexCount?.(verts, sphereFallback)
    },
    [onMeshVertexCount],
  )
  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      {settledVertexStress !== null ? (
        <div
          style={{
            position: 'absolute',
            top: 10,
            left: 10,
            zIndex: 2,
            background: 'rgba(255,255,255,0.88)',
            border: '1px solid #d4dfec',
            borderRadius: 10,
            padding: '8px 12px',
            fontSize: 11,
            color: '#1d3553',
            pointerEvents: 'none',
          }}
        >
          <div style={{ fontWeight: 700, marginBottom: 6 }}>Contact stress</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ color: '#2255aa' }}>low</span>
            <div
              style={{
                width: 80,
                height: 10,
                borderRadius: 3,
                background: 'linear-gradient(90deg, #0000ff, #00ff00, #ff0000)',
              }}
            />
            <span style={{ color: '#aa2222' }}>high</span>
          </div>
          <div style={{ marginTop: 4, color: '#55779f' }}>blue = no contact · red = max stress</div>
        </div>
      ) : null}
      {settledVertexStress !== null && (sphereFallbackMesh || (meshVerts !== null && meshVerts <= 25)) ? (
        <div
          style={{
            position: 'absolute',
            bottom: 10,
            left: 10,
            zIndex: 2,
            background: 'rgba(255, 248, 220, 0.95)',
            border: '1px solid #e6c35c',
            borderRadius: 10,
            padding: '8px 12px',
            fontSize: 11,
            color: '#7a5a00',
            pointerEvents: 'none',
            maxWidth: 280,
          }}
        >
          ⚠ particle.obj not found — stress colors may be misaligned
        </div>
      ) : null}
      <Canvas style={{ width: '100%', height: '100%' }} camera={{ position: [0.55, 0.85, 0.85], fov: 50 }} gl={{ alpha: false }}>
        <Scene
          count={particleCount}
          liveFrameRef={liveFrameRef}
          simRunId={simRunId}
          settledVertexStress={settledVertexStress}
          onMeshVertexCount={reportMeshVerts}
          environmentType={environmentType}
          plateSize={plateSize}
          wallThickness={wallThickness}
          plateWallHeight={plateWallHeight}
          cylinderDiameter={cylinderDiameter}
          cylinderHeight={cylinderHeight}
          cylinderSegments={cylinderSegments}
          transparentContainer={transparentContainer}
        />
      </Canvas>
    </div>
  )
}
