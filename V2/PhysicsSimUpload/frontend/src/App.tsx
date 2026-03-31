import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import Viewer3D from './Viewer3D'
import Dashboard from './Dashboard'
import ResultsOverlay from './ResultsOverlay'
import type { ExportPayload } from './ResultsOverlay'
import { getSharedSimulationSocket } from './simulationSocket'
import type { LivePhysicsFrame, MetricsResponse, ParticleData, SimMetrics, SimulationConfig } from './types'

type ResultsPayload = {
  particles: ParticleData[]
}

type ParticlesResponse = {
  particles: string[]
  default?: string
}

/** Default throughput tuning (matches prior UI baseline). */
const STANDARD_PHYSICS_TUNING: Pick<
  SimulationConfig,
  'SETTLE_THRESHOLD' | 'DT' | 'SUBSTEPS' | 'ANALYTICAL_MODE' | 'DROP_HEIGHT' | 'DROP_SPREAD' | 'GRAVITY'
> = {
  SETTLE_THRESHOLD: 1e-3,
  DT: 1 / 240,
  SUBSTEPS: 4,
  ANALYTICAL_MODE: true,
  DROP_HEIGHT: 0.2,
  DROP_SPREAD: 0,
  GRAVITY: [0, -9.81, 0],
}

/** Faster settling for rigid-body runs: looser settle, coarser step, no analytical handoff, shorter drop + spread, higher |g|. */
const FAST_PHYSICS_TUNING: Pick<
  SimulationConfig,
  'SETTLE_THRESHOLD' | 'DT' | 'SUBSTEPS' | 'ANALYTICAL_MODE' | 'DROP_HEIGHT' | 'DROP_SPREAD' | 'GRAVITY'
> = {
  SETTLE_THRESHOLD: 5e-3,
  DT: 1 / 120,
  SUBSTEPS: 2,
  ANALYTICAL_MODE: false,
  DROP_HEIGHT: 0.05,
  DROP_SPREAD: 0.3,
  GRAVITY: [0, -20.0, 0],
}

function matchesFastPhysicsPreset(c: SimulationConfig): boolean {
  const f = FAST_PHYSICS_TUNING
  if (Math.abs(c.SETTLE_THRESHOLD - f.SETTLE_THRESHOLD) >= 1e-9) return false
  if (Math.abs(c.DT - f.DT) >= 1e-9) return false
  if (c.SUBSTEPS !== f.SUBSTEPS) return false
  if (c.ANALYTICAL_MODE !== false) return false
  if (Math.abs(c.DROP_HEIGHT - f.DROP_HEIGHT) >= 1e-9) return false
  if (Math.abs(c.DROP_SPREAD - f.DROP_SPREAD) >= 1e-9) return false
  return c.GRAVITY.every((v, i) => Math.abs(v - f.GRAVITY[i]) < 1e-6)
}

const MODE_LABEL: Record<'idle' | 'falling' | 'settled', string> = {
  idle: 'Ready',
  falling: 'Running',
  settled: 'Finished',
}

const FAST_MODE_HELP =
  'Rigid-style preset: looser settle threshold, dt 1/120, 2 substeps, analytical off, shorter drop, Vogel spread, stronger gravity. Uncheck to restore standard settings.'

type LengthUnit = 'um' | 'mm' | 'cm'

const LENGTH_UNITS: Record<LengthUnit, { label: string; short: string; metersPerUnit: number }> = {
  um: { label: 'Micrometers', short: 'µm', metersPerUnit: 1e-6 },
  mm: { label: 'Millimeters', short: 'mm', metersPerUnit: 0.001 },
  cm: { label: 'Centimeters', short: 'cm', metersPerUnit: 0.01 },
}

export default function App() {
  const [mode, setMode] = useState<'idle' | 'falling' | 'settled'>('idle')
  const [logs, setLogs] = useState<string[]>([])
  const [metrics, setMetrics] = useState<SimMetrics>({
    Z: 0,
    total_pp: 0,
    total_pc: 0,
    n_isolated: 0,
    n_container_touch: 0,
    system_pressure: 0,
  })
  const [runSignal, setRunSignal] = useState(0)
  const [keSeries, setKeSeries] = useState<Array<{ t: number; kinetic_energy: number }>>([])
  const [metricsResponse, setMetricsResponse] = useState<MetricsResponse>({
    Z: 0,
    total_pp: 0,
    total_pc: 0,
    n_isolated: 0,
    n_container_touch: 0,
    system_pressure: 0,
    z_history: [],
    max_vel_history: [],
    contact_graph_dict: {},
    contact_graph_links: [],
  })
  const [pressureSeries, setPressureSeries] = useState<Array<{ t: number; system_pressure: number }>>([])
  const [liveHud, setLiveHud] = useState({
    t: 0,
    max_vel: 0,
    kinetic_energy: 0,
    system_pressure: 0,
  })
  const [loadProgress, setLoadProgress] = useState<{ phase: string; pct: number; detail: string } | null>(null)
  const [settledVertexStress, setSettledVertexStress] = useState<Record<string, number[]> | null>(null)
  const [availableParticles, setAvailableParticles] = useState<string[]>(['particle.obj'])
  const [simConfig, setSimConfig] = useState<SimulationConfig>({
    PARTICLE_FILE: 'particle.obj',
    N_PARTICLES: 100,
    YOUNGS_MODULUS: 200e6,
    POISSON_RATIO: 0.45,
    ...STANDARD_PHYSICS_TUNING,
    SEQUENTIAL_DROP: false,
    SEQUENTIAL_STAGE_DURATION: null,
    SIM_DURATION: 5.0,
    ENVIRONMENT_TYPE: 'plate',
    CYLINDER_DIAMETER: 0.2,
    CYLINDER_HEIGHT: 0.3,
    PLATE_SIZE: 0.6,
    WALL_THICKNESS: 0.02,
    PLATE_WALL_HEIGHT: 0.15,
    STRESS_SIGMA: 0.4,
  })
  const [lengthUnit, setLengthUnit] = useState<LengthUnit>('cm')
  const { metersPerUnit, short: lengthUnitLabel } = LENGTH_UNITS[lengthUnit]
  const lengthScale = 1 / metersPerUnit
  const [showResults, setShowResults] = useState(false)
  const [screenshotDataUrl, setScreenshotDataUrl] = useState<string | null>(null)
  const [exportData, setExportData] = useState<ExportPayload | null>(null)

  const logBoxRef = useRef<HTMLPreElement | null>(null)
  const liveFrameRef = useRef<LivePhysicsFrame>({ step: -99999, t: 0, particles: [], serial: 0 })
  const liveFrameSerialRef = useRef(0)
  const pendingStartRef = useRef<string | null>(null)
  const [transparentContainer, setTransparentContainer] = useState(false)
  const [wsStatus, setWsStatus] = useState<'connecting' | 'open' | 'closed'>(() => {
    const ws = getSharedSimulationSocket()
    if (ws.readyState === WebSocket.OPEN) return 'open'
    if (ws.readyState === WebSocket.CONNECTING) return 'connecting'
    return 'closed'
  })
  const [sectionOpen, setSectionOpen] = useState({ simulation: true, environment: true, advanced: false })
  const [logOpen, setLogOpen] = useState(false)
  const [showAdvancedHud, setShowAdvancedHud] = useState(false)

  const refreshParticleList = useCallback(async () => {
    const resp = (await fetch('http://localhost:8000/particles').then((r) => r.json())) as ParticlesResponse
    const list = Array.isArray(resp.particles) ? resp.particles.filter((v) => typeof v === 'string' && v.length > 0) : []
    const defaultName = typeof resp.default === 'string' && resp.default.length > 0 ? resp.default : 'particle.obj'
    const normalized = list.length > 0 ? list : [defaultName]
    setAvailableParticles(normalized)
    setSimConfig((prev) => {
      const chosen = prev.PARTICLE_FILE ?? defaultName
      return { ...prev, PARTICLE_FILE: normalized.includes(chosen) ? chosen : defaultName }
    })
  }, [])

  const refreshResults = useCallback(async () => {
    const [mRes, rRes] = await Promise.all([fetch('http://localhost:8000/metrics'), fetch('http://localhost:8000/results')])
    const mJson = (await mRes.json()) as MetricsResponse
    const rJson = (await rRes.json()) as ResultsPayload
    setMetrics(mJson)
    setMetricsResponse(mJson)
    if (mJson.kinetic_energy_history?.length) {
      setKeSeries(mJson.kinetic_energy_history.map((p) => ({ t: Number(p.t), kinetic_energy: Number(p.kinetic_energy) })))
    }
    if (mJson.pressure_history?.length) {
      setPressureSeries(mJson.pressure_history.map((p) => ({ t: Number(p.t), system_pressure: Number(p.system_pressure) })))
    }
    return rJson.particles || []
  }, [])

  const appendLog = useCallback((line: string) => {
    setLogs((prev) => [...prev, line])
  }, [])

  const copyLogs = useCallback(async () => {
    const text = logs.join('\n')
    if (!text) return
    try {
      await navigator.clipboard.writeText(text)
    } catch {
      const ta = document.createElement('textarea')
      ta.value = text
      ta.style.position = 'fixed'
      ta.style.left = '-9999px'
      document.body.appendChild(ta)
      ta.select()
      try {
        document.execCommand('copy')
      } finally {
        document.body.removeChild(ta)
      }
    }
  }, [logs])

  const handleRunComplete = useCallback(async () => {
    const particles = await refreshResults()
    setSettledVertexStress((prev) => {
      if (prev && Object.keys(prev).length > 0) return prev
      if (!particles?.length) return null
      const maxC = Math.max(...particles.map((p) => p.n_contacts), 1)
      const fallback: Record<string, number[]> = {}
      for (const p of particles) {
        fallback[String(p.id)] = [p.n_contacts / maxC]
      }
      return fallback
    })
  }, [refreshResults])

  const handleFrameMetrics = useCallback((point: { t: number; max_vel: number }) => {
    setLiveHud((h) => ({ ...h, t: point.t, max_vel: point.max_vel }))
  }, [])

  const handleLiveMetrics = useCallback(
    (point: { t: number; kinetic_energy: number; system_pressure: number }) => {
      setLiveHud((h) => ({
        ...h,
        t: point.t,
        kinetic_energy: point.kinetic_energy,
        system_pressure: point.system_pressure,
      }))
      setKeSeries((prev) => [...prev, { t: point.t, kinetic_energy: point.kinetic_energy }])
      setPressureSeries((prev) => [...prev, { t: point.t, system_pressure: point.system_pressure }])
    },
    [],
  )

  const onFrameMetricsRef = useRef(handleFrameMetrics)
  const onLiveMetricsRef = useRef(handleLiveMetrics)
  const onLogsRef = useRef(appendLog)
  const onModeChangeRef = useRef(setMode)
  const onRunCompleteRef = useRef(handleRunComplete)

  useEffect(() => {
    onFrameMetricsRef.current = handleFrameMetrics
  }, [handleFrameMetrics])
  useEffect(() => {
    onLiveMetricsRef.current = handleLiveMetrics
  }, [handleLiveMetrics])
  useEffect(() => {
    onLogsRef.current = appendLog
  }, [appendLog])
  useEffect(() => {
    onModeChangeRef.current = setMode
  }, [setMode])
  useEffect(() => {
    onRunCompleteRef.current = handleRunComplete
  }, [handleRunComplete])

  useEffect(() => {
    fetch('http://localhost:8000/metrics')
      .then((r) => r.json())
      .then((m: MetricsResponse) => {
        if (m.kinetic_energy_history?.length) {
          setKeSeries(m.kinetic_energy_history.map((p) => ({ t: p.t, kinetic_energy: p.kinetic_energy })))
        }
        if (m.pressure_history?.length) {
          setPressureSeries(m.pressure_history.map((p) => ({ t: p.t, system_pressure: p.system_pressure })))
        }
      })
      .catch(() => {})
  }, [])

  useEffect(() => {
    let cancelled = false
    refreshParticleList().catch(() => {
      if (cancelled) return
      setAvailableParticles(['particle.obj'])
    })
    return () => {
      cancelled = true
    }
  }, [refreshParticleList])

  useEffect(() => {
    if (runSignal === 0) return
    setSettledVertexStress(null)
    // Reset overlay state when a new run starts
    setShowResults(false)
    setScreenshotDataUrl(null)
    setExportData(null)
  }, [runSignal])

  useEffect(() => {
    if (mode !== 'settled') return

    // Schedule canvas capture for the next animation frame so Three.js has
    // finished rendering the settled frame (preserveDrawingBuffer=true keeps
    // the backbuffer valid, but rAF ensures we grab the most recent render).
    let rafId: number
    rafId = requestAnimationFrame(() => {
      const canvas = document.querySelector('canvas')
      if (canvas) {
        try {
          setScreenshotDataUrl(canvas.toDataURL('image/png'))
        } catch {
          // canvas may be cross-origin tainted; ignore
        }
      }
    })

    // Fetch full export payload
    fetch('http://localhost:8000/export')
      .then((r) => r.json())
      .then((data: ExportPayload) => {
        setExportData(data)
        setShowResults(true)
      })
      .catch((err) => {
        appendLog(`[warn] Could not load results overlay: ${err}`)
      })

    return () => cancelAnimationFrame(rafId)
  }, [mode, appendLog])

  useEffect(() => {
    const ws = getSharedSimulationSocket()
    const sync = () => {
      if (ws.readyState === WebSocket.OPEN) setWsStatus('open')
      else if (ws.readyState === WebSocket.CONNECTING) setWsStatus('connecting')
      else setWsStatus('closed')
    }
    ws.addEventListener('open', sync)
    ws.addEventListener('close', sync)
    ws.addEventListener('error', sync)
    sync()
    return () => {
      ws.removeEventListener('open', sync)
      ws.removeEventListener('close', sync)
      ws.removeEventListener('error', sync)
    }
  }, [])

  useEffect(() => {
    const ws = getSharedSimulationSocket()
    const flushPending = () => {
      if (pendingStartRef.current && ws.readyState === WebSocket.OPEN) {
        ws.send(pendingStartRef.current)
        pendingStartRef.current = null
      }
    }
    ws.addEventListener('open', flushPending)

    const onMessage = async (evt: MessageEvent) => {
      let parsed: unknown = null
      try {
        parsed = JSON.parse(String(evt.data ?? ''))
      } catch {
        onLogsRef.current(String(evt.data ?? ''))
        return
      }
      const msg = parsed as {
        type?: string
        line?: string
        message?: string
        t?: number
        particles?: LivePhysicsFrame['particles']
        Z?: number
        max_vel?: number
      }
      if (msg.type === 'progress') {
        const m = msg as { type: string; phase?: string; pct?: number; detail?: string }
        setLoadProgress({
          phase: typeof m.phase === 'string' ? m.phase : '',
          pct: typeof m.pct === 'number' && Number.isFinite(m.pct) ? Math.min(1, Math.max(0, m.pct)) : 0,
          detail: typeof m.detail === 'string' ? m.detail : '',
        })
        return
      }
      if (msg.type === 'log' && msg.line) {
        onLogsRef.current(msg.line)
        return
      }
      if (msg.type === 'live_metrics') {
        const lm = msg as {
          type: string
          t?: number
          Z?: number
          n_rattlers?: number
          kinetic_energy?: number
          system_pressure?: number
        }
        onLiveMetricsRef.current({
          t: typeof lm.t === 'number' ? lm.t : 0,
          kinetic_energy: typeof lm.kinetic_energy === 'number' ? lm.kinetic_energy : 0,
          system_pressure: typeof lm.system_pressure === 'number' ? lm.system_pressure : 0,
        })
        return
      }
      if (msg.type === 'frame') {
        const fm = msg as { step?: number; t?: number; particles?: LivePhysicsFrame['particles'] }
        liveFrameSerialRef.current += 1
        liveFrameRef.current = {
          step: typeof fm.step === 'number' ? fm.step : -1,
          t: typeof fm.t === 'number' ? fm.t : 0,
          particles: fm.particles ?? [],
          serial: liveFrameSerialRef.current,
        }
        onFrameMetricsRef.current({
          t: typeof msg.t === 'number' ? msg.t : 0,
          max_vel: typeof msg.max_vel === 'number' ? msg.max_vel : 0,
        })
        return
      }
      if (msg.type === 'error') {
        setLoadProgress(null)
        onLogsRef.current(`[error] ${msg.message ?? 'unknown error'}`)
        onModeChangeRef.current('idle')
        return
      }
      if (msg.type === 'complete') {
        setLoadProgress(null)
        onLogsRef.current('Simulation complete')
        const done = msg as {
          vertex_stress?: Record<string, number[]>
        }
        if (done.vertex_stress && Object.keys(done.vertex_stress).length > 0) {
          setSettledVertexStress(done.vertex_stress)
        }
        try {
          await onRunCompleteRef.current()
        } catch (e) {
          onLogsRef.current(`[warn] metrics refresh failed: ${e}`)
        }
        onModeChangeRef.current('settled')
        return
      }
      if (msg.type === 'idle') {
        setLoadProgress(null)
        return
      }
      if (msg.type === 'cancelled') {
        setLoadProgress(null)
        onModeChangeRef.current('idle')
        return
      }
    }

    const onError = () => {
      onLogsRef.current('WebSocket error (is the API running on port 8000?)')
      onModeChangeRef.current('idle')
    }

    ws.addEventListener('message', onMessage)
    ws.addEventListener('error', onError)
    flushPending()

    return () => {
      ws.removeEventListener('open', flushPending)
      ws.removeEventListener('message', onMessage)
      ws.removeEventListener('error', onError)
    }
  }, [])

  useEffect(() => {
    const el = logBoxRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [logs])

  const cancelSimulation = useCallback(async () => {
    if (mode !== 'falling') return
    try {
      await fetch('http://localhost:8000/cancel', { method: 'POST' })
      appendLog('Cancellation requested...')
    } catch {
      appendLog('[error] Failed to request cancellation')
    }
  }, [appendLog, mode])

  const simConfigRef = useRef(simConfig)
  useEffect(() => {
    simConfigRef.current = simConfig
  }, [simConfig])

  useEffect(() => {
    if (runSignal === 0) return
    const ws = getSharedSimulationSocket()
    const payload = JSON.stringify({
      type: 'start',
      frame_every: 30,
      live_metrics_every: 60,
      config: simConfigRef.current,
    })
    liveFrameSerialRef.current = 0
    liveFrameRef.current = { step: -99999, t: 0, particles: [], serial: 0 }
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(payload)
    } else {
      pendingStartRef.current = payload
    }
  }, [runSignal])

  const sidebarStyle: React.CSSProperties = {
    width: 340,
    background: '#f3f7fc',
    color: '#16314d',
    padding: 16,
    display: 'flex',
    flexDirection: 'column',
    gap: 12,
    boxSizing: 'border-box',
    borderRight: '1px solid #d1ddec',
  }

  const buttonStyle: React.CSSProperties = {
    background: mode === 'falling' ? '#7ca8e0' : '#2b6cff',
    border: '1px solid #1e5ccf',
    color: '#fff',
    padding: '10px 12px',
    borderRadius: 10,
    cursor: mode === 'falling' ? 'not-allowed' : 'pointer',
    fontWeight: 700,
  }

  const labelStyle: React.CSSProperties = { color: '#4d6b8f', fontSize: 12 }
  const valueStyle: React.CSSProperties = { color: '#173454', fontWeight: 800 }

  const metricsGridStyle: React.CSSProperties = {
    display: 'grid',
    gridTemplateColumns: 'repeat(3, 1fr)',
    gap: 6,
    background: '#edf4fc',
    border: '1px solid #d4dfec',
    borderRadius: 8,
    padding: '6px 8px',
  }

  const metricsLabelStyle: React.CSSProperties = { color: '#6b829f', fontSize: 10, lineHeight: 1.2 }
  const metricsValueStyle: React.CSSProperties = { color: '#173454', fontWeight: 700, fontSize: 12, lineHeight: 1.2 }

  const sectionToggleStyle: React.CSSProperties = {
    width: '100%',
    border: 'none',
    background: 'transparent',
    color: '#1d3553',
    fontWeight: 700,
    fontSize: 12,
    textAlign: 'left',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    cursor: 'pointer',
    padding: '2px 0 6px',
  }

  const formatPressure = (pa: number | undefined) => {
    if (pa == null || !Number.isFinite(pa)) return '—'
    const a = Math.abs(pa)
    if (a >= 1e6 || a < 1e-2) return pa.toExponential(2)
    return pa.toFixed(1)
  }

  const modularJammingHeuristic = useMemo(
    () =>
      liveHud.max_vel < 0.05 &&
      liveHud.kinetic_energy < 1e-2 &&
      Number.isFinite(liveHud.system_pressure) &&
      liveHud.system_pressure > 10,
    [liveHud.max_vel, liveHud.kinetic_energy, liveHud.system_pressure],
  )
  const particleObjectUrl = useMemo(() => {
    const file = encodeURIComponent(simConfig.PARTICLE_FILE ?? 'particle.obj')
    // Bust browser + R3F loader caches so edited/replaced OBJ files are reloaded.
    return `http://localhost:8000/particles/${file}?run=${runSignal}`
  }, [simConfig.PARTICLE_FILE, runSignal])

  return (
    <div
      style={{
        height: '100vh',
        display: 'flex',
        flexDirection: 'column',
        fontFamily: 'ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial',
      }}
    >
      <div style={{ flexGrow: 1, display: 'flex', minHeight: 0 }}>
        <div style={sidebarStyle}>
          <div style={{ fontSize: 18, fontWeight: 900, letterSpacing: 0.2 }}>Particle Sim</div>

          <button
            type="button"
            style={buttonStyle}
            onClick={() => {
              if (mode === 'falling') return
              setMode('falling')
              setLogs([])
              setKeSeries([])
              setPressureSeries([])
              setLiveHud({ t: 0, max_vel: 0, kinetic_energy: 0, system_pressure: 0 })
              setLoadProgress({ phase: 'start', pct: 0, detail: 'Connecting…' })
              setRunSignal((v) => v + 1)
            }}
          >
            {mode === 'falling' ? 'Running…' : 'Run Simulation'}
          </button>
          <button
            type="button"
            style={{
              ...buttonStyle,
              background: mode === 'falling' ? '#d94848' : '#b8c7dc',
              border: mode === 'falling' ? '1px solid #b42828' : '1px solid #9fb2ca',
              cursor: mode === 'falling' ? 'pointer' : 'not-allowed',
            }}
            onClick={cancelSimulation}
            disabled={mode !== 'falling'}
          >
            Cancel Simulation
          </button>

          {mode === 'falling' && loadProgress ? (
            <div style={{ marginTop: 4 }}>
              <div style={{ ...labelStyle, marginBottom: 4 }}>
                {loadProgress.phase === 'simulate' ? 'Simulation' : 'Loading'}{' '}
                {loadProgress.detail ? `— ${loadProgress.detail}` : ''}
              </div>
              <div
                style={{
                  height: 8,
                  borderRadius: 6,
                  background: '#d4dfec',
                  overflow: 'hidden',
                  border: '1px solid #bccbe0',
                }}
              >
                <div
                  style={{
                    height: '100%',
                    width: `${Math.round(loadProgress.pct * 100)}%`,
                    background: 'linear-gradient(90deg, #2b6cff, #5a9cff)',
                    transition: 'width 0.2s ease-out',
                  }}
                />
              </div>
            </div>
          ) : null}

          <div
            style={{
              flex: '0 1 auto',
              minHeight: 0,
              maxHeight: '38vh',
              overflowY: 'auto',
              paddingRight: 4,
              display: 'flex',
              flexDirection: 'column',
              gap: 8,
            }}
          >
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10, flexWrap: 'wrap' }}>
              <div style={{ flex: '1 1 120px' }}>
                <div style={labelStyle}>Status</div>
                <div style={valueStyle}>{MODE_LABEL[mode]}</div>
              </div>
              <div style={{ flex: '1 1 120px' }}>
                <div style={labelStyle}>Server stream</div>
                <div style={{ ...valueStyle, display: 'flex', alignItems: 'center', gap: 6, fontSize: 12 }}>
                  <span
                    aria-hidden
                    style={{
                      width: 8,
                      height: 8,
                      borderRadius: 999,
                      flexShrink: 0,
                      background:
                        wsStatus === 'open' ? '#2a9d8f' : wsStatus === 'connecting' ? '#e9c46a' : '#9aa5b5',
                    }}
                  />
                  {wsStatus === 'open' ? 'Connected' : wsStatus === 'connecting' ? 'Connecting…' : 'Disconnected'}
                </div>
              </div>
            </div>

            <div style={{ display: 'grid', gap: 8 }}>
              <div style={{ ...labelStyle, marginTop: 0 }}>Configuration</div>
              <label style={labelStyle} title="Geometry & the 3D viewer are scaled to these display units (physics still runs in meters).">
                Length units (visual + input)
                <select
                  value={lengthUnit}
                  onChange={(e) => setLengthUnit(e.target.value as LengthUnit)}
                  style={{ width: '100%', border: '1px solid #bccbe0', borderRadius: 6, padding: '4px 6px' }}
                >
                  <option value="um">{LENGTH_UNITS.um.label} ({LENGTH_UNITS.um.short})</option>
                  <option value="mm">{LENGTH_UNITS.mm.label} ({LENGTH_UNITS.mm.short})</option>
                  <option value="cm">{LENGTH_UNITS.cm.label} ({LENGTH_UNITS.cm.short})</option>
                </select>
              </label>
              <label style={labelStyle} title="Select an OBJ from the backend Particles folder.">
                Particle mesh
                <div style={{ display: 'flex', gap: 6 }}>
                  <select
                    value={simConfig.PARTICLE_FILE ?? 'particle.obj'}
                    onChange={(e) => setSimConfig((p) => ({ ...p, PARTICLE_FILE: e.target.value }))}
                    style={{ flex: 1, border: '1px solid #bccbe0', borderRadius: 6, padding: '4px 6px' }}
                  >
                    {availableParticles.map((name) => (
                      <option key={name} value={name}>
                        {name}
                      </option>
                    ))}
                  </select>
                  <button
                    type="button"
                    title="Reload files from the backend Particles folder"
                    onClick={() => {
                      refreshParticleList().catch(() => setAvailableParticles(['particle.obj']))
                    }}
                    style={{
                      border: '1px solid #bccbe0',
                      borderRadius: 6,
                      padding: '4px 8px',
                      background: '#fff',
                      color: '#1d3553',
                      cursor: 'pointer',
                      fontSize: 12,
                      fontWeight: 700,
                    }}
                  >
                    Refresh
                  </button>
                </div>
              </label>

              <button
                type="button"
                onClick={() => setSectionOpen((s) => ({ ...s, simulation: !s.simulation }))}
                style={sectionToggleStyle}
              >
                <span>Simulation</span>
                <span style={{ color: '#55779f', fontSize: 11 }}>{sectionOpen.simulation ? 'Collapse' : 'Expand'}</span>
              </button>
              {sectionOpen.simulation ? (
                <>
                  <label style={{ ...labelStyle, display: 'flex', alignItems: 'flex-start', gap: 8 }} title={FAST_MODE_HELP}>
                    <input
                      type="checkbox"
                      checked={matchesFastPhysicsPreset(simConfig)}
                      disabled={simConfig.SEQUENTIAL_DROP === true}
                      onChange={(e) =>
                        setSimConfig((p) => ({
                          ...p,
                          ...(e.target.checked ? FAST_PHYSICS_TUNING : STANDARD_PHYSICS_TUNING),
                        }))
                      }
                    />
                    <span>Fast preset (rigid-style settling)</span>
                  </label>
                  <label style={labelStyle}>
                    Particle count
                    <input
                      type="number"
                      min={1}
                      value={simConfig.N_PARTICLES}
                      onChange={(e) => setSimConfig((p) => ({ ...p, N_PARTICLES: Math.max(1, Number(e.target.value || 1)) }))}
                      style={{ width: '100%', border: '1px solid #bccbe0', borderRadius: 6, padding: '4px 6px' }}
                    />
                  </label>
                  <label style={labelStyle}>
                    Young&apos;s modulus (Pa); ≥1e8 ≈ rigid: {simConfig.YOUNGS_MODULUS.toExponential(2)}
                    <input
                      type="range"
                      min={4}
                      max={9}
                      step={0.01}
                      value={Math.log10(Math.max(1e4, Math.min(1e9, simConfig.YOUNGS_MODULUS)))}
                      onChange={(e) =>
                        setSimConfig((p) => ({
                          ...p,
                          YOUNGS_MODULUS: Math.min(1e9, 10 ** Number(e.target.value)),
                        }))
                      }
                      style={{ width: '100%' }}
                    />
                  </label>
                  <label style={labelStyle}>
                    Stress spread σ (rad): {simConfig.STRESS_SIGMA?.toFixed(2) ?? '0.40'}
                    <input
                      type="range"
                      min={0.1}
                      max={1.5}
                      step={0.05}
                      value={simConfig.STRESS_SIGMA ?? 0.4}
                      onChange={(e) =>
                        setSimConfig((p) => ({
                          ...p,
                          STRESS_SIGMA: Number(e.target.value),
                        }))
                      }
                      style={{ width: '100%' }}
                    />
                  </label>
                  <label style={labelStyle}>
                    Poisson ratio
                    <input
                      type="number"
                      step={0.01}
                      min={0}
                      max={0.49}
                      value={simConfig.POISSON_RATIO}
                      onChange={(e) => setSimConfig((p) => ({ ...p, POISSON_RATIO: Number(e.target.value) }))}
                      style={{ width: '100%', border: '1px solid #bccbe0', borderRadius: 6, padding: '4px 6px' }}
                    />
                  </label>
                  <label style={labelStyle}>
                    Max duration (s)
                    <input
                      type="number"
                      step={0.5}
                      min={0.5}
                      value={simConfig.SIM_DURATION}
                      onChange={(e) => setSimConfig((p) => ({ ...p, SIM_DURATION: Number(e.target.value) }))}
                      style={{ width: '100%', border: '1px solid #bccbe0', borderRadius: 6, padding: '4px 6px' }}
                    />
                  </label>
                  <label style={{ ...labelStyle, display: 'flex', alignItems: 'center', gap: 8 }}>
                    <input
                      type="checkbox"
                      checked={simConfig.SEQUENTIAL_DROP === true}
                      onChange={(e) =>
                        setSimConfig((p) => ({
                          ...p,
                          SEQUENTIAL_DROP: e.target.checked,
                        }))
                      }
                    />
                    <span>Sequential drop (one particle per rebuild)</span>
                  </label>
                  <label style={labelStyle}>
                    Stage duration (s, blank = auto)
                    <input
                      type="number"
                      min={0.05}
                      step={0.1}
                      placeholder="auto"
                      value={simConfig.SEQUENTIAL_STAGE_DURATION ?? ''}
                      disabled={simConfig.SEQUENTIAL_DROP !== true}
                      onChange={(e) => {
                        const raw = e.target.value
                        setSimConfig((p) => ({
                          ...p,
                          SEQUENTIAL_STAGE_DURATION: raw === '' ? null : Math.max(0.05, Number(raw)),
                        }))
                      }}
                      style={{ width: '100%', border: '1px solid #bccbe0', borderRadius: 6, padding: '4px 6px' }}
                    />
                  </label>
                </>
              ) : null}

              <button
                type="button"
                onClick={() => setSectionOpen((s) => ({ ...s, environment: !s.environment }))}
                style={sectionToggleStyle}
              >
                <span>Environment &amp; drop</span>
                <span style={{ color: '#55779f', fontSize: 11 }}>{sectionOpen.environment ? 'Collapse' : 'Expand'}</span>
              </button>
              {sectionOpen.environment ? (
                <>
                  <label style={labelStyle}>
                    Container
                    <select
                      value={simConfig.ENVIRONMENT_TYPE}
                      onChange={(e) => {
                        const v = e.target.value as SimulationConfig['ENVIRONMENT_TYPE']
                        setSimConfig((p) => ({ ...p, ENVIRONMENT_TYPE: v }))
                        if (v === 'plate') setTransparentContainer(false)
                      }}
                      style={{ width: '100%', border: '1px solid #bccbe0', borderRadius: 6, padding: '4px 6px' }}
                    >
                      <option value="plate">Flat plate</option>
                      <option value="cylinder">Cylinder</option>
                    </select>
                  </label>
                  {simConfig.ENVIRONMENT_TYPE === 'plate' ? (
                    <>
                      <label style={labelStyle}>
                        Plate size ({lengthUnitLabel})
                        <input
                          type="number"
                          step={0.05 / metersPerUnit}
                          min={0.1 / metersPerUnit}
                          value={simConfig.PLATE_SIZE / metersPerUnit}
                          onChange={(e) =>
                            setSimConfig((p) => ({ ...p, PLATE_SIZE: Number(e.target.value) * metersPerUnit }))
                          }
                          style={{ width: '100%', border: '1px solid #bccbe0', borderRadius: 6, padding: '4px 6px' }}
                        />
                      </label>
                      <label style={labelStyle}>
                        Rim wall height ({lengthUnitLabel})
                        <input
                          type="number"
                          step={0.01 / metersPerUnit}
                          min={0 / metersPerUnit}
                          value={simConfig.PLATE_WALL_HEIGHT / metersPerUnit}
                          onChange={(e) =>
                            setSimConfig((p) => ({ ...p, PLATE_WALL_HEIGHT: Number(e.target.value) * metersPerUnit }))
                          }
                          style={{ width: '100%', border: '1px solid #bccbe0', borderRadius: 6, padding: '4px 6px' }}
                        />
                      </label>
                    </>
                  ) : (
                    <>
                      <label style={labelStyle}>
                        Cylinder diameter ({lengthUnitLabel})
                        <input
                          type="number"
                          step={0.01 / metersPerUnit}
                          min={0.05 / metersPerUnit}
                          value={simConfig.CYLINDER_DIAMETER / metersPerUnit}
                          onChange={(e) =>
                            setSimConfig((p) => ({ ...p, CYLINDER_DIAMETER: Number(e.target.value) * metersPerUnit }))
                          }
                          style={{ width: '100%', border: '1px solid #bccbe0', borderRadius: 6, padding: '4px 6px' }}
                        />
                      </label>
                      <label style={labelStyle}>
                        Cylinder height ({lengthUnitLabel})
                        <input
                          type="number"
                          step={0.01 / metersPerUnit}
                          min={0.01 / metersPerUnit}
                          value={simConfig.CYLINDER_HEIGHT / metersPerUnit}
                          onChange={(e) =>
                            setSimConfig((p) => ({ ...p, CYLINDER_HEIGHT: Number(e.target.value) * metersPerUnit }))
                          }
                          style={{ width: '100%', border: '1px solid #bccbe0', borderRadius: 6, padding: '4px 6px' }}
                        />
                      </label>
                    </>
                  )}
                  <label style={labelStyle}>
                    Wall thickness ({lengthUnitLabel})
                    <input
                      type="number"
                      step={0.005 / metersPerUnit}
                      min={0.001 / metersPerUnit}
                      value={simConfig.WALL_THICKNESS / metersPerUnit}
                      onChange={(e) =>
                        setSimConfig((p) => ({ ...p, WALL_THICKNESS: Number(e.target.value) * metersPerUnit }))
                      }
                      style={{ width: '100%', border: '1px solid #bccbe0', borderRadius: 6, padding: '4px 6px' }}
                    />
                  </label>
                  <label style={labelStyle}>
                    Drop height ({lengthUnitLabel})
                    <input
                      type="number"
                      step={0.01 / metersPerUnit}
                      min={0.01 / metersPerUnit}
                      value={simConfig.DROP_HEIGHT / metersPerUnit}
                      onChange={(e) =>
                        setSimConfig((p) => ({ ...p, DROP_HEIGHT: Number(e.target.value) * metersPerUnit }))
                      }
                      style={{ width: '100%', border: '1px solid #bccbe0', borderRadius: 6, padding: '4px 6px' }}
                    />
                  </label>
                  <label style={labelStyle}>
                    Horizontal spread (0 = column at origin)
                    <input
                      type="number"
                      step={0.05}
                      min={0}
                      max={1}
                      value={simConfig.DROP_SPREAD}
                      onChange={(e) => setSimConfig((p) => ({ ...p, DROP_SPREAD: Number(e.target.value) }))}
                      style={{ width: '100%', border: '1px solid #bccbe0', borderRadius: 6, padding: '4px 6px' }}
                    />
                  </label>
                </>
              ) : null}

              <button
                type="button"
                onClick={() => setSectionOpen((s) => ({ ...s, advanced: !s.advanced }))}
                style={sectionToggleStyle}
              >
                <span>Advanced solver</span>
                <span style={{ color: '#55779f', fontSize: 11 }}>{sectionOpen.advanced ? 'Collapse' : 'Expand'}</span>
              </button>
              {sectionOpen.advanced ? (
                <>
                  <label style={{ ...labelStyle, display: 'flex', alignItems: 'center', gap: 8 }}>
                    <input
                      type="checkbox"
                      checked={simConfig.ANALYTICAL_MODE !== false}
                      disabled={simConfig.SEQUENTIAL_DROP === true || matchesFastPhysicsPreset(simConfig)}
                      title={
                        simConfig.SEQUENTIAL_DROP
                          ? 'Disabled while sequential drop is on.'
                          : matchesFastPhysicsPreset(simConfig)
                            ? 'Fast preset keeps analytical handoff off; uncheck Fast preset to edit.'
                            : undefined
                      }
                      onChange={(e) => setSimConfig((p) => ({ ...p, ANALYTICAL_MODE: e.target.checked }))}
                    />
                    <span>Analytical handoff (fast fall, then fine contact)</span>
                  </label>
                  <label style={labelStyle}>
                    Time step Δt (s): {simConfig.DT.toFixed(5)} (default 1/240)
                    <input
                      type="range"
                      min={0.001}
                      max={0.01}
                      step={0.00025}
                      value={simConfig.DT}
                      onChange={(e) => setSimConfig((p) => ({ ...p, DT: Number(e.target.value) }))}
                      style={{ width: '100%' }}
                    />
                  </label>
                  <label style={labelStyle}>
                    Substeps per frame: {simConfig.SUBSTEPS}
                    <input
                      type="range"
                      min={1}
                      max={20}
                      step={1}
                      value={simConfig.SUBSTEPS}
                      onChange={(e) => setSimConfig((p) => ({ ...p, SUBSTEPS: Math.round(Number(e.target.value)) }))}
                      style={{ width: '100%' }}
                    />
                  </label>
                  <label style={labelStyle}>
                    Settle speed threshold (m/s)
                    <input
                      type="number"
                      step={0.0005}
                      min={1e-6}
                      value={simConfig.SETTLE_THRESHOLD}
                      onChange={(e) => setSimConfig((p) => ({ ...p, SETTLE_THRESHOLD: Number(e.target.value) }))}
                      style={{ width: '100%', border: '1px solid #bccbe0', borderRadius: 6, padding: '4px 6px' }}
                    />
                  </label>
                  <label style={labelStyle}>
                    Gravity (m/s²) X, Y, Z
                    <div style={{ display: 'flex', gap: 6 }}>
                      {([0, 1, 2] as const).map((i) => (
                        <input
                          key={i}
                          type="number"
                          step={0.1}
                          value={simConfig.GRAVITY[i]}
                          onChange={(e) => {
                            const v = Number(e.target.value)
                            setSimConfig((p) => {
                              const g: [number, number, number] = [p.GRAVITY[0], p.GRAVITY[1], p.GRAVITY[2]]
                              g[i] = v
                              return { ...p, GRAVITY: g }
                            })
                          }}
                          style={{ flex: 1, border: '1px solid #bccbe0', borderRadius: 6, padding: '4px 6px' }}
                        />
                      ))}
                    </div>
                  </label>
                </>
              ) : null}
            </div>
          </div>

          <div style={{ flex: '0 0 auto' }}>
            <div style={{ ...labelStyle, marginBottom: 4 }}>Last run metrics</div>
            <div style={metricsGridStyle}>
              <div>
                <div style={metricsLabelStyle}>Coordination Z</div>
                <div style={metricsValueStyle}>{Number.isFinite(metrics.Z) ? metrics.Z.toFixed(2) : '—'}</div>
              </div>
              <div>
                <div style={metricsLabelStyle}>Pressure (Pa)</div>
                <div style={metricsValueStyle}>{formatPressure(metrics.system_pressure)}</div>
              </div>
              <div>
                <div style={metricsLabelStyle}>Particle–particle</div>
                <div style={metricsValueStyle}>{metrics.total_pp}</div>
              </div>
              <div>
                <div style={metricsLabelStyle}>Particle–wall</div>
                <div style={metricsValueStyle}>{metrics.total_pc}</div>
              </div>
              <div>
                <div style={metricsLabelStyle}>Isolated</div>
                <div style={metricsValueStyle}>{metrics.n_isolated}</div>
              </div>
              <div>
                <div style={metricsLabelStyle}>Wall contacts</div>
                <div style={metricsValueStyle}>{metrics.n_container_touch}</div>
              </div>
            </div>
          </div>

          {exportData && (
            <button
              type="button"
              onClick={() => setShowResults(true)}
              style={{
                background: mode === 'settled' ? '#1d3553' : 'white',
                border: '1px solid #bccbe0',
                color: mode === 'settled' ? 'white' : '#1d3553',
                padding: '9px 12px',
                borderRadius: 10,
                cursor: 'pointer',
                fontWeight: 700,
                fontSize: 12,
                width: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 6,
              }}
            >
              <span style={{ fontSize: 14 }}>📊</span> View Results
            </button>
          )}

          <div style={{ flex: logOpen ? '1 1 55%' : '0 0 auto', minHeight: logOpen ? 220 : 0, minWidth: 0, display: 'flex', flexDirection: 'column' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6, gap: 8, flexWrap: 'wrap' }}>
              <button
                type="button"
                onClick={() => setLogOpen((o) => !o)}
                style={{
                  border: 'none',
                  background: 'transparent',
                  padding: 0,
                  color: '#4d6b8f',
                  fontSize: 12,
                  fontWeight: 700,
                  cursor: 'pointer',
                  textAlign: 'left',
                }}
              >
                Developer log {logOpen ? '▼' : '▶'} {logs.length > 0 ? `(${logs.length})` : ''}
              </button>
              {logOpen ? (
                <button
                  type="button"
                  onClick={() => void copyLogs()}
                  style={{
                    flexShrink: 0,
                    fontSize: 11,
                    fontWeight: 700,
                    padding: '4px 10px',
                    borderRadius: 8,
                    border: '1px solid #bccbe0',
                    background: '#fff',
                    color: '#1d3553',
                    cursor: 'pointer',
                  }}
                >
                  Copy
                </button>
              ) : null}
            </div>
            {logOpen ? (
              <pre
                ref={logBoxRef}
                style={{
                  flex: 1,
                  minHeight: 0,
                  minWidth: 0,
                  background: '#f8fbff',
                  border: '1px solid #d4dfec',
                  borderRadius: 12,
                  padding: 10,
                  margin: 0,
                  overflow: 'auto',
                  color: '#274465',
                  fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
                  fontSize: 12,
                  lineHeight: 1.35,
                  whiteSpace: 'pre-wrap',
                  overflowWrap: 'break-word',
                  userSelect: 'text',
                  WebkitUserSelect: 'text',
                  cursor: 'text',
                }}
              >
                {logs.join('\n')}
              </pre>
            ) : null}
          </div>
        </div>

        <div style={{ flexGrow: 1, background: '#eaf1f9', position: 'relative', display: 'grid', gridTemplateRows: '1fr auto' }}>
          {mode === 'idle' ? (
            <div style={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center', color: '#5b7598', fontSize: 14 }}>
              Click “Run Simulation” to stream frames
            </div>
          ) : null}

          <div
            style={{
              minHeight: 380,
              height: '100%',
              position: 'relative',
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            <div
              style={{
                position: 'absolute',
                top: 10,
                right: 10,
                zIndex: 2,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'flex-end',
                gap: 8,
                pointerEvents: 'none',
              }}
            >
              <div
                style={{
                  background: 'rgba(255,255,255,0.9)',
                  border: '1px solid #d4dfec',
                  borderRadius: 10,
                  padding: '10px 12px',
                  fontSize: 12,
                  color: '#1d3553',
                  boxShadow: '0 4px 12px rgba(20,40,80,0.12)',
                  pointerEvents: 'auto',
                }}
              >
                <div style={{ fontWeight: 800, marginBottom: 6 }}>Live view</div>
                <div style={{ display: 'grid', gap: 4 }}>
                  <div>
                    <span style={{ color: '#55779f' }}>Time </span>
                    {liveHud.t.toFixed(3)} s
                  </div>
                  <div>
                    <span style={{ color: '#55779f' }}>Kinetic energy </span>
                    {liveHud.kinetic_energy.toExponential(2)} J
                  </div>
                  <div>
                    <span style={{ color: '#55779f' }}>Pressure </span>
                    {liveHud.system_pressure.toExponential(2)} Pa
                  </div>
                  <button
                    type="button"
                    onClick={() => setShowAdvancedHud((v) => !v)}
                    style={{
                      marginTop: 4,
                      alignSelf: 'flex-start',
                      fontSize: 11,
                      fontWeight: 700,
                      padding: '4px 8px',
                      borderRadius: 6,
                      border: '1px solid #bccbe0',
                      background: '#fff',
                      color: '#1d3553',
                      cursor: 'pointer',
                    }}
                  >
                    {showAdvancedHud ? 'Hide' : 'Show'} advanced
                  </button>
                  {showAdvancedHud ? (
                    <div style={{ marginTop: 2, fontSize: 11, fontWeight: 700, color: modularJammingHeuristic ? '#2a9d8f' : '#55779f' }}>
                      {modularJammingHeuristic ? '● Heuristic: jamming-like (low KE, high P)' : '○ Heuristic: still settling'}
                    </div>
                  ) : null}
                </div>
              </div>
              {simConfig.ENVIRONMENT_TYPE === 'cylinder' ? (
                <label
                  style={{
                    pointerEvents: 'auto',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 8,
                    cursor: 'pointer',
                    background: 'rgba(255,255,255,0.92)',
                    border: '1px solid #d4dfec',
                    borderRadius: 10,
                    padding: '8px 12px',
                    fontSize: 12,
                    color: '#1d3553',
                    boxShadow: '0 4px 12px rgba(20,40,80,0.1)',
                    userSelect: 'none',
                  }}
                >
                  <input
                    type="checkbox"
                    checked={transparentContainer}
                    onChange={(e) => setTransparentContainer(e.target.checked)}
                  />
                  Transparent container
                </label>
              ) : null}
            </div>
            <div style={{ flex: 1, minHeight: 380, position: 'relative', overflow: 'hidden' }}>
              <Viewer3D
                particleCount={simConfig.N_PARTICLES}
                particleObjectUrl={particleObjectUrl}
                liveFrameRef={liveFrameRef}
                simRunId={runSignal}
                settledVertexStress={settledVertexStress}
                environmentType={simConfig.ENVIRONMENT_TYPE}
                plateSize={simConfig.PLATE_SIZE}
                wallThickness={simConfig.WALL_THICKNESS}
                plateWallHeight={simConfig.PLATE_WALL_HEIGHT}
                cylinderDiameter={simConfig.CYLINDER_DIAMETER}
                cylinderHeight={simConfig.CYLINDER_HEIGHT}
                cylinderSegments={32}
                transparentContainer={transparentContainer}
                lengthScale={lengthScale}
              />
            </div>
          </div>
          <div style={{ padding: 12 }}>
            <Dashboard keSeries={keSeries} pressureSeries={pressureSeries} metrics={metricsResponse} />
          </div>
        </div>
      </div>

      {showResults && exportData && (
        <ResultsOverlay
          screenshot={screenshotDataUrl}
          data={exportData}
          onClose={() => setShowResults(false)}
        />
      )}
    </div>
  )
}
