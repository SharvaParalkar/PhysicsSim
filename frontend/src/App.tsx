import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import Viewer3D from './Viewer3D'
import Dashboard from './Dashboard'
import { getSharedSimulationSocket } from './simulationSocket'
import type { LivePhysicsFrame, MetricsResponse, ParticleData, SimMetrics, SimulationConfig } from './types'

type ResultsPayload = {
  particles: ParticleData[]
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
  const [simTime, setSimTime] = useState<number>(0)
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
  const [simConfig, setSimConfig] = useState<SimulationConfig>({
    N_PARTICLES: 100,
    YOUNGS_MODULUS: 200e6,
    POISSON_RATIO: 0.45,
    DT: 1 / 240,
    SUBSTEPS: 4,
    ANALYTICAL_MODE: true,
    SEQUENTIAL_DROP: false,
    SEQUENTIAL_STAGE_DURATION: null,
    SIM_DURATION: 5.0,
    ENVIRONMENT_TYPE: 'plate',
    CYLINDER_DIAMETER: 0.2,
    DROP_HEIGHT: 0.2,
    DROP_SPREAD: 0,
    PLATE_SIZE: 0.6,
    WALL_THICKNESS: 0.02,
    STRESS_SIGMA: 0.4,
  })
  const logBoxRef = useRef<HTMLPreElement | null>(null)
  const liveFrameRef = useRef<LivePhysicsFrame>({ step: -99999, t: 0, particles: [], serial: 0 })
  const liveFrameSerialRef = useRef(0)
  const pendingStartRef = useRef<string | null>(null)
  const [transparentContainer, setTransparentContainer] = useState(false)

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
    await refreshResults()
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

  const onSimTimeRef = useRef(setSimTime)
  const onFrameMetricsRef = useRef(handleFrameMetrics)
  const onLiveMetricsRef = useRef(handleLiveMetrics)
  const onLogsRef = useRef(appendLog)
  const onModeChangeRef = useRef(setMode)
  const onRunCompleteRef = useRef(handleRunComplete)

  useEffect(() => {
    onSimTimeRef.current = setSimTime
  }, [setSimTime])
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
    if (runSignal === 0) return
    setSettledVertexStress(null)
  }, [runSignal])

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
        onSimTimeRef.current(typeof fm.t === 'number' ? fm.t : 0)
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
        await onRunCompleteRef.current()
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
      frame_every: 2,
      live_metrics_every: 10,
      config: simConfigRef.current,
    })
    setSimTime(0)
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
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap: 6,
    background: '#edf4fc',
    border: '1px solid #d4dfec',
    borderRadius: 8,
    padding: '6px 8px',
  }

  const metricsLabelStyle: React.CSSProperties = { color: '#6b829f', fontSize: 10, lineHeight: 1.2 }
  const metricsValueStyle: React.CSSProperties = { color: '#173454', fontWeight: 700, fontSize: 12, lineHeight: 1.2 }

  const modularJammingHeuristic = useMemo(
    () =>
      liveHud.max_vel < 0.05 &&
      liveHud.kinetic_energy < 1e-2 &&
      Number.isFinite(liveHud.system_pressure) &&
      liveHud.system_pressure > 10,
    [liveHud.max_vel, liveHud.kinetic_energy, liveHud.system_pressure],
  )

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
              setSimTime(0)
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
            <div style={{ display: 'flex', gap: 10 }}>
              <div style={{ flex: 1 }}>
                <div style={labelStyle}>mode</div>
                <div style={valueStyle}>{mode}</div>
              </div>
              <div style={{ flex: 1 }}>
                <div style={labelStyle}>t (s)</div>
                <div style={valueStyle}>{simTime.toFixed(2)}</div>
              </div>
            </div>

            <div style={{ display: 'grid', gap: 8 }}>
              <div style={{ ...labelStyle, marginTop: 0 }}>Simulation Config</div>
            <label style={labelStyle}>
              N_PARTICLES
              <input
                type="number"
                min={1}
                value={simConfig.N_PARTICLES}
                onChange={(e) => setSimConfig((p) => ({ ...p, N_PARTICLES: Math.max(1, Number(e.target.value || 1)) }))}
                style={{ width: '100%', border: '1px solid #bccbe0', borderRadius: 6, padding: '4px 6px' }}
              />
            </label>
            <label style={labelStyle}>
              YOUNGS_MODULUS (Pa, &gt;1e8 = rigid): {simConfig.YOUNGS_MODULUS.toExponential(2)}
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
              Stress spread (STRESS_SIGMA, rad): {simConfig.STRESS_SIGMA?.toFixed(2) ?? '0.40'}
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
            <label style={{ ...labelStyle, display: 'flex', alignItems: 'center', gap: 8 }}>
              <input
                type="checkbox"
                checked={simConfig.ANALYTICAL_MODE !== false}
                disabled={simConfig.SEQUENTIAL_DROP === true}
                onChange={(e) => setSimConfig((p) => ({ ...p, ANALYTICAL_MODE: e.target.checked }))}
              />
              <span>Analytical mode (fast fall, then 500 Hz / 16 substeps)</span>
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
              <span>Sequential drop (FEM: one new particle per scene rebuild)</span>
            </label>
            <label style={labelStyle}>
              SEQUENTIAL_STAGE_DURATION (s, blank = auto)
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
            <label style={labelStyle}>
              DT (s): {simConfig.DT.toFixed(5)} (throughput default 1/240)
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
              SUBSTEPS: {simConfig.SUBSTEPS}
              <input
                type="range"
                min={4}
                max={20}
                step={1}
                value={simConfig.SUBSTEPS}
                onChange={(e) => setSimConfig((p) => ({ ...p, SUBSTEPS: Math.round(Number(e.target.value)) }))}
                style={{ width: '100%' }}
              />
            </label>
            <label style={labelStyle}>
              SIM_DURATION (s)
              <input
                type="number"
                step={0.5}
                min={0.5}
                value={simConfig.SIM_DURATION}
                onChange={(e) => setSimConfig((p) => ({ ...p, SIM_DURATION: Number(e.target.value) }))}
                style={{ width: '100%', border: '1px solid #bccbe0', borderRadius: 6, padding: '4px 6px' }}
              />
            </label>
            <label style={labelStyle}>
              POISSON_RATIO
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
              ENVIRONMENT_TYPE
              <select
                value={simConfig.ENVIRONMENT_TYPE}
                onChange={(e) => {
                  const v = e.target.value as SimulationConfig['ENVIRONMENT_TYPE']
                  setSimConfig((p) => ({ ...p, ENVIRONMENT_TYPE: v }))
                  if (v === 'plate') setTransparentContainer(false)
                }}
                style={{ width: '100%', border: '1px solid #bccbe0', borderRadius: 6, padding: '4px 6px' }}
              >
                <option value="plate">plate</option>
                <option value="cylinder">cylinder</option>
              </select>
            </label>
            <label style={labelStyle}>
              CYLINDER_DIAMETER (m)
              <input
                type="number"
                step={0.01}
                min={0.05}
                value={simConfig.CYLINDER_DIAMETER}
                onChange={(e) => setSimConfig((p) => ({ ...p, CYLINDER_DIAMETER: Number(e.target.value) }))}
                style={{ width: '100%', border: '1px solid #bccbe0', borderRadius: 6, padding: '4px 6px' }}
              />
            </label>
            <label style={labelStyle}>
              DROP_HEIGHT (m)
              <input
                type="number"
                step={0.01}
                min={0.01}
                value={simConfig.DROP_HEIGHT}
                onChange={(e) => setSimConfig((p) => ({ ...p, DROP_HEIGHT: Number(e.target.value) }))}
                style={{ width: '100%', border: '1px solid #bccbe0', borderRadius: 6, padding: '4px 6px' }}
              />
            </label>
            <label style={labelStyle}>
              DROP_SPREAD (0 = column at origin)
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
            </div>
          </div>

          <div style={{ flex: '0 0 auto' }}>
            <div style={{ ...labelStyle, marginBottom: 4 }}>Metrics (post-run)</div>
            <div style={metricsGridStyle}>
              <div>
                <div style={metricsLabelStyle}>total_pp</div>
                <div style={metricsValueStyle}>{metrics.total_pp}</div>
              </div>
              <div>
                <div style={metricsLabelStyle}>total_pc</div>
                <div style={metricsValueStyle}>{metrics.total_pc}</div>
              </div>
            </div>
          </div>

          <div style={{ flex: '1 1 55%', minHeight: 220, minWidth: 0, display: 'flex', flexDirection: 'column' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6, gap: 8 }}>
              <div style={labelStyle}>Log</div>
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
            </div>
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
                }}
              >
                <div style={{ fontWeight: 800, marginBottom: 6 }}>Live HUD</div>
                <div style={{ display: 'grid', gap: 4 }}>
                  <div>
                    <span style={{ color: '#55779f' }}>t </span>
                    {liveHud.t.toFixed(3)} s
                  </div>
                  <div>
                    <span style={{ color: '#55779f' }}>KE </span>
                    {liveHud.kinetic_energy.toExponential(2)} J
                  </div>
                  <div>
                    <span style={{ color: '#55779f' }}>P </span>
                    {liveHud.system_pressure.toExponential(2)} Pa
                  </div>
                  <div style={{ marginTop: 6, fontSize: 11, fontWeight: 700, color: modularJammingHeuristic ? '#2a9d8f' : '#55779f' }}>
                    {modularJammingHeuristic ? '● Modular jamming (heuristic)' : '○ Settling / not jammed'}
                  </div>
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
                liveFrameRef={liveFrameRef}
                simRunId={runSignal}
                settledVertexStress={settledVertexStress}
                environmentType={simConfig.ENVIRONMENT_TYPE}
                plateSize={simConfig.PLATE_SIZE}
                wallThickness={simConfig.WALL_THICKNESS}
                cylinderDiameter={simConfig.CYLINDER_DIAMETER}
                cylinderHeight={0.3}
                cylinderSegments={32}
                transparentContainer={transparentContainer}
              />
            </div>
          </div>
          <div style={{ padding: 12 }}>
            <Dashboard keSeries={keSeries} pressureSeries={pressureSeries} metrics={metricsResponse} />
          </div>
        </div>
      </div>
    </div>
  )
}
