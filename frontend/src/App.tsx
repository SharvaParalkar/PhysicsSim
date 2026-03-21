import { useCallback, useEffect, useRef, useState } from 'react'
import Viewer3D from './Viewer3D'
import Dashboard from './Dashboard'
import { getSharedSimulationSocket } from './simulationSocket'
import type { MetricsResponse, ParticleData, SimMetrics, SimulationConfig, WsFrameParticle } from './types'

type ResultsPayload = {
  particles: ParticleData[]
}

export default function App() {
  const [mode, setMode] = useState<'idle' | 'falling' | 'settled'>('idle')
  const [logs, setLogs] = useState<string[]>([])
  const [metrics, setMetrics] = useState<SimMetrics>({ Z: 0, total_pp: 0, total_pc: 0, n_isolated: 0, n_container_touch: 0 })
  const [simTime, setSimTime] = useState<number>(0)
  const [runSignal, setRunSignal] = useState(0)
  const [zSeries, setZSeries] = useState<Array<{ t: number; Z: number }>>([])
  const [maxVelSeries, setMaxVelSeries] = useState<Array<{ t: number; max_vel: number }>>([])
  const [metricsResponse, setMetricsResponse] = useState<MetricsResponse>({
    Z: 0,
    total_pp: 0,
    total_pc: 0,
    n_isolated: 0,
    n_container_touch: 0,
    z_history: [],
    max_vel_history: [],
    contact_graph_dict: {},
    contact_graph_links: [],
  })
  const [liveHud, setLiveHud] = useState({ t: 0, Z: 0, max_vel: 0 })
  const [simConfig, setSimConfig] = useState<SimulationConfig>({
    N_PARTICLES: 100,
    YOUNGS_MODULUS: 1e8,
    POISSON_RATIO: 0.45,
    DT: 1 / 400,
    SUBSTEPS: 10,
    SIM_DURATION: 5.0,
    ENVIRONMENT_TYPE: 'plate',
    CYLINDER_DIAMETER: 0.2,
    DROP_HEIGHT: 0.15,
  })
  const logBoxRef = useRef<HTMLPreElement | null>(null)
  const liveTransformsRef = useRef<WsFrameParticle[]>([])
  const pendingStartRef = useRef<string | null>(null)

  const refreshResults = useCallback(async () => {
    const [mRes, rRes] = await Promise.all([fetch('http://localhost:8000/metrics'), fetch('http://localhost:8000/results')])
    const mJson = (await mRes.json()) as MetricsResponse
    const rJson = (await rRes.json()) as ResultsPayload
    setMetrics(mJson)
    setMetricsResponse(mJson)
    return rJson.particles || []
  }, [])

  const appendLog = useCallback((line: string) => {
    setLogs((prev) => [...prev, line])
  }, [])

  const handleRunComplete = useCallback(async () => {
    await refreshResults()
  }, [refreshResults])

  const handleFrameMetrics = useCallback((point: { t: number; Z: number; max_vel: number }) => {
    setLiveHud({ t: point.t, Z: point.Z, max_vel: point.max_vel })
    setZSeries((prev) => [...prev, { t: point.t, Z: point.Z }])
    setMaxVelSeries((prev) => [...prev, { t: point.t, max_vel: point.max_vel }])
  }, [])

  const onSimTimeRef = useRef(setSimTime)
  const onFrameMetricsRef = useRef(handleFrameMetrics)
  const onLogsRef = useRef(appendLog)
  const onModeChangeRef = useRef(setMode)
  const onRunCompleteRef = useRef(handleRunComplete)
  const onCancelledRef = useRef<() => void>(() => {})

  useEffect(() => {
    onSimTimeRef.current = setSimTime
  }, [setSimTime])
  useEffect(() => {
    onFrameMetricsRef.current = handleFrameMetrics
  }, [handleFrameMetrics])
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
    onCancelledRef.current = () => {
      appendLog('Simulation cancelled')
      setMode('idle')
    }
  }, [appendLog, setMode])

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
        particles?: WsFrameParticle[]
        Z?: number
        max_vel?: number
      }
      if (msg.type === 'log' && msg.line) {
        onLogsRef.current(msg.line)
        return
      }
      if (msg.type === 'frame') {
        liveTransformsRef.current = msg.particles || []
        onSimTimeRef.current(typeof msg.t === 'number' ? msg.t : 0)
        onFrameMetricsRef.current({
          t: typeof msg.t === 'number' ? msg.t : 0,
          Z: typeof msg.Z === 'number' ? msg.Z : 0,
          max_vel: typeof msg.max_vel === 'number' ? msg.max_vel : 0,
        })
        return
      }
      if (msg.type === 'error') {
        onLogsRef.current(`[error] ${msg.message ?? 'unknown error'}`)
        onModeChangeRef.current('idle')
        return
      }
      if (msg.type === 'complete') {
        onLogsRef.current('Simulation complete')
        await onRunCompleteRef.current(liveTransformsRef.current as unknown as ParticleData[])
        onModeChangeRef.current('settled')
        return
      }
      if (msg.type === 'idle') {
        return
      }
      if (msg.type === 'cancelled') {
        onCancelledRef.current()
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
      config: simConfigRef.current,
    })
    setMode('falling')
    setSimTime(0)
    liveTransformsRef.current = []
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
    gridTemplateColumns: '1fr 1fr',
    gap: 10,
    background: '#edf4fc',
    border: '1px solid #d4dfec',
    borderRadius: 12,
    padding: 12,
  }

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
            style={buttonStyle}
            onClick={() => {
              if (mode === 'falling') return
              setLogs([])
              setSimTime(0)
              setZSeries([])
              setMaxVelSeries([])
              setLiveHud({ t: 0, Z: 0, max_vel: 0 })
              setRunSignal((v) => v + 1)
            }}
          >
            {mode === 'falling' ? 'Running…' : 'Run Simulation'}
          </button>
          <button
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
            <div style={{ ...labelStyle, marginTop: 4 }}>Simulation Config</div>
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
              YOUNGS_MODULUS (Pa, ≤1e8 FEM): {simConfig.YOUNGS_MODULUS.toExponential(2)}
              <input
                type="range"
                min={4}
                max={8}
                step={0.01}
                value={Math.log10(Math.max(1e4, Math.min(1e8, simConfig.YOUNGS_MODULUS)))}
                onChange={(e) =>
                  setSimConfig((p) => ({
                    ...p,
                    YOUNGS_MODULUS: Math.min(1e8, 10 ** Number(e.target.value)),
                  }))
                }
                style={{ width: '100%' }}
              />
            </label>
            <label style={labelStyle}>
              DT (s): {simConfig.DT.toFixed(5)} (default 1/400)
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
                onChange={(e) => setSimConfig((p) => ({ ...p, ENVIRONMENT_TYPE: e.target.value as SimulationConfig['ENVIRONMENT_TYPE'] }))}
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
          </div>

          <div>
            <div style={{ ...labelStyle, marginBottom: 6 }}>Metrics (post-run)</div>
            <div style={metricsGridStyle}>
              <div>
                <div style={labelStyle}>Z</div>
                <div style={valueStyle}>{metrics.Z.toFixed(3)}</div>
              </div>
              <div>
                <div style={labelStyle}>total_pp</div>
                <div style={valueStyle}>{metrics.total_pp}</div>
              </div>
              <div>
                <div style={labelStyle}>total_pc</div>
                <div style={valueStyle}>{metrics.total_pc}</div>
              </div>
              <div>
                <div style={labelStyle}>n_isolated</div>
                <div style={valueStyle}>{metrics.n_isolated}</div>
              </div>
            </div>
          </div>

          <div style={{ flex: 1, minHeight: 0, minWidth: 0, display: 'flex', flexDirection: 'column' }}>
            <div style={{ ...labelStyle, marginBottom: 6 }}>Log</div>
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

          <div style={{ minHeight: 380, position: 'relative' }}>
            <div
              style={{
                position: 'absolute',
                top: 10,
                right: 10,
                zIndex: 2,
                background: 'rgba(255,255,255,0.9)',
                border: '1px solid #d4dfec',
                borderRadius: 10,
                padding: '10px 12px',
                fontSize: 12,
                color: '#1d3553',
                boxShadow: '0 4px 12px rgba(20,40,80,0.12)',
                pointerEvents: 'none',
              }}
            >
              <div style={{ fontWeight: 800, marginBottom: 6 }}>Live HUD</div>
              <div style={{ display: 'grid', gap: 4 }}>
                <div>
                  <span style={{ color: '#55779f' }}>t </span>
                  {liveHud.t.toFixed(3)} s
                </div>
                <div>
                  <span style={{ color: '#55779f' }}>Z </span>
                  {liveHud.Z.toFixed(3)}
                </div>
                <div>
                  <span style={{ color: '#55779f' }}>max |v| </span>
                  {liveHud.max_vel.toFixed(5)} m/s
                </div>
              </div>
            </div>
            <Viewer3D particleCount={simConfig.N_PARTICLES} liveTransformsRef={liveTransformsRef} />
          </div>
          <div style={{ padding: 12 }}>
            <Dashboard zSeries={zSeries} maxVelSeries={maxVelSeries} metrics={metricsResponse} />
          </div>
        </div>
      </div>
    </div>
  )
}
