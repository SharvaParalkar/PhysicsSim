import React, { useState, useRef, useEffect, useCallback } from "react";
import { MaterialPresetSlider } from "./MaterialPresets";
import { Viewer3D } from "./Viewer3D";
import { ContactGraph } from "./ContactGraph";
import type { EnvironmentType } from "./Viewer3D";
import type { SimMetrics, SimulationResults } from "./types";

// ── Constants ──────────────────────────────────────────────────────────────

const API_BASE_URL = "http://localhost:8000";
const WEBSOCKET_URL = "ws://localhost:8000/ws";
const DEFAULT_YOUNGS_MODULUS = 1_000_000;

// ── Metric row helper ──────────────────────────────────────────────────────

type MetricRowProps = {
  label: string;
  value: string;
};

const MetricRow: React.FC<MetricRowProps> = ({ label, value }) => (
  <div
    style={{
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
      padding: "6px 10px",
      background: "#0a0a14",
      borderRadius: 6,
      border: "1px solid #2d2d3f",
      gap: 12,
    }}
  >
    <span style={{ fontSize: 12, color: "#94a3b8", flexShrink: 0 }}>{label}</span>
    <span style={{ fontSize: 13, fontWeight: 600, color: "#e2e8f0" }}>{value}</span>
  </div>
);

// ── Section heading helper ─────────────────────────────────────────────────

type SectionHeadingProps = {
  children: React.ReactNode;
};

const SectionHeading: React.FC<SectionHeadingProps> = ({ children }) => (
  <h2
    style={{
      margin: "0 0 10px",
      fontSize: 11,
      fontWeight: 700,
      textTransform: "uppercase",
      letterSpacing: "0.1em",
      color: "#64748b",
    }}
  >
    {children}
  </h2>
);

// ── App ────────────────────────────────────────────────────────────────────

/**
 * Root dashboard for the Granular Simulation tool.
 *
 * Layout:
 *   ┌──────────────────────┬────────────────────────────────┐
 *   │  Sidebar (320 px)    │  3-D Viewer      (top ~60 %)   │
 *   │  · Material slider   ├────────────────────────────────┤
 *   │  · Start button      │  Contact Graph   (btm ~40 %)   │
 *   │  · Metrics           │  (visible only after results)  │
 *   │  · Log output        │                                │
 *   └──────────────────────┴────────────────────────────────┘
 */
export default function App() {
  // ── State ────────────────────────────────────────────────────────────────

  const [simulationLogs, setSimulationLogs] = useState<string[]>([]);
  const [isSimulating, setIsSimulating] = useState<boolean>(false);
  const [metrics, setMetrics] = useState<SimMetrics | null>(null);
  const [E, setE] = useState<number>(DEFAULT_YOUNGS_MODULUS);
  const [environmentType, setEnvironmentType] = useState<EnvironmentType>("cylinder");
  const [results, setResults] = useState<SimulationResults | null>(null);

  // Ref to the active WebSocket so we can close it on unmount.
  const socketRef = useRef<WebSocket | null>(null);
  // Sentinel element at the bottom of the log pane for auto-scrolling.
  const logsBottomRef = useRef<HTMLDivElement | null>(null);

  // Auto-scroll the log pane whenever new lines arrive.
  useEffect(() => {
    logsBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [simulationLogs]);

  // Close any open socket when the component unmounts.
  useEffect(() => {
    return () => {
      socketRef.current?.close();
    };
  }, []);

  // ── fetchResults ─────────────────────────────────────────────────────────

  /**
   * Calls GET /metrics and GET /results in parallel, then stores both in state.
   * Called automatically when the server signals "SIMULATION_DONE".
   */
  const fetchResults = useCallback(async () => {
    try {
      const [metricsResponse, resultsResponse] = await Promise.all([
        fetch(`${API_BASE_URL}/metrics`),
        fetch(`${API_BASE_URL}/results`),
      ]);

      if (!metricsResponse.ok) {
        throw new Error(`GET /metrics failed with status ${metricsResponse.status}`);
      }
      if (!resultsResponse.ok) {
        throw new Error(`GET /results failed with status ${resultsResponse.status}`);
      }

      const metricsData: SimMetrics = await metricsResponse.json();
      // The server key is snake_case; map to our camelCase SimulationResults shape.
      const rawResults = await resultsResponse.json();

      setMetrics(metricsData);
      setResults({
        particles: rawResults.particles,
        contactPairs: rawResults.contact_pairs,
        contactPoints: rawResults.contact_points,
      });
    } catch (fetchError) {
      const errorMessage =
        fetchError instanceof Error ? fetchError.message : String(fetchError);
      setSimulationLogs((previousLogs) => [
        ...previousLogs,
        `[ERROR] Could not fetch results: ${errorMessage}`,
      ]);
    }
  }, []);

  // ── handleStart ──────────────────────────────────────────────────────────

  /**
   * Opens a WebSocket to the backend, sends "start", and streams log lines
   * into `simulationLogs`.  When the server sends "SIMULATION_DONE" the
   * socket is closed and `fetchResults` is called.
   */
  const handleStart = useCallback(() => {
    // Guard: don't open a second connection while one is active.
    if (isSimulating || socketRef.current !== null) return;

    setIsSimulating(true);
    setSimulationLogs([]);
    setMetrics(null);
    setResults(null);

    const socket = new WebSocket(WEBSOCKET_URL);
    socketRef.current = socket;

    socket.onopen = () => {
      socket.send("start");
    };

    socket.onmessage = (event: MessageEvent<string>) => {
      const incomingMessage = event.data;

      if (incomingMessage === "SIMULATION_DONE") {
        socket.close();
        setIsSimulating(false);
        fetchResults();
      } else {
        setSimulationLogs((previousLogs) => [...previousLogs, incomingMessage]);
      }
    };

    socket.onerror = () => {
      setIsSimulating(false);
      setSimulationLogs((previousLogs) => [
        ...previousLogs,
        "[ERROR] WebSocket error — is the server running on port 8000?",
      ]);
    };

    socket.onclose = () => {
      socketRef.current = null;
    };
  }, [isSimulating, fetchResults]);

  // ── Render ───────────────────────────────────────────────────────────────

  return (
    <div
      style={{
        display: "flex",
        height: "100vh",
        overflow: "hidden",
        fontFamily: "'Inter', system-ui, sans-serif",
        background: "#0a0a14",
        color: "#e2e8f0",
      }}
    >
      {/* ── Sidebar ── */}
      <aside
        style={{
          width: 320,
          flexShrink: 0,
          display: "flex",
          flexDirection: "column",
          gap: 20,
          padding: "20px 16px",
          background: "#141420",
          borderRight: "1px solid #1e1e30",
          overflowY: "auto",
        }}
      >
        {/* App title */}
        <div>
          <h1 style={{ margin: 0, fontSize: 17, fontWeight: 700, color: "#a78bfa" }}>
            Granular Simulation
          </h1>
          <p style={{ margin: "4px 0 0", fontSize: 12, color: "#4b5563" }}>
            Contact network analyser
          </p>
        </div>

        {/* ── Material stiffness ── */}
        <section>
          <SectionHeading>Material</SectionHeading>
          <MaterialPresetSlider value={E} onChangeE={setE} />
        </section>

        {/* ── Environment ── */}
        <section>
          <SectionHeading>Environment</SectionHeading>
          <div style={{ display: "flex", gap: 8 }}>
            {(["cylinder", "plate"] as EnvironmentType[]).map((type) => (
              <button
                key={type}
                onClick={() => setEnvironmentType(type)}
                style={{
                  flex: 1,
                  padding: "7px 10px",
                  borderRadius: 6,
                  border: `1px solid ${environmentType === type ? "#7c3aed" : "#2d2d3f"}`,
                  background: environmentType === type ? "#1e1230" : "#0a0a14",
                  color: environmentType === type ? "#a78bfa" : "#64748b",
                  fontSize: 12,
                  fontWeight: 600,
                  cursor: "pointer",
                  textTransform: "capitalize",
                  transition: "all 0.15s",
                }}
              >
                {type}
              </button>
            ))}
          </div>
        </section>

        {/* ── Run control ── */}
        <section>
          <button
            onClick={handleStart}
            disabled={isSimulating}
            style={{
              width: "100%",
              padding: "11px 16px",
              borderRadius: 8,
              border: "none",
              background: isSimulating ? "#1e1e30" : "#7c3aed",
              color: isSimulating ? "#4b5563" : "#ffffff",
              fontSize: 14,
              fontWeight: 600,
              cursor: isSimulating ? "not-allowed" : "pointer",
              transition: "background 0.2s",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: 8,
            }}
          >
            {isSimulating ? (
              <>
                <SpinnerIcon />
                Simulating…
              </>
            ) : (
              "▶  Start Simulation"
            )}
          </button>
        </section>

        {/* ── Metrics ── */}
        {metrics !== null && (
          <section>
            <SectionHeading>Results</SectionHeading>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              <MetricRow
                label="Avg contacts / particle (Z)"
                value={metrics.Z.toFixed(3)}
              />
              <MetricRow
                label="Particle–particle contacts"
                value={metrics.total_pp.toLocaleString()}
              />
              <MetricRow
                label="Particle–container contacts"
                value={metrics.total_pc.toLocaleString()}
              />
              <MetricRow
                label="Isolated particles (Z = 0)"
                value={metrics.n_isolated.toLocaleString()}
              />
              <MetricRow
                label="Touching container wall"
                value={metrics.n_container_touch.toLocaleString()}
              />
            </div>
          </section>
        )}

        {/* ── Simulation log ── */}
        <section
          style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0 }}
        >
          <SectionHeading>
            Simulation Log
            {simulationLogs.length > 0 && (
              <span style={{ marginLeft: 6, color: "#374151", fontWeight: 400 }}>
                ({simulationLogs.length} lines)
              </span>
            )}
          </SectionHeading>

          <div
            style={{
              flex: 1,
              minHeight: 140,
              maxHeight: 300,
              overflowY: "auto",
              background: "#0a0a14",
              borderRadius: 6,
              border: "1px solid #1e1e30",
              padding: "8px 10px",
              fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
              fontSize: 11,
              lineHeight: 1.65,
              color: "#86efac",
            }}
          >
            {simulationLogs.length === 0 ? (
              <span style={{ color: "#374151", fontStyle: "italic" }}>
                No output yet…
              </span>
            ) : (
              simulationLogs.map((logLine, lineIndex) => {
                const isError = logLine.startsWith("[ERROR]");
                return (
                  <div
                    key={lineIndex}
                    style={{ color: isError ? "#f87171" : undefined }}
                  >
                    {logLine}
                  </div>
                );
              })
            )}
            {/* Sentinel for auto-scroll */}
            <div ref={logsBottomRef} />
          </div>
        </section>
      </aside>

      {/* ── Right panel: 3-D viewer + contact graph ── */}
      <main
        style={{
          flex: 1,
          minWidth: 0,
          display: "flex",
          flexDirection: "column",
        }}
      >
        {/* 3-D viewer – shrinks to give room to the graph once results arrive */}
        <div
          style={{
            flex: results ? "3 1 0" : "1 1 0",
            position: "relative",
            minHeight: 0,
            transition: "flex 0.3s ease",
          }}
        >
          <Viewer3D
            particles={results?.particles ?? []}
            contactPairs={results?.contactPairs ?? []}
            contactPoints={results?.contactPoints ?? []}
            environmentType={environmentType}
          />
        </div>

        {/* Contact graph panel – only rendered after results are available */}
        {results && (
          <div
            style={{
              flex: "2 1 0",
              minHeight: 0,
              borderTop: "1px solid #1e1e30",
              background: "#0d0d1a",
              display: "flex",
              flexDirection: "column",
            }}
          >
            {/* Panel header */}
            <div
              style={{
                flexShrink: 0,
                padding: "6px 14px",
                borderBottom: "1px solid #1e1e30",
                display: "flex",
                alignItems: "center",
                gap: 8,
              }}
            >
              <span
                style={{
                  fontSize: 10,
                  fontWeight: 700,
                  textTransform: "uppercase",
                  letterSpacing: "0.1em",
                  color: "#64748b",
                }}
              >
                Contact Graph
              </span>
              <span
                style={{
                  fontSize: 10,
                  color: "#374151",
                  fontFamily: "'JetBrains Mono', monospace",
                }}
              >
                {results.particles.length} nodes · {results.contactPairs.length}{" "}
                edges
              </span>
            </div>

            {/* D3 graph fills the remaining height */}
            <div style={{ flex: 1, minHeight: 0 }}>
              <ContactGraph
                particles={results.particles}
                contactPairs={results.contactPairs}
              />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

// ── Spinner icon (pure CSS, no extra dep) ──────────────────────────────────

const spinnerKeyframes = `
@keyframes spin {
  from { transform: rotate(0deg); }
  to   { transform: rotate(360deg); }
}`;

// Inject the keyframe rule once.
if (typeof document !== "undefined") {
  const styleElement = document.createElement("style");
  styleElement.textContent = spinnerKeyframes;
  document.head.appendChild(styleElement);
}

const SpinnerIcon: React.FC = () => (
  <span
    style={{
      display: "inline-block",
      width: 14,
      height: 14,
      border: "2px solid #374151",
      borderTopColor: "#6d28d9",
      borderRadius: "50%",
      animation: "spin 0.75s linear infinite",
    }}
  />
);
