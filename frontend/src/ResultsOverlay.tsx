import { createPortal } from 'react-dom'
import { useEffect, useRef, useState } from 'react'

// ── Types ─────────────────────────────────────────────────────────────────────

export interface ExportParticle {
  id: number
  x: number | null
  y: number | null
  z: number | null
  qx: number | null
  qy: number | null
  qz: number | null
  qw: number | null
  n_contacts: number | null
  neighbors: number[]
}

export interface ExportContact {
  particle_a: number | null
  particle_b: number | null
  depth: number | null
  force: number | null
  px: number | null
  py: number | null
  pz: number | null
  nx: number | null
  ny: number | null
  nz: number | null
}

export interface ExportSummary {
  n_particles?: number
  Z?: number
  total_pp?: number
  total_pc?: number
  n_isolated?: number
  n_container_touch?: number
  system_pressure?: number
  contact_efficiency?: number
  total_particle_volume?: number
  single_particle_volume_m3?: number
  timestamp?: string
}

export interface ExportPayload {
  summary: ExportSummary
  particles: ExportParticle[]
  contacts: ExportContact[]
}

interface Props {
  screenshot: string | null
  data: ExportPayload
  onClose: () => void
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmt(v: number | null | undefined, decimals = 4): string {
  if (v == null || !Number.isFinite(v)) return '—'
  return v.toFixed(decimals)
}

function fmtSci(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return '—'
  const a = Math.abs(v)
  if (a === 0) return '0'
  if (a >= 1e4 || (a < 1e-2 && a > 0)) return v.toExponential(3)
  return v.toFixed(4)
}

function fmtInt(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return '—'
  return String(Math.round(v))
}

const BASE = 'http://localhost:8000'

const PORTAL_ID = 'results-print-portal'

function getOrCreatePortalRoot(): HTMLElement {
  let el = document.getElementById(PORTAL_ID)
  if (!el) {
    el = document.createElement('div')
    el.id = PORTAL_ID
    document.body.appendChild(el)
  }
  return el
}

// ── Sub-components ────────────────────────────────────────────────────────────

function StatTile({ label, value }: { label: string; value: string }) {
  return (
    <div style={{
      background: 'white',
      border: '1px solid #d4dfec',
      borderRadius: 10,
      padding: '14px 16px',
      minWidth: 0,
    }}>
      <div style={{ fontSize: 11, color: '#55779f', marginBottom: 4, lineHeight: 1.3 }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 800, color: '#1d3553', lineHeight: 1.1 }}>{value}</div>
    </div>
  )
}

function SectionHeader({ children }: { children: React.ReactNode }) {
  return (
    <div className="ro-section-header" style={{
      fontSize: 11,
      fontWeight: 800,
      letterSpacing: '0.08em',
      color: '#55779f',
      textTransform: 'uppercase' as const,
      borderBottom: '1px solid #d4dfec',
      paddingBottom: 6,
      marginBottom: 12,
      marginTop: 24,
    }}>
      {children}
    </div>
  )
}

function DownloadBtn({ href, label, prominent }: { href: string; label: string; prominent?: boolean }) {
  if (prominent) {
    return (
      <a
        href={href}
        className="ro-no-print"
        style={{
          display: 'block',
          background: '#1d3553',
          color: 'white',
          borderRadius: 10,
          padding: '14px 20px',
          fontSize: 14,
          fontWeight: 700,
          width: '100%',
          textAlign: 'center' as const,
          textDecoration: 'none',
          boxSizing: 'border-box' as const,
          transition: 'background 0.15s',
        }}
        onMouseEnter={(e) => (e.currentTarget.style.background = '#274465')}
        onMouseLeave={(e) => (e.currentTarget.style.background = '#1d3553')}
      >
        {label}
      </a>
    )
  }
  return (
    <a
      href={href}
      className="ro-no-print"
      style={{
        display: 'inline-block',
        background: 'white',
        border: '1px solid #bccbe0',
        borderRadius: 8,
        padding: '8px 14px',
        fontSize: 12,
        fontWeight: 600,
        color: '#1d3553',
        textDecoration: 'none',
        transition: 'background 0.15s',
        whiteSpace: 'nowrap' as const,
      }}
      onMouseEnter={(e) => (e.currentTarget.style.background = '#eaf1f9')}
      onMouseLeave={(e) => (e.currentTarget.style.background = 'white')}
    >
      ↓ {label}
    </a>
  )
}

const thStyle: React.CSSProperties = {
  background: '#eaf1f9',
  fontSize: 11,
  fontWeight: 700,
  color: '#1d3553',
  padding: '6px 8px',
  textAlign: 'left' as const,
  position: 'sticky' as const,
  top: 0,
  whiteSpace: 'nowrap' as const,
}

const tdStyle = (even: boolean): React.CSSProperties => ({
  fontSize: 12,
  color: '#2d4a6a',
  borderBottom: '1px solid #eaf1f9',
  padding: '5px 8px',
  background: even ? '#f8fbff' : 'white',
  fontVariantNumeric: 'tabular-nums',
  whiteSpace: 'nowrap' as const,
})

// ── Print style tag ───────────────────────────────────────────────────────────
// Injected once into <head>. Hides the entire React root (#root) during print
// and reveals only the portal that lives directly in <body>.

const PRINT_CSS = `
@page { size: A4 landscape; margin: 16mm; }
@media print {
  #root { display: none !important; }
  #${PORTAL_ID} { display: block !important; }
  .ro-backdrop {
    position: static !important;
    background: none !important;
    display: block !important;
    padding: 0 !important;
    align-items: unset !important;
    justify-content: unset !important;
  }
  .ro-panel {
    position: static !important;
    max-height: none !important;
    overflow: visible !important;
    border: none !important;
    box-shadow: none !important;
    border-radius: 0 !important;
    width: 100% !important;
    padding: 0 !important;
  }
  .ro-no-print { display: none !important; }
  .ro-scrollable { max-height: none !important; overflow: visible !important; }
  .ro-print-footer { display: block !important; font-size: 9pt; color: #55779f; }
  .ro-section-header { font-size: 14pt !important; }
  th, td { font-size: 10pt !important; }
}
@media screen {
  .ro-print-footer { display: none; }
  #${PORTAL_ID} { display: contents; }
}
`

function ensurePrintStyles() {
  if (document.getElementById('ro-print-styles')) return
  const tag = document.createElement('style')
  tag.id = 'ro-print-styles'
  tag.textContent = PRINT_CSS
  document.head.appendChild(tag)
}

// ── Main component ────────────────────────────────────────────────────────────

export default function ResultsOverlay({ screenshot, data, onClose }: Props) {
  const { summary, particles, contacts } = data
  const [portalRoot] = useState(getOrCreatePortalRoot)
  const [jsonExpanded, setJsonExpanded] = useState(false)

  useEffect(() => {
    ensurePrintStyles()
  }, [])

  const handlePrint = () => window.print()

  const handleExportAll = () => {
    const json = JSON.stringify(data, null, 2)
    const blob = new Blob([json], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `simulation_export_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const z = summary?.Z ?? 0
  const pp = summary?.total_pp ?? 0
  const pc = summary?.total_pc ?? 0
  const isolated = summary?.n_isolated ?? 0
  const wallTouch = summary?.n_container_touch ?? 0
  const contactEff = summary?.contact_efficiency ?? 0
  const nParticles = summary?.n_particles ?? particles.length
  const ts = summary?.timestamp ?? ''

  const overlay = (
    /* ── Backdrop ──────────────────────────────────────────────────────── */
    <div
      className="ro-backdrop"
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 1000,
        background: 'rgba(10, 20, 40, 0.72)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
      onClick={(e) => { if (e.target === e.currentTarget) onClose() }}
    >
      {/* ── Panel ──────────────────────────────────────────────────────── */}
      <div
        className="ro-panel"
        style={{
          background: '#f8fbff',
          border: '1px solid #d4dfec',
          borderRadius: 16,
          width: 'min(860px, 96vw)',
          maxHeight: '92vh',
          overflowY: 'auto',
          padding: 28,
          position: 'relative',
          boxShadow: '0 24px 64px rgba(10,20,60,0.3)',
        }}
      >
        {/* ── Header row ─────────────────────────────────────────────── */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16, paddingRight: 36 }}>
          <div style={{ fontSize: 16, fontWeight: 900, color: '#1d3553', letterSpacing: 0.3, flex: 1 }}>
            Granular Jamming Results
          </div>
          <button
            type="button"
            className="ro-no-print"
            onClick={handleExportAll}
            style={{
              background: '#1d3553',
              border: 'none',
              fontSize: 12,
              padding: '6px 14px',
              borderRadius: 8,
              cursor: 'pointer',
              color: 'white',
              fontWeight: 700,
            }}
            onMouseEnter={(e) => (e.currentTarget.style.background = '#274465')}
            onMouseLeave={(e) => (e.currentTarget.style.background = '#1d3553')}
            title="Download full simulation_export.json — importable into Analysis.html"
          >
            ↓ Export All Data
          </button>
          <button
            type="button"
            className="ro-no-print"
            onClick={handlePrint}
            style={{
              background: 'white',
              border: '1px solid #bccbe0',
              fontSize: 12,
              padding: '6px 14px',
              borderRadius: 8,
              cursor: 'pointer',
              color: '#1d3553',
              fontWeight: 600,
            }}
          >
            Print
          </button>
        </div>

        {/* ── Close button ───────────────────────────────────────────── */}
        <button
          type="button"
          className="ro-no-print"
          onClick={onClose}
          style={{
            position: 'absolute',
            top: 16,
            right: 16,
            background: 'transparent',
            border: 'none',
            fontSize: 20,
            color: '#55779f',
            cursor: 'pointer',
            lineHeight: 1,
            padding: 4,
          }}
          aria-label="Close"
        >
          ✕
        </button>

        {/* ── Screenshot ─────────────────────────────────────────────── */}
        {screenshot ? (
          <img
            src={screenshot}
            alt="Settled simulation"
            style={{
              width: '100%',
              aspectRatio: '16/9',
              objectFit: 'cover',
              borderRadius: 8,
              border: '1px solid #d4dfec',
              display: 'block',
              marginBottom: 4,
            }}
          />
        ) : (
          <div style={{
            width: '100%',
            aspectRatio: '16/9',
            borderRadius: 8,
            border: '1px solid #d4dfec',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: '#eaf1f9',
            color: '#8aa3be',
            fontSize: 13,
            marginBottom: 4,
          }}>
            Screenshot not available
          </div>
        )}

        {/* ── Population summary ─────────────────────────────────────── */}
        <SectionHeader>Population Summary</SectionHeader>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(3, 1fr)',
          gap: 10,
          marginBottom: 4,
        }}>
          <StatTile label="P–P Contacts" value={fmtInt(pp)} />
          <StatTile label="P–Container Contacts" value={fmtInt(pc)} />
          <StatTile label="Avg Z (per particle)" value={fmt(z, 2)} />
          <StatTile label="Isolated Particles" value={fmtInt(isolated)} />
          <StatTile label="Container Touching" value={fmtInt(wallTouch)} />
          <StatTile label="Contact Efficiency /m³" value={fmtSci(contactEff)} />
        </div>

        {/* ── Per-particle table ──────────────────────────────────────── */}
        <SectionHeader>Per-Particle Data</SectionHeader>
        <div className="ro-scrollable" style={{ maxHeight: 220, overflowY: 'auto', borderRadius: 8, border: '1px solid #d4dfec' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                {['ID', 'X', 'Y', 'Z', 'Z#', 'Neighbors'].map((h) => (
                  <th key={h} style={thStyle}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {particles.map((p, i) => (
                <tr key={p.id}>
                  <td style={tdStyle(i % 2 === 1)}>{p.id}</td>
                  <td style={tdStyle(i % 2 === 1)}>{fmt(p.x, 4)}</td>
                  <td style={tdStyle(i % 2 === 1)}>{fmt(p.y, 4)}</td>
                  <td style={tdStyle(i % 2 === 1)}>{fmt(p.z, 4)}</td>
                  <td style={tdStyle(i % 2 === 1)}>{p.n_contacts ?? '—'}</td>
                  <td style={{ ...tdStyle(i % 2 === 1), maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {p.neighbors.length > 0
                      ? p.neighbors.slice(0, 6).join(', ') + (p.neighbors.length > 6 ? '…' : '')
                      : '—'}
                  </td>
                </tr>
              ))}
              {particles.length === 0 && (
                <tr><td colSpan={6} style={{ ...tdStyle(false), textAlign: 'center', color: '#8aa3be' }}>No data</td></tr>
              )}
            </tbody>
          </table>
        </div>

        {/* ── Contact data table ─────────────────────────────────────── */}
        <SectionHeader>Contact Data</SectionHeader>
        <div className="ro-scrollable" style={{ maxHeight: 200, overflowY: 'auto', borderRadius: 8, border: '1px solid #d4dfec' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                {['Pair', 'Depth mm', 'Force N', 'Contact Point'].map((h) => (
                  <th key={h} style={thStyle}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {contacts.map((c, i) => {
                const depthMm = c.depth != null && Number.isFinite(c.depth) ? (c.depth * 1000).toFixed(3) : '—'
                const forceN = c.force != null && Number.isFinite(c.force) ? Number(c.force).toFixed(4) : '—'
                const pt = (c.px != null && c.py != null && c.pz != null)
                  ? `(${fmt(c.px, 3)}, ${fmt(c.py, 3)}, ${fmt(c.pz, 3)})`
                  : '—'
                return (
                  <tr key={i}>
                    <td style={tdStyle(i % 2 === 1)}>{c.particle_a}↔{c.particle_b}</td>
                    <td style={tdStyle(i % 2 === 1)}>{depthMm}</td>
                    <td style={tdStyle(i % 2 === 1)}>{forceN}</td>
                    <td style={{ ...tdStyle(i % 2 === 1), fontFamily: 'monospace', fontSize: 11 }}>{pt}</td>
                  </tr>
                )
              })}
              {contacts.length === 0 && (
                <tr><td colSpan={4} style={{ ...tdStyle(false), textAlign: 'center', color: '#8aa3be' }}>No contact data</td></tr>
              )}
            </tbody>
          </table>
        </div>

        {/* ── Downloads ──────────────────────────────────────────────── */}
        <SectionHeader>Downloads</SectionHeader>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 10, marginBottom: 12 }}>
          <DownloadBtn href={`${BASE}/download/particles-csv`} label="particles.csv" />
          <DownloadBtn href={`${BASE}/download/contacts-csv`} label="contact_pairs.csv" />
          <DownloadBtn href={`${BASE}/download/contact-points-csv`} label="contact_points.csv" />
          <DownloadBtn href={`${BASE}/download/summary-json`} label="summary.json" />
        </div>
        <DownloadBtn
          href={`${BASE}/download/obj`}
          label={`↓  DOWNLOAD SETTLED GEOMETRY (.OBJ)  —  ${nParticles} particles · world-space coords + contact_network.obj in ZIP`}
          prominent
        />

        {/* ── JSON Statistics ─────────────────────────────────────────── */}
        <SectionHeader>JSON Statistics</SectionHeader>
        <button
          type="button"
          className="ro-no-print"
          onClick={() => setJsonExpanded(v => !v)}
          style={{
            width: '100%',
            background: '#f0f7ff',
            border: '1px solid #d4dfec',
            borderRadius: 8,
            padding: '8px 14px',
            fontSize: 12,
            fontWeight: 600,
            color: '#1d3553',
            cursor: 'pointer',
            textAlign: 'left',
            marginBottom: jsonExpanded ? 0 : 4,
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <span>Summary Statistics (JSON)</span>
          <span style={{ fontSize: 10, color: '#8aa3be' }}>{jsonExpanded ? '▲ collapse' : '▼ expand'}</span>
        </button>
        {jsonExpanded && (
          <div style={{
            background: '#f8fbff',
            border: '1px solid #d4dfec',
            borderTop: 'none',
            borderRadius: '0 0 8px 8px',
            padding: '12px 14px',
            maxHeight: 260,
            overflowY: 'auto',
            marginBottom: 8,
          }}>
            <pre style={{
              margin: 0,
              fontSize: 11,
              fontFamily: '"JetBrains Mono", "Fira Code", monospace',
              color: '#1d3553',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-all',
              lineHeight: 1.6,
            }}>
              {JSON.stringify({
                n_particles: nParticles,
                Z_avg_coordination: z,
                total_pp_contacts: pp,
                total_pc_contacts: pc,
                n_isolated_rattlers: isolated,
                n_container_touching: wallTouch,
                contact_efficiency_per_m3: contactEff,
                system_pressure_pa: summary?.system_pressure ?? null,
                total_particle_volume_m3: summary?.total_particle_volume ?? null,
                single_particle_volume_m3: summary?.single_particle_volume_m3 ?? null,
                n_contact_pairs: contacts.length,
                timestamp: ts || null,
              }, null, 2)}
            </pre>
          </div>
        )}
        <p className="ro-no-print" style={{ fontSize: 11, color: '#8aa3be', marginBottom: 4 }}>
          ↳ Use <strong>Export All Data</strong> to download the full payload (summary + per-particle + contacts), then drag it into <strong>Analysis.html</strong> for advanced visualization.
        </p>

        {/* ── Print footer ───────────────────────────────────────────── */}
        <div className="ro-print-footer" style={{ marginTop: 24, borderTop: '1px solid #d4dfec', paddingTop: 8 }}>
          {ts ? `Simulation: ${ts}` : ''} &nbsp;·&nbsp; {nParticles} particles &nbsp;·&nbsp; Avg Z = {fmt(z, 2)}
        </div>
      </div>
    </div>
  )

  return createPortal(overlay, portalRoot)
}
