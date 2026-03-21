import { useEffect, useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { ContactGraphLink, MetricsResponse } from './types'

function normalizeGraphLinks(metrics: MetricsResponse): ContactGraphLink[] {
  if (metrics.contact_graph_links?.length) return metrics.contact_graph_links
  const links: ContactGraphLink[] = []
  for (const [source, neighbors] of Object.entries(metrics.contact_graph_dict || {})) {
    for (const [target, attr] of Object.entries(neighbors || {})) {
      const s = Number(source)
      const t = Number(target)
      if (!Number.isFinite(s) || !Number.isFinite(t) || s >= t) continue
      links.push({
        source: s,
        target: t,
        depth: Number(attr?.depth ?? 0),
        force: typeof attr?.force === 'number' ? attr.force : null,
        area: typeof attr?.area === 'number' ? attr.area : null,
      })
    }
  }
  return links
}

export default function Dashboard(props: {
  zSeries: Array<{ t: number; Z: number }>
  maxVelSeries: Array<{ t: number; max_vel: number }>
  metrics: MetricsResponse
}) {
  const { zSeries, maxVelSeries, metrics } = props
  const [open, setOpen] = useState({ z: true, velocity: true, chains: false })
  const graphRef = useRef<SVGSVGElement | null>(null)
  const links = useMemo(() => normalizeGraphLinks(metrics), [metrics])
  const forceValues = useMemo(
    () => links.map((l) => (typeof l.force === 'number' ? Math.abs(l.force) : NaN)).filter((v) => Number.isFinite(v)),
    [links]
  )
  const minForce = forceValues.length ? Math.min(...forceValues) : 0
  const maxForce = forceValues.length ? Math.max(...forceValues) : 1

  useEffect(() => {
    const svgEl = graphRef.current
    if (!svgEl) return
    const width = 480
    const height = 180
    const nodesSet = new Set<number>()
    for (const l of links) {
      nodesSet.add(Number(l.source))
      nodesSet.add(Number(l.target))
    }
    const nodes = Array.from(nodesSet).map((id) => ({ id }))
    const simLinks = links.map((l) => ({ ...l })) as Array<d3.SimulationLinkDatum<{ id: number }>>

    const svg = d3.select(svgEl)
    svg.selectAll('*').remove()
    svg.attr('viewBox', `0 0 ${width} ${height}`)
    if (!nodes.length) {
      svg.append('text').attr('x', width / 2).attr('y', height / 2).attr('text-anchor', 'middle').attr('fill', '#5b7598').text('No contact graph yet')
      return
    }

    const forceDomainMax = maxForce > minForce ? maxForce : minForce + 1e-6
    const forceToColor = d3.scaleSequential(d3.interpolateTurbo).domain([minForce, forceDomainMax])
    const forceToWidth = d3.scaleLinear().domain([minForce, forceDomainMax]).range([1.0, 4.0]).clamp(true)
    const simulation = d3
      .forceSimulation(nodes)
      .force('link', d3.forceLink(simLinks).id((d) => String((d as { id: number }).id)).distance(24))
      .force('charge', d3.forceManyBody().strength(-80))
      .force('center', d3.forceCenter(width / 2, height / 2))

    const edge = svg
      .append('g')
      .selectAll('line')
      .data(simLinks)
      .enter()
      .append('line')
      .attr('stroke', (d) => {
        const f = Number((d as ContactGraphLink).force)
        if (Number.isFinite(f)) return forceToColor(Math.abs(f))
        return '#6e7b9c'
      })
      .attr('stroke-opacity', 0.75)
      .attr('stroke-width', (d) => {
        const f = Number((d as ContactGraphLink).force)
        if (Number.isFinite(f)) return forceToWidth(Math.abs(f))
        return Math.max(1, Math.min(3.5, Number((d as ContactGraphLink).depth || 0) * 2000))
      })

    const node = svg.append('g').selectAll('circle').data(nodes).enter().append('circle').attr('r', 4).attr('fill', '#3a86ff')
    simulation.on('tick', () => {
      edge
        .attr('x1', (d) => ((d.source as { x?: number }).x ?? width / 2))
        .attr('y1', (d) => ((d.source as { y?: number }).y ?? height / 2))
        .attr('x2', (d) => ((d.target as { x?: number }).x ?? width / 2))
        .attr('y2', (d) => ((d.target as { y?: number }).y ?? height / 2))
      node.attr('cx', (d) => (d as { x?: number }).x ?? width / 2).attr('cy', (d) => (d as { y?: number }).y ?? height / 2)
    })
    return () => simulation.stop()
  }, [links, minForce, maxForce])

  const cardStyle: React.CSSProperties = { background: '#f8fbff', border: '1px solid #d4dfec', borderRadius: 10, padding: 10 }
  const panelHeader = (title: string, key: keyof typeof open) => (
    <button
      onClick={() => setOpen((p) => ({ ...p, [key]: !p[key] }))}
      style={{ width: '100%', border: 'none', background: 'transparent', color: '#1d3553', fontWeight: 700, textAlign: 'left', display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer', padding: 0, marginBottom: 8 }}
    >
      <span>{title}</span>
      <span style={{ color: '#55779f', fontSize: 12 }}>{open[key] ? 'Collapse' : 'Expand'}</span>
    </button>
  )

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
      <div style={cardStyle}>
        {panelHeader('Coordination Number (Z) over time', 'z')}
        {open.z ? (
          <div style={{ width: '100%', height: 140 }}>
            <ResponsiveContainer>
              <LineChart data={zSeries}>
                <XAxis dataKey="t" tick={{ fill: '#4f6f93', fontSize: 11 }} />
                <YAxis tick={{ fill: '#4f6f93', fontSize: 11 }} />
                <Tooltip />
                <Line type="monotone" dataKey="Z" stroke="#3a86ff" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : null}
      </div>
      <div style={cardStyle}>
        {panelHeader('Max velocity over time', 'velocity')}
        {open.velocity ? (
          <div style={{ width: '100%', height: 140 }}>
            <ResponsiveContainer>
              <LineChart data={maxVelSeries}>
                <XAxis dataKey="t" tick={{ fill: '#4f6f93', fontSize: 11 }} />
                <YAxis tick={{ fill: '#4f6f93', fontSize: 11 }} />
                <Tooltip />
                <Line type="monotone" dataKey="max_vel" stroke="#ff6b35" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : null}
      </div>
      <div style={{ ...cardStyle, gridColumn: '1 / span 2' }}>
        {panelHeader('Contact force chains (D3)', 'chains')}
        {open.chains ? (
          <>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
              <div style={{ color: '#5b7598', fontSize: 11 }}>low force</div>
              <div style={{ flex: 1, height: 8, borderRadius: 999, background: 'linear-gradient(90deg, #30123b 0%, #28bceb 50%, #f9d423 75%, #f44f38 100%)' }} />
              <div style={{ color: '#5b7598', fontSize: 11 }}>high force</div>
            </div>
            <div style={{ color: '#5b7598', fontSize: 11, marginBottom: 8 }}>|F| range: {minForce.toExponential(2)} to {maxForce.toExponential(2)}</div>
            <svg ref={graphRef} style={{ width: '100%', height: 170 }} />
          </>
        ) : (
          <div style={{ color: '#5b7598', fontSize: 12 }}>Expand to view the force-chain graph.</div>
        )}
      </div>
    </div>
  )
}
