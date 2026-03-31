import React, { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import type { ParticleData, ContactPair } from "./types";

// ── Types ──────────────────────────────────────────────────────────────────

interface GraphNode extends d3.SimulationNodeDatum {
  id: number;
  degree: number;
}

interface GraphLink extends d3.SimulationLinkDatum<GraphNode> {
  force: number;
}

export interface ContactGraphProps {
  particles: ParticleData[];
  contactPairs: ContactPair[];
}

// ── Constants ──────────────────────────────────────────────────────────────

const NODE_RADIUS = 6;
const LINK_DISTANCE = 35;
const CHARGE_STRENGTH = -80;
const COLLISION_RADIUS = NODE_RADIUS + 2;

// ── ContactGraph ───────────────────────────────────────────────────────────

/**
 * Renders a D3 force-directed graph of the particle contact network.
 * Nodes represent particles; edges represent contact pairs.
 * Node colour encodes degree (number of contacts): yellow → red (low → high).
 */
export const ContactGraph: React.FC<ContactGraphProps> = ({
  particles,
  contactPairs,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const simulationRef = useRef<d3.Simulation<GraphNode, GraphLink> | null>(null);

  // Track container dimensions so the simulation re-centres on resize.
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });

  // Tooltip state (React-managed for clean teardown).
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    nodeId: number;
    degree: number;
  } | null>(null);

  // ── Resize observer ────────────────────────────────────────────────────

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const ro = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setContainerSize({ width, height });
    });

    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // ── Build / rebuild the D3 simulation ─────────────────────────────────

  useEffect(() => {
    const { width, height } = containerSize;
    if (width === 0 || height === 0 || !svgRef.current) return;

    // Stop any running simulation before rebuilding.
    simulationRef.current?.stop();

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    svg.attr("width", width).attr("height", height);

    // ── Empty state ──────────────────────────────────────────────────────

    if (particles.length === 0) return;

    // ── Build graph data ─────────────────────────────────────────────────

    // Degree = particle-particle contact count (not n_contacts which may
    // include container contacts).
    const degreeMap = new Map<number, number>(particles.map((p) => [p.id, 0]));
    for (const cp of contactPairs) {
      degreeMap.set(cp.particle_a, (degreeMap.get(cp.particle_a) ?? 0) + 1);
      degreeMap.set(cp.particle_b, (degreeMap.get(cp.particle_b) ?? 0) + 1);
    }

    const nodes: GraphNode[] = particles.map((p) => ({
      id: p.id,
      degree: degreeMap.get(p.id) ?? 0,
    }));

    const links: GraphLink[] = contactPairs.map((cp) => ({
      source: cp.particle_a,
      target: cp.particle_b,
      force: cp.force ?? 0,
    }));

    const maxDegree = Math.max(...nodes.map((n) => n.degree), 1);
    const maxForce = Math.max(...links.map((l) => l.force), 1);

    // Yellow (isolated) → orange → red (highly connected / jammed).
    const colorScale = d3
      .scaleSequential(d3.interpolateYlOrRd)
      .domain([0, maxDegree]);

    // ── SVG structure ────────────────────────────────────────────────────

    // Zoom / pan layer.
    const zoomBehaviour = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.2, 8])
      .on("zoom", (event) => {
        g.attr("transform", event.transform);
      });

    svg.call(zoomBehaviour);

    const g = svg.append("g").attr("class", "graph-root");

    // ── Links ────────────────────────────────────────────────────────────

    const linkGroup = g.append("g").attr("class", "links");
    const linkElements = linkGroup
      .selectAll<SVGLineElement, GraphLink>("line")
      .data(links)
      .join("line")
      .attr("stroke", "#4b5563")
      .attr("stroke-opacity", 0.6)
      // Thicker lines for higher-force contacts.
      .attr("stroke-width", (d) =>
        Math.max(0.5, Math.min(3, 0.5 + (d.force / maxForce) * 2.5))
      );

    // ── Nodes ────────────────────────────────────────────────────────────

    const nodeGroup = g.append("g").attr("class", "nodes");
    const nodeElements = nodeGroup
      .selectAll<SVGCircleElement, GraphNode>("circle")
      .data(nodes, (d) => d.id)
      .join("circle")
      .attr("r", NODE_RADIUS)
      .attr("fill", (d) => colorScale(d.degree))
      .attr("stroke", "#1f2937")
      .attr("stroke-width", 0.8)
      .style("cursor", "grab")
      .on("mouseover", (event: MouseEvent, d: GraphNode) => {
        d3.select(event.currentTarget as SVGCircleElement)
          .raise()
          .attr("stroke", "#e2e8f0")
          .attr("stroke-width", 1.5);
        const rect = containerRef.current!.getBoundingClientRect();
        setTooltip({
          x: event.clientX - rect.left,
          y: event.clientY - rect.top,
          nodeId: d.id,
          degree: d.degree,
        });
      })
      .on("mousemove", (event: MouseEvent) => {
        const rect = containerRef.current!.getBoundingClientRect();
        setTooltip((prev) =>
          prev
            ? { ...prev, x: event.clientX - rect.left, y: event.clientY - rect.top }
            : null
        );
      })
      .on("mouseout", (event: MouseEvent) => {
        d3.select(event.currentTarget as SVGCircleElement)
          .attr("stroke", "#1f2937")
          .attr("stroke-width", 0.8);
        setTooltip(null);
      })
      .call(
        d3
          .drag<SVGCircleElement, GraphNode>()
          .on("start", (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on("drag", (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on("end", (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          })
      );

    // ── Force simulation ─────────────────────────────────────────────────

    const simulation = d3
      .forceSimulation<GraphNode, GraphLink>(nodes)
      .force(
        "link",
        d3
          .forceLink<GraphNode, GraphLink>(links)
          .id((d) => d.id)
          .distance(LINK_DISTANCE)
          .strength(0.8)
      )
      .force("charge", d3.forceManyBody<GraphNode>().strength(CHARGE_STRENGTH))
      .force("center", d3.forceCenter(width / 2, height / 2).strength(0.05))
      .force("collision", d3.forceCollide<GraphNode>(COLLISION_RADIUS));

    simulationRef.current = simulation;

    simulation.on("tick", () => {
      linkElements
        .attr("x1", (d) => (d.source as GraphNode).x ?? 0)
        .attr("y1", (d) => (d.source as GraphNode).y ?? 0)
        .attr("x2", (d) => (d.target as GraphNode).x ?? 0)
        .attr("y2", (d) => (d.target as GraphNode).y ?? 0);

      nodeElements
        .attr("cx", (d) => d.x ?? 0)
        .attr("cy", (d) => d.y ?? 0);
    });

    // ── Degree colour legend ─────────────────────────────────────────────

    const LEGEND_W = 140;
    const LEGEND_H = 10;
    const LEGEND_X = width - LEGEND_W - 16;
    const LEGEND_Y = height - 38;

    const defs = svg.append("defs");
    const gradId = "cg-degree-gradient";

    const grad = defs
      .append("linearGradient")
      .attr("id", gradId)
      .attr("x1", "0%")
      .attr("x2", "100%");

    d3.range(0, 1.01, 0.1).forEach((t) => {
      grad
        .append("stop")
        .attr("offset", `${t * 100}%`)
        .attr("stop-color", colorScale(t * maxDegree));
    });

    const legend = svg
      .append("g")
      .attr("class", "legend")
      .attr("transform", `translate(${LEGEND_X},${LEGEND_Y})`);

    legend
      .append("rect")
      .attr("width", LEGEND_W)
      .attr("height", LEGEND_H)
      .attr("rx", 2)
      .style("fill", `url(#${gradId})`);

    const labelStyle = {
      fontSize: "10px",
      fill: "#9ca3af",
      fontFamily: "'Inter', system-ui, sans-serif",
    };

    legend
      .append("text")
      .attr("x", 0)
      .attr("y", -4)
      .attr("text-anchor", "start")
      .style("font-size", labelStyle.fontSize)
      .style("fill", labelStyle.fill)
      .style("font-family", labelStyle.fontFamily)
      .text("0 contacts");

    legend
      .append("text")
      .attr("x", LEGEND_W)
      .attr("y", -4)
      .attr("text-anchor", "end")
      .style("font-size", labelStyle.fontSize)
      .style("fill", labelStyle.fill)
      .style("font-family", labelStyle.fontFamily)
      .text(`${maxDegree} contacts`);

    legend
      .append("text")
      .attr("x", LEGEND_W / 2)
      .attr("y", LEGEND_H + 14)
      .attr("text-anchor", "middle")
      .style("font-size", "9px")
      .style("fill", "#6b7280")
      .style("font-family", labelStyle.fontFamily)
      .text("particle degree (p–p contacts)");

    return () => {
      simulation.stop();
    };
  }, [particles, contactPairs, containerSize]);

  // ── Render ─────────────────────────────────────────────────────────────

  return (
    <div
      ref={containerRef}
      style={{
        position: "relative",
        width: "100%",
        height: "100%",
        overflow: "hidden",
      }}
    >
      {/* Empty state */}
      {particles.length === 0 && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            gap: 8,
            pointerEvents: "none",
          }}
        >
          <svg
            width="40"
            height="40"
            viewBox="0 0 24 24"
            fill="none"
            stroke="#374151"
            strokeWidth="1.5"
          >
            <circle cx="5" cy="12" r="2" />
            <circle cx="19" cy="5" r="2" />
            <circle cx="19" cy="19" r="2" />
            <line x1="7" y1="12" x2="17" y2="6" stroke="#374151" />
            <line x1="7" y1="12" x2="17" y2="18" stroke="#374151" />
          </svg>
          <span style={{ fontSize: 12, color: "#374151" }}>
            Run a simulation to see the contact graph
          </span>
        </div>
      )}

      {/* D3 canvas */}
      <svg
        ref={svgRef}
        style={{ width: "100%", height: "100%", display: "block" }}
      />

      {/* Hover tooltip */}
      {tooltip !== null && (
        <div
          style={{
            position: "absolute",
            left: tooltip.x + 14,
            top: tooltip.y - 10,
            background: "rgba(10,10,20,0.92)",
            border: "1px solid #2d2d3f",
            borderRadius: 6,
            padding: "5px 9px",
            fontSize: 12,
            color: "#e2e8f0",
            pointerEvents: "none",
            whiteSpace: "nowrap",
            lineHeight: 1.6,
          }}
        >
          <span style={{ color: "#94a3b8" }}>Particle </span>
          <strong>{tooltip.nodeId}</strong>
          <br />
          <span style={{ color: "#94a3b8" }}>Contacts </span>
          <strong>{tooltip.degree}</strong>
        </div>
      )}

      {/* Zoom hint */}
      {particles.length > 0 && (
        <span
          style={{
            position: "absolute",
            top: 10,
            right: 12,
            fontSize: 10,
            color: "#374151",
            pointerEvents: "none",
          }}
        >
          scroll to zoom · drag nodes
        </span>
      )}
    </div>
  );
};
