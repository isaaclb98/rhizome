import { useEffect, useRef, useMemo } from 'react';
import * as d3 from 'd3';

const DOMAIN_COLORS = {
  Modernism: '#6366f1',
  Postmodernism: '#a855f7',
  'Critical theory': '#f97316',
  Unknown: '#6b7280',
};

function getDomainColor(domain) {
  return DOMAIN_COLORS[domain] || DOMAIN_COLORS.Unknown;
}

export default function Graph({ path, selectedChunkId, onNodeClick }) {
  const svgRef = useRef(null);
  const containerRef = useRef(null);

  const { nodes, pathEdges, knnEdges } = useMemo(() => {
    const nodes = path.map((step, i) => ({
      id: step.chunk_id,
      label: step.article_title.length > 20
        ? step.article_title.slice(0, 20) + '…'
        : step.article_title,
      fullTitle: step.article_title,
      domain: step.domain,
      color: getDomainColor(step.domain),
      stepIndex: i,
      forced_jump: step.forced_jump,
      similarity: step.similarity,
    }));

    const pathEdges = [];
    for (let i = 0; i < path.length - 1; i++) {
      pathEdges.push({
        source: path[i].chunk_id,
        target: path[i + 1].chunk_id,
        forced: path[i].forced_jump,
      });
    }

    // kNN edges — only within path
    const nodeIds = new Set(nodes.map(n => n.id));
    const knnEdges = [];
    for (let i = 0; i < path.length; i++) {
      for (const cand of (path[i].candidates || [])) {
        if (nodeIds.has(cand.chunk_id) && cand.chunk_id !== path[i + 1]?.chunk_id) {
          knnEdges.push({
            source: path[i].chunk_id,
            target: cand.chunk_id,
          });
        }
      }
    }

    return { nodes, pathEdges, knnEdges };
  }, [path]);

  useEffect(() => {
    if (!svgRef.current || !containerRef.current || nodes.length === 0) return;

    const container = containerRef.current;
    const { width, height } = container.getBoundingClientRect();
    if (width === 0 || height === 0) return;

    d3.select(svgRef.current).selectAll('*').remove();

    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height);

    const nodeCount = nodes.length;
    const padding = { top: 16, bottom: 16 };
    const innerHeight = height - padding.top - padding.bottom;

    // Pre-fix y positions: distribute nodes evenly top-to-bottom
    const nodeById = new Map();
    nodes.forEach((n, i) => {
      const y = padding.top + (i / Math.max(nodeCount - 1, 1)) * innerHeight;
      // x is centered with small jitter based on step index to separate overlapping nodes
      const jitter = ((i % 3) - 1) * (width * 0.08);
      nodeById.set(n.id, { ...n, x: width / 2 + jitter, y });
    });

    // Very weak simulation only for x spread, y is fixed
    const simulation = d3.forceSimulation(nodes)
      .force('x', d3.forceX((d) => nodeById.get(d.id).x).strength(0.6))
      .force('y', d3.forceY((d) => nodeById.get(d.id).y).strength(0.95))
      .force('charge', d3.forceManyBody().strength(-20))
      .alphaDecay(0.08);

    // Draw kNN edges (behind)
    const knnLine = svg.append('g').selectAll('line')
      .data(knnEdges)
      .join('line')
      .attr('stroke', '#3f4155')
      .attr('stroke-width', 0.5)
      .attr('opacity', 0.4)
      .attr('stroke-dasharray', '2,3');

    // Draw path edges
    const pathLine = svg.append('g').selectAll('line')
      .data(pathEdges)
      .join('line')
      .attr('stroke', (d) => d.forced ? '#f97316' : '#525766')
      .attr('stroke-width', (d) => d.forced ? 1.5 : 1)
      .attr('stroke-dasharray', (d) => d.forced ? '4,3' : null)
      .attr('opacity', 0.9);

    // Draw nodes as compact circles
    const node = svg.append('g').selectAll('g')
      .data(nodes)
      .join('g')
      .attr('cursor', 'pointer')
      .call(
        d3.drag()
          .on('start drag end', (event, d) => {
            if (event.sourceEvent) {
              nodeById.get(d.id).x = event.x;
              nodeById.get(d.id).y = Math.max(padding.top, Math.min(height - padding.bottom, event.y));
              simulation.alpha(0.1).restart();
            }
          })
      );

    node.append('circle')
      .attr('r', 6)
      .attr('fill', (d) => d.color)
      .attr('stroke', (d) => d.id === selectedChunkId ? '#fff' : '#0f1117')
      .attr('stroke-width', (d) => d.id === selectedChunkId ? 2.5 : 1.5)
      .attr('opacity', 0.95);

    // Step number inside circle
    node.append('text')
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'central')
      .attr('fill', '#fff')
      .attr('font-size', '7px')
      .attr('font-weight', '600')
      .attr('font-family', 'ui-sans-serif, system-ui, sans-serif')
      .text((d) => d.stepIndex + 1);

    // Label to the right of node
    node.append('text')
      .attr('dx', 10)
      .attr('dy', 3)
      .attr('fill', '#9ca3af')
      .attr('font-size', '9px')
      .attr('font-family', 'ui-sans-serif, system-ui, sans-serif')
      .text((d) => d.label);

    // Tooltip
    const tooltip = d3.select(container)
      .selectAll('.graph-tooltip')
      .data([null])
      .join('div')
      .attr('class', 'graph-tooltip')
      .style('position', 'absolute')
      .style('pointer-events', 'none')
      .style('opacity', 0)
      .style('background', '#1a1d27')
      .style('border', '1px solid #3f4155')
      .style('border-radius', '6px')
      .style('padding', '8px 12px')
      .style('font-size', '11px')
      .style('color', '#e5e7eb')
      .style('max-width', '220px')
      .style('box-shadow', '0 4px 12px rgba(0,0,0,0.5)')
      .style('z-index', 20);

    node
      .on('mouseenter', (event, d) => {
        tooltip.style('opacity', 1).html(
          `<div style="font-weight:600;color:${d.color};margin-bottom:3px;font-size:11px">${d.fullTitle}</div>
           <div style="color:#9ca3af;margin-bottom:3px">${d.domain}</div>
           <div style="color:#6b7280;font-size:10px">step ${d.stepIndex + 1} · sim ${d.similarity.toFixed(3)}${d.forced_jump ? ' · forced jump' : ''}</div>`
        );
      })
      .on('mousemove', (event) => {
        const rect = container.getBoundingClientRect();
        tooltip
          .style('left', `${Math.min(event.clientX - rect.left + 8, rect.width - 230)}px`)
          .style('top', `${Math.max(0, event.clientY - rect.top - 40)}px`);
      })
      .on('mouseleave', () => tooltip.style('opacity', 0))
      .on('click', (event, d) => {
        event.stopPropagation();
        onNodeClick(d);
      });

    simulation.on('tick', () => {
      knnLine
        .attr('x1', (d) => nodeById.get(d.source)?.x ?? 0)
        .attr('y1', (d) => nodeById.get(d.source)?.y ?? 0)
        .attr('x2', (d) => nodeById.get(d.target)?.x ?? 0)
        .attr('y2', (d) => nodeById.get(d.target)?.y ?? 0);

      pathLine
        .attr('x1', (d) => nodeById.get(d.source)?.x ?? 0)
        .attr('y1', (d) => nodeById.get(d.source)?.y ?? 0)
        .attr('x2', (d) => nodeById.get(d.target)?.x ?? 0)
        .attr('y2', (d) => nodeById.get(d.target)?.y ?? 0);

      node.attr('transform', (d) => {
        const nd = nodeById.get(d.id);
        return `translate(${nd?.x ?? 0},${nd?.y ?? 0})`;
      });
    });

    const observer = new ResizeObserver(() => {
      const { width: w, height: h } = container.getBoundingClientRect();
      svg.attr('width', w).attr('height', h);
    });
    observer.observe(container);

    return () => {
      observer.disconnect();
      simulation.stop();
    };
  }, [nodes, pathEdges, knnEdges, selectedChunkId, onNodeClick]);

  return (
    <div ref={containerRef} className="flex-1 relative overflow-hidden bg-bg-primary">
      <svg ref={svgRef} className="w-full h-full" />
    </div>
  );
}
