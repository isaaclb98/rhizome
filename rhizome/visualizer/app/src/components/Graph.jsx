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

function similarityToWidth(score) {
  const clamped = Math.max(0.5, Math.min(1.0, score));
  return 1 + ((clamped - 0.5) / 0.5) * 4;
}

export default function Graph({ path, selectedChunkId, onNodeClick }) {
  const svgRef = useRef(null);
  const containerRef = useRef(null);

  // Build nodes and edges from path + candidates
  const { nodes, pathEdges, knnEdges } = useMemo(() => {
    // Index map: chunk_id → node
    const nodeMap = new Map();
    const nodes = path.map((step, i) => {
      const node = {
        id: step.chunk_id,
        label: step.article_title.length > 28
          ? step.article_title.slice(0, 28) + '…'
          : step.article_title,
        fullTitle: step.article_title,
        domain: step.domain,
        color: getDomainColor(step.domain),
        stepIndex: i,
        forced_jump: step.forced_jump,
        similarity: step.similarity,
      };
      nodeMap.set(step.chunk_id, node);
      return node;
    });

    // Path edges: consecutive steps in the traversal
    const pathEdges = [];
    for (let i = 0; i < path.length - 1; i++) {
      pathEdges.push({
        source: path[i].chunk_id,
        target: path[i + 1].chunk_id,
        forced: path[i].forced_jump,
        similarity: path[i].similarity,
      });
    }

    // kNN edges: from each step to its top_k candidates
    const knnEdges = [];
    const candidateIds = new Set();
    for (let i = 0; i < path.length; i++) {
      for (const cand of (path[i].candidates || [])) {
        // Only draw kNN edge if candidate is in the path AND not already the path edge
        const candId = cand.chunk_id;
        if (nodeMap.has(candId) && candId !== path[i + 1]?.chunk_id) {
          knnEdges.push({
            source: path[i].chunk_id,
            target: candId,
            similarity: cand.similarity,
          });
          candidateIds.add(candId);
        }
      }
    }

    return { nodes, pathEdges, knnEdges };
  }, [path]);

  useEffect(() => {
    if (!svgRef.current || !containerRef.current || nodes.length === 0) return;

    const container = containerRef.current;
    const { width, height } = container.getBoundingClientRect();

    d3.select(svgRef.current).selectAll('*').remove();

    const svg = d3
      .select(svgRef.current)
      .attr('width', width)
      .attr('height', height);

    // Arrow markers
    const defs = svg.append('defs');

    defs.append('marker')
      .attr('id', 'arrowhead')
      .attr('viewBox', '-0 -5 10 10')
      .attr('refX', 18)
      .attr('refY', 0)
      .attr('orient', 'auto')
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .append('path')
      .attr('d', 'M 0,-5 L 10,0 L 0,5')
      .attr('fill', '#525766');

    defs.append('marker')
      .attr('id', 'arrowhead-jump')
      .attr('viewBox', '-0 -5 10 10')
      .attr('refX', 18)
      .attr('refY', 0)
      .attr('orient', 'auto')
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .append('path')
      .attr('d', 'M 0,-5 L 10,0 L 0,5')
      .attr('fill', '#f97316');

    const g = svg.append('g');

    // Zoom
    const zoom = d3.zoom()
      .scaleExtent([0.2, 4])
      .on('zoom', (event) => g.attr('transform', event.transform));
    svg.call(zoom);

    // Initial centered transform
    svg.call(zoom.transform, d3.zoomIdentity.translate(0, 0));

    // Pre-set node positions: fixed x by stepIndex, y distributed vertically
    const nodeCount = nodes.length;
    const nodeById = new Map(nodes.map((n, i) => [n.id, { ...n, x: i * (width / Math.max(nodeCount - 1, 1)), y: height / 2 }]));

    // Force simulation with weak horizontal link force to keep order
    const simulation = d3.forceSimulation(nodes)
      .force('x', d3.forceX((d) => {
        const widthPerNode = width / Math.max(nodeCount - 1, 1);
        return d.stepIndex * widthPerNode;
      }).strength(0.4))
      .force('y', d3.forceY(height / 2).strength(0.1))
      .force('charge', d3.forceManyBody().strength(-80))
      .force('collision', d3.forceCollide().radius(20))
      .alphaDecay(0.03);

    // Draw kNN edges first (behind)
    const knnLine = g.append('g').selectAll('line')
      .data(knnEdges)
      .join('line')
      .attr('stroke', '#3f4155')
      .attr('stroke-width', 0.5)
      .attr('opacity', 0.5)
      .attr('stroke-dasharray', '2,3');

    // Draw path edges on top
    const pathLine = g.append('g').selectAll('line')
      .data(pathEdges)
      .join('line')
      .attr('stroke', (d) => d.forced ? '#f97316' : '#525766')
      .attr('stroke-width', (d) => similarityToWidth(d.similarity))
      .attr('stroke-dasharray', (d) => d.forced ? '5,4' : null)
      .attr('marker-end', (d) => d.forced ? 'url(#arrowhead-jump)' : 'url(#arrowhead)')
      .attr('opacity', 0.9);

    // Draw nodes
    const node = g.append('g').selectAll('g')
      .data(nodes)
      .join('g')
      .attr('cursor', 'pointer')
      .call(
        d3.drag()
          .on('start', (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x; d.fy = d.y;
          })
          .on('drag', (event, d) => { d.fx = event.x; d.fy = event.y; })
          .on('end', (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null; d.fy = null;
          })
      );

    node.append('circle')
      .attr('r', 8)
      .attr('fill', (d) => d.color)
      .attr('stroke', (d) => d.id === selectedChunkId ? '#fff' : '#0f1117')
      .attr('stroke-width', (d) => d.id === selectedChunkId ? 3 : 2)
      .attr('opacity', 0.95);

    node.append('text')
      .attr('dx', 12)
      .attr('dy', 4)
      .attr('fill', '#9ca3af')
      .attr('font-size', '11px')
      .attr('font-family', 'ui-sans-serif, system-ui, sans-serif')
      .text((d) => d.label);

    // Hover tooltip
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
      .style('font-size', '12px')
      .style('color', '#e5e7eb')
      .style('max-width', '280px')
      .style('box-shadow', '0 4px 12px rgba(0,0,0,0.4)')
      .style('z-index', 10);

    node
      .on('mouseenter', (event, d) => {
        tooltip.style('opacity', 1).html(
          `<div style="font-weight:600;color:${d.color};margin-bottom:4px">${d.fullTitle}</div>
           <div style="color:#9ca3af;font-size:11px;margin-bottom:4px">${d.domain}</div>
           <div style="color:#9ca3af;font-size:11px">step ${d.stepIndex + 1} · sim ${d.similarity.toFixed(3)}</div>`
        );
      })
      .on('mousemove', (event) => {
        const rect = container.getBoundingClientRect();
        const x = event.clientX - rect.left + 12;
        const y = event.clientY - rect.top - 10;
        tooltip
          .style('left', `${Math.min(x, rect.width - 300)}px`)
          .style('top', `${Math.max(0, y)}px`);
      })
      .on('mouseleave', () => tooltip.style('opacity', 0))
      .on('click', (event, d) => {
        event.stopPropagation();
        onNodeClick(d);
      });

    // Click on background deselects
    svg.on('click', () => onNodeClick(null));

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

      node.attr('transform', (d) => `translate(${d.x ?? 0},${d.y ?? 0})`);
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
    <div ref={containerRef} className="absolute inset-0">
      <svg ref={svgRef} className="w-full h-full" />
    </div>
  );
}
