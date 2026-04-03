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

    // Clip path
    const clipId = `graph-clip-${Math.random().toString(36).slice(2)}`;
    svg.append('defs').append('clipPath').attr('id', clipId)
      .append('rect').attr('width', width).attr('height', height);

    const g = svg.append('g').attr('clip-path', `url(#${clipId})`);

    const padTop = 16, padBottom = 16;
    const usableHeight = height - padTop - padBottom;

    // Position nodes: vertical line at x = 0 (center), evenly spaced vertically
    const nodeById = new Map();
    nodes.forEach((n, i) => {
      const x = 0;
      const y = padTop + (i / Math.max(nodes.length - 1, 1)) * usableHeight;
      nodeById.set(n.id, { ...n, x, y });
    });

    const getX = (id) => nodeById.get(id)?.x ?? 0;
    const getY = (id) => nodeById.get(id)?.y ?? 0;

    // Edges
    const knnLine = g.append('g')
      .selectAll('line').data(knnEdges).join('line')
      .attr('stroke', '#3f4155').attr('stroke-width', 0.5)
      .attr('opacity', 0.35).attr('stroke-dasharray', '2,3')
      .attr('x1', (d) => getX(d.source)).attr('y1', (d) => getY(d.source))
      .attr('x2', (d) => getX(d.target)).attr('y2', (d) => getY(d.target));

    const pathLine = g.append('g')
      .selectAll('line').data(pathEdges).join('line')
      .attr('stroke', (d) => d.forced ? '#f97316' : '#525766')
      .attr('stroke-width', (d) => d.forced ? 1.5 : 1)
      .attr('stroke-dasharray', (d) => d.forced ? '4,3' : null)
      .attr('opacity', 0.9)
      .attr('x1', (d) => getX(d.source)).attr('y1', (d) => getY(d.source))
      .attr('x2', (d) => getX(d.target)).attr('y2', (d) => getY(d.target));

    // Nodes
    const node = g.append('g')
      .selectAll('g').data(nodes).join('g')
      .attr('cursor', 'pointer')
      .call(
        d3.drag()
          .on('start', (event) => event.sourceEvent.stopPropagation())
          .on('drag', (event, d) => {
            const nx = Math.max(padLeft, Math.min(width - padRight, event.x));
            const ny = Math.max(padTop, Math.min(height - padBottom, event.y));
            nodeById.get(d.id).x = nx;
            nodeById.get(d.id).y = ny;
            refreshPositions();
          })
      );

    function refreshPositions() {
      knnLine
        .attr('x1', (e) => getX(e.source)).attr('y1', (e) => getY(e.source))
        .attr('x2', (e) => getX(e.target)).attr('y2', (e) => getY(e.target));
      pathLine
        .attr('x1', (e) => getX(e.source)).attr('y1', (e) => getY(e.source))
        .attr('x2', (e) => getX(e.target)).attr('y2', (e) => getY(e.target));
      node.attr('transform', (d) => {
        const nd = nodeById.get(d.id);
        return `translate(${nd?.x ?? 0},${nd?.y ?? 0})`;
      });
    }

    node.attr('transform', (d) => {
      const nd = nodeById.get(d.id);
      return `translate(${nd?.x ?? 0},${nd?.y ?? 0})`;
    });

    node.append('circle')
      .attr('r', 7)
      .attr('fill', (d) => d.color)
      .attr('stroke', (d) => d.id === selectedChunkId ? '#fff' : '#0f1117')
      .attr('stroke-width', (d) => d.id === selectedChunkId ? 2.5 : 1.5)
      .attr('opacity', 0.95);

    node.append('text')
      .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
      .attr('fill', '#fff').attr('font-size', '7px')
      .attr('font-weight', '600')
      .attr('font-family', 'ui-sans-serif, system-ui, sans-serif')
      .text((d) => d.stepIndex + 1);

    node.append('text')
      .attr('dx', 11).attr('dy', 3)
      .attr('fill', '#9ca3af').attr('font-size', '9px')
      .attr('font-family', 'ui-sans-serif, system-ui, sans-serif')
      .text((d) => d.label);

    // Tooltip
    const tooltip = d3.select(container)
      .selectAll('.graph-tooltip').data([null]).join('div')
      .attr('class', 'graph-tooltip')
      .style('position', 'absolute').style('pointer-events', 'none')
      .style('opacity', 0).style('background', '#1a1d27')
      .style('border', '1px solid #3f4155').style('border-radius', '6px')
      .style('padding', '8px 12px').style('font-size', '11px')
      .style('color', '#e5e7eb').style('max-width', '220px')
      .style('box-shadow', '0 4px 12px rgba(0,0,0,0.5)')
      .style('z-index', 20);

    // Transform state
    let scale = 1;
    let panX = 0;
    let panY = 0;
    const minScale = 0.3;
    const maxScale = 1.5;

    function applyTransform() {
      g.attr('transform', `translate(${panX},${panY}) scale(${scale})`);
    }

    // Initial centering — compute bounding box from node positions, center it
    const xs = nodes.map((n) => nodeById.get(n.id).x);
    const ys = nodes.map((n) => nodeById.get(n.id).y);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const gW = maxX - minX || 1;
    const gH = maxY - minY || 1;

    // Scale to fit panel, centered — center the visual mass (mean of node positions) at the panel center
    scale = Math.min(width / gW, height / gH, 1.5);
    // Use mean x of all nodes for better centering (bbox center is biased when nodes are unevenly distributed)
    const meanX = xs.reduce((a, b) => a + b, 0) / xs.length;
    const meanY = ys.reduce((a, b) => a + b, 0) / ys.length;
    panX = width / 2 - meanX * scale;
    panY = height / 2 - meanY * scale;
    applyTransform();

    // Native wheel zoom
    container.addEventListener('wheel', (e) => {
      e.preventDefault();
      const factor = e.deltaY > 0 ? 0.9 : 1.1;
      const newScale = Math.max(minScale, Math.min(maxScale, scale * factor));
      if (newScale !== scale) {
        // Zoom toward mouse position
        const rect = container.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        panX = mx - (mx - panX) * (newScale / scale);
        panY = my - (my - panY) * (newScale / scale);
        scale = newScale;
        applyTransform();
      }
    }, { passive: false });

    // Native pan
    let isPanning = false;
    let lastX = 0, lastY = 0;

    container.addEventListener('mousedown', (e) => {
      if (e.button !== 0) return;
      // Only pan if clicking on the container background (not a node)
      if (e.target !== svgRef.current && !e.target.classList.contains('graph-bg')) {
        return;
      }
      isPanning = true;
      lastX = e.clientX;
      lastY = e.clientY;
      container.style.cursor = 'grabbing';
    });

    window.addEventListener('mousemove', (e) => {
      if (!isPanning) return;
      panX += e.clientX - lastX;
      panY += e.clientY - lastY;
      lastX = e.clientX;
      lastY = e.clientY;
      applyTransform();
    });

    window.addEventListener('mouseup', () => {
      if (isPanning) {
        isPanning = false;
        container.style.cursor = '';
      }
    });

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
        const mx = (event.clientX - rect.left - panX) / scale;
        const my = (event.clientY - rect.top - panY) / scale;
        tooltip
          .style('left', `${Math.min(mx + 8, rect.width - 230)}px`)
          .style('top', `${Math.max(0, my - 40)}px`);
      })
      .on('mouseleave', () => tooltip.style('opacity', 0))
      .on('click', (event, d) => {
        event.stopPropagation();
        onNodeClick(d);
      });

    svg.on('click', () => onNodeClick(null));

    // Resize observer
    const observer = new ResizeObserver(() => {
      const { width: w, height: h } = container.getBoundingClientRect();
      svg.attr('width', w).attr('height', h);
      svg.select(`#${clipId} rect`).attr('width', w).attr('height', h);
    });
    observer.observe(container);

    return () => observer.disconnect();
  }, [nodes, pathEdges, knnEdges, selectedChunkId, onNodeClick]);

  return (
    <div ref={containerRef} className="flex-1 relative bg-bg-primary overflow-hidden">
      <svg ref={svgRef} className="w-full h-full graph-bg" style={{ display: 'block' }} />
      <div className="absolute bottom-2 right-2 text-[10px] text-gray-600 pointer-events-none select-none">
        scroll to zoom · drag to pan
      </div>
    </div>
  );
}
