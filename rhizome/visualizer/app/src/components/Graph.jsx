import { useEffect, useRef, useMemo, useCallback } from 'react';
import * as d3 from 'd3';

// Uses CSS var --node-color set by data-theme on document root

export default function Graph({ path, selectedChunkId, onNodeClick }) {
  const svgRef = useRef(null);
  const containerRef = useRef(null);
  const zoomRef = useRef(null);  // store zoom behavior for programmatic control

  const { nodes, pathEdges, knnEdges } = useMemo(() => {
    const nodes = path.map((step, i) => ({
      id: step.chunk_id,
      label: step.article_title.length > 20
        ? step.article_title.slice(0, 20) + '…'
        : step.article_title,
      fullTitle: step.article_title,
      color: 'var(--node-color)',
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

  const drawGraph = useCallback(() => {
    if (!svgRef.current || !containerRef.current || nodes.length === 0) return;

    const container = containerRef.current;
    const rect = container.getBoundingClientRect();
    const W = Math.round(rect.width);
    const H = Math.round(rect.height);
    if (W === 0 || H === 0) return;

    d3.select(svgRef.current).selectAll('*').remove();

    const svg = d3.select(svgRef.current)
      .attr('width', W)
      .attr('height', H);

    // Clip path
    const clipId = `graph-clip-${Math.random().toString(36).slice(2)}`;
    svg.append('defs').append('clipPath').attr('id', clipId)
      .append('rect').attr('width', W).attr('height', H);

    // Pan+zoom group — all content goes here
    const panGroup = svg.append('g').attr('clip-path', `url(#${clipId})`);

    const padTop = 20, padBottom = 20;
    const usableH = H - padTop - padBottom;

    // Node positions: all at center x=W/2
    const cx = W / 2;
    const nodeById = new Map();
    nodes.forEach((n, i) => {
      const x = cx;
      const y = padTop + (i / Math.max(nodes.length - 1, 1)) * usableH;
      nodeById.set(n.id, { ...n, x, y });
    });

    const getX = (id) => nodeById.get(id)?.x ?? 0;
    const getY = (id) => nodeById.get(id)?.y ?? 0;

    // Edges
    const knnLine = panGroup.append('g')
      .selectAll('line').data(knnEdges).join('line')
      .attr('stroke', 'var(--border-color)').attr('stroke-width', 0.5)
      .attr('opacity', 0.35).attr('stroke-dasharray', '2,3')
      .attr('x1', (d) => getX(d.source)).attr('y1', (d) => getY(d.source))
      .attr('x2', (d) => getX(d.target)).attr('y2', (d) => getY(d.target));

    const pathLine = panGroup.append('g')
      .selectAll('line').data(pathEdges).join('line')
      .attr('stroke', 'var(--text-muted)')
      .attr('stroke-width', 1)
      .attr('opacity', 0.9)
      .attr('x1', (d) => getX(d.source)).attr('y1', (d) => getY(d.source))
      .attr('x2', (d) => getX(d.target)).attr('y2', (d) => getY(d.target));

    // Nodes
    const node = panGroup.append('g')
      .selectAll('g').data(nodes).join('g')
      .attr('cursor', 'pointer');

    node.attr('transform', (d) => {
      const nd = nodeById.get(d.id);
      return `translate(${nd?.x ?? 0},${nd?.y ?? 0})`;
    });

    node.append('circle')
      .attr('r', 7)
      .attr('fill', (d) => d.color)
      .attr('stroke', (d) => d.id === selectedChunkId ? 'var(--bg-primary)' : 'var(--border-color)')
      .attr('stroke-width', (d) => d.id === selectedChunkId ? 2.5 : 1.5)
      .attr('opacity', 0.95);

    node.append('text')
      .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
      .attr('fill', '#fff').attr('font-size', '7px')
      .attr('font-weight', '600')
      .attr('font-family', 'JetBrains Mono, ui-monospace, monospace')
      .text((d) => d.stepIndex + 1);

    node.append('text')
      .attr('dx', 11).attr('dy', 3)
      .attr('fill', 'var(--text-muted)').attr('font-size', '9px')
      .attr('font-family', 'Inter, ui-sans-serif, system-ui, sans-serif')
      .text((d) => d.label);

    // Zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.5, 4])
      .on('zoom', (event) => {
        panGroup.attr('transform', event.transform);
      });

    zoomRef.current = zoom;
    svg.call(zoom);

    // Initial mount: center the last node at viewport center, scale 1.5
    if (nodes.length > 0) {
      const lastNode = nodes[nodes.length - 1];
      const lastNd = nodeById.get(lastNode.id);
      if (lastNd) {
        const centerTx = W / 2 - 1.5 * lastNd.x;
        const centerTy = H / 2 - 1.5 * lastNd.y;
        svg.call(zoom.transform, d3.zoomIdentity.translate(centerTx, centerTy).scale(1.5));
      }
    } else {
      svg.call(zoom.transform, d3.zoomIdentity.scale(1.5));
    }

    // Node drag: individual node repositioning using SVG-space coords (d3.pointer)
    node.call(
      d3.drag()
        .on('start', function(event) {
          event.sourceEvent.stopPropagation();
        })
        .on('drag', function(event, d) {
          const nd = nodeById.get(d.id);
          if (!nd) return;
          // d3.pointer gives SVG-space coords regardless of active zoom transform
          const [svgX, svgY] = d3.pointer(event, svgRef.current);
          const nx = Math.max(0, Math.min(W, svgX));
          const ny = Math.max(padTop, Math.min(H - padBottom, svgY));
          nd.x = nx;
          nd.y = ny;
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

    // Tooltip
    const tooltip = d3.select(container)
      .selectAll('.graph-tooltip').data([null]).join('div')
      .attr('class', 'graph-tooltip')
      .style('position', 'absolute').style('pointer-events', 'none')
      .style('opacity', 0).style('background', 'var(--bg-secondary)')
      .style('border', '1px solid var(--border-color)').style('border-radius', '6px')
      .style('padding', '8px 12px').style('font-size', '11px')
      .style('color', 'var(--text-primary)').style('max-width', '220px')
      .style('box-shadow', '0 4px 12px rgba(0,0,0,0.15)')
      .style('z-index', 20);

    node
      .on('mouseenter', (event, d) => {
        tooltip.style('opacity', 1).html(
          `<div style="font-weight:600;color:${d.color};margin-bottom:3px;font-size:11px">${d.fullTitle}</div>
           <div style="color:#6b7280;font-size:10px">step ${d.stepIndex + 1} · sim ${d.similarity.toFixed(3)}</div>`
        );
      })
      .on('mousemove', (event) => {
        const r = container.getBoundingClientRect();
        tooltip
          .style('left', `${Math.min(event.clientX - r.left + 8, r.width - 230)}px`)
          .style('top', `${Math.max(0, event.clientY - r.top - 40)}px`);
      })
      .on('mouseleave', () => tooltip.style('opacity', 0))
      .on('click', (event, d) => {
        event.stopPropagation();
        onNodeClick(d);
      });

    svg.on('click', () => onNodeClick(null));
  }, [nodes, pathEdges, knnEdges, selectedChunkId, onNodeClick]);

  // When selectedChunkId changes, animate view to center the clicked node
  useEffect(() => {
    if (!selectedChunkId || !svgRef.current || !containerRef.current) return;
    if (!zoomRef.current) return;

    const container = containerRef.current;
    const rect = container.getBoundingClientRect();
    const W = Math.round(rect.width);
    const H = Math.round(rect.height);

    // Find the node position
    const stepIndex = nodes.findIndex((n) => n.id === selectedChunkId);
    if (stepIndex === -1) return;

    const padTop = 20, padBottom = 20;
    const usableH = H - padTop - padBottom;
    const cx = W / 2;
    const nodeY = padTop + (stepIndex / Math.max(nodes.length - 1, 1)) * usableH;

    // Get current zoom state
    const svg = d3.select(svgRef.current);
    const currentTransform = d3.zoomTransform(svgRef.current);
    const currentScale = currentTransform.k;

    // Center node at viewport center, preserving current zoom scale
    const targetTransform = d3.zoomIdentity
      .translate(W / 2 - currentScale * cx, H / 2 - currentScale * nodeY)
      .scale(currentScale);

    svg.transition().duration(300).call(zoomRef.current.transform, targetTransform);
  }, [selectedChunkId, nodes]);

  useEffect(() => {
    drawGraph();

    if (!containerRef.current) return;
    const ro = new ResizeObserver(() => drawGraph());
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, [drawGraph]);

  return (
    <div ref={containerRef} className="flex-1 relative overflow-hidden bg-bg-primary">
      <svg ref={svgRef} style={{ display: 'block', width: '100%', height: '100%' }} />
      <div className="absolute bottom-2 right-2 text-[10px] text-text-muted pointer-events-none select-none">
        scroll to zoom · drag to pan
      </div>
    </div>
  );
}
