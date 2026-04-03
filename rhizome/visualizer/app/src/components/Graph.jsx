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
  // Map similarity score [0.5, 1.0] → [1, 5]px
  const clamped = Math.max(0.5, Math.min(1.0, score));
  return 1 + ((clamped - 0.5) / 0.5) * 4;
}

export default function Graph({ path, stats, selectedNode, onNodeClick, onNodeHover }) {
  const svgRef = useRef(null);
  const containerRef = useRef(null);

  // Build nodes and links from path
  const { nodes, links } = useMemo(() => {
    const nodes = path.map((step, i) => ({
      id: step.chunk_id,
      label: step.article_title.length > 28
        ? step.article_title.slice(0, 28) + '…'
        : step.article_title,
      fullTitle: step.article_title,
      domain: step.domain,
      color: getDomainColor(step.domain),
      text: step.text,
      depth: step.depth,
      similarity: step.similarity,
      forced_jump: step.forced_jump,
      excerpt: step.text.slice(0, 100) + (step.text.length > 100 ? '…' : ''),
    }));

    const links = [];
    for (let i = 0; i < path.length - 1; i++) {
      const src = path[i];
      const dst = path[i + 1];
      links.push({
        source: src.chunk_id,
        target: dst.chunk_id,
        similarity: src.similarity,
        forced_jump: src.forced_jump,
      });
    }

    return { nodes, links };
  }, [path]);

  useEffect(() => {
    if (!svgRef.current || !containerRef.current || nodes.length === 0) return;

    const container = containerRef.current;
    const { width, height } = container.getBoundingClientRect();

    // Clear previous
    d3.select(svgRef.current).selectAll('*').remove();

    const svg = d3
      .select(svgRef.current)
      .attr('width', width)
      .attr('height', height);

    // Arrow marker for directed edges
    svg
      .append('defs')
      .append('marker')
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

    // Dashed marker for forced jumps
    svg
      .select('defs')
      .append('marker')
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

    // Zoom behavior
    const zoom = d3
      .zoom()
      .scaleExtent([0.3, 3])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });

    svg.call(zoom);

    // Initial zoom to fit
    const initialScale = Math.min(0.9, Math.min(width / 600, height / 400));
    svg.call(
      zoom.transform,
      d3.zoomIdentity
        .translate(width / 2, height / 2)
        .scale(initialScale)
        .translate(-width / 2, -height / 2)
    );

    // Force simulation
    const simulation = d3
      .forceSimulation(nodes)
      .force(
        'link',
        d3
          .forceLink(links)
          .id((d) => d.id)
          .distance(100)
      )
      .force('charge', d3.forceManyBody().strength(-200))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(30));

    // Draw links
    const link = g
      .append('g')
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke', (d) => (d.forced_jump ? '#f97316' : '#525766'))
      .attr('stroke-width', (d) => similarityToWidth(d.similarity))
      .attr('stroke-dasharray', (d) => (d.forced_jump ? '5,4' : null))
      .attr('marker-end', (d) =>
        d.forced_jump ? 'url(#arrowhead-jump)' : 'url(#arrowhead)'
      )
      .attr('opacity', 0.8);

    // Draw nodes
    const node = g
      .append('g')
      .selectAll('g')
      .data(nodes)
      .join('g')
      .attr('cursor', 'pointer')
      .call(
        d3
          .drag()
          .on('start', (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on('drag', (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on('end', (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          })
      );

    // Node circles
    node
      .append('circle')
      .attr('r', 8)
      .attr('fill', (d) => d.color)
      .attr('stroke', '#0f1117')
      .attr('stroke-width', 2)
      .attr('opacity', 0.95);

    // Node labels
    node
      .append('text')
      .attr('dx', 12)
      .attr('dy', 4)
      .attr('fill', '#9ca3af')
      .attr('font-size', '11px')
      .attr('font-family', 'ui-sans-serif, system-ui, sans-serif')
      .text((d) => d.label);

    // Hover tooltip
    const tooltip = d3
      .select(container)
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
        tooltip
          .style('opacity', 1)
          .html(
            `<div style="font-weight:600;color:${d.color};margin-bottom:4px">${d.fullTitle}</div>
             <div style="color:#9ca3af;font-size:11px;margin-bottom:4px">${d.domain}</div>
             <div style="line-height:1.4">${d.excerpt}</div>`
          );
        onNodeHover(d);
      })
      .on('mousemove', (event) => {
        const rect = container.getBoundingClientRect();
        const x = event.clientX - rect.left + 12;
        const y = event.clientY - rect.top - 10;
        tooltip
          .style('left', `${Math.min(x, rect.width - 300)}px`)
          .style('top', `${Math.max(0, y)}px`);
      })
      .on('mouseleave', () => {
        tooltip.style('opacity', 0);
      })
      .on('click', (event, d) => {
        onNodeClick(d);
      });

    // Highlight selected node
    node.each(function (d) {
      if (selectedNode && d.id === selectedNode.chunk_id) {
        d3.select(this)
          .select('circle')
          .attr('stroke', '#fff')
          .attr('stroke-width', 3)
          .attr('r', 10);
      }
    });

    // Tick
    simulation.on('tick', () => {
      link
        .attr('x1', (d) => d.source.x)
        .attr('y1', (d) => d.source.y)
        .attr('x2', (d) => d.target.x)
        .attr('y2', (d) => d.target.y);

      node.attr('transform', (d) => `translate(${d.x},${d.y})`);
    });

    // Resize observer
    const observer = new ResizeObserver(() => {
      const { width: w, height: h } = container.getBoundingClientRect();
      svg.attr('width', w).attr('height', h);
      simulation.force('center', d3.forceCenter(w / 2, h / 2));
      simulation.alpha(0.3).restart();
    });
    observer.observe(container);

    return () => {
      observer.disconnect();
      simulation.stop();
    };
  }, [nodes, links, selectedNode, onNodeClick, onNodeHover]);

  return (
    <div ref={containerRef} className="absolute inset-0">
      <svg ref={svgRef} className="w-full h-full" />
    </div>
  );
}
