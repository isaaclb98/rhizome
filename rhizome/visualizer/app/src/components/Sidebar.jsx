const DOMAIN_COLORS = {
  Modernism: '#6366f1',
  Postmodernism: '#a855f7',
  'Critical theory': '#f97316',
  Unknown: '#6b7280',
};

function DomainBadge({ domain }) {
  const color = DOMAIN_COLORS[domain] || DOMAIN_COLORS.Unknown;
  return (
    <span
      className="inline-block px-2 py-0.5 rounded text-xs font-medium"
      style={{ backgroundColor: `${color}20`, color }}
    >
      {domain}
    </span>
  );
}

export default function Sidebar({ selectedNode, stats, pathIndex }) {
  if (!selectedNode) {
    return (
      <aside className="w-80 border-l border-bg-tertiary flex items-center justify-center text-gray-600 text-sm p-6 text-center">
        Click a node to inspect a chunk
      </aside>
    );
  }

  return (
    <aside className="w-80 border-l border-bg-tertiary flex flex-col overflow-hidden">
      {/* Header */}
      <div className="flex-none px-4 pt-4 pb-2 border-b border-bg-tertiary">
        <div className="flex items-start justify-between gap-2 mb-2">
          <h2 className="text-sm font-semibold text-gray-200 leading-tight">
            {selectedNode.article_title}
          </h2>
          <a
            href={selectedNode.article_url}
            target="_blank"
            rel="noopener noreferrer"
            className="flex-none text-gray-500 hover:text-gray-300 text-xs mt-0.5"
            title="Open in Wikipedia"
          >
            ↗
          </a>
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          <DomainBadge domain={selectedNode.domain} />
          {selectedNode.forced_jump && (
            <span className="inline-block px-2 py-0.5 rounded text-xs font-medium bg-orange-900/40 text-orange-400">
              Forced jump
            </span>
          )}
        </div>
      </div>

      {/* Chunk content */}
      <div className="flex-1 overflow-y-auto px-4 py-3">
        <div className="text-xs text-gray-500 mb-1">
          Chunk {pathIndex + 1}
          {stats && ` of ${stats.depth}`}
          {selectedNode.depth !== undefined && ` · depth ${selectedNode.depth}`}
        </div>

        <p className="text-sm text-gray-300 leading-relaxed whitespace-pre-wrap">
          {selectedNode.text}
        </p>

        {/* Metadata */}
        <div className="mt-4 pt-3 border-t border-bg-tertiary space-y-1">
          <div className="flex items-center justify-between text-xs">
            <span className="text-gray-500">Similarity</span>
            <span className="text-gray-300 font-mono">
              {selectedNode.similarity?.toFixed(4) ?? '—'}
            </span>
          </div>
          <div className="flex items-center justify-between text-xs">
            <span className="text-gray-500">Chunk ID</span>
            <span className="text-gray-400 font-mono text-xs">
              {selectedNode.chunk_id}
            </span>
          </div>
        </div>
      </div>
    </aside>
  );
}
