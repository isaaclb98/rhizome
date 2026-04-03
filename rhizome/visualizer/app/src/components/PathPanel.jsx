import { useRef, useEffect } from 'react';

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
      className="inline-block px-1.5 py-0.5 rounded text-xs font-medium flex-shrink-0"
      style={{ backgroundColor: `${color}20`, color }}
    >
      {domain}
    </span>
  );
}

function PathItem({ step, index, isSelected, onClick }) {
  const isForced = step.forced_jump;
  return (
    <button
      onClick={onClick}
      className={`w-full text-left px-3 py-2.5 border-b border-bg-tertiary transition-colors ${
        isSelected
          ? 'bg-bg-tertiary'
          : 'hover:bg-bg-secondary'
      }`}
    >
      <div className="flex items-start gap-2 mb-1">
        <span className={`flex-shrink-0 w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold ${
          isForced ? 'bg-orange-900/60 text-orange-400' : 'bg-bg-tertiary text-gray-400'
        }`}>
          {index + 1}
        </span>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5 flex-wrap">
            <span className="text-sm font-medium text-gray-200 leading-tight truncate">
              {step.article_title}
            </span>
            <DomainBadge domain={step.domain} />
          </div>
          {isForced && (
            <span className="inline-block mt-0.5 text-xs text-orange-400">
              forced jump · sim {step.similarity.toFixed(3)}
            </span>
          )}
        </div>
      </div>
      <p className="text-xs text-gray-500 leading-relaxed pl-7 line-clamp-2">
        {step.text}
      </p>
      {isSelected && (
        <div className="mt-2 pl-7">
          <p className="text-sm text-gray-300 leading-relaxed whitespace-pre-wrap">
            {step.text}
          </p>
          <div className="mt-2 flex items-center gap-3 text-xs text-gray-500">
            <span>sim <span className="text-gray-400 font-mono">{step.similarity.toFixed(4)}</span></span>
            <a
              href={step.article_url}
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-gray-300 underline"
            >
              Wikipedia ↗
            </a>
          </div>
        </div>
      )}
    </button>
  );
}

export default function PathPanel({ path, selectedChunkId, onSelectChunk }) {
  const listRef = useRef(null);
  const selectedRef = useRef(null);

  // Auto-scroll selected item into view
  useEffect(() => {
    if (selectedRef.current) {
      selectedRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [selectedChunkId]);

  if (path.length === 0) {
    return (
      <aside className="flex-1 border-r border-bg-tertiary flex items-center justify-center text-gray-600 text-sm p-6 text-center">
        Run a traversal to see the path
      </aside>
    );
  }

  return (
    <aside className="flex-1 border-r border-bg-tertiary flex flex-col overflow-hidden bg-bg-primary">
      <div className="flex-none px-3 py-2 border-b border-bg-tertiary bg-bg-secondary">
        <div className="text-xs text-gray-400 font-medium">
          Path — {path.length} chunks
        </div>
      </div>
      <div ref={listRef} className="flex-1 overflow-y-auto">
        {path.map((step, i) => (
          <div
            key={step.chunk_id}
            ref={step.chunk_id === selectedChunkId ? selectedRef : null}
          >
            <PathItem
              step={step}
              index={i}
              isSelected={step.chunk_id === selectedChunkId}
              onClick={() => onSelectChunk(step)}
            />
          </div>
        ))}
      </div>
    </aside>
  );
}
