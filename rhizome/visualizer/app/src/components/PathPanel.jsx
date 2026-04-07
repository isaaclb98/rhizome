import { useRef, useEffect } from 'react';

function PathItem({ step, index, isSelected, onClick }) {
  return (
    <div
      onClick={onClick}
      className={`w-full text-left px-3 py-2.5 border-b border-bg-tertiary transition-colors cursor-pointer ${
        isSelected ? 'bg-bg-tertiary border-l-2 border-l-accent' : 'hover:bg-bg-secondary'
      }`}
      style={{ userSelect: 'text' }}
    >
      <div className="flex items-start gap-2">
        <span className={`flex-shrink-0 w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold mt-0.5 ${
          isSelected ? 'bg-accent text-white' : 'bg-bg-tertiary text-text-muted'
        }`}>
          {index + 1}
        </span>
        <div className="flex-1 min-w-0 select-text">
          <div className="flex items-center gap-1.5 flex-wrap mb-1">
            <span className="text-sm font-medium text-text-primary leading-tight select-text">
              {step.article_title}
            </span>
          </div>
          <p className="text-xs text-text-muted leading-relaxed select-text">
            {step.text}
          </p>
          {isSelected && (
            <div className="mt-2 flex items-center gap-3 text-xs text-text-muted">
              <span>sim <span className="text-text-secondary font-mono select-text">{step.similarity.toFixed(4)}</span></span>
              <a
                href={step.article_url}
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-text-primary underline"
              >
                Wikipedia ↗
              </a>
            </div>
          )}
        </div>
      </div>
    </div>
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
      <aside className="h-full border-r border-border flex items-center justify-center bg-bg-primary">
        <span className="text-text-muted text-sm text-center px-6">Run a traversal to see the path</span>
      </aside>
    );
  }

  return (
    <aside className="h-full border-r border-border flex flex-col bg-bg-primary">
      <div className="flex-none px-3 py-2 border-b border-border bg-bg-secondary">
        <div className="text-xs text-text-muted font-medium">
          Path — {path.length} chunks
        </div>
      </div>
      <div ref={listRef} className="flex-1 min-h-0 overflow-y-auto">
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
