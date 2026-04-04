import { useState, useCallback, useEffect } from 'react';
import Controls from './components/Controls.jsx';
import Graph from './components/Graph.jsx';
import PathPanel from './components/PathPanel.jsx';
import Legend from './components/Legend.jsx';

const DEFAULT_PARAMS = {
  query: 'the tension between modernism and postmodernism',
  depth: 8,
  epsilon: 0.1,
  top_k: 20,
  temperature: 1.0,
  max_same_article_consecutive: 2,
};

export default function App() {
  const [path, setPath] = useState([]);
  const [stats, setStats] = useState(null);
  const [selectedChunkId, setSelectedChunkId] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [params, setParams] = useState(DEFAULT_PARAMS);
  const [domains, setDomains] = useState([]);

  // Fetch unique domains after first successful traversal (lazy, cached in state)
  const fetchDomainsIfNeeded = useCallback(() => {
    if (domains.length > 0) return;
    fetch('/domains')
      .then((r) => r.ok ? r.json() : null)
      .then((d) => { if (d?.domains) setDomains(d.domains); })
      .catch(() => {});
  }, [domains.length]);

  const handleTraverse = useCallback(async (requestParams) => {
    setIsLoading(true);
    setError(null);
    setSelectedChunkId(null);

    try {
      const response = await fetch('/traverse', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestParams),
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(err.detail || `HTTP ${response.status}`);
      }

      const data = await response.json();
      setPath(data.path);
      setStats(data.stats);
      setParams(requestParams);
      fetchDomainsIfNeeded();
    } catch (err) {
      setError(err.message || 'Traversal failed');
      setPath([]);
      setStats(null);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleNodeClick = useCallback((node) => {
    const id = node?.id ?? node?.chunk_id ?? null;
    setSelectedChunkId((prev) => (prev === id ? prev : id));
  }, []);

  return (
    <div className="flex flex-col h-screen bg-bg-primary overflow-hidden">
      {/* Header */}
      <header className="flex-none border-b border-bg-tertiary px-4 py-3">
        <div className="flex items-center gap-3 mb-3">
          <h1 className="text-2xl font-bold tracking-tight text-white">
            Rhizome
          </h1>
          <span className="text-xs text-gray-500 font-mono">
            Wikipedia semantic traversal
          </span>
        </div>
        <Controls params={params} onTraverse={handleTraverse} isLoading={isLoading} />
        <div className="mt-2">
          <Legend domains={domains} />
        </div>
      </header>

      {/* Error banner */}
      {error && (
        <div className="flex-none bg-red-900/30 border-b border-red-800 px-4 py-2 text-red-300 text-sm">
          {error}
        </div>
      )}

      {/* Main content: two-column layout */}
      <div className="flex-1 min-h-0 flex overflow-hidden">
        {/* Left: path text panel */}
        <div className="flex-1 min-h-0 overflow-hidden">
          <PathPanel
            path={path}
            selectedChunkId={selectedChunkId}
            onSelectChunk={handleNodeClick}
          />
        </div>

        {/* Right: graph strip — always visible on desktop, collapses on mobile */}
        <div className="hidden lg:flex lg:flex-col lg:w-80 xl:w-96 border-l border-bg-tertiary overflow-hidden flex-shrink-0">
          {path.length > 0 ? (
            <Graph
              path={path}
              selectedChunkId={selectedChunkId}
              onNodeClick={handleNodeClick}
              domains={domains}
            />
          ) : (
            <div className="relative flex-1">
              <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-600 text-xs text-center gap-2">
                <span>Graph appears here</span>
                <span>after traversal</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      {stats && (
        <footer className="flex-none border-t border-bg-tertiary px-4 py-1.5 flex items-center gap-4 text-xs text-gray-500">
          <div className="flex items-center gap-1.5">
            <span>Depth</span>
            <span className="text-gray-300">{stats.depth}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span>ε</span>
            <span className="text-gray-300">{stats.epsilon}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span>top_k</span>
            <span className="text-gray-300">{stats.top_k}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span>temp</span>
            <span className="text-gray-300">{stats.temperature}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span>same-art</span>
            <span className="text-gray-300">{stats.max_same_article_consecutive}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-orange-400">●</span>
            <span>Forced jumps</span>
            <span className="text-gray-300">{stats.forced_jumps}</span>
          </div>
        </footer>
      )}
    </div>
  );
}
