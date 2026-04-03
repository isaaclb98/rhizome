import { useState, useCallback } from 'react';
import Controls from './components/Controls.jsx';
import Graph from './components/Graph.jsx';
import PathPanel from './components/PathPanel.jsx';
import Legend from './components/Legend.jsx';

const DEFAULT_PARAMS = {
  query: 'the tension between modernism and postmodernism',
  depth: 8,
  epsilon: 0.1,
  top_k: 5,
};

export default function App() {
  const [path, setPath] = useState([]);
  const [stats, setStats] = useState(null);
  const [selectedChunkId, setSelectedChunkId] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [params, setParams] = useState(DEFAULT_PARAMS);

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
    } catch (err) {
      setError(err.message || 'Traversal failed');
      setPath([]);
      setStats(null);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleNodeClick = useCallback((node) => {
    setSelectedChunkId((prev) =>
      prev && node && prev === node.id ? null : (node ? node.id : null)
    );
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
          <Legend />
        </div>
      </header>

      {/* Error banner */}
      {error && (
        <div className="flex-none bg-red-900/30 border-b border-red-800 px-4 py-2 text-red-300 text-sm">
          {error}
        </div>
      )}

      {/* Main content: two-column layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left: path text panel */}
        <div className="flex-1 overflow-hidden">
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
            />
          ) : (
            <div className="flex-1 flex items-center justify-center text-gray-600 text-xs p-4 text-center">
              Graph appears here after traversal
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
            <span className="text-orange-400">●</span>
            <span>Forced jumps</span>
            <span className="text-gray-300">{stats.forced_jumps}</span>
          </div>
        </footer>
      )}
    </div>
  );
}
