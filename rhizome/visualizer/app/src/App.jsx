import { useState, useCallback, useEffect, useRef } from 'react';
import Controls from './components/Controls.jsx';
import Graph from './components/Graph.jsx';
import PathPanel from './components/PathPanel.jsx';

const DEFAULT_PARAMS = {
  query: 'the tension between modernism and postmodernism',
  depth: 20,
  epsilon: 0.1,
  top_k: 30,
  temperature: 1.0,
  max_same_article_consecutive: 2,
};

export default function App() {
  const [path, setPath] = useState([]);
  const [stats, setStats] = useState(null);
  const [selectedChunkId, setSelectedChunkId] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState(null);
  const [params, setParams] = useState(DEFAULT_PARAMS);
  const abortControllerRef = useRef(null);
  const forcedJumpsRef = useRef(0);

  // Theme: default light, read from localStorage on mount
  useEffect(() => {
    const stored = localStorage.getItem('rhizome-theme');
    const initial = stored || 'light';
    document.documentElement.setAttribute('data-theme', initial);
  }, []);

  const [theme, setTheme] = useState('light');
  const toggleTheme = useCallback(() => {
    setTheme(prev => {
      const next = prev === 'light' ? 'dark' : 'light';
      document.documentElement.setAttribute('data-theme', next);
      localStorage.setItem('rhizome-theme', next);
      return next;
    });
  }, []);

  const handleStreamTraverse = useCallback(async (requestParams) => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;

    setIsLoading(true);
    setIsStreaming(true);
    setError(null);
    setSelectedChunkId(null);
    setPath([]);
    forcedJumpsRef.current = 0;
    setStats({
      depth: requestParams.depth,
      epsilon: requestParams.epsilon,
      top_k: requestParams.top_k,
      temperature: requestParams.temperature,
      max_same_article_consecutive: requestParams.max_same_article_consecutive,
      forced_jumps: 0,
    });
    setParams(requestParams);

    try {
      const response = await fetch('/traverse/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestParams),
        signal: controller.signal,
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(err.detail || `HTTP ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const raw = line.slice(6);
          if (!raw.trim()) continue;

          let data;
          try {
            data = JSON.parse(raw);
          } catch {
            continue;
          }

          if (data.type === 'step') {
            if (data.forced_jump) {
              forcedJumpsRef.current += 1;
              setStats((prev) =>
                prev ? { ...prev, forced_jumps: forcedJumpsRef.current } : prev
              );
            }
            const step = {
              chunk_id: data.chunk_id,
              text: data.text,
              article_title: data.article_title,
              article_url: data.article_url || '',
              depth: data.depth,
              similarity: data.similarity,
              forced_jump: data.forced_jump,
              candidates: data.candidates || [],
            };
            setPath((prev) => {
              if (prev.some((s) => s.chunk_id === step.chunk_id)) return prev;
              return [...prev, step];
            });
          } else if (data.type === 'done') {
            setStats((prev) =>
              prev ? { ...prev, forced_jumps: forcedJumpsRef.current } : prev
            );
            setIsStreaming(false);
            setIsLoading(false);
          } else if (data.type === 'error') {
            if (data.code === 'ABORTED') {
              // Client disconnect — silent
              setIsStreaming(false);
              setIsLoading(false);
            } else {
              setError(data.message || 'Traversal error');
              setIsStreaming(false);
              setIsLoading(false);
            }
          }
        }
      }
    } catch (err) {
      if (err.name === 'AbortError' || err.message === 'The user aborted a request.') {
        setIsStreaming(false);
        setIsLoading(false);
        return;
      }
      setError(err.message || 'Traversal failed');
      setPath([]);
      setStats(null);
      setIsStreaming(false);
    } finally {
      setIsLoading(false);
      setIsStreaming(false);
    }
  }, []);

  const handleNodeClick = useCallback((node) => {
    const id = node?.id ?? node?.chunk_id ?? null;
    setSelectedChunkId((prev) => (prev === id ? prev : id));
  }, []);

  // Abort on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  return (
    <div className="flex flex-col h-screen bg-bg-primary overflow-hidden">
      {/* Header */}
      <header className="flex-none bg-bg-secondary border-b border-border px-4 py-3">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold tracking-tight text-text-primary">
              Rhizome
            </h1>
            <span className="text-xs text-text-muted font-mono">
              Wikipedia semantic traversal
            </span>
          </div>
          <button
            type="button"
            onClick={toggleTheme}
            className="px-3 py-2 text-sm text-text-muted hover:text-text-primary border border-border rounded transition-colors cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
            title={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
          >
            {theme === 'light' ? (
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
              </svg>
            ) : (
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="5"/>
                <line x1="12" y1="1" x2="12" y2="3"/>
                <line x1="12" y1="21" x2="12" y2="23"/>
                <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
                <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
                <line x1="1" y1="12" x2="3" y2="12"/>
                <line x1="21" y1="12" x2="23" y2="12"/>
                <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
                <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
              </svg>
            )}
          </button>
        </div>
        <Controls params={params} onTraverse={handleStreamTraverse} isLoading={isLoading} />
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
        <div className="flex-1 min-h-0 overflow-hidden bg-bg-secondary">
          <PathPanel
            path={path}
            selectedChunkId={selectedChunkId}
            onSelectChunk={handleNodeClick}
          />
        </div>

        {/* Right: graph strip */}
        <div className="hidden lg:flex lg:flex-col lg:w-80 xl:w-96 border-l border-border overflow-hidden flex-shrink-0">
          {path.length > 0 ? (
            <Graph
              path={path}
              selectedChunkId={selectedChunkId}
              onNodeClick={handleNodeClick}
              depth={stats?.depth ?? params.depth}
            />
          ) : (
            <div className="relative flex-1">
              <div className="absolute inset-0 flex flex-col items-center justify-center text-text-muted text-xs text-center gap-2">
                <span>Graph appears here</span>
                <span>after traversal</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      {stats && (
        <footer className="flex-none border-t border-border px-4 py-1.5 flex items-center gap-4 text-xs text-text-muted">
          <div className="flex items-center gap-1.5">
            <span>Depth</span>
            <span className="text-text-primary">{stats.depth}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span>ε</span>
            <span className="text-text-primary">{stats.epsilon}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span>top_k</span>
            <span className="text-text-primary">{stats.top_k}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span>temp</span>
            <span className="text-text-primary">{stats.temperature}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span>same-art</span>
            <span className="text-text-primary">{stats.max_same_article_consecutive}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-accent">●</span>
            <span>Forced jumps</span>
            <span className="text-text-primary">{stats.forced_jumps}</span>
          </div>
        </footer>
      )}
    </div>
  );
}
