import { useState, useCallback, useEffect, useRef } from 'react';
import Controls from './Controls.jsx';
import Graph from './Graph.jsx';
import PathPanel from './PathPanel.jsx';

const EXAMPLE_QUERIES = [
  'the tension between modernism and postmodernism',
  'what is the relationship between art and technology',
  'the aesthetics of modernism',
  'how did Abstract Expressionism influence contemporary art',
  'the death of the author and literary theory',
  'indeterminacy in art and philosophy',
  'fragmentation',
];

const DEFAULT_PARAMS = {
  query: EXAMPLE_QUERIES[Math.floor(Math.random() * EXAMPLE_QUERIES.length)],
  depth: 10,
  epsilon: 0.1,
  top_k: 30,
  temperature: 1.0,
  max_same_article_consecutive: 2,
};

export default function TraverseTab({ params, setParams }) {
  const [path, setPath] = useState([]);
  const [stats, setStats] = useState(null);
  const [selectedChunkId, setSelectedChunkId] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState(null);
  const abortControllerRef = useRef(null);
  const forcedJumpsRef = useRef(0);

  // Theme lives in App.jsx — read it once on mount for any data-* attributes
  // that depend on the current value, but do not own the state here.

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
    if (setParams) setParams(requestParams);

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
              prev ? { ...prev, ...(data.stats || {}), forced_jumps: forcedJumpsRef.current } : prev
            );
            setIsStreaming(false);
            setIsLoading(false);
          } else if (data.type === 'error') {
            if (data.code === 'ABORTED') {
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
  }, [setParams]);

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
    <>
      {/* Form row (no internal section title — the tab already names the section) */}
      <section className="flex-none bg-bg-secondary border-b border-border px-4 py-3">
        <Controls params={params} onTraverse={handleStreamTraverse} isLoading={isLoading} />
      </section>

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
              <div className="absolute inset-0 flex flex-col items-center justify-center text-text-muted text-xs text-center gap-2 px-6">
                <span>Graph of the walk's path</span>
                <span>appears here after traversal</span>
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
    </>
  );
}
