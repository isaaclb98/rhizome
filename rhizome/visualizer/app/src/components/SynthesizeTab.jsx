import { useState, useCallback, useEffect, useRef } from 'react';
import Markdown from 'react-markdown';

export default function SynthesizeTab({ params, setParams }) {
  const [thesis, setThesis] = useState({ main_thesis: '', content: '' });
  const [path, setPath] = useState([]);
  const [stats, setStats] = useState(null);
  const [walkProgress, setWalkProgress] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const abortControllerRef = useRef(null);

  const [localParams, setLocalParams] = useState({
    query: params.query,
    depth: params.depth,
    epsilon: params.epsilon,
    temperature: params.temperature,
    max_same_article_consecutive: params.max_same_article_consecutive,
    inject_seed: params.inject_seed,
  });

  const updateParam = (key, value) => {
    setLocalParams((prev) => ({ ...prev, [key]: value }));
  };

  const handleSynthesize = useCallback(async () => {
    if (!localParams.query.trim()) return;

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;

    setThesis({ main_thesis: '', content: '' });
    setPath([]);
    setStats(null);
    setWalkProgress({ walked: 0, total: localParams.depth });
    setError(null);
    setIsLoading(true);

    if (setParams) {
      setParams({
        ...params,
        query: localParams.query,
        depth: localParams.depth,
        epsilon: localParams.epsilon,
        temperature: localParams.temperature,
        max_same_article_consecutive: localParams.max_same_article_consecutive,
      });
    }

    try {
      const response = await fetch('/idea/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(localParams),
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
            setPath((prev) => {
              if (prev.some((s) => s.chunk_id === data.chunk_id)) return prev;
              return [...prev, data];
            });
            setWalkProgress((prev) =>
              prev ? { ...prev, walked: prev.walked + 1 } : prev
            );
          } else if (data.type === 'thesis') {
            setThesis({
              main_thesis: data.main_thesis || '',
              content: data.content || '',
            });
          } else if (data.type === 'done') {
            setStats(data.stats || null);
          } else if (data.type === 'error') {
            setError(data.detail || 'Synthesis error');
          }
        }
      }
    } catch (err) {
      if (err.name === 'AbortError' || err.message === 'The user aborted a request.') {
        // User-initiated cancellation
      } else {
        setError(err.message || 'Synthesis failed');
      }
    } finally {
      setIsLoading(false);
    }
  }, [localParams, params, setParams]);

  // Abort on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  const inputClass = `w-full bg-bg-secondary border border-border rounded px-3 py-2 text-sm text-text-primary placeholder-text-muted focus:outline-none focus:border-accent focus:ring-1 focus:ring-accent transition-colors`;
  const labelClass = 'block text-xs text-text-muted mb-1';

  return (
    <>
      {/* Header */}
      <header className="flex-none bg-bg-secondary border-b border-border px-4 py-3">
        <div className="flex items-center gap-3 mb-3">
          <h2 className="text-lg font-bold tracking-tight text-text-primary">
            Synthesize
          </h2>
          <span className="text-xs text-text-muted font-mono">
            Walk the corpus, then forge a thesis from the material
          </span>
        </div>

        {/* Inline form */}
        <form
          onSubmit={(e) => { e.preventDefault(); handleSynthesize(); }}
          className="flex items-end gap-4 flex-wrap"
        >
          <div className="flex-1 min-w-64">
            <label className={labelClass} htmlFor="seed">Seed</label>
            <input
              id="seed"
              type="text"
              value={localParams.query}
              onChange={(e) => updateParam('query', e.target.value)}
              placeholder="e.g. the tension between structure and event"
              className={inputClass}
              disabled={isLoading}
            />
          </div>
          <div className="w-20">
            <label className={labelClass} htmlFor="depth">Depth</label>
            <input
              id="depth" type="number" min="1" max="50"
              value={localParams.depth}
              onChange={(e) => updateParam('depth', Number(e.target.value))}
              className={inputClass}
              disabled={isLoading}
            />
          </div>
          <div className="w-20">
            <label className={labelClass} htmlFor="epsilon">ε</label>
            <input
              id="epsilon" type="number" min="0" max="1" step="0.05"
              value={localParams.epsilon}
              onChange={(e) => updateParam('epsilon', Number(e.target.value))}
              className={inputClass}
              disabled={isLoading}
            />
          </div>
          <div className="w-20">
            <label className={labelClass} htmlFor="temp">temp</label>
            <input
              id="temp" type="number" min="0" max="3" step="0.1"
              value={localParams.temperature}
              onChange={(e) => updateParam('temperature', Number(e.target.value))}
              className={inputClass}
              disabled={isLoading}
            />
          </div>
          <div className="w-24">
            <label className={labelClass} htmlFor="same-art">same-art</label>
            <input
              id="same-art" type="number" min="0" max="10"
              value={localParams.max_same_article_consecutive}
              onChange={(e) => updateParam('max_same_article_consecutive', Number(e.target.value))}
              className={inputClass}
              disabled={isLoading}
            />
          </div>

          <label
            className="inline-flex items-center gap-2 pb-2.5 text-sm text-text-secondary cursor-pointer select-none"
            title="When on, the seed text is included in the LLM prompt as a synthesis lens. Off: the model only sees the fragments."
          >
            <input
              type="checkbox"
              checked={localParams.inject_seed}
              onChange={(e) => updateParam('inject_seed', e.target.checked)}
              disabled={isLoading}
              className="h-4 w-4 rounded border-border bg-bg-secondary text-accent focus:ring-2 focus:ring-accent cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
            />
            Include seed in prompt
          </label>

          <button
            type="submit"
            disabled={isLoading || !localParams.query.trim()}
            className="px-4 py-2 text-sm bg-accent text-white rounded hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
          >
            {isLoading ? 'Synthesizing…' : 'Synthesize'}
          </button>
        </form>
      </header>

      {/* Error banner */}
      {error && (
        <div className="flex-none bg-red-900/30 border-b border-red-800 px-4 py-2 text-red-300 text-sm">
          {error}
        </div>
      )}

      {/* Main: thesis + walk material */}
      <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
        {/* Thesis panel */}
        <div className="flex-1 min-h-0 overflow-y-auto px-6 py-6 bg-bg-primary">
          {thesis.content ? (
            <article className="markdown-body max-w-none text-text-primary">
              {thesis.main_thesis ? (
                <header className="mb-6 pb-5 border-b border-border">
                  <div className="text-xs uppercase tracking-wider text-text-muted mb-2">
                    Thesis
                  </div>
                  <p className="text-lg font-semibold leading-snug text-text-primary">
                    {thesis.main_thesis}
                  </p>
                </header>
              ) : null}
              <Markdown>{thesis.content}</Markdown>
            </article>
          ) : isLoading ? (
            <div className="text-text-muted text-sm">
              Synthesizing…
            </div>
          ) : (
            <div className="text-text-muted text-sm">
              Run a synthesis to begin.
            </div>
          )}
        </div>

        {/* Walk material disclosure */}
        {path.length > 0 && (
          <details className="flex-none border-t border-border bg-bg-secondary">
            <summary className="cursor-pointer px-4 py-2 text-xs text-text-muted hover:text-text-primary select-none">
              Walk material — {path.length} step(s){stats ? `, ${stats.forced_jumps} jump(s)` : ''}
              {walkProgress && walkProgress.walked < walkProgress.total && isLoading ? (
                <span className="ml-2 text-accent">
                  walking {walkProgress.walked}/{walkProgress.total}…
                </span>
              ) : null}
            </summary>
            <ol className="max-h-64 overflow-y-auto px-4 py-2 space-y-2">
              {path.map((step, idx) => (
                <li key={step.chunk_id || idx} className="text-xs">
                  <div className="flex items-center gap-2 text-text-muted">
                    <span>[{idx}]</span>
                    <a
                      href={step.article_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-accent hover:underline"
                    >
                      {step.article_title}
                    </a>
                    {step.forced_jump && (
                      <span className="text-amber-400" title="Forced jump — unrelated to the preceding fragment">↯</span>
                    )}
                    <span className="text-text-muted">sim {step.similarity?.toFixed?.(3) ?? '–'}</span>
                  </div>
                </li>
              ))}
            </ol>
          </details>
        )}
      </div>
    </>
  );
}
