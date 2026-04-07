import { useState, useEffect, useCallback } from 'react';

export default function Controls({ params, onTraverse, isLoading }) {
  const [query, setQuery] = useState(params.query);
  const [depth, setDepth] = useState(params.depth);
  const [epsilon, setEpsilon] = useState(params.epsilon);
  const [topK, setTopK] = useState(params.top_k);
  const [temperature, setTemperature] = useState(params.temperature);
  const [maxSameArticle, setMaxSameArticle] = useState(params.max_same_article_consecutive);

  // Read theme from the document root (set by App on mount) to avoid read race
  const [theme, setTheme] = useState(() => document.documentElement.getAttribute('data-theme') || 'light');

  useEffect(() => {
    const observer = new MutationObserver(() => {
      const t = document.documentElement.getAttribute('data-theme');
      if (t) setTheme(t);
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    return () => observer.disconnect();
  }, []);

  const toggleTheme = useCallback(() => {
    const next = theme === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('rhizome-theme', next);
  }, [theme]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    onTraverse({
      query: query.trim(),
      depth,
      epsilon,
      top_k: topK,
      temperature,
      max_same_article_consecutive: maxSameArticle,
    });
  };

  const inputClass = `w-full bg-bg-secondary border border-border rounded px-3 py-2 text-sm text-text-primary placeholder-text-muted focus:outline-none focus:border-accent focus:ring-1 focus:ring-accent transition-colors`;
  const labelClass = 'block text-xs text-text-muted mb-1';

  return (
    <form onSubmit={handleSubmit} className="flex items-end gap-4 flex-wrap">
      {/* Query input */}
      <div className="flex-1 min-w-64">
        <label className={labelClass} htmlFor="query">
          Query
        </label>
        <input
          id="query"
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="e.g. the tension between modernism and postmodernism"
          className={inputClass}
          disabled={isLoading}
        />
      </div>

      {/* Depth */}
      <div className="w-24">
        <label className={labelClass} htmlFor="depth">
          Depth <span className="text-text-muted">(1-100)</span>
        </label>
        <input
          id="depth"
          type="number"
          min={1}
          max={100}
          value={depth}
          onChange={(e) => setDepth(Number(e.target.value))}
          className={inputClass}
          disabled={isLoading}
        />
      </div>

      {/* Epsilon */}
      <div className="w-28">
        <label className={labelClass} htmlFor="epsilon">
          Epsilon <span className="text-text-muted">(0-1)</span>
        </label>
        <input
          id="epsilon"
          type="number"
          min={0}
          max={1}
          step={0.05}
          value={epsilon}
          onChange={(e) => setEpsilon(Number(e.target.value))}
          className={inputClass}
          disabled={isLoading}
        />
      </div>

      {/* Top K */}
      <div className="w-24">
        <label className={labelClass} htmlFor="topK">
          Top K <span className="text-text-muted">(1-50)</span>
        </label>
        <input
          id="topK"
          type="number"
          min={1}
          max={50}
          value={topK}
          onChange={(e) => setTopK(Number(e.target.value))}
          className={inputClass}
          disabled={isLoading}
        />
      </div>

      {/* Temperature */}
      <div className="w-24">
        <label className={labelClass} htmlFor="temperature">
          Temp <span className="text-text-muted">(0-3)</span>
        </label>
        <input
          id="temperature"
          type="number"
          min={0}
          max={3}
          step={0.1}
          value={temperature}
          onChange={(e) => setTemperature(Number(e.target.value))}
          className={inputClass}
          disabled={isLoading}
        />
      </div>

      {/* Max Same Article */}
      <div className="w-28">
        <label className={labelClass} htmlFor="maxSameArticle">
          Same Art. <span className="text-text-muted">(0-20)</span>
        </label>
        <input
          id="maxSameArticle"
          type="number"
          min={0}
          max={20}
          value={maxSameArticle}
          onChange={(e) => setMaxSameArticle(Number(e.target.value))}
          className={inputClass}
          disabled={isLoading}
        />
      </div>

      {/* Theme toggle */}
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

      {/* Submit */}
      <button
        type="submit"
        disabled={isLoading || !query.trim()}
        className="px-5 py-2 bg-accent hover:bg-accent-hover disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium rounded transition-colors cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-bg-primary"
      >
        {isLoading ? 'Traversing...' : 'Traverse'}
      </button>
    </form>
  );
}
