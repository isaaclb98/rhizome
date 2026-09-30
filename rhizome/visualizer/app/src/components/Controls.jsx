import { useState } from 'react';
import Slider from './Slider.jsx';
import { explorationToParams } from '../exploration.js';

export default function Controls({ params, onTraverse, isLoading }) {
  const [query, setQuery] = useState(params.query ?? '');
  const [depth, setDepth] = useState(params.depth);
  const [exploration, setExploration] = useState(params.exploration);
  const [maxSameArticle, setMaxSameArticle] = useState(params.max_same_article_consecutive);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    const { epsilon, temperature } = explorationToParams(exploration);
    onTraverse({
      query: query.trim(),
      depth,
      exploration,
      epsilon,
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

      {/* Length */}
      <div className="w-24">
        <label className={labelClass} htmlFor="depth">
          Length <span className="text-text-muted">(1-50)</span>
        </label>
        <input
          id="depth"
          type="number"
          min={1}
          max={50}
          value={depth}
          onChange={(e) => setDepth(Number(e.target.value))}
          className={inputClass}
          disabled={isLoading}
          title="How many steps the walk takes. Each step reads one chunk of Wikipedia. Default 10."
        />
      </div>

      {/* Exploration slider */}
      <div className="w-44">
        <div className="flex items-baseline justify-between mb-1">
          <label className={labelClass}>Exploration</label>
          <span className="text-xs text-text-muted font-mono">
            {exploration.toFixed(2)}
          </span>
        </div>
        <Slider
          value={exploration}
          onChange={setExploration}
          min={0}
          max={1}
          step={0.01}
          disabled={isLoading}
          ariaLabel="Exploration"
        />
        <p className="text-[11px] text-text-muted mt-0.5">stay on topic ↔ wander</p>
      </div>

      {/* Same Article Limit */}
      <div className="w-28">
        <label className={labelClass} htmlFor="maxSameArticle">
          Same Art. <span className="text-text-muted">(0-10)</span>
        </label>
        <input
          id="maxSameArticle"
          type="number"
          min={0}
          max={10}
          value={maxSameArticle}
          onChange={(e) => setMaxSameArticle(Number(e.target.value))}
          className={inputClass}
          disabled={isLoading}
          title="Force a new article after this many consecutive steps. 0 disables the rule. Default 2."
        />
      </div>

      {/* Submit */}
      <button
        type="submit"
        disabled={isLoading || !query.trim()}
        className="px-5 py-2 bg-accent hover:bg-accent-hover disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium rounded transition-colors cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-bg-secondary"
      >
        {isLoading ? 'Traversing…' : 'Traverse'}
      </button>
    </form>
  );
}
