import { useState } from 'react';

export default function Controls({ params, onTraverse, isLoading }) {
  const [query, setQuery] = useState(params.query);
  const [depth, setDepth] = useState(params.depth);
  const [epsilon, setEpsilon] = useState(params.epsilon);
  const [topK, setTopK] = useState(params.top_k);
  const [temperature, setTemperature] = useState(params.temperature);
  const [maxSameArticle, setMaxSameArticle] = useState(params.max_same_article_consecutive);

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
    <form onSubmit={handleSubmit} className="space-y-3">
      {/* Query input — its own row, full width */}
      <div>
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

      {/* Dials row */}
      <div className="flex flex-wrap gap-3">
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
            title="How many steps the walk takes from the starting concept. Each step picks the next chunk from the corpus based on similarity. Default 10."
          />
          <p className="text-[11px] text-text-muted mt-0.5">walk length</p>
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
            title="Exploration probability. 0 = always pick the most-similar chunk (greedy). 1 = always pick at random. 0.1 is a small nudge toward surprise."
          />
          <p className="text-[11px] text-text-muted mt-0.5">explore</p>
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
            title="How many candidates Qdrant returns per step. The walker picks one from these. Larger = more options but slower."
          />
          <p className="text-[11px] text-text-muted mt-0.5">candidates</p>
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
            title="Softness of the pick. 0 = always the most-similar non-blocked candidate. Higher = more likely to pick a less-similar one. Affects randomness independently of epsilon."
          />
          <p className="text-[11px] text-text-muted mt-0.5">pick softness</p>
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
            title="How many consecutive steps can come from the same Wikipedia article. After this many, the walker is forced to jump to a different article. 0 disables the rule."
          />
          <p className="text-[11px] text-text-muted mt-0.5">stay limit</p>
        </div>
      </div>

      {/* Submit */}
      <div className="flex justify-end pt-1">
        <button
          type="submit"
          disabled={isLoading || !query.trim()}
          className="px-5 py-2 bg-accent hover:bg-accent-hover disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium rounded transition-colors cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-bg-secondary"
        >
          {isLoading ? 'Traversing…' : 'Traverse'}
        </button>
      </div>
    </form>
  );
}
