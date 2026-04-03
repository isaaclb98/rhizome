import { useState } from 'react';

export default function Controls({ params, onTraverse, isLoading }) {
  const [query, setQuery] = useState(params.query);
  const [depth, setDepth] = useState(params.depth);
  const [epsilon, setEpsilon] = useState(params.epsilon);
  const [topK, setTopK] = useState(params.top_k);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    onTraverse({ query: query.trim(), depth, epsilon, top_k: topK });
  };

  return (
    <form onSubmit={handleSubmit} className="flex items-end gap-4 flex-wrap">
      {/* Query input */}
      <div className="flex-1 min-w-64">
        <label className="block text-xs text-gray-400 mb-1" htmlFor="query">
          Query
        </label>
        <input
          id="query"
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="e.g. the tension between modernism and postmodernism"
          className="w-full bg-bg-secondary border border-bg-tertiary rounded px-3 py-2 text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-colors"
          disabled={isLoading}
        />
      </div>

      {/* Depth */}
      <div className="w-24">
        <label className="block text-xs text-gray-400 mb-1" htmlFor="depth">
          Depth <span className="text-gray-600">(1-20)</span>
        </label>
        <input
          id="depth"
          type="number"
          min={1}
          max={20}
          value={depth}
          onChange={(e) => setDepth(Number(e.target.value))}
          className="w-full bg-bg-secondary border border-bg-tertiary rounded px-3 py-2 text-sm text-gray-200 focus:outline-none focus:border-blue-500 transition-colors"
          disabled={isLoading}
        />
      </div>

      {/* Epsilon */}
      <div className="w-28">
        <label className="block text-xs text-gray-400 mb-1" htmlFor="epsilon">
          Epsilon <span className="text-gray-600">(0-1)</span>
        </label>
        <input
          id="epsilon"
          type="number"
          min={0}
          max={1}
          step={0.05}
          value={epsilon}
          onChange={(e) => setEpsilon(Number(e.target.value))}
          className="w-full bg-bg-secondary border border-bg-tertiary rounded px-3 py-2 text-sm text-gray-200 focus:outline-none focus:border-blue-500 transition-colors"
          disabled={isLoading}
        />
      </div>

      {/* Top K */}
      <div className="w-24">
        <label className="block text-xs text-gray-400 mb-1" htmlFor="topK">
          Top K <span className="text-gray-600">(1-20)</span>
        </label>
        <input
          id="topK"
          type="number"
          min={1}
          max={20}
          value={topK}
          onChange={(e) => setTopK(Number(e.target.value))}
          className="w-full bg-bg-secondary border border-bg-tertiary rounded px-3 py-2 text-sm text-gray-200 focus:outline-none focus:border-blue-500 transition-colors"
          disabled={isLoading}
        />
      </div>

      {/* Submit */}
      <button
        type="submit"
        disabled={isLoading || !query.trim()}
        className="px-5 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-blue-900/40 disabled:text-gray-600 text-white text-sm font-medium rounded transition-colors cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 focus-visible:ring-offset-2 focus-visible:ring-offset-bg-primary"
      >
        {isLoading ? 'Traversing...' : 'Traverse'}
      </button>
    </form>
  );
}
