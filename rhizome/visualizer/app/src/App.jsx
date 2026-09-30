import { useState } from 'react';
import TraverseTab from './components/TraverseTab.jsx';
import SynthesizeTab from './components/SynthesizeTab.jsx';

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
  depth: 20,
  epsilon: 0.1,
  top_k: 30,
  temperature: 1.0,
  max_same_article_consecutive: 2,
};

export default function App() {
  const [tab, setTab] = useState('synthesize');
  const [params, setParams] = useState(DEFAULT_PARAMS);

  return (
    <div className="flex flex-col h-screen bg-bg-primary overflow-hidden">
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
        </div>
        <nav className="flex gap-1" role="tablist">
          <button
            type="button"
            role="tab"
            aria-selected={tab === 'synthesize'}
            onClick={() => setTab('synthesize')}
            className={`px-4 py-2 text-sm rounded-t border border-b-0 transition-colors cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
              tab === 'synthesize'
                ? 'bg-bg-primary border-border text-text-primary'
                : 'bg-transparent border-transparent text-text-muted hover:text-text-primary'
            }`}
          >
            Synthesize
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={tab === 'traverse'}
            onClick={() => setTab('traverse')}
            className={`px-4 py-2 text-sm rounded-t border border-b-0 transition-colors cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
              tab === 'traverse'
                ? 'bg-bg-primary border-border text-text-primary'
                : 'bg-transparent border-transparent text-text-muted hover:text-text-primary'
            }`}
          >
            Traverse
          </button>
        </nav>
      </header>

      <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
        {tab === 'synthesize' ? (
          <SynthesizeTab params={params} setParams={setParams} />
        ) : (
          <TraverseTab params={params} setParams={setParams} />
        )}
      </div>
    </div>
  );
}
