import { useState, useEffect, useCallback } from 'react';
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
  depth: 10,
  epsilon: 0.1,
  top_k: 30,
  temperature: 1.0,
  max_same_article_consecutive: 2,
  inject_seed: false,
};

const THEME_STORAGE_KEY = 'rhizome-theme';

function readInitialTheme() {
  const stored = localStorage.getItem(THEME_STORAGE_KEY);
  if (stored === 'light' || stored === 'dark') return stored;
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function ThemeButton({ theme, onToggle }) {
  const next = theme === 'light' ? 'dark' : 'light';
  return (
    <button
      type="button"
      onClick={onToggle}
      className="w-full flex items-center gap-3 px-3 py-2 text-sm text-text-secondary hover:text-text-primary hover:bg-bg-tertiary rounded-md transition-colors cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
      title={`Switch to ${next} mode`}
      aria-label="Toggle theme"
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
          <line x1="18.36" y1="5.64" x2="19.78" y2="18.36"/>
        </svg>
      )}
    </button>
  );
}

export default function App() {
  const [params, setParams] = useState(DEFAULT_PARAMS);
  const [theme, setTheme] = useState(readInitialTheme);

  // Initial tab can come from ?tab=synthesize|traverse — useful for headless
  // tests and deep links. Falls back to synthesize.
  const [tab, setTab] = useState(() => {
    const fromUrl = new URLSearchParams(window.location.search).get('tab');
    return fromUrl === 'traverse' ? 'traverse' : 'synthesize';
  });

  // Apply theme to <html data-theme> whenever it changes. Single source of
  // truth so tab switches don't reset it and a stale localStorage value
  // doesn't surprise the user.
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme((prev) => {
      const next = prev === 'light' ? 'dark' : 'light';
      localStorage.setItem(THEME_STORAGE_KEY, next);
      return next;
    });
  }, []);

  return (
    <div className="flex h-screen bg-bg-primary overflow-hidden">
      {/* Left sidebar — mode switcher */}
      <aside className="flex-none w-56 bg-bg-secondary border-r border-border flex flex-col">
        <div className="px-4 py-4 border-b border-border">
          <h1 className="text-xl font-bold tracking-tight text-text-primary">
            Rhizome
          </h1>
          <span className="text-xs text-text-muted">
            Wikipedia semantic traversal
          </span>
        </div>

        <nav className="flex-1 px-2 py-3 space-y-1" role="tablist" aria-label="Sections">
          <button
            type="button"
            role="tab"
            aria-selected={tab === 'synthesize'}
            onClick={() => setTab('synthesize')}
            className={`w-full flex items-center gap-3 px-3 py-2 text-sm rounded-md transition-colors cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
              tab === 'synthesize'
                ? 'bg-bg-tertiary text-text-primary font-medium'
                : 'text-text-secondary hover:bg-bg-tertiary hover:text-text-primary'
            }`}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 19l7-7 3 3-7 7-3-3z"/>
              <path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z"/>
              <path d="M2 2l7.586 7.586"/>
              <circle cx="11" cy="11" r="2"/>
            </svg>
            Synthesize
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={tab === 'traverse'}
            onClick={() => setTab('traverse')}
            className={`w-full flex items-center gap-3 px-3 py-2 text-sm rounded-md transition-colors cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
              tab === 'traverse'
                ? 'bg-bg-tertiary text-text-primary font-medium'
                : 'text-text-secondary hover:bg-bg-tertiary hover:text-text-primary'
            }`}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="6" cy="6" r="3"/>
              <circle cx="18" cy="6" r="3"/>
              <circle cx="12" cy="18" r="3"/>
              <line x1="8.5" y1="7.5" x2="10.5" y2="15.5"/>
              <line x1="15.5" y1="7.5" x2="13.5" y2="15.5"/>
              <line x1="6" y1="6" x2="18" y2="6"/>
            </svg>
            Traverse
          </button>
        </nav>

        <div className="px-2 py-3 border-t border-border">
          <ThemeButton theme={theme} onToggle={toggleTheme} />
        </div>
      </aside>

      {/* Main content area */}
      <main className="flex-1 min-w-0 flex flex-col overflow-hidden">
        <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
          {tab === 'synthesize' ? (
            <SynthesizeTab params={params} setParams={setParams} />
          ) : (
            <TraverseTab params={params} setParams={setParams} />
          )}
        </div>
      </main>
    </div>
  );
}
