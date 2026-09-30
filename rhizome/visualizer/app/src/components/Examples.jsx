import React from 'react';

const EXAMPLES = [
  {
    title: 'The aesthetics of absence',
    body: 'How negative space, the void, and silence became first-class aesthetic objects across 20th-century art and music.',
  },
  {
    title: 'The archive as a technology of memory',
    body: 'What it means for memory to live in a substrate — and what that substrate decides.',
  },
  {
    title: 'Why late style resists summary',
    body: 'Beethoven, Beckett, Matisse at the end: works that refuse to round off the arc of a career.',
  },
  {
    title: 'Indeterminacy in art and philosophy',
    body: 'From John Cage to quantum mechanics: a shared fascination with the undecided.',
  },
  {
    title: 'How did Abstract Expressionism influence contemporary art',
    body: 'The traces of gestural painting and scale in art made fifty years later.',
  },
  {
    title: 'The economics of attention',
    body: 'What it costs a mind to focus, and how that cost has been quietly re-priced.',
  },
];

export default function Examples({ onPick, disabled }) {
  return (
    <div className="h-full overflow-y-auto px-6 py-8 bg-bg-primary">
      <div className="max-w-3xl mx-auto">
        <div className="text-text-muted text-xs uppercase tracking-[0.18em] mb-4">
          Or start with an example
        </div>
        <ul className="grid grid-cols-1 md:grid-cols-2 gap-2">
          {EXAMPLES.map((ex) => (
            <li key={ex.title}>
              <button
                type="button"
                onClick={() => onPick(ex.title)}
                disabled={disabled}
                className="w-full text-left px-4 py-3 bg-bg-secondary border border-border rounded-md hover:border-accent hover:bg-bg-tertiary transition-colors disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent group"
              >
                <div className="text-sm font-medium text-text-primary group-hover:text-accent transition-colors">
                  {ex.title}
                </div>
                <div className="text-xs text-text-muted mt-1 leading-relaxed">
                  {ex.body}
                </div>
              </button>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
