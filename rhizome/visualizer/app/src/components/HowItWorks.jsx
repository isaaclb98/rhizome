import React from 'react';

const STEPS_SYNTHESIZE = [
  {
    n: 1,
    title: 'Walk',
    body: 'A vector search reads chunks of Wikipedia in order. Each chunk is a step; the next step is found by similarity to the previous one.',
  },
  {
    n: 2,
    title: 'Collect',
    body: 'After the walk finishes, the fragments are gathered as raw material. Length and shape vary with depth and randomness.',
  },
  {
    n: 3,
    title: 'Forge',
    body: 'The fragments — and optionally the seed — are sent to an LLM, which argues one thesis from them. Assertive, no hedging, no citations.',
  },
];

const STEPS_TRAVERSE = [
  {
    n: 1,
    title: 'Walk',
    body: 'A vector search reads chunks of Wikipedia in order. Each chunk is a step; the next step is found by similarity to the previous one.',
  },
  {
    n: 2,
    title: 'List',
    body: 'The path appears on the left as a vertical list — each step shows its title, a similarity score, and the chunk itself.',
  },
  {
    n: 3,
    title: 'Visualise',
    body: 'When the walk is done, the right side renders the path as a graph: each node is a step, each edge the similarity jump between them.',
  },
];

export default function HowItWorks({ mode }) {
  const steps = mode === 'synthesize' ? STEPS_SYNTHESIZE : STEPS_TRAVERSE;
  return (
    <section className="mt-3 bg-bg-secondary border border-border rounded-lg p-4">
      <header className="mb-3 flex items-baseline gap-2">
        <h3 className="text-sm font-semibold tracking-tight text-text-primary">
          How this works
        </h3>
        <span className="text-[11px] text-text-muted">
          the three things that will happen
        </span>
      </header>
      <ol className="grid grid-cols-3 gap-3">
        {steps.map((s) => (
          <li key={s.n} className="flex flex-col gap-1">
            <div className="flex items-baseline gap-2">
              <span className="text-[11px] font-mono text-text-muted w-4">
                {String(s.n).padStart(2, '0')}
              </span>
              <span className="text-sm font-medium text-text-primary">
                {s.title}
              </span>
            </div>
            <p className="text-xs text-text-secondary leading-relaxed pl-6">
              {s.body}
            </p>
          </li>
        ))}
      </ol>
    </section>
  );
}
