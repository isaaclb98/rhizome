// Deterministic color palette for domain coloring
const DOMAIN_PALETTE = [
  '#6366f1', // indigo
  '#a855f7', // purple
  '#f97316', // orange
  '#22c55e', // green
  '#06b6d4', // cyan
  '#ec4899', // pink
  '#eab308', // yellow
  '#14b8a6', // teal
  '#f43f5e', // rose
  '#8b5cf6', // violet
  '#84cc16', // lime
  '#0ea5e9', // sky
];

function getDomainColor(domain, index) {
  return DOMAIN_PALETTE[index % DOMAIN_PALETTE.length];
}

export default function Legend({ domains = [] }) {
  if (domains.length === 0) {
    return (
      <div className="flex items-center gap-4 flex-wrap">
        <span className="text-xs text-gray-600 italic">loading domains…</span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-4 flex-wrap">
      {domains.map((name, i) => (
        <div key={name} className="flex items-center gap-1.5">
          <span
            className="inline-block w-2.5 h-2.5 rounded-full flex-shrink-0"
            style={{ backgroundColor: getDomainColor(name, i) }}
          />
          <span className="text-xs text-gray-400">{name}</span>
        </div>
      ))}
    </div>
  );
}
