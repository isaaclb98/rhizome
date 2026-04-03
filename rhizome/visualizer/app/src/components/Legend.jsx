const DOMAINS = [
  { name: 'Modernism', color: '#6366f1' },
  { name: 'Postmodernism', color: '#a855f7' },
  { name: 'Critical theory', color: '#f97316' },
  { name: 'Unknown', color: '#6b7280' },
];

export default function Legend() {
  return (
    <div className="flex items-center gap-4 flex-wrap">
      {DOMAINS.map(({ name, color }) => (
        <div key={name} className="flex items-center gap-1.5">
          <span
            className="inline-block w-2.5 h-2.5 rounded-full flex-shrink-0"
            style={{ backgroundColor: color }}
          />
          <span className="text-xs text-gray-400">{name}</span>
        </div>
      ))}
    </div>
  );
}
