import React, { useCallback, useRef } from 'react';

/**
 * Single-thumb range slider.
 *
 * value: number in [min, max]
 * onChange: (next) => void
 * step: numeric increment (defaults to (max-min)/100)
 *
 * Renders an accessible <input type="range"> underneath so screen readers
 * and keyboard users get the standard behaviour; the visible track and
 * thumb are absolutely positioned over it for styling control.
 */
export default function Slider({
  value,
  onChange,
  min = 0,
  max = 1,
  step,
  disabled = false,
  ariaLabel,
}) {
  const trackRef = useRef(null);
  const inputValue = value;

  const fillPct = ((inputValue - min) / (max - min)) * 100;

  const handlePointerDown = useCallback(
    (e) => {
      if (disabled) return;
      const track = trackRef.current;
      if (!track) return;
      const rect = track.getBoundingClientRect();
      const updateFromX = (clientX) => {
        const ratio = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
        const raw = min + ratio * (max - min);
        const inc = step ?? (max - min) / 100;
        const snapped = Math.round(raw / inc) * inc;
        const clamped = Math.max(min, Math.min(max, snapped));
        onChange(Number(clamped.toFixed(3)));
      };
      updateFromX(e.clientX);
      const onMove = (ev) => updateFromX(ev.clientX);
      const onUp = () => {
        window.removeEventListener('pointermove', onMove);
        window.removeEventListener('pointerup', onUp);
      };
      window.addEventListener('pointermove', onMove);
      window.addEventListener('pointerup', onUp);
    },
    [disabled, min, max, step, onChange]
  );

  return (
    <div
      ref={trackRef}
      onPointerDown={handlePointerDown}
      className={`relative h-6 flex items-center select-none ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
    >
      <div className="absolute inset-x-0 h-1 bg-bg-tertiary rounded-full pointer-events-none" />
      <div
        className="absolute left-0 h-1 bg-accent rounded-full pointer-events-none"
        style={{ width: `${fillPct}%` }}
      />
      <div
        className="absolute w-4 h-4 bg-accent border-2 border-bg-secondary rounded-full shadow-sm pointer-events-none"
        style={{ left: `calc(${fillPct}% - 8px)` }}
      />
      <input
        type="range"
        min={min}
        max={max}
        step={step ?? (max - min) / 100}
        value={inputValue}
        onChange={(e) => onChange(Number(e.target.value))}
        disabled={disabled}
        aria-label={ariaLabel}
        className="absolute inset-0 w-full opacity-0 cursor-pointer disabled:cursor-not-allowed"
      />
    </div>
  );
}
