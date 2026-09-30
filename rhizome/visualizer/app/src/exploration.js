/**
 * Linear mapping from a single user-facing Exploration slider (0-1) to the
 * engine's two parameters. Higher exploration = more frequent AND bigger
 * random deviations from the most-similar chunk.
 *
 *   exploration = 0.0  →  epsilon=0.00, temperature=0.0  (fully greedy)
 *   exploration = 0.10 →  epsilon=0.08, temperature=0.4  (default — small texture)
 *   exploration = 0.50 →  epsilon=0.40, temperature=2.0  (half-and-half)
 *   exploration = 1.00 →  epsilon=0.80, temperature=4.0  (very chaotic)
 *
 * The cap of 4.0 on temperature is deliberate — past that point the softmax
 * is already nearly uniform, so further increases barely change the walk.
 */
export function explorationToParams(exploration) {
  const x = Math.max(0, Math.min(1, exploration));
  return {
    epsilon: Number((x * 0.8).toFixed(4)),
    temperature: Number((x * 4).toFixed(3)),
  };
}

/** Reverse mapping for displaying the current epsilon/temperature as a
 *  slider position when an existing walk is replayed. */
export function paramsToExploration(epsilon, temperature) {
  // Prefer epsilon — it's the more meaningful axis. If epsilon is 0 the
  // slider should be at 0 regardless of temperature.
  if (epsilon <= 0) return 0;
  return Math.max(0, Math.min(1, epsilon / 0.8));
}
