import type { ResultMessage, StreamMessage, Verdict } from '../config'

const seeded = (seed: number): number => {
  const value = Math.sin(seed * 12.9898) * 43758.5453
  return value - Math.floor(value)
}

export function mockMessageAt(step: number): StreamMessage {
  const seconds = Number(((step + 1) * 0.5).toFixed(1))
  if (seconds < 2) return { type: 'warmup', seconds }

  const flagged = seconds > 6 && seconds <= 9
  const baseScore = flagged ? 2.1 : 0.35
  const score = Number((baseScore + (seeded(step + 7) - 0.5) * 0.18).toFixed(4))
  const verdict: Verdict = flagged ? 'synthetic' : 'genuine'
  const windowBins = Math.min(8, Math.floor(seconds / 0.5))
  const result: ResultMessage = {
    type: 'result',
    seq: Math.max(0, step - 3),
    t: seconds,
    score,
    smoothed_score: score,
    verdict,
    bin_scores: Array.from({ length: windowBins }, (_, index) => {
      const absoluteBin = Math.floor(seconds / 0.5) - windowBins + index
      return absoluteBin >= 12 && absoluteBin < 18 ? 2.1 : 0.35
    }),
    seconds,
    latency_ms: 18 + Math.round(seeded(step + 19) * 8),
    model_version: 'mock-scripted-cnn',
    source: 'mock',
  }
  return result
}

export function mockPeakAt(step: number): number {
  return 0.18 + seeded(step + 101) * 0.55
}
