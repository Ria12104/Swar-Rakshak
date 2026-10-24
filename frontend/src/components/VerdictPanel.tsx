import type { Verdict } from '../config'
import { ScoreBar } from './ScoreBar'

type Props = {
  verdict: Verdict | null
  score: number | null
  thresholds: { uncertainLow: number; syntheticHigh: number }
}

const labels: Record<Verdict, string> = {
  genuine: 'Likely genuine',
  uncertain: 'Uncertain',
  synthetic: 'Likely synthetic',
}

export function VerdictPanel({ verdict, score, thresholds }: Props) {
  return (
    <section className="verdict-panel" aria-live="polite">
      <div className="section-kicker">Assessment</div>
      <h1 className={verdict ? `verdict-word ${verdict}` : 'verdict-word empty'}>{verdict ? labels[verdict] : 'No result yet'}</h1>
      <ScoreBar score={score} {...thresholds} />
      <div className="raw-score">Raw score {score === null ? 'No score' : score.toFixed(4)}</div>
    </section>
  )
}
