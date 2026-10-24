type Props = {
  duration: number
  latency: number | null
  connection: string
  warmup: string
  source: string
  format?: string
  warnings?: string[]
}

export function StatsList({ duration, latency, connection, warmup, source, format, warnings = [] }: Props) {
  return (
    <dl className="stats-list">
      <div><dt>Analysed</dt><dd>{duration.toFixed(1)} s</dd></div>
      <div><dt>Latency</dt><dd>{latency === null ? 'Not available' : `${Math.round(latency)} ms`}</dd></div>
      <div><dt>Connection</dt><dd>{connection}</dd></div>
      <div><dt>Warm-up</dt><dd>{warmup}</dd></div>
      <div><dt>Source</dt><dd>{source}</dd></div>
      {format && <div><dt>Format</dt><dd>{format}</dd></div>}
      {warnings.length > 0 && <div><dt>Warnings</dt><dd>{warnings.join(', ')}</dd></div>}
    </dl>
  )
}
