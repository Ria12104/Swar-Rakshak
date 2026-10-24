import { useEffect, useRef } from 'react'
import { binSeconds, scoreMax, scoreMin, syntheticHigh, uncertainLow } from '../config'

type Props = {
  bins: Array<number | null>
  duration: number
  thresholds: { uncertainLow: number; syntheticHigh: number }
}

function colourForScore(score: number | null, thresholds: Props['thresholds']): string {
  if (score === null) return '#2A2B26'
  if (score >= thresholds.syntheticHigh) return '#E8502F'
  if (score >= thresholds.uncertainLow) return '#E5A935'
  return '#4FBF73'
}

export function getFlaggedSentence(bins: Array<number | null>, thresholds: Props['thresholds']): string {
  const flagged = bins
    .map((score, index) => (score !== null && score >= thresholds.syntheticHigh ? index : -1))
    .filter((index) => index >= 0)
  if (flagged.length === 0) return 'No segment flagged'
  const ranges: Array<[number, number]> = []
  let start = flagged[0]
  let previous = flagged[0]
  for (const index of flagged.slice(1)) {
    if (index !== previous + 1) {
      ranges.push([start, previous + 1])
      start = index
    }
    previous = index
  }
  ranges.push([start, previous + 1])
  const format = (range: [number, number]) => `${(range[0] * binSeconds).toFixed(1)} to ${(range[1] * binSeconds).toFixed(1)} s`
  return `Bins ${ranges.map(format).join(', ')} flagged`
}

export function TimelineStrip({ bins, duration, thresholds }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const binsRef = useRef(bins)
  binsRef.current = bins

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    let frame = 0
    let running = true
    const draw = () => {
      if (!running) return
      const ratio = window.devicePixelRatio || 1
      const width = canvas.clientWidth
      const height = canvas.clientHeight
      canvas.width = Math.max(1, Math.floor(width * ratio))
      canvas.height = Math.max(1, Math.floor(height * ratio))
      const context = canvas.getContext('2d')
      if (!context) return
      context.scale(ratio, ratio)
      context.fillStyle = '#151613'
      context.fillRect(0, 0, width, height)
      const currentBins = binsRef.current
      const cellWidth = Math.max(4, width / Math.max(1, currentBins.length || Math.ceil(duration / binSeconds)))
      currentBins.forEach((score, index) => {
        context.fillStyle = colourForScore(score, thresholds)
        context.globalAlpha = score === null ? 0.25 : 1
        context.fillRect(index * cellWidth + 1, 8, Math.max(2, cellWidth - 2), height - 22)
      })
      context.globalAlpha = 1
      context.fillStyle = '#9A978B'
      context.font = '11px IBM Plex Mono, monospace'
      context.textAlign = 'center'
      const tickCount = Math.max(1, Math.floor(width / 80))
      for (let index = 0; index <= tickCount; index += 1) {
        const x = (index / tickCount) * width
        context.fillRect(x, height - 12, 1, 4)
        context.fillText(`${index}s`, x, height - 1)
      }
      frame = window.requestAnimationFrame(draw)
    }
    draw()
    return () => {
      running = false
      window.cancelAnimationFrame(frame)
    }
  }, [duration, thresholds])

  return (
    <div className="timeline-wrap">
      <canvas className="timeline-canvas" ref={canvasRef} aria-label="Authenticity timeline" />
      <div className="timeline-legend" aria-label="Timeline legend">
        <span><i className="legend-swatch genuine" /> genuine &lt; {thresholds.uncertainLow.toFixed(2)}</span>
        <span><i className="legend-swatch uncertain" /> uncertain</span>
        <span><i className="legend-swatch synthetic" /> synthetic ≥ {thresholds.syntheticHigh.toFixed(2)}</span>
      </div>
    </div>
  )
}

export { scoreMax, scoreMin, syntheticHigh, uncertainLow }
