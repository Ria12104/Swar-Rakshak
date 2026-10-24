import { useEffect, useRef, useState } from 'react'
import { scoreMax, scoreMin } from '../config'

type Props = {
  score: number | null
  uncertainLow: number
  syntheticHigh: number
}

function percentage(score: number): number {
  return Math.min(100, Math.max(0, ((score - scoreMin) / (scoreMax - scoreMin)) * 100))
}

export function ScoreBar({ score, uncertainLow, syntheticHigh }: Props) {
  const barRef = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(0)
  useEffect(() => {
    const element = barRef.current
    if (!element) return
    const update = () => setWidth(element.clientWidth)
    update()
    const observer = new ResizeObserver(update)
    observer.observe(element)
    return () => observer.disconnect()
  }, [])

  const genuineRaw = percentage(uncertainLow)
  const uncertainRaw = Math.max(0, percentage(syntheticHigh) - genuineRaw)
  const syntheticRaw = Math.max(0, 100 - genuineRaw - uncertainRaw)
  const minimumUncertain = width > 0 ? 8 : 0
  const uncertainWidth = width > 0 ? Math.max(minimumUncertain, width * uncertainRaw / 100) : 0
  const remaining = Math.max(0, width - uncertainWidth)
  const genuineWidth = width > 0 ? remaining * genuineRaw / Math.max(0.001, genuineRaw + syntheticRaw) : 0
  const syntheticWidth = Math.max(0, remaining - genuineWidth)

  return (
    <div className="scorebar-block">
      <div className="scorebar" ref={barRef}>
        <div className="scorebar-zone genuine" style={width ? { width: `${genuineWidth}px` } : undefined} />
        <div className="scorebar-zone uncertain" style={width ? { width: `${uncertainWidth}px` } : undefined} />
        <div className="scorebar-zone synthetic" style={width ? { width: `${syntheticWidth}px` } : undefined} />
        {score !== null && <span className="scorebar-marker" style={{ left: `${percentage(score)}%` }} />}
      </div>
      <div className="scorebar-labels">
        <span>genuine</span>
        <span>uncertain</span>
        <span>synthetic</span>
      </div>
      <div className="scorebar-range"><span>{scoreMin}</span><span>raw score</span><span>+{scoreMax}</span></div>
    </div>
  )
}
