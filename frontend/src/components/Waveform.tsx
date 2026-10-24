import { useEffect, useRef } from 'react'

type Props = {
  peaks: number[]
  active: boolean
}

function fitCanvas(canvas: HTMLCanvasElement): CanvasRenderingContext2D | null {
  const ratio = window.devicePixelRatio || 1
  const width = canvas.clientWidth
  const height = canvas.clientHeight
  canvas.width = Math.max(1, Math.floor(width * ratio))
  canvas.height = Math.max(1, Math.floor(height * ratio))
  const context = canvas.getContext('2d')
  context?.scale(ratio, ratio)
  return context
}

export function Waveform({ peaks, active }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const peaksRef = useRef(peaks)
  peaksRef.current = peaks

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    let frame = 0
    let running = true

    const draw = () => {
      if (!running) return
      const context = fitCanvas(canvas)
      if (!context) return
      const width = canvas.clientWidth
      const height = canvas.clientHeight
      context.fillStyle = '#151613'
      context.fillRect(0, 0, width, height)
      context.strokeStyle = '#2A2B26'
      context.lineWidth = 1
      context.beginPath()
      context.moveTo(0, height / 2 + 0.5)
      context.lineTo(width, height / 2 + 0.5)
      context.stroke()

      const points = peaksRef.current
      if (points.length > 0) {
        context.strokeStyle = '#ECEAE3'
        context.lineWidth = 1.5
        context.beginPath()
        points.forEach((peak, index) => {
          const x = (index / Math.max(1, points.length - 1)) * width
          const amplitude = Math.min(1, peak) * height * 0.42
          context.moveTo(x, height / 2 - amplitude)
          context.lineTo(x, height / 2 + amplitude)
        })
        context.stroke()
      }
      frame = window.requestAnimationFrame(draw)
    }

    draw()
    return () => {
      running = false
      window.cancelAnimationFrame(frame)
    }
  }, [active])

  return <canvas className="waveform-canvas" ref={canvasRef} aria-label="Audio waveform" />
}
