import { useCallback, useEffect, useRef, useState } from 'react'
import { BackendSettings } from './components/BackendSettings'
import { TimelineStrip, getFlaggedSentence } from './components/TimelineStrip'
import { UploadBox } from './components/UploadBox'
import { VerdictPanel } from './components/VerdictPanel'
import { Waveform } from './components/Waveform'
import { StatsList } from './components/StatsList'
import {
  binSeconds,
  fetchHealth,
  getApiUrl,
  isMockRequested,
  saveApiUrl,
  type ResultMessage,
  type StreamMessage,
  type Verdict,
  uncertainLow as fallbackUncertainLow,
  syntheticHigh as fallbackSyntheticHigh,
} from './config'
import { useStream } from './hooks/useStream'
import { downsampleForCanvas } from './lib/wav'

type Mode = 'live' | 'upload'
type Connection = 'connecting' | 'connected' | 'unreachable'

const unreachableMessage = 'Backend unreachable. Check the Backend URL below.'

function statusLabel(connection: Connection): string {
  if (connection === 'connected') return 'Connected'
  if (connection === 'unreachable') return 'Unreachable'
  return 'Connecting'
}

function normaliseResult(message: ResultMessage, fallbackSource: 'mic' | 'mock'): ResultMessage {
  return { ...message, source: message.source || fallbackSource }
}

export default function App() {
  const [mode, setMode] = useState<Mode>('live')
  const [mock, setMock] = useState(isMockRequested())
  const [apiUrl, setApiUrl] = useState(getApiUrl())
  const [connection, setConnection] = useState<Connection>(isMockRequested() ? 'connected' : 'connecting')
  const [errorMessage, setErrorMessage] = useState('')
  const [thresholds, setThresholds] = useState({ uncertainLow: fallbackUncertainLow, syntheticHigh: fallbackSyntheticHigh })
  const [verdict, setVerdict] = useState<Verdict | null>(null)
  const [score, setScore] = useState<number | null>(null)
  const [duration, setDuration] = useState(0)
  const [latency, setLatency] = useState<number | null>(null)
  const [warmupSeconds, setWarmupSeconds] = useState(0)
  const [source, setSource] = useState<'mic' | 'upload' | 'mock'>(isMockRequested() ? 'mock' : 'mic')
  const [format, setFormat] = useState<string | undefined>(undefined)
  const [warnings, setWarnings] = useState<string[]>([])
  const [uploadBusy, setUploadBusy] = useState(false)
  const [visualVersion, setVisualVersion] = useState(0)
  const peaksRef = useRef<number[]>([])
  const binsRef = useRef<Array<number | null>>([])
  const durationRef = useRef(0)

  const applyResult = useCallback((incoming: ResultMessage) => {
    const result = normaliseResult(incoming, mock ? 'mock' : 'mic')
    const resultSource = result.source === 'upload' ? 'upload' : result.source === 'mock' ? 'mock' : 'mic'
    const startIndex = Math.max(0, Math.round(result.t / binSeconds) - result.bin_scores.length)
    result.bin_scores.forEach((binScore, index) => {
      binsRef.current[startIndex + index] = binScore
    })
    durationRef.current = result.seconds
    setDuration(result.seconds)
    setScore(result.smoothed_score ?? result.score)
    setVerdict(result.verdict)
    setLatency(result.latency_ms)
    setWarmupSeconds(2)
    setSource(resultSource)
    setVisualVersion((version) => version + 1)
  }, [mock])

  const handleMessage = useCallback((message: StreamMessage) => {
    if (message.type === 'warmup') {
      setWarmupSeconds(message.seconds)
      durationRef.current = message.seconds
      setDuration(message.seconds)
      setVisualVersion((version) => version + 1)
      return
    }
    if (message.type === 'error') {
      setErrorMessage(message.message)
      return
    }
    applyResult(message)
  }, [applyResult])

  const runHealthCheck = useCallback(async (nextApiUrl = apiUrl) => {
    if (mock) {
      setConnection('connected')
      setErrorMessage('')
      return
    }
    setConnection('connecting')
    try {
      const health = await fetchHealth(nextApiUrl)
      if (health.thresholds) {
        setThresholds({
          uncertainLow: health.thresholds.uncertain_low,
          syntheticHigh: health.thresholds.synthetic_high,
        })
      }
      setConnection('connected')
      setErrorMessage('')
    } catch {
      setConnection('unreachable')
      setErrorMessage(unreachableMessage)
    }
  }, [apiUrl, mock])

  useEffect(() => {
    void runHealthCheck()
  }, [runHealthCheck])

  const handlePeak = useCallback((peak: number) => {
    peaksRef.current.push(peak)
    const maxPoints = 720
    if (peaksRef.current.length > maxPoints) peaksRef.current.splice(0, peaksRef.current.length - maxPoints)
    setVisualVersion((version) => version + 1)
  }, [])

  const { recording, start, stop } = useStream({
    apiUrl,
    mock,
    onMessage: handleMessage,
    onPeak: handlePeak,
    onStatus: setConnection,
    onError: setErrorMessage,
  })

  const handleRecord = async () => {
    setErrorMessage('')
    setMode('live')
    peaksRef.current = []
    binsRef.current = []
    setVerdict(null)
    setScore(null)
    setDuration(0)
    setLatency(null)
    setWarmupSeconds(0)
    setSource(mock ? 'mock' : 'mic')
    setFormat(undefined)
    setWarnings([])
    await start()
  }

  const handleClear = () => {
    void stop()
    peaksRef.current = []
    binsRef.current = []
    durationRef.current = 0
    setVerdict(null)
    setScore(null)
    setDuration(0)
    setLatency(null)
    setWarmupSeconds(0)
    setFormat(undefined)
    setWarnings([])
    setSource(mock ? 'mock' : 'mic')
    setErrorMessage('')
    setVisualVersion((version) => version + 1)
  }

  const handleUpload = async (file: File, uploadPeaks: number[], uploadDuration: number) => {
    setMode('upload')
    peaksRef.current = uploadPeaks
    binsRef.current = []
    setUploadBusy(true)
    setErrorMessage('')
    setDuration(uploadDuration)
    setSource(mock ? 'mock' : 'upload')
    setVisualVersion((version) => version + 1)
    try {
      if (mock) {
        const mockScore = 0.35
        const mockBins = Array.from({ length: Math.max(1, Math.ceil(uploadDuration / binSeconds)) }, () => mockScore)
        binsRef.current = mockBins
        setScore(mockScore)
        setVerdict('genuine')
        setLatency(20)
        setWarmupSeconds(2)
        setWarnings(['Mock data'])
        return
      }
      const form = new FormData()
      form.append('file', file)
      const response = await fetch(`${apiUrl.replace(/\/+$/, '')}/predict`, {
        method: 'POST',
        headers: { 'ngrok-skip-browser-warning': '1' },
        body: form,
      })
      if (!response.ok) throw new Error(`Upload failed with ${response.status}`)
      const body = await response.json() as {
        score: number
        verdict: Verdict
        duration?: number
        duration_s?: number
        format?: string
        warnings?: string[]
        bin_scores?: number[]
        latency_ms?: number
      }
      const resultDuration = body.duration_s ?? body.duration ?? uploadDuration
      binsRef.current = body.bin_scores ?? []
      setDuration(resultDuration)
      durationRef.current = resultDuration
      setScore(body.score)
      setVerdict(body.verdict)
      setLatency(body.latency_ms ?? null)
      setFormat(body.format)
      setWarnings(body.warnings ?? [])
      setWarmupSeconds(2)
      setVisualVersion((version) => version + 1)
    } catch {
      setConnection('unreachable')
      setErrorMessage(unreachableMessage)
    } finally {
      setUploadBusy(false)
    }
  }

  const handleApiChange = (value: string) => {
    setApiUrl(value)
    saveApiUrl(value)
  }
  const handleReconnect = () => void runHealthCheck(apiUrl)
  const handleMockChange = (value: boolean) => {
    setMock(value)
    setSource(value ? 'mock' : 'mic')
    if (value) {
      setConnection('connected')
      setErrorMessage('')
    } else {
      void fetchHealth(apiUrl).then(() => {
        setConnection('connected')
        setErrorMessage('')
      }).catch(() => {
        setConnection('unreachable')
        setErrorMessage(unreachableMessage)
      })
    }
  }

  const flaggedSentence = getFlaggedSentence(binsRef.current, thresholds)
  void visualVersion

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="wordmark">Swar Rakshak</div>
        <div className="header-status">
          {mock && <span className="mock-label">Mock data</span>}
          <span className={`status-dot ${connection}`} />
          <span>{statusLabel(connection)}</span>
        </div>
      </header>

      <main className="main-grid">
        <section className="workspace">
          <nav className="mode-tabs" aria-label="Mode">
            <button className={mode === 'live' ? 'tab active' : 'tab'} onClick={() => setMode('live')}>Live</button>
            <button className={mode === 'upload' ? 'tab active' : 'tab'} onClick={() => setMode('upload')}>Upload</button>
          </nav>

          <div className="signal-area">
            <Waveform peaks={peaksRef.current} active={recording || mode === 'live'} />
            <TimelineStrip bins={binsRef.current} duration={duration} thresholds={thresholds} />
            <p className="flagged-sentence">{flaggedSentence}</p>
          </div>

          <div className="controls-row">
            {mode === 'live' ? (
              <>
                {!recording ? (
                  <button className="button solid" type="button" onClick={() => void handleRecord()}>Record</button>
                ) : (
                  <button className="button solid stop-button" type="button" onClick={() => void stop()}>Stop</button>
                )}
                {recording && <span className="listening"><i /> Listening</span>}
              </>
            ) : (
              <UploadBox busy={uploadBusy} onUpload={handleUpload} />
            )}
            <button className="button plain" type="button" onClick={handleClear}>Clear</button>
          </div>
          {errorMessage && <div className="error-line"><span>{errorMessage}</span>{errorMessage === unreachableMessage && <button className="button plain" onClick={handleReconnect}>Reconnect</button>}</div>}
        </section>

        <aside className="results-panel">
          <VerdictPanel verdict={verdict} score={score} thresholds={thresholds} />
          <StatsList
            duration={duration}
            latency={latency}
            connection={statusLabel(connection)}
            warmup={warmupSeconds >= 2 ? 'Ready' : `Warming up ${warmupSeconds.toFixed(1)} / 2.0 s`}
            source={source}
            format={format}
            warnings={warnings}
          />
        </aside>
      </main>

      <footer>
        <BackendSettings value={apiUrl} onChange={handleApiChange} onReconnect={() => { setApiUrl(apiUrl); handleReconnect() }} mock={mock} onMockChange={handleMockChange} />
      </footer>
    </div>
  )
}
