import { useCallback, useEffect, useRef, useState } from 'react'
import { getWsUrl, type StreamMessage } from '../config'
import { mockMessageAt, mockPeakAt } from '../lib/mock'

type Status = 'connecting' | 'connected' | 'unreachable'

type Options = {
  apiUrl: string
  mock: boolean
  onMessage: (message: StreamMessage) => void
  onPeak: (peak: number) => void
  onStatus: (status: Status) => void
  onError: (message: string) => void
}

const unreachableMessage = 'Backend unreachable. Check the Backend URL below.'

export function useStream({ apiUrl, mock, onMessage, onPeak, onStatus, onError }: Options) {
  const [recording, setRecording] = useState(false)
  const socketRef = useRef<WebSocket | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null)
  const workletRef = useRef<AudioWorkletNode | null>(null)
  const silentGainRef = useRef<GainNode | null>(null)
  const mockTimerRef = useRef<number | null>(null)
  const intentionalStopRef = useRef(false)
  const callbacksRef = useRef({ onMessage, onPeak, onStatus, onError })
  callbacksRef.current = { onMessage, onPeak, onStatus, onError }

  const cleanupAudio = useCallback(async () => {
    sourceRef.current?.disconnect()
    workletRef.current?.disconnect()
    silentGainRef.current?.disconnect()
    mediaStreamRef.current?.getTracks().forEach((track) => track.stop())
    mediaStreamRef.current = null
    sourceRef.current = null
    workletRef.current = null
    silentGainRef.current = null
    if (audioContextRef.current) {
      await audioContextRef.current.close()
      audioContextRef.current = null
    }
  }, [])

  const stop = useCallback(async () => {
    if (mockTimerRef.current !== null) {
      window.clearInterval(mockTimerRef.current)
      mockTimerRef.current = null
    }
    intentionalStopRef.current = true
    const socket = socketRef.current
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: 'stop' }))
      window.setTimeout(() => {
        if (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING) socket.close()
      }, 800)
    } else {
      socket?.close()
    }
    socketRef.current = null
    await cleanupAudio()
    setRecording(false)
  }, [cleanupAudio])

  const start = useCallback(async () => {
    await stop()
    intentionalStopRef.current = false
    if (mock) {
      callbacksRef.current.onStatus('connected')
      setRecording(true)
      let step = 0
      const tick = () => {
        callbacksRef.current.onPeak(mockPeakAt(step))
        callbacksRef.current.onMessage(mockMessageAt(step))
        step += 1
      }
      tick()
      mockTimerRef.current = window.setInterval(tick, 500)
      return
    }

    callbacksRef.current.onStatus('connecting')
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          channelCount: 1,
        },
      })
      mediaStreamRef.current = mediaStream
      const context = new AudioContext()
      audioContextRef.current = context
      await context.audioWorklet.addModule('/pcm-worklet.js')
      const socket = new WebSocket(getWsUrl(apiUrl))
      socket.binaryType = 'arraybuffer'
      socketRef.current = socket
      socket.onopen = () => {
        socket.send(JSON.stringify({ type: 'config', sample_rate: context.sampleRate, channels: 1, format: 'int16' }))
        callbacksRef.current.onStatus('connected')
        setRecording(true)
        const source = context.createMediaStreamSource(mediaStream)
        const worklet = new AudioWorkletNode(context, 'pcm-worklet')
        const silentGain = context.createGain()
        silentGain.gain.value = 0
        worklet.port.onmessage = (event: MessageEvent<{ type: string; buffer?: ArrayBuffer; value?: number }>) => {
          if (event.data.type === 'pcm' && event.data.buffer) socket.send(event.data.buffer)
          if (event.data.type === 'peak' && typeof event.data.value === 'number') callbacksRef.current.onPeak(event.data.value)
        }
        source.connect(worklet)
        worklet.connect(silentGain)
        silentGain.connect(context.destination)
        sourceRef.current = source
        workletRef.current = worklet
        silentGainRef.current = silentGain
      }
      socket.onmessage = (event) => {
        try {
          callbacksRef.current.onMessage(JSON.parse(event.data) as StreamMessage)
        } catch {
          callbacksRef.current.onError('Received an invalid message from the backend.')
        }
      }
      socket.onerror = () => {
        callbacksRef.current.onStatus('unreachable')
        callbacksRef.current.onError(unreachableMessage)
      }
      socket.onclose = () => {
        socketRef.current = null
        if (!intentionalStopRef.current) {
          callbacksRef.current.onStatus('unreachable')
          callbacksRef.current.onError(unreachableMessage)
        }
        setRecording(false)
      }
    } catch (error) {
      await cleanupAudio()
      setRecording(false)
      callbacksRef.current.onStatus('unreachable')
      callbacksRef.current.onError(error instanceof DOMException && error.name === 'NotAllowedError' ? 'Microphone access was denied.' : unreachableMessage)
    }
  }, [apiUrl, cleanupAudio, mock, stop])

  useEffect(() => () => {
    void stop()
  }, [stop])

  return { recording, start, stop }
}
