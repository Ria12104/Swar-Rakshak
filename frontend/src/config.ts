export const uncertainLow = 1.22
export const syntheticHigh = 1.29
export const scoreMin = -6
export const scoreMax = 6
export const binSeconds = 0.5

export type Verdict = 'genuine' | 'uncertain' | 'synthetic'
export type Source = 'mic' | 'upload' | 'mock'

export type BackendThresholds = {
  uncertain_low: number
  synthetic_high: number
}

export type HealthResponse = {
  status: string
  model_version: string
  device: string
  thresholds?: BackendThresholds
  thresholds_placeholder?: boolean
}

export type WarmupMessage = {
  type: 'warmup'
  seconds: number
}

export type ResultMessage = {
  type?: 'result'
  seq: number
  t: number
  score: number
  smoothed_score?: number
  verdict: Verdict
  bin_scores: number[]
  seconds: number
  latency_ms: number
  model_version: string
  source: Source
}

export type ErrorMessage = {
  type: 'error'
  message: string
}

export type StreamMessage = WarmupMessage | ResultMessage | ErrorMessage

export type UploadResponse = {
  score: number
  smoothed_score?: number
  verdict: Verdict
  duration?: number
  duration_s?: number
  format?: string
  model_version: string
  warnings?: string[]
  bin_scores?: number[]
  bin_s?: number
  source?: 'upload'
  thresholds_placeholder?: boolean
}

function readQuery(name: string): string | null {
  try {
    return new URLSearchParams(window.location.search).get(name)
  } catch {
    return null
  }
}

function readStorage(key: string): string | null {
  try {
    return window.localStorage.getItem(key)
  } catch {
    return null
  }
}

function normalizeApiUrl(value: string): string {
  return value.trim().replace(/\/+$/, '')
}

export function getApiUrl(): string {
  const queryApi = readQuery('api')
  const storedApi = readStorage('swar-rakshak-api-url')
  const envApi = import.meta.env.VITE_API_URL as string | undefined
  return normalizeApiUrl(queryApi || storedApi || envApi || 'http://localhost:8000')
}

export function saveApiUrl(value: string): string {
  const normalized = normalizeApiUrl(value)
  try {
    window.localStorage.setItem('swar-rakshak-api-url', normalized)
  } catch {
    // Storage can be unavailable in private browsing or embedded contexts.
  }
  return normalized
}

export function getWsUrl(apiUrl: string): string {
  const normalized = normalizeApiUrl(apiUrl)
  const url = new URL(normalized)
  url.protocol = window.location.protocol === 'https:' || url.protocol === 'https:' ? 'wss:' : 'ws:'
  url.pathname = `${url.pathname.replace(/\/$/, '')}/stream`
  return url.toString()
}

export function isMockRequested(): boolean {
  return readQuery('mock') === '1'
}

export async function fetchHealth(apiUrl: string): Promise<HealthResponse> {
  const response = await fetch(`${normalizeApiUrl(apiUrl)}/health`, {
    headers: { 'ngrok-skip-browser-warning': '1' },
  })
  if (!response.ok) throw new Error(`Health check failed with ${response.status}`)
  return response.json() as Promise<HealthResponse>
}
