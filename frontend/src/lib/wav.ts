export type DecodedAudio = {
  samples: Float32Array
  duration: number
  sampleRate: number
}

export async function decodeAudioFile(file: File): Promise<DecodedAudio> {
  const context = new AudioContext()
  try {
    const buffer = await context.decodeAudioData(await file.arrayBuffer())
    const source = buffer.getChannelData(0)
    return {
      samples: new Float32Array(source),
      duration: buffer.duration,
      sampleRate: buffer.sampleRate,
    }
  } finally {
    await context.close()
  }
}

export function downsampleForCanvas(samples: Float32Array, points: number): number[] {
  if (points <= 0 || samples.length === 0) return []
  const result: number[] = []
  const blockSize = Math.max(1, Math.floor(samples.length / points))
  for (let point = 0; point < points; point += 1) {
    const start = point * blockSize
    const end = Math.min(samples.length, start + blockSize)
    let peak = 0
    for (let index = start; index < end; index += 1) peak = Math.max(peak, Math.abs(samples[index]))
    result.push(peak)
  }
  return result
}
