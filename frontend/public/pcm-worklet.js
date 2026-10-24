class PcmWorkletProcessor extends AudioWorkletProcessor {
  constructor() {
    super()
    this.chunkSize = Math.max(1, Math.round(sampleRate * 0.5))
    this.chunk = new Float32Array(this.chunkSize)
    this.chunkIndex = 0
    this.peakSamples = 0
    this.peak = 0
  }

  process(inputs) {
    const input = inputs[0]
    const channel = input && input[0]
    if (!channel) return true

    for (let index = 0; index < channel.length; index += 1) {
      const sample = channel[index]
      this.chunk[this.chunkIndex] = sample
      this.chunkIndex += 1
      this.peak = Math.max(this.peak, Math.abs(sample))
      this.peakSamples += 1

      if (this.peakSamples >= 1024) {
        this.port.postMessage({ type: 'peak', value: this.peak })
        this.peakSamples = 0
        this.peak = 0
      }

      if (this.chunkIndex >= this.chunkSize) {
        const pcm = new Int16Array(this.chunkSize)
        for (let sampleIndex = 0; sampleIndex < this.chunkSize; sampleIndex += 1) {
          const clipped = Math.max(-1, Math.min(1, this.chunk[sampleIndex]))
          pcm[sampleIndex] = clipped < 0 ? clipped * 32768 : clipped * 32767
        }
        this.port.postMessage({ type: 'pcm', buffer: pcm.buffer }, [pcm.buffer])
        this.chunkIndex = 0
      }
    }
    return true
  }
}

registerProcessor('pcm-worklet', PcmWorkletProcessor)
