import { useRef, useState } from 'react'
import { decodeAudioFile, downsampleForCanvas } from '../lib/wav'

type Props = {
  busy: boolean
  onUpload: (file: File, peaks: number[], duration: number) => Promise<void>
}

export function UploadBox({ busy, onUpload }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [warning, setWarning] = useState('')
  const [error, setError] = useState('')

  const selectFile = async (file?: File) => {
    if (!file) return
    setWarning('')
    setError('')
    if (!file.type.startsWith('audio/') && !/\.(wav|flac|mp3|ogg)$/i.test(file.name)) {
      setError('Choose an audio file.')
      return
    }
    if (file.size >= 10 * 1024 * 1024) {
      setError('File must be under 10 MB.')
      return
    }
    try {
      const decoded = await decodeAudioFile(file)
      if (decoded.duration < 2 || decoded.duration > 10) setWarning('Clip is outside the recommended 2 to 10 s range.')
      await onUpload(file, downsampleForCanvas(decoded.samples, 720), decoded.duration)
    } catch {
      setError('Could not decode this audio file.')
    }
  }

  return (
    <div className="upload-box">
      <input
        ref={inputRef}
        type="file"
        accept="audio/*,.wav,.flac,.mp3,.ogg"
        hidden
        onChange={(event) => void selectFile(event.target.files?.[0])}
      />
      <button className="button solid" type="button" onClick={() => inputRef.current?.click()} disabled={busy}>
        {busy ? 'Analysing' : 'Upload clip'}
      </button>
      <span className="upload-hint">wav preferred, under 10 MB</span>
      {warning && <p className="field-warning">{warning}</p>}
      {error && <p className="field-error">{error}</p>}
    </div>
  )
}
