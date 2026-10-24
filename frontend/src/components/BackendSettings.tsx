type Props = {
  value: string
  onChange: (value: string) => void
  onReconnect: () => void
  mock: boolean
  onMockChange: (value: boolean) => void
}

export function BackendSettings({ value, onChange, onReconnect, mock, onMockChange }: Props) {
  return (
    <div className="footer-settings">
      <label className="backend-field">
        <span>Backend URL</span>
        <input value={value} onChange={(event) => onChange(event.target.value)} onBlur={onReconnect} />
      </label>
      <label className="toggle-field">
        <span>Mock data</span>
        <input type="checkbox" checked={mock} onChange={(event) => onMockChange(event.target.checked)} />
        <span className="toggle-track" aria-hidden="true"><span /></span>
      </label>
    </div>
  )
}
