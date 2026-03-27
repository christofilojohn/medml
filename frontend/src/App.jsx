import { useState, useEffect, useRef, useCallback } from 'react'

// ═══════════════════════════════════════════════════════════════
// MedML Forge — On-Device Clinical AI Pipeline Dashboard
// ═══════════════════════════════════════════════════════════════
// Architecture:
//   React UI → Python ML Worker (scan/train/cleanup)
//            → Qwen LLM via llama.cpp (reasoning)
//   All local. Zero patient data leaves the machine.
// ═══════════════════════════════════════════════════════════════

const ML_API = '' // proxied through vite dev server

const STAGES = [
  { id: 'scan',    icon: '⬡', label: 'Data Scan',    desc: 'Point to your local data directory' },
  { id: 'preview', icon: '◈', label: 'Data Preview',  desc: 'Inspect and understand your dataset' },
  { id: 'clean',   icon: '◇', label: 'Cleanup',       desc: 'Fix quality issues' },
  { id: 'config',  icon: '⬢', label: 'Configure',     desc: 'Model & training settings' },
  { id: 'train',   icon: '△', label: 'Training',      desc: 'On-device model training' },
  { id: 'eval',    icon: '○', label: 'Evaluate',      desc: 'Performance assessment' },
]

// ── Tiny chart component ────────────────────────────────────
function MiniChart({ data, color = '#10b981', height = 80, label }) {
  if (!data || data.length < 2) return null
  const max = Math.max(...data)
  const min = Math.min(...data)
  const range = max - min || 1
  const w = 100
  const points = data.map((v, i) => {
    const x = (i / (data.length - 1)) * w
    const y = height - ((v - min) / range) * (height - 8) - 4
    return `${x},${y}`
  }).join(' ')
  const areaPoints = `0,${height} ${points} ${w},${height}`

  return (
    <div style={{ position: 'relative' }}>
      {label && <div style={{ fontSize: 10, color: '#64748b', marginBottom: 4 }}>{label}</div>}
      <svg viewBox={`0 0 ${w} ${height}`} style={{ width: '100%', height }}>
        <defs>
          <linearGradient id={`g-${label}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.3" />
            <stop offset="100%" stopColor={color} stopOpacity="0" />
          </linearGradient>
        </defs>
        <polygon points={areaPoints} fill={`url(#g-${label})`} />
        <polyline points={points} fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        <circle cx={(data.length - 1) / (data.length - 1) * w} cy={height - ((data[data.length - 1] - min) / range) * (height - 8) - 4}
          r="3" fill={color} />
      </svg>
    </div>
  )
}

// ── Data table component ────────────────────────────────────
function DataTable({ columns, rows, maxRows = 12 }) {
  if (!columns || !rows) return null
  const display = rows.slice(0, maxRows)
  return (
    <div style={{ overflowX: 'auto', borderRadius: 8, border: '1px solid #e2e8f0' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, fontFamily: "'JetBrains Mono', monospace" }}>
        <thead>
          <tr>
            {columns.map((c, i) => (
              <th key={i} style={{
                padding: '8px 12px', textAlign: 'left', background: '#f8fafc',
                color: '#64748b', fontWeight: 600, borderBottom: '1px solid #e2e8f0',
                whiteSpace: 'nowrap', position: 'sticky', top: 0, fontSize: 11,
              }}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {display.map((row, ri) => (
            <tr key={ri} style={{ background: ri % 2 === 0 ? '#ffffff' : '#f8fafc' }}>
              {row.map((cell, ci) => (
                <td key={ci} style={{
                  padding: '6px 12px', color: '#1e293b', borderBottom: '1px solid #e2e8f022',
                  whiteSpace: 'nowrap', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis',
                }}>
                  {cell === null || cell === undefined ? <span style={{ color: '#94a3b8' }}>null</span> : String(cell)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length > maxRows && (
        <div style={{ padding: '6px 12px', fontSize: 11, color: '#64748b', background: '#f8fafc', textAlign: 'center' }}>
          Showing {maxRows} of {rows.length} rows
        </div>
      )}
    </div>
  )
}

// ── Image grid for previewing image datasets ────────────────
function ImageGrid({ images }) {
  if (!images || images.length === 0) return null
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(120px, 1fr))', gap: 10 }}>
      {images.map((img, i) => (
        <div key={i} style={{
          background: '#f8fafc', borderRadius: 8, padding: 8,
          border: '1px solid #e2e8f0', textAlign: 'center'
        }}>
          <img src={`data:image/png;base64,${img.b64}`} alt={img.class}
            style={{ width: '100%', height: 80, objectFit: 'contain', borderRadius: 4, marginBottom: 6 }} />
          <div style={{ fontSize: 11, color: '#10b981', fontWeight: 600 }}>{img.class}</div>
          <div style={{ fontSize: 10, color: '#64748b' }}>{img.original_size}</div>
        </div>
      ))}
    </div>
  )
}

// ── Column inspector ────────────────────────────────────────
function ColumnInspector({ columns }) {
  if (!columns || columns.length === 0) return null
  return (
    <div style={{ display: 'grid', gap: 6, maxHeight: 350, overflowY: 'auto' }}>
      {columns.map((col, i) => (
        <div key={i} style={{
          display: 'flex', alignItems: 'center', gap: 12, padding: '8px 12px',
          background: '#f8fafc', borderRadius: 6, border: '1px solid #e2e8f0',
        }}>
          <div style={{ flex: 1 }}>
            <span style={{ fontSize: 13, fontWeight: 600, color: '#1e293b', fontFamily: "'JetBrains Mono', monospace" }}>
              {col.name}
            </span>
            <span style={{
              marginLeft: 8, fontSize: 10, padding: '2px 6px', borderRadius: 4,
              background: col.dtype?.includes('float') || col.dtype?.includes('int') ? '#d1fae5' : '#e0e7ff',
              color: col.dtype?.includes('float') || col.dtype?.includes('int') ? '#065f46' : '#3730a3',
            }}>
              {col.dtype}
            </span>
          </div>
          {col.missing_pct > 0 && (
            <span style={{ fontSize: 11, color: '#f59e0b' }}>{col.missing_pct}% missing</span>
          )}
          <span style={{ fontSize: 11, color: '#64748b' }}>{col.unique} unique</span>
        </div>
      ))}
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════
// Main App
// ═══════════════════════════════════════════════════════════════
export default function App() {
  const [stage, setStage] = useState(0)
  const [completed, setCompleted] = useState(new Set())
  const [backend, setBackend] = useState({ online: false, device: 'checking...' })

  // Data state
  const [dataPath, setDataPath] = useState('')
  const [scanning, setScanning] = useState(false)
  const [scanResult, setScanResult] = useState(null)
  const [dataset, setDataset] = useState(null)

  // Training state
  const [trainConfig, setTrainConfig] = useState({
    epochs: 50, lr: 0.001, batch_size: 32, model_type: 'auto', target_column: '',
  })
  const [training, setTraining] = useState(null)
  const [trainHistory, setTrainHistory] = useState([])

  // Feature selection state
  const [featureMode, setFeatureMode] = useState('manual') // 'auto' | 'manual'
  const [autoFeatures, setAutoFeatures] = useState(null)
  const [autoRunning, setAutoRunning] = useState(false)
  const [selectedFeatures, setSelectedFeatures] = useState([]) // [] = all

  // AI reasoning
  const [aiMessages, setAiMessages] = useState([])
  const [aiLoading, setAiLoading] = useState(false)
  const [aiInput, setAiInput] = useState('')

  // Cleanup
  const [cleanupResult, setCleanupResult] = useState(null)

  const aiChatRef = useRef(null)
  const sseRef = useRef(null)

  // ── Check backend health ──────────────────────────────────
  useEffect(() => {
    const check = async () => {
      try {
        const r = await fetch('/health')
        const d = await r.json()
        setBackend({ online: true, device: d.device, torch: d.torch, gpu: d.gpu_type })
      } catch {
        setBackend({ online: false, device: 'offline' })
      }
    }
    check()
    const iv = setInterval(check, 10000)
    return () => clearInterval(iv)
  }, [])

  // ── Scroll AI chat ────────────────────────────────────────
  useEffect(() => {
    if (aiChatRef.current) aiChatRef.current.scrollTop = aiChatRef.current.scrollHeight
  }, [aiMessages])

  // ── Stage management ──────────────────────────────────────
  const completeStage = useCallback((idx) => {
    setCompleted(prev => new Set([...prev, idx]))
  }, [])

  const goNext = useCallback(() => {
    setStage(s => {
      completeStage(s)
      return Math.min(s + 1, STAGES.length - 1)
    })
  }, [completeStage])

  // ── Scan data ─────────────────────────────────────────────
  const runScan = async () => {
    if (!dataPath.trim()) return
    setScanning(true)
    try {
      const r = await fetch('/scan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: dataPath }),
      })
      const data = await r.json()
      if (data.error) {
        setAiMessages(prev => [...prev, { role: 'system', text: `Scan error: ${data.error}` }])
      } else {
        setScanResult(data)
        setDataset(data)
        completeStage(0)
        setStage(1)
        askAI(`Scan complete. Analyze this dataset and suggest next steps.`, data.summary)
      }
    } catch (e) {
      setAiMessages(prev => [...prev, { role: 'system', text: `Connection error: ${e.message}` }])
    }
    setScanning(false)
  }

  // ── Ask AI (Qwen reasoning) ───────────────────────────────
  const askAI = async (prompt, context = null) => {
    setAiMessages(prev => [...prev, { role: 'user', text: prompt }])
    setAiLoading(true)
    try {
      const r = await fetch('/reason', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          context: context || scanResult?.summary || {},
        }),
      })
      const data = await r.json()
      setAiMessages(prev => [...prev, { role: 'ai', text: data.reply }])
    } catch {
      setAiMessages(prev => [...prev, { role: 'ai', text: 'LLM offline — using default recommendations.' }])
    }
    setAiLoading(false)
  }

  // ── Run cleanup ───────────────────────────────────────────
  const runCleanup = async () => {
    try {
      const r = await fetch('/cleanup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ actions: ['duplicates', 'missing', 'outliers', 'corrupted'] }),
      })
      const data = await r.json()
      setCleanupResult(data)
      completeStage(2)
      askAI('Cleanup completed. Here are the results. What should we do next?', data)
    } catch (e) {
      setAiMessages(prev => [...prev, { role: 'system', text: `Cleanup error: ${e.message}` }])
    }
  }

  // ── Start training (SSE) ──────────────────────────────────
  const startTraining = async () => {
    try {
      const r = await fetch('/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data_path: dataPath,
          target_column: trainConfig.target_column,
          epochs: trainConfig.epochs,
          model_type: trainConfig.model_type,
          batch_size: trainConfig.batch_size,
          lr: trainConfig.lr,
          feature_columns: selectedFeatures.length > 0 ? selectedFeatures : null,
        }),
      })
      const data = await r.json()
      if (data.error) {
        setAiMessages(prev => [...prev, { role: 'system', text: `Train error: ${data.error}` }])
        return
      }

      // Subscribe to SSE stream
      setStage(4)
      setTrainHistory([])
      const sse = new EventSource('/train/stream')
      sseRef.current = sse

      sse.onmessage = (e) => {
        const d = JSON.parse(e.data)
        setTraining(d)
        if (d.metrics) {
          setTrainHistory(prev => {
            const next = [...prev, d.metrics]
            return next.slice(-200)
          })
        }
        if (!d.active && d.status === 'complete') {
          sse.close()
          completeStage(4)
          setStage(5)
          askAI(`Training complete! Final metrics: accuracy=${d.metrics.accuracy}, f1=${d.metrics.f1}, val_loss=${d.metrics.val_loss}. What are the next steps?`)
        }
      }
      sse.onerror = () => {
        sse.close()
        const poll = setInterval(async () => {
          try {
            const r = await fetch('/train/status')
            const d = await r.json()
            setTraining(d)
            if (d.metrics) setTrainHistory(prev => [...prev, d.metrics].slice(-200))
            if (!d.active) {
              clearInterval(poll)
              if (d.status === 'complete') {
                completeStage(4)
                setStage(5)
              }
            }
          } catch {}
        }, 500)
      }
    } catch (e) {
      setAiMessages(prev => [...prev, { role: 'system', text: `Error: ${e.message}` }])
    }
  }

  const stopTraining = async () => {
    try { await fetch('/train/stop', { method: 'POST' }) } catch {}
    if (sseRef.current) sseRef.current.close()
  }

  // ── Render ────────────────────────────────────────────────
  return (
    <div style={{
      fontFamily: "'DM Sans', system-ui, sans-serif",
      background: '#f0f4f8', color: '#1e293b', minHeight: '100vh',
      display: 'flex', flexDirection: 'column',
    }}>
      <style>{`
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 5px; height: 5px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
        input:focus, textarea:focus, select:focus { outline: none; border-color: #10b981 !important; }
        select { appearance: auto; background: #ffffff; color: #1e293b; }
        @keyframes spin { to { transform: rotate(360deg) } }
        @keyframes fadeUp { from { opacity:0; transform:translateY(12px) } to { opacity:1; transform:translateY(0) } }
        .fade-up { animation: fadeUp 0.4s ease both; }
      `}</style>

      {/* ═══ HEADER ═══ */}
      <header style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '12px 24px', borderBottom: '1px solid #e2e8f0', background: '#ffffff',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{
            width: 32, height: 32, borderRadius: 8,
            background: 'linear-gradient(135deg, #10b981, #6366f1)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 15, fontWeight: 700, color: '#fff',
          }}>M</div>
          <div>
            <div style={{ fontSize: 15, fontWeight: 700, letterSpacing: '-0.02em' }}>MedML Forge</div>
            <div style={{ fontSize: 10, color: '#64748b', letterSpacing: '0.05em', textTransform: 'uppercase' }}>
              On-Device Clinical AI Pipeline
            </div>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, fontSize: 11 }}>
          <span style={{
            display: 'flex', alignItems: 'center', gap: 5, padding: '4px 10px',
            borderRadius: 6, background: backend.online ? '#d1fae5' : '#fee2e2',
            color: backend.online ? '#065f46' : '#991b1b',
          }}>
            <span style={{
              width: 6, height: 6, borderRadius: '50%',
              background: backend.online ? '#10b981' : '#ef4444',
            }} />
            {backend.online ? backend.device : 'Offline'}
          </span>
          <span style={{ padding: '4px 10px', borderRadius: 6, background: '#e2e8f0', color: '#475569' }}>
            🔒 LOCAL
          </span>
        </div>
      </header>

      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>

        {/* ═══ LEFT RAIL — PIPELINE PROGRESS ═══ */}
        <nav style={{
          width: 220, flexShrink: 0, background: '#f1f5f9',
          borderRight: '1px solid #e2e8f0', display: 'flex', flexDirection: 'column',
          padding: '20px 0',
        }}>
          <div style={{
            fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.1em',
            color: '#94a3b8', padding: '0 16px', marginBottom: 12, fontWeight: 600,
          }}>Pipeline</div>

          {STAGES.map((s, i) => {
            const active = i === stage
            const done = completed.has(i)
            const locked = i > stage + 1 && !done
            return (
              <div key={s.id}
                onClick={() => !locked && setStage(i)}
                style={{
                  display: 'flex', alignItems: 'center', gap: 10,
                  padding: '10px 16px', cursor: locked ? 'default' : 'pointer',
                  borderLeft: active ? '2px solid #10b981' : '2px solid transparent',
                  background: active ? '#ecfdf5' : 'transparent',
                  opacity: locked ? 0.3 : 1,
                  transition: 'all 0.15s',
                }}
              >
                <div style={{
                  width: 26, height: 26, borderRadius: 6,
                  background: done ? '#d1fae5' : active ? '#e2e8f0' : '#ffffff',
                  border: `1px solid ${done ? '#6ee7b7' : active ? '#cbd5e1' : '#e2e8f0'}`,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: 12, color: done ? '#10b981' : active ? '#1e293b' : '#64748b',
                  fontWeight: 600,
                }}>
                  {done ? '✓' : s.icon}
                </div>
                <div>
                  <div style={{ fontSize: 12, fontWeight: active ? 600 : 400, color: active ? '#10b981' : done ? '#64748b' : '#94a3b8' }}>
                    {s.label}
                  </div>
                  <div style={{ fontSize: 10, color: '#94a3b8' }}>{s.desc}</div>
                </div>
              </div>
            )
          })}

          {/* Overall progress bar */}
          <div style={{ marginTop: 'auto', padding: '16px' }}>
            <div style={{ fontSize: 10, color: '#94a3b8', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Progress
            </div>
            <div style={{ height: 4, borderRadius: 2, background: '#e2e8f0', overflow: 'hidden' }}>
              <div style={{
                height: '100%', borderRadius: 2,
                background: 'linear-gradient(90deg, #10b981, #6366f1)',
                width: `${(completed.size / STAGES.length) * 100}%`,
                transition: 'width 0.5s ease',
              }} />
            </div>
            <div style={{ fontSize: 11, color: '#64748b', marginTop: 6 }}>
              {completed.size}/{STAGES.length} stages
            </div>
          </div>
        </nav>

        {/* ═══ MAIN WORKSPACE ═══ */}
        <main style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>

          {/* ── Center panel: stage content ─── */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '24px 32px' }}>

            {/* STAGE 0: SCAN */}
            {stage === 0 && (
              <div className="fade-up">
                <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 4 }}>Point to your data</h2>
                <p style={{ fontSize: 13, color: '#64748b', marginBottom: 20 }}>
                  Enter the path to your local dataset. Only metadata leaves this box — never raw patient data.
                </p>

                <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
                  <input
                    value={dataPath} onChange={e => setDataPath(e.target.value)}
                    placeholder="/path/to/your/dataset"
                    style={{
                      flex: 1, padding: '10px 14px', background: '#ffffff', border: '1px solid #e2e8f0',
                      borderRadius: 8, color: '#1e293b', fontSize: 13, fontFamily: "'JetBrains Mono', monospace",
                    }}
                    onKeyDown={e => e.key === 'Enter' && runScan()}
                  />
                  <button
                    onClick={async () => {
                      try {
                        const res = await fetch('http://localhost:8081/pick-folder', { method: 'POST' })
                        const data = await res.json()
                        if (data.path) setDataPath(data.path)
                      } catch {
                        alert('Could not open folder picker — is the backend running?')
                      }
                    }}
                    style={{
                      padding: '10px 14px', borderRadius: 8, border: '1px solid #e2e8f0',
                      background: '#ffffff', color: '#475569', fontSize: 13, cursor: 'pointer', whiteSpace: 'nowrap',
                    }}
                  >
                    📁 Browse
                  </button>
                  <button onClick={runScan} disabled={scanning || !dataPath.trim()} style={{
                    padding: '10px 20px', borderRadius: 8, border: 'none', fontWeight: 600, fontSize: 13,
                    background: scanning ? '#e2e8f0' : '#10b981', color: scanning ? '#64748b' : '#fff',
                    cursor: scanning ? 'wait' : 'pointer',
                  }}>
                    {scanning ? '⏳ Scanning...' : '▶ Scan'}
                  </button>
                </div>

                <div style={{
                  background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: 16,
                  fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: '#64748b', lineHeight: 1.8,
                }}>
                  <div style={{ color: '#94a3b8', marginBottom: 8 }}># What the scanner does locally:</div>
                  <div style={{ color: '#10b981' }}>→ Counts files, detects formats (CSV, DICOM, PNG...)</div>
                  <div style={{ color: '#10b981' }}>→ Reads column names & types (tabular)</div>
                  <div style={{ color: '#10b981' }}>→ Detects class folders & resolution (images)</div>
                  <div style={{ color: '#10b981' }}>→ Computes distributions, missing %, duplicates</div>
                  <div style={{ color: '#10b981' }}>→ Generates sample thumbnails (images only)</div>
                  <div style={{ marginTop: 8, color: '#ef4444' }}>✗ Never reads raw cell values or pixel data</div>
                  <div style={{ color: '#ef4444' }}>✗ Never uploads anything to any server</div>
                </div>
              </div>
            )}

            {/* STAGE 1: PREVIEW */}
            {stage === 1 && dataset && (
              <div className="fade-up">
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
                  <div>
                    <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 4 }}>Dataset Preview</h2>
                    <p style={{ fontSize: 13, color: '#64748b' }}>
                      {dataset.type === 'tabular' ? `${dataset.summary.total_rows?.toLocaleString()} rows × ${dataset.summary.total_columns} columns` :
                       dataset.type === 'image' ? `${dataset.summary.total_files?.toLocaleString()} images across ${dataset.summary.num_classes} classes` :
                       'Dataset loaded'}
                    </p>
                  </div>
                  <button onClick={goNext} style={{
                    padding: '8px 16px', borderRadius: 8, border: 'none', background: '#10b981',
                    color: '#fff', fontWeight: 600, fontSize: 13, cursor: 'pointer',
                  }}>Continue →</button>
                </div>

                {/* Metrics row */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 10, marginBottom: 20 }}>
                  {dataset.type === 'tabular' ? <>
                    <MetricBox label="Rows" value={dataset.summary.total_rows?.toLocaleString()} />
                    <MetricBox label="Columns" value={dataset.summary.total_columns} />
                    <MetricBox label="Missing" value={`${dataset.summary.missing_pct}%`} color={dataset.summary.missing_pct > 5 ? '#f59e0b' : '#10b981'} />
                    <MetricBox label="Duplicates" value={dataset.summary.duplicates} color={dataset.summary.duplicates > 0 ? '#ef4444' : '#10b981'} />
                    <MetricBox label="Numeric" value={dataset.summary.dtypes?.numeric} />
                    <MetricBox label="Categorical" value={dataset.summary.dtypes?.categorical} />
                  </> : <>
                    <MetricBox label="Images" value={dataset.summary.total_files?.toLocaleString()} />
                    <MetricBox label="Classes" value={dataset.summary.num_classes} />
                    <MetricBox label="Avg Resolution" value={dataset.summary.avg_resolution} />
                    <MetricBox label="Corrupted" value={dataset.summary.corrupted} color={dataset.summary.corrupted > 0 ? '#ef4444' : '#10b981'} />
                  </>}
                </div>

                {/* Tabular preview */}
                {dataset.type === 'tabular' && dataset.preview && (
                  <div style={{ marginBottom: 20 }}>
                    <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8, color: '#64748b' }}>Data Sample</div>
                    <DataTable columns={dataset.preview.columns} rows={dataset.preview.rows} />
                  </div>
                )}

                {/* Column inspector */}
                {dataset.type === 'tabular' && dataset.columns && (
                  <div style={{ marginBottom: 20 }}>
                    <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8, color: '#64748b' }}>Column Inspector</div>
                    <ColumnInspector columns={dataset.columns} />
                  </div>
                )}

                {/* Image preview */}
                {dataset.type === 'image' && dataset.preview && (
                  <div style={{ marginBottom: 20 }}>
                    <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8, color: '#64748b' }}>Class Samples</div>
                    <ImageGrid images={dataset.preview} />
                  </div>
                )}

                {/* Class distribution for images */}
                {dataset.type === 'image' && dataset.summary.classes && (
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8, color: '#64748b' }}>Class Distribution</div>
                    {Object.entries(dataset.summary.classes).map(([cls, cnt]) => {
                      const pct = (cnt / dataset.summary.total_files * 100)
                      return (
                        <div key={cls} style={{ marginBottom: 6 }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: '#64748b', marginBottom: 3 }}>
                            <span>{cls}</span>
                            <span style={{ fontFamily: "'JetBrains Mono', monospace" }}>{cnt.toLocaleString()} ({pct.toFixed(1)}%)</span>
                          </div>
                          <div style={{ height: 4, borderRadius: 2, background: '#e2e8f0' }}>
                            <div style={{ height: '100%', width: `${pct}%`, borderRadius: 2, background: '#10b981', transition: 'width 0.5s' }} />
                          </div>
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>
            )}

            {/* STAGE 2: CLEANUP */}
            {stage === 2 && (
              <div className="fade-up">
                <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 4 }}>Data Cleanup</h2>
                <p style={{ fontSize: 13, color: '#64748b', marginBottom: 20 }}>
                  Fix data quality issues before training. The AI agent will recommend specific actions.
                </p>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 20 }}>
                  {['Remove duplicates', 'Impute missing values', 'Cap outliers (3σ)', 'Remove corrupted files'].map((action, i) => (
                    <div key={i} style={{
                      padding: '12px 16px', background: '#ffffff', borderRadius: 8,
                      border: '1px solid #e2e8f0', display: 'flex', alignItems: 'center', gap: 10,
                    }}>
                      <div style={{
                        width: 18, height: 18, borderRadius: 4, background: '#d1fae5',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontSize: 11, color: '#10b981',
                      }}>✓</div>
                      <span style={{ fontSize: 13 }}>{action}</span>
                    </div>
                  ))}
                </div>

                {cleanupResult && (
                  <div style={{ background: '#ffffff', borderRadius: 8, border: '1px solid #e2e8f0', padding: 16, marginBottom: 16 }}>
                    <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>Cleanup Results</div>
                    {cleanupResult.actions_taken?.map((a, i) => (
                      <div key={i} style={{ fontSize: 12, color: '#10b981', marginBottom: 4 }}>✓ {a}</div>
                    ))}
                    <div style={{ fontSize: 12, color: '#64748b', marginTop: 8 }}>
                      {cleanupResult.rows_before?.toLocaleString()} → {cleanupResult.rows_after?.toLocaleString()} entries
                    </div>
                  </div>
                )}

                <div style={{ display: 'flex', gap: 8 }}>
                  <button onClick={runCleanup} style={{
                    padding: '10px 20px', borderRadius: 8, border: 'none', background: '#10b981',
                    color: '#fff', fontWeight: 600, fontSize: 13, cursor: 'pointer',
                  }}>Run Cleanup</button>
                  <button onClick={goNext} style={{
                    padding: '10px 20px', borderRadius: 8, border: '1px solid #e2e8f0', background: 'transparent',
                    color: '#64748b', fontWeight: 600, fontSize: 13, cursor: 'pointer',
                  }}>Skip →</button>
                </div>
              </div>
            )}

            {/* STAGE 3: CONFIGURE */}
            {stage === 3 && (
              <div className="fade-up">
                <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 4 }}>Training Configuration</h2>
                <p style={{ fontSize: 13, color: '#64748b', marginBottom: 20 }}>
                  Configure your model, select features, and set training parameters.
                </p>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 24 }}>
                  <ConfigField label="Target Column" value={trainConfig.target_column}
                    onChange={v => setTrainConfig(c => ({ ...c, target_column: v }))}
                    options={scanResult?.summary?.target_candidates}
                    placeholder="auto-detect" />
                  <ConfigField label="Model Type" value={trainConfig.model_type}
                    onChange={v => setTrainConfig(c => ({ ...c, model_type: v }))}
                    options={['auto', 'mlp', 'cnn', 'resnet', 'logistic']} />
                  <ConfigField label="Epochs" value={trainConfig.epochs} type="number"
                    onChange={v => setTrainConfig(c => ({ ...c, epochs: parseInt(v) || 50 }))} />
                  <ConfigField label="Learning Rate" value={trainConfig.lr} type="number" step="0.0001"
                    onChange={v => setTrainConfig(c => ({ ...c, lr: parseFloat(v) || 0.001 }))} />
                  <ConfigField label="Batch Size" value={trainConfig.batch_size} type="number"
                    onChange={v => setTrainConfig(c => ({ ...c, batch_size: parseInt(v) || 32 }))} />
                </div>

                {/* ── Feature Selection ─────────────────────── */}
                {dataset?.type === 'tabular' && dataset?.columns?.length > 0 && (
                  <div style={{ marginBottom: 24, border: '1px solid #e2e8f0', borderRadius: 10, overflow: 'hidden' }}>
                    {/* Header row */}
                    <div style={{
                      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                      padding: '12px 16px', background: '#f8fafc', borderBottom: '1px solid #e2e8f0',
                    }}>
                      <div style={{ fontWeight: 600, fontSize: 13 }}>Feature Selection
                        {selectedFeatures.length > 0 && (
                          <span style={{ marginLeft: 8, fontSize: 11, color: '#10b981', fontWeight: 400 }}>
                            {selectedFeatures.length} selected
                          </span>
                        )}
                      </div>
                      {/* Mode toggle */}
                      <div style={{ display: 'flex', gap: 4, background: '#e2e8f0', borderRadius: 6, padding: 3 }}>
                        {['manual', 'auto'].map(m => (
                          <button key={m} onClick={() => setFeatureMode(m)} style={{
                            padding: '4px 12px', borderRadius: 4, border: 'none', fontSize: 12, fontWeight: 600,
                            cursor: 'pointer',
                            background: featureMode === m ? '#ffffff' : 'transparent',
                            color: featureMode === m ? '#0f172a' : '#64748b',
                            boxShadow: featureMode === m ? '0 1px 3px #0001' : 'none',
                          }}>
                            {m === 'auto' ? '⚡ Auto (AI)' : '✋ Manual'}
                          </button>
                        ))}
                      </div>
                    </div>

                    <div style={{ padding: 16 }}>
                      {featureMode === 'auto' ? (
                        <div>
                          <div style={{ fontSize: 12, color: '#64748b', marginBottom: 12 }}>
                            Runs PyImpetus (PIMP) statistical importance testing, falling back to Random Forest if not installed.
                          </div>
                          <button
                            disabled={autoRunning}
                            onClick={async () => {
                              setAutoRunning(true)
                              try {
                                const res = await fetch('/feature-select', {
                                  method: 'POST',
                                  headers: { 'Content-Type': 'application/json' },
                                  body: JSON.stringify({ target_column: trainConfig.target_column }),
                                })
                                const data = await res.json()
                                if (data.error) { alert(data.error); return }
                                setAutoFeatures(data)
                                setSelectedFeatures(data.selected)
                              } catch(e) { alert('Feature selection failed: ' + e.message) }
                              finally { setAutoRunning(false) }
                            }}
                            style={{
                              padding: '8px 16px', borderRadius: 7, border: 'none', fontWeight: 600,
                              fontSize: 12, cursor: autoRunning ? 'wait' : 'pointer',
                              background: autoRunning ? '#e2e8f0' : '#6366f1',
                              color: autoRunning ? '#64748b' : '#fff',
                              marginBottom: autoFeatures ? 14 : 0,
                            }}>
                            {autoRunning ? '⏳ Running...' : '▶ Run Feature Selection'}
                          </button>

                          {autoFeatures && (
                            <div>
                              <div style={{ fontSize: 11, color: '#64748b', marginBottom: 10 }}>
                                Method: <strong>{autoFeatures.method}</strong> · {autoFeatures.selected.length}/{autoFeatures.all_features.length} features selected
                              </div>
                              <div style={{ display: 'grid', gap: 5, maxHeight: 260, overflowY: 'auto' }}>
                                {autoFeatures.all_features.map(feat => {
                                  const imp = autoFeatures.importances[feat] || 0
                                  const maxImp = Math.max(...Object.values(autoFeatures.importances))
                                  const isSelected = selectedFeatures.includes(feat)
                                  return (
                                    <div key={feat} onClick={() => setSelectedFeatures(prev =>
                                      prev.includes(feat) ? prev.filter(f => f !== feat) : [...prev, feat]
                                    )} style={{
                                      display: 'flex', alignItems: 'center', gap: 10, padding: '7px 10px',
                                      borderRadius: 6, cursor: 'pointer',
                                      border: `1px solid ${isSelected ? '#10b981' : '#e2e8f0'}`,
                                      background: isSelected ? '#f0fdf4' : '#fafafa',
                                    }}>
                                      <input type="checkbox" readOnly checked={isSelected}
                                        style={{ accentColor: '#10b981', cursor: 'pointer' }} />
                                      <span style={{ fontSize: 12, fontFamily: 'monospace', flex: 1, color: '#1e293b' }}>{feat}</span>
                                      <div style={{ width: 80, height: 6, background: '#e2e8f0', borderRadius: 3, overflow: 'hidden' }}>
                                        <div style={{ width: `${(imp / maxImp) * 100}%`, height: '100%', background: '#10b981', borderRadius: 3 }} />
                                      </div>
                                      <span style={{ fontSize: 10, color: '#64748b', width: 42, textAlign: 'right', fontFamily: 'monospace' }}>
                                        {imp.toFixed(3)}
                                      </span>
                                    </div>
                                  )
                                })}
                              </div>
                            </div>
                          )}
                        </div>
                      ) : (
                        /* Manual mode — checkbox list of all columns */
                        <div>
                          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
                            <div style={{ fontSize: 12, color: '#64748b' }}>Check the features to include in training.</div>
                            <div style={{ display: 'flex', gap: 8 }}>
                              <button onClick={() => setSelectedFeatures(
                                dataset.columns.filter(c => c.name !== trainConfig.target_column).map(c => c.name)
                              )} style={{ fontSize: 11, padding: '3px 8px', borderRadius: 4, border: '1px solid #e2e8f0', background: '#fff', color: '#475569', cursor: 'pointer' }}>
                                All
                              </button>
                              <button onClick={() => setSelectedFeatures([])}
                                style={{ fontSize: 11, padding: '3px 8px', borderRadius: 4, border: '1px solid #e2e8f0', background: '#fff', color: '#475569', cursor: 'pointer' }}>
                                None
                              </button>
                            </div>
                          </div>
                          <div style={{ display: 'grid', gap: 4, maxHeight: 260, overflowY: 'auto' }}>
                            {dataset.columns
                              .filter(c => c.name !== trainConfig.target_column)
                              .map(col => {
                                const allCols = dataset.columns.filter(c => c.name !== trainConfig.target_column).map(c => c.name)
                                const isSelected = selectedFeatures.length === 0 || selectedFeatures.includes(col.name)
                                return (
                                  <div key={col.name} onClick={() => setSelectedFeatures(prev => {
                                    const cur = prev.length === 0 ? allCols : prev
                                    return cur.includes(col.name) ? cur.filter(f => f !== col.name) : [...cur, col.name]
                                  })} style={{
                                    display: 'flex', alignItems: 'center', gap: 10, padding: '7px 10px',
                                    borderRadius: 6, cursor: 'pointer',
                                    border: `1px solid ${isSelected ? '#6366f1' : '#e2e8f0'}`,
                                    background: isSelected ? '#eef2ff' : '#fafafa',
                                  }}>
                                    <input type="checkbox" readOnly checked={isSelected}
                                      style={{ accentColor: '#6366f1', cursor: 'pointer' }} />
                                    <span style={{ fontSize: 12, fontFamily: 'monospace', flex: 1, color: '#1e293b' }}>{col.name}</span>
                                    <span style={{ fontSize: 10, padding: '2px 6px', borderRadius: 4,
                                      background: col.dtype?.includes('float') || col.dtype?.includes('int') ? '#d1fae5' : '#e0e7ff',
                                      color: col.dtype?.includes('float') || col.dtype?.includes('int') ? '#065f46' : '#3730a3',
                                    }}>{col.dtype}</span>
                                    {col.missing_pct > 0 && (
                                      <span style={{ fontSize: 10, color: '#f59e0b' }}>{col.missing_pct}% missing</span>
                                    )}
                                  </div>
                                )
                              })}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                <div style={{ display: 'flex', gap: 8 }}>
                  <button onClick={() => {
                    completeStage(3)
                    startTraining()
                  }} style={{
                    padding: '10px 24px', borderRadius: 8, border: 'none', background: '#10b981',
                    color: '#fff', fontWeight: 600, fontSize: 13, cursor: 'pointer',
                    display: 'flex', alignItems: 'center', gap: 6,
                  }}>
                    ⚡ Start Training
                  </button>
                  <button onClick={() => askAI('What model architecture and hyperparameters do you recommend for this dataset?')} style={{
                    padding: '10px 20px', borderRadius: 8, border: '1px solid #6366f1', background: 'transparent',
                    color: '#6366f1', fontWeight: 600, fontSize: 13, cursor: 'pointer',
                  }}>
                    🧠 Ask AI
                  </button>
                </div>
              </div>
            )}

            {/* STAGE 4: TRAINING */}
            {stage === 4 && (
              <div className="fade-up">
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
                  <div>
                    <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 4 }}>Training</h2>
                    <p style={{ fontSize: 13, color: '#64748b' }}>
                      {training?.active ? `Epoch ${training.epoch}/${training.total_epochs}` : training?.status === 'complete' ? 'Complete' : 'Starting...'}
                    </p>
                  </div>
                  {training?.active && (
                    <button onClick={stopTraining} style={{
                      padding: '8px 16px', borderRadius: 8, border: '1px solid #ef4444', background: 'transparent',
                      color: '#ef4444', fontWeight: 600, fontSize: 12, cursor: 'pointer',
                    }}>■ Stop</button>
                  )}
                </div>

                {/* Progress */}
                {training && (
                  <div style={{ marginBottom: 20 }}>
                    <div style={{ height: 6, borderRadius: 3, background: '#e2e8f0', overflow: 'hidden' }}>
                      <div style={{
                        height: '100%', borderRadius: 3,
                        background: training.active
                          ? 'linear-gradient(90deg, #10b981, #34d399)'
                          : training.status === 'complete' ? '#10b981' : '#ef4444',
                        width: `${training.total_epochs > 0 ? (training.epoch / training.total_epochs) * 100 : 0}%`,
                        transition: 'width 0.3s',
                      }} />
                    </div>
                  </div>
                )}

                {/* Live metrics */}
                {training?.metrics && (
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10, marginBottom: 20 }}>
                    <MetricBox label="Train Loss" value={training.metrics.train_loss?.toFixed(4)} color="#f59e0b" mono />
                    <MetricBox label="Val Loss" value={training.metrics.val_loss?.toFixed(4)} color="#10b981" mono />
                    <MetricBox label="Accuracy" value={training.metrics.accuracy ? `${(training.metrics.accuracy * 100).toFixed(1)}%` : '—'} color="#10b981" mono />
                    <MetricBox label="F1 Score" value={training.metrics.f1?.toFixed(3)} color="#6366f1" mono />
                  </div>
                )}

                {/* Loss & accuracy charts */}
                {trainHistory.length > 2 && (
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
                    <div style={{ background: '#ffffff', borderRadius: 10, padding: 16, border: '1px solid #e2e8f0' }}>
                      <MiniChart data={trainHistory.map(h => h.val_loss)} color="#10b981" height={90} label="Validation Loss" />
                    </div>
                    <div style={{ background: '#ffffff', borderRadius: 10, padding: 16, border: '1px solid #e2e8f0' }}>
                      <MiniChart data={trainHistory.map(h => h.accuracy)} color="#6366f1" height={90} label="Accuracy" />
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* STAGE 5: EVALUATE */}
            {stage === 5 && training?.metrics && (
              <div className="fade-up">
                <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 4 }}>Evaluation Results</h2>
                <p style={{ fontSize: 13, color: '#64748b', marginBottom: 20 }}>
                  Training complete. Review performance before deciding next steps.
                </p>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10, marginBottom: 20 }}>
                  <MetricBox label="Final Accuracy" value={`${(training.metrics.accuracy * 100).toFixed(1)}%`} color="#10b981" mono large />
                  <MetricBox label="F1 Score" value={training.metrics.f1?.toFixed(3)} color="#6366f1" mono large />
                  <MetricBox label="Train Loss" value={training.metrics.train_loss?.toFixed(4)} color="#f59e0b" mono large />
                  <MetricBox label="Val Loss" value={training.metrics.val_loss?.toFixed(4)} mono large />
                </div>

                {trainHistory.length > 2 && (
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
                    <div style={{ background: '#ffffff', borderRadius: 10, padding: 16, border: '1px solid #e2e8f0' }}>
                      <MiniChart data={trainHistory.map(h => h.train_loss)} color="#f59e0b" height={100} label="Train Loss (full history)" />
                    </div>
                    <div style={{ background: '#ffffff', borderRadius: 10, padding: 16, border: '1px solid #e2e8f0' }}>
                      <MiniChart data={trainHistory.map(h => h.accuracy)} color="#10b981" height={100} label="Accuracy (full history)" />
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* ═══ RIGHT PANEL — AI CO-PILOT ═══ */}
          <aside style={{
            width: 340, flexShrink: 0, borderLeft: '1px solid #e2e8f0',
            background: '#f1f5f9', display: 'flex', flexDirection: 'column',
            overflow: 'hidden', minHeight: 0,
          }}>
            <div style={{
              padding: '12px 16px', borderBottom: '1px solid #e2e8f0',
              fontSize: 12, fontWeight: 600, color: '#64748b',
              display: 'flex', alignItems: 'center', gap: 6,
            }}>
              <span style={{ fontSize: 14 }}>🧠</span> AI Co-Pilot
              {aiLoading && <span style={{ color: '#10b981', fontSize: 10, animation: 'spin 1s linear infinite', display: 'inline-block' }}>◌</span>}
              <span style={{
                marginLeft: 'auto', fontSize: 10, padding: '2px 6px', borderRadius: 4,
                background: '#e2e8f0', color: '#64748b',
              }}>Qwen 2.5</span>
            </div>

            <div ref={aiChatRef} style={{ flex: 1, overflowY: 'auto', padding: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
              {aiMessages.length === 0 && (
                <div style={{ fontSize: 12, color: '#94a3b8', textAlign: 'center', padding: '40px 16px', lineHeight: 1.7 }}>
                  The AI co-pilot will analyze your data and provide recommendations as you progress through the pipeline.
                </div>
              )}
              {aiMessages.map((m, i) => (
                <div key={i} style={{
                  padding: '8px 12px', borderRadius: m.role === 'user' ? '10px 10px 2px 10px' : m.role === 'system' ? 8 : '2px 10px 10px 10px',
                  background: m.role === 'user' ? '#10b981' : m.role === 'system' ? '#eef2ff' : '#ffffff',
                  border: m.role === 'ai' ? '1px solid #e2e8f0' : m.role === 'system' ? '1px solid #c7d2fe' : 'none',
                  color: m.role === 'user' ? '#fff' : m.role === 'system' ? '#6366f1' : '#1e293b',
                  fontSize: 12, lineHeight: 1.6, maxWidth: m.role === 'system' ? '100%' : '90%',
                  alignSelf: m.role === 'user' ? 'flex-end' : m.role === 'system' ? 'center' : 'flex-start',
                  whiteSpace: 'pre-wrap', wordBreak: 'break-word',
                }}>
                  {m.text}
                </div>
              ))}
            </div>

            <div style={{ padding: '8px 12px', borderTop: '1px solid #e2e8f0' }}>
              <div style={{ display: 'flex', gap: 6 }}>
                <input
                  value={aiInput} onChange={e => setAiInput(e.target.value)}
                  onKeyDown={e => {
                    if (e.key === 'Enter' && aiInput.trim()) {
                      askAI(aiInput)
                      setAiInput('')
                    }
                  }}
                  placeholder="Ask the AI co-pilot..."
                  style={{
                    flex: 1, padding: '8px 10px', background: '#ffffff', border: '1px solid #e2e8f0',
                    borderRadius: 6, color: '#1e293b', fontSize: 12,
                  }}
                />
                <button
                  onClick={() => { if (aiInput.trim()) { askAI(aiInput); setAiInput('') } }}
                  disabled={aiLoading}
                  style={{
                    padding: '8px 12px', borderRadius: 6, border: 'none', background: '#10b981',
                    color: '#fff', fontWeight: 600, fontSize: 11, cursor: 'pointer',
                  }}
                >→</button>
              </div>
            </div>
          </aside>
        </main>
      </div>
    </div>
  )
}


// ── Small reusable pieces ───────────────────────────────────

function MetricBox({ label, value, color, mono, large }) {
  return (
    <div style={{
      background: '#ffffff', borderRadius: 8, padding: '12px 14px',
      border: '1px solid #e2e8f0',
    }}>
      <div style={{ fontSize: 10, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: 4 }}>
        {label}
      </div>
      <div style={{
        fontSize: large ? 22 : 18, fontWeight: 700,
        color: color || '#1e293b',
        fontFamily: mono ? "'JetBrains Mono', monospace" : 'inherit',
      }}>
        {value ?? '—'}
      </div>
    </div>
  )
}

function ConfigField({ label, value, onChange, type = 'text', options, placeholder, step }) {
  return (
    <div>
      <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
        {label}
      </div>
      {options ? (
        <select
          value={value}
          onChange={e => onChange(e.target.value)}
          style={{
            width: '100%', padding: '8px 10px', background: '#ffffff', border: '1px solid #e2e8f0',
            borderRadius: 6, color: '#1e293b', fontSize: 13,
          }}
        >
          {options.map(o => <option key={o} value={o}>{o}</option>)}
        </select>
      ) : (
        <input
          type={type} value={value} step={step}
          onChange={e => onChange(e.target.value)}
          placeholder={placeholder}
          style={{
            width: '100%', padding: '8px 10px', background: '#ffffff', border: '1px solid #e2e8f0',
            borderRadius: 6, color: '#1e293b', fontSize: 13,
            fontFamily: type === 'number' ? "'JetBrains Mono', monospace" : 'inherit',
          }}
        />
      )}
    </div>
  )
}
