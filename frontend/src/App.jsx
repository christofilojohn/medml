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
  { id: 'scan',    icon: 'biotech',                  label: 'Scan',      desc: 'Point to your local data directory' },
  { id: 'preview', icon: 'visibility',               label: 'Preview',   desc: 'Inspect and understand your dataset' },
  { id: 'clean',   icon: 'cleaning_services',        label: 'Cleanup',   desc: 'Fix quality issues' },
  { id: 'config',  icon: 'settings_input_component', label: 'Configure', desc: 'Model & training settings' },
  { id: 'train',   icon: 'model_training',           label: 'Training',  desc: 'On-device model training' },
  { id: 'eval',    icon: 'analytics',                label: 'Evaluate',  desc: 'Performance assessment' },
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
    <div className="font-body text-on-surface min-h-screen bg-surface">

      {/* ═══ SIDEBAR ═══ */}
      <aside className="fixed left-0 top-0 h-screen flex flex-col py-8 gap-2 bg-slate-50 border-r border-slate-200/50 w-64 z-50">
        <div className="px-6 mb-6">
          <h2 className="font-body text-xs font-black uppercase tracking-widest text-primary">PIPELINE</h2>
        </div>
        <nav className="flex-1 flex flex-col gap-1">
          {STAGES.map((s, i) => {
            const active = i === stage
            const done = completed.has(i)
            const locked = i > stage + 1 && !done
            return (
              <button key={s.id}
                onClick={() => !locked && setStage(i)}
                className={`flex items-center px-4 py-3 mx-2 rounded-md text-left transition-all duration-200 ${
                  active
                    ? 'bg-primary text-white shadow-sm'
                    : locked
                    ? 'text-slate-300 cursor-default'
                    : 'text-slate-600 hover:translate-x-1 cursor-pointer'
                }`}
              >
                <span className="material-symbols-outlined mr-3 text-xl">
                  {done && !active ? 'check_circle' : s.icon}
                </span>
                <span className="font-body text-sm font-medium uppercase tracking-widest">{s.label}</span>
              </button>
            )
          })}
        </nav>
        <div className="px-6 pt-4 border-t border-slate-200/50">
          <p className="font-label text-[10px] uppercase tracking-widest text-on-surface-variant mb-2">Progress</p>
          <div className="h-1 bg-surface-container rounded-full overflow-hidden">
            <div className="h-full bg-primary rounded-full transition-all duration-500"
              style={{ width: `${(completed.size / STAGES.length) * 100}%` }} />
          </div>
          <p className="text-xs text-on-surface-variant mt-2">{completed.size}/{STAGES.length} stages</p>
        </div>
      </aside>

      <div className="ml-64 min-h-screen flex flex-col">

        {/* ═══ HEADER ═══ */}
        <header className="sticky top-0 z-40 bg-slate-50 flex justify-between items-center w-full px-8 py-4 border-b border-slate-200/50">
          <div className="flex items-center gap-3">
            <span className="material-symbols-outlined text-primary">lock</span>
            <h1 className="font-headline font-bold tracking-tight text-xl text-primary">MedML Forge</h1>
          </div>
          <div className="flex items-center gap-4">
            <div className="px-3 py-1 bg-surface-container rounded-full flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${backend.online ? 'bg-tertiary-fixed-dim' : 'bg-error'}`}
                style={backend.online ? { boxShadow: '0 0 8px rgba(136,217,130,0.6)' } : {}} />
              <span className="font-label text-xs font-bold tracking-wider text-primary">
                {backend.online ? (backend.device?.toUpperCase() || 'LOCAL') : 'OFFLINE'}
              </span>
            </div>
            <button className="material-symbols-outlined text-slate-500 hover:text-primary transition-colors">account_circle</button>
          </div>
        </header>

        {/* ═══ MAIN WORKSPACE ═══ */}
        <div className="flex flex-1 overflow-hidden">

          {/* ── Center panel: stage content ─── */}
          <div className="flex-1 overflow-y-auto p-8 bg-surface">
            <div className="max-w-4xl mx-auto space-y-8">

              {/* STAGE 0: SCAN */}
              {stage === 0 && (
                <div className="fade-up space-y-6">
                  <div>
                    <p className="font-label text-xs uppercase tracking-widest text-on-surface-variant mb-1">Data Ingestion</p>
                    <h2 className="font-headline text-3xl font-extrabold text-primary tracking-tighter">Point to Your Data</h2>
                  </div>

                  <div className="bg-surface-container-low p-8 rounded-xl space-y-6">
                    <div className="flex items-center gap-3 border-b border-outline-variant/20 pb-4">
                      <span className="material-symbols-outlined text-primary">folder_open</span>
                      <h3 className="font-headline text-lg font-bold text-primary">Dataset Location</h3>
                    </div>
                    <p className="text-sm text-on-surface-variant">
                      Enter the path to your local dataset. Only metadata is analyzed — never raw patient data.
                    </p>
                    <div className="flex gap-3">
                      <input
                        value={dataPath} onChange={e => setDataPath(e.target.value)}
                        placeholder="/path/to/your/dataset"
                        onKeyDown={e => e.key === 'Enter' && runScan()}
                        className="flex-1 px-4 py-2.5 bg-surface-container-lowest border border-outline-variant/40 rounded-lg text-on-surface text-sm font-mono focus:outline-none focus:border-primary"
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
                        className="px-4 py-2.5 bg-surface-container-high text-primary font-bold text-sm rounded-lg hover:bg-surface-container-highest transition-colors whitespace-nowrap flex items-center gap-2"
                      >
                        <span className="material-symbols-outlined text-base">folder</span>
                        Browse
                      </button>
                      <button onClick={runScan} disabled={scanning || !dataPath.trim()}
                        className={`px-6 py-2.5 font-bold text-sm rounded-lg flex items-center gap-2 transition-all ${
                          scanning || !dataPath.trim()
                            ? 'bg-surface-container text-on-surface-variant cursor-not-allowed'
                            : 'bg-primary text-on-primary hover:opacity-90 shadow-lg'
                        }`}
                      >
                        <span className="material-symbols-outlined text-base">
                          {scanning ? 'hourglass_empty' : 'play_arrow'}
                        </span>
                        {scanning ? 'Scanning...' : 'Scan'}
                      </button>
                    </div>

                    <div className="bg-surface p-5 rounded-lg border-l-4 border-primary font-mono text-xs text-on-surface-variant leading-relaxed space-y-1">
                      <p className="text-outline mb-2"># What the scanner does locally:</p>
                      <p className="text-on-tertiary-container">→ Counts files, detects formats (CSV, DICOM, PNG...)</p>
                      <p className="text-on-tertiary-container">→ Reads column names &amp; types (tabular)</p>
                      <p className="text-on-tertiary-container">→ Detects class folders &amp; resolution (images)</p>
                      <p className="text-on-tertiary-container">→ Computes distributions, missing %, duplicates</p>
                      <p className="text-on-tertiary-container">→ Generates sample thumbnails (images only)</p>
                      <p className="text-error mt-2">✗ Never reads raw cell values or pixel data</p>
                      <p className="text-error">✗ Never uploads anything to any server</p>
                    </div>
                  </div>
                </div>
              )}

              {/* STAGE 1: PREVIEW */}
              {stage === 1 && dataset && (
                <div className="fade-up space-y-6">
                  <div className="flex items-end justify-between">
                    <div>
                      <p className="font-label text-xs uppercase tracking-widest text-on-surface-variant mb-1">Data Inspection</p>
                      <h2 className="font-headline text-3xl font-extrabold text-primary tracking-tighter">Dataset Preview</h2>
                      <p className="text-sm text-on-surface-variant mt-1">
                        {dataset.type === 'tabular'
                          ? `${dataset.summary.total_rows?.toLocaleString()} rows × ${dataset.summary.total_columns} columns`
                          : dataset.type === 'image'
                          ? `${dataset.summary.total_files?.toLocaleString()} images across ${dataset.summary.num_classes} classes`
                          : 'Dataset loaded'}
                      </p>
                    </div>
                    <button onClick={goNext}
                      className="px-6 py-2 bg-primary text-on-primary font-bold text-sm rounded-md hover:opacity-90 transition-all flex items-center gap-2">
                      Continue
                      <span className="material-symbols-outlined text-base">arrow_forward</span>
                    </button>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    {dataset.type === 'tabular' ? <>
                      <MetricBox label="Rows" value={dataset.summary.total_rows?.toLocaleString()} />
                      <MetricBox label="Columns" value={dataset.summary.total_columns} />
                      <MetricBox label="Missing" value={`${dataset.summary.missing_pct}%`} warn={dataset.summary.missing_pct > 5} />
                      <MetricBox label="Duplicates" value={dataset.summary.duplicates} warn={dataset.summary.duplicates > 0} />
                      <MetricBox label="Numeric" value={dataset.summary.dtypes?.numeric} />
                      <MetricBox label="Categorical" value={dataset.summary.dtypes?.categorical} />
                    </> : <>
                      <MetricBox label="Images" value={dataset.summary.total_files?.toLocaleString()} />
                      <MetricBox label="Classes" value={dataset.summary.num_classes} />
                      <MetricBox label="Avg Resolution" value={dataset.summary.avg_resolution} />
                      <MetricBox label="Corrupted" value={dataset.summary.corrupted} warn={dataset.summary.corrupted > 0} />
                    </>}
                  </div>

                  {dataset.type === 'tabular' && dataset.preview && (
                    <div className="bg-surface-container-low p-6 rounded-xl">
                      <h3 className="font-headline font-bold text-primary text-sm mb-4 flex items-center gap-2">
                        <span className="material-symbols-outlined text-sm">table</span>
                        Data Sample
                      </h3>
                      <DataTable columns={dataset.preview.columns} rows={dataset.preview.rows} />
                    </div>
                  )}

                  {dataset.type === 'tabular' && dataset.columns && (
                    <div className="bg-surface-container-low p-6 rounded-xl">
                      <h3 className="font-headline font-bold text-primary text-sm mb-4 flex items-center gap-2">
                        <span className="material-symbols-outlined text-sm">schema</span>
                        Column Inspector
                      </h3>
                      <ColumnInspector columns={dataset.columns} />
                    </div>
                  )}

                  {dataset.type === 'image' && dataset.preview && (
                    <div className="bg-surface-container-low p-6 rounded-xl">
                      <h3 className="font-headline font-bold text-primary text-sm mb-4">Class Samples</h3>
                      <ImageGrid images={dataset.preview} />
                    </div>
                  )}

                  {dataset.type === 'image' && dataset.summary.classes && (
                    <div className="bg-surface-container-low p-6 rounded-xl">
                      <h3 className="font-headline font-bold text-primary text-sm mb-4">Class Distribution</h3>
                      {Object.entries(dataset.summary.classes).map(([cls, cnt]) => {
                        const pct = (cnt / dataset.summary.total_files * 100)
                        return (
                          <div key={cls} className="mb-4">
                            <div className="flex justify-between text-xs text-on-surface-variant mb-1">
                              <span>{cls}</span>
                              <span className="font-mono">{cnt.toLocaleString()} ({pct.toFixed(1)}%)</span>
                            </div>
                            <div className="h-1 bg-surface-container rounded-full overflow-hidden">
                              <div className="h-full bg-primary rounded-full transition-all duration-500"
                                style={{ width: `${pct}%` }} />
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
                <div className="fade-up space-y-6">
                  <div>
                    <p className="font-label text-xs uppercase tracking-widest text-on-surface-variant mb-1">Quality Assurance</p>
                    <h2 className="font-headline text-3xl font-extrabold text-primary tracking-tighter">Data Cleanup</h2>
                  </div>

                  <div className="bg-surface-container-low p-8 rounded-xl space-y-6">
                    <div className="flex items-center gap-3 border-b border-outline-variant/20 pb-4">
                      <span className="material-symbols-outlined text-primary">cleaning_services</span>
                      <h3 className="font-headline text-lg font-bold text-primary">Cleanup Actions</h3>
                    </div>
                    <p className="text-sm text-on-surface-variant">Fix data quality issues before training. The AI agent will recommend specific actions.</p>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      {['Remove duplicates', 'Impute missing values', 'Cap outliers (3σ)', 'Remove corrupted files'].map((action, i) => (
                        <div key={i} className="bg-surface-container-lowest p-4 rounded-xl flex items-center gap-3 border border-outline-variant/20">
                          <div className="w-8 h-8 rounded bg-tertiary-fixed/30 flex items-center justify-center">
                            <span className="material-symbols-outlined text-on-tertiary-fixed-variant text-base">check</span>
                          </div>
                          <span className="text-sm text-on-surface">{action}</span>
                        </div>
                      ))}
                    </div>

                    {cleanupResult && (
                      <div className="bg-surface p-5 rounded-lg border-l-4 border-on-tertiary-fixed-variant">
                        <p className="font-headline font-bold text-primary text-sm mb-3">Cleanup Results</p>
                        {cleanupResult.actions_taken?.map((a, i) => (
                          <p key={i} className="text-xs text-on-tertiary-fixed-variant mb-1">✓ {a}</p>
                        ))}
                        <p className="text-xs text-on-surface-variant mt-3">
                          {cleanupResult.rows_before?.toLocaleString()} → {cleanupResult.rows_after?.toLocaleString()} entries
                        </p>
                      </div>
                    )}

                    <div className="flex gap-3 pt-2">
                      <button onClick={runCleanup}
                        className="px-6 py-2 bg-primary text-on-primary font-bold text-sm rounded-md hover:opacity-90 transition-all flex items-center gap-2">
                        <span className="material-symbols-outlined text-base">play_arrow</span>
                        Run Cleanup
                      </button>
                      <button onClick={goNext}
                        className="px-6 py-2 bg-surface-container-high text-primary font-bold text-sm rounded-md hover:bg-surface-container-highest transition-colors">
                        Skip →
                      </button>
                    </div>
                  </div>
                </div>
              )}

              {/* STAGE 3: CONFIGURE */}
              {stage === 3 && (
                <div className="fade-up space-y-6">
                  <div className="flex items-end justify-between">
                    <div>
                      <p className="font-label text-xs uppercase tracking-widest text-on-surface-variant mb-1">Configuration Engine</p>
                      <h2 className="font-headline text-3xl font-extrabold text-primary tracking-tighter">Hyperparameter Calibration</h2>
                    </div>
                    <div className="flex gap-3">
                      <button onClick={() => setTrainConfig({ epochs: 50, lr: 0.001, batch_size: 32, model_type: 'auto', target_column: '' })}
                        className="px-5 py-2 bg-surface-container-high text-primary font-bold text-sm rounded-md hover:bg-surface-container-highest transition-colors">
                        RESET DEFAULTS
                      </button>
                      <button onClick={() => { completeStage(3); startTraining() }}
                        className="px-5 py-2 bg-primary text-on-primary font-bold text-sm rounded-md shadow-lg hover:opacity-90 transition-all flex items-center gap-2">
                        <span className="material-symbols-outlined text-base">play_arrow</span>
                        INITIALIZE ENGINE
                      </button>
                    </div>
                  </div>

                  {/* Core sliders */}
                  <div className="bg-surface-container-low p-8 rounded-xl space-y-8">
                    <div className="flex items-center gap-3 border-b border-outline-variant/20 pb-4">
                      <span className="material-symbols-outlined text-primary">tune</span>
                      <h3 className="font-headline text-lg font-bold text-primary">Core Model Sensitivity</h3>
                    </div>

                    {/* Learning rate */}
                    <div className="space-y-3">
                      <div className="flex justify-between items-baseline">
                        <label className="font-headline font-bold text-primary text-sm flex items-center gap-2">
                          Clinical Sensitivity
                          <span className="font-label text-[10px] text-on-surface-variant bg-surface-container px-2 py-0.5 rounded tracking-tighter">(LEARNING RATE)</span>
                        </label>
                        <span className="font-label font-bold text-primary">{trainConfig.lr}</span>
                      </div>
                      <input type="range" className="w-full h-1"
                        min="0.0001" max="0.1" step="0.0001"
                        value={trainConfig.lr}
                        onChange={e => setTrainConfig(c => ({ ...c, lr: parseFloat(e.target.value) }))}
                      />
                      <div className="bg-surface p-4 rounded-lg flex items-start gap-3 border-l-4 border-primary">
                        <span className="material-symbols-outlined text-primary text-sm mt-0.5">info</span>
                        <p className="text-xs text-on-surface-variant leading-relaxed">Lower sensitivity allows for more stable, long-term pattern recognition. Recommended for longitudinal patient studies.</p>
                      </div>
                    </div>

                    {/* Epochs */}
                    <div className="space-y-3">
                      <div className="flex justify-between items-baseline">
                        <label className="font-headline font-bold text-primary text-sm flex items-center gap-2">
                          Diagnostic Depth
                          <span className="font-label text-[10px] text-on-surface-variant bg-surface-container px-2 py-0.5 rounded tracking-tighter">(EPOCHS)</span>
                        </label>
                        <span className="font-label font-bold text-primary">{trainConfig.epochs}</span>
                      </div>
                      <input type="range" className="w-full h-1"
                        min="10" max="1000" step="10"
                        value={trainConfig.epochs}
                        onChange={e => setTrainConfig(c => ({ ...c, epochs: parseInt(e.target.value) }))}
                      />
                      <div className="bg-surface p-4 rounded-lg flex items-start gap-3 border-l-4 border-primary">
                        <span className="material-symbols-outlined text-primary text-sm mt-0.5">info</span>
                        <p className="text-xs text-on-surface-variant leading-relaxed">How many times the model reviews each clinical record. Higher depth increases comprehension but risks overfitting to rare cases.</p>
                      </div>
                    </div>

                    {/* Batch size */}
                    <div className="space-y-3">
                      <div className="flex justify-between items-baseline">
                        <label className="font-headline font-bold text-primary text-sm flex items-center gap-2">
                          Sample Volume
                          <span className="font-label text-[10px] text-on-surface-variant bg-surface-container px-2 py-0.5 rounded tracking-tighter">(BATCH SIZE)</span>
                        </label>
                        <span className="font-label font-bold text-primary">{trainConfig.batch_size}</span>
                      </div>
                      <input type="range" className="w-full h-1"
                        min="16" max="512" step="16"
                        value={trainConfig.batch_size}
                        onChange={e => setTrainConfig(c => ({ ...c, batch_size: parseInt(e.target.value) }))}
                      />
                      <div className="bg-surface p-4 rounded-lg flex items-start gap-3 border-l-4 border-primary">
                        <span className="material-symbols-outlined text-primary text-sm mt-0.5">info</span>
                        <p className="text-xs text-on-surface-variant leading-relaxed">Number of records analyzed simultaneously for each step. Balances computational load with training smoothness.</p>
                      </div>
                    </div>
                  </div>

                  {/* Model + target selects */}
                  <div className="bg-surface-container-low p-8 rounded-xl">
                    <div className="flex items-center gap-3 border-b border-outline-variant/20 pb-4 mb-6">
                      <span className="material-symbols-outlined text-primary">settings</span>
                      <h3 className="font-headline text-lg font-bold text-primary">Model Parameters</h3>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                      <ConfigField label="Target Column" value={trainConfig.target_column}
                        onChange={v => setTrainConfig(c => ({ ...c, target_column: v }))}
                        options={scanResult?.summary?.target_candidates}
                        placeholder="auto-detect" />
                      <ConfigField label="Model Type" value={trainConfig.model_type}
                        onChange={v => setTrainConfig(c => ({ ...c, model_type: v }))}
                        options={['auto', 'mlp', 'cnn', 'resnet', 'logistic']} />
                    </div>
                  </div>

                  {/* Feature Selection */}
                  {dataset?.type === 'tabular' && dataset?.columns?.length > 0 && (
                    <div className="bg-surface-container-low p-8 rounded-xl">
                      <div className="flex items-center justify-between border-b border-outline-variant/20 pb-4 mb-6">
                        <div className="flex items-center gap-3">
                          <span className="material-symbols-outlined text-secondary">account_tree</span>
                          <h3 className="font-headline text-lg font-bold text-primary">
                            Explainable Selection
                            {selectedFeatures.length > 0 && (
                              <span className="ml-2 text-sm text-secondary font-normal">{selectedFeatures.length} selected</span>
                            )}
                          </h3>
                        </div>
                        <div className="flex gap-1 bg-surface-container rounded-lg p-1">
                          {['manual', 'auto'].map(m => (
                            <button key={m} onClick={() => setFeatureMode(m)}
                              className={`px-3 py-1.5 rounded text-xs font-bold transition-all ${
                                featureMode === m
                                  ? 'bg-surface-container-lowest text-on-surface shadow-sm'
                                  : 'text-on-surface-variant'
                              }`}>
                              {m === 'auto' ? 'Auto (AI)' : 'Manual'}
                            </button>
                          ))}
                        </div>
                      </div>

                      {featureMode === 'auto' ? (
                        <div className="space-y-4">
                          <p className="text-xs text-on-surface-variant">
                            Runs PyImpetus (PIMP) statistical importance testing, falling back to Random Forest if not installed.
                          </p>
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
                            className={`px-5 py-2 font-bold text-sm rounded-md flex items-center gap-2 transition-all ${
                              autoRunning
                                ? 'bg-surface-container text-on-surface-variant cursor-wait'
                                : 'bg-secondary text-on-secondary hover:opacity-90'
                            }`}>
                            <span className="material-symbols-outlined text-base">
                              {autoRunning ? 'hourglass_empty' : 'play_arrow'}
                            </span>
                            {autoRunning ? 'Running...' : 'Run Feature Selection'}
                          </button>

                          {autoFeatures && (
                            <div>
                              <p className="text-xs text-on-surface-variant mb-3">
                                Method: <strong>{autoFeatures.method}</strong> · {autoFeatures.selected.length}/{autoFeatures.all_features.length} features selected
                              </p>
                              <div className="space-y-2 max-h-64 overflow-y-auto">
                                {autoFeatures.all_features.map(feat => {
                                  const imp = autoFeatures.importances[feat] || 0
                                  const maxImp = Math.max(...Object.values(autoFeatures.importances))
                                  const isSelected = selectedFeatures.includes(feat)
                                  return (
                                    <div key={feat} onClick={() => setSelectedFeatures(prev =>
                                      prev.includes(feat) ? prev.filter(f => f !== feat) : [...prev, feat]
                                    )} className={`flex items-center gap-3 p-3 rounded-xl cursor-pointer transition-all border ${
                                      isSelected
                                        ? 'border-on-tertiary-fixed-variant bg-tertiary-fixed/10'
                                        : 'border-outline-variant/30 bg-surface-container-lowest'
                                    }`}>
                                      <input type="checkbox" readOnly checked={isSelected} className="rounded" />
                                      <span className="text-xs font-mono flex-1 text-on-surface">{feat}</span>
                                      <div className="w-20 h-1.5 bg-surface-container rounded-full overflow-hidden">
                                        <div className="h-full bg-on-tertiary-fixed-variant rounded-full"
                                          style={{ width: `${(imp / maxImp) * 100}%` }} />
                                      </div>
                                      <span className="text-[10px] text-on-surface-variant font-mono w-10 text-right">{imp.toFixed(3)}</span>
                                    </div>
                                  )
                                })}
                              </div>
                            </div>
                          )}
                        </div>
                      ) : (
                        <div className="space-y-4">
                          <div className="flex items-center justify-between">
                            <p className="text-xs text-on-surface-variant">Select features to include in training.</p>
                            <div className="flex gap-2">
                              <button onClick={() => setSelectedFeatures(
                                dataset.columns.filter(c => c.name !== trainConfig.target_column).map(c => c.name)
                              )} className="text-xs px-3 py-1 rounded border border-outline-variant/40 bg-surface-container-lowest text-on-surface-variant hover:bg-surface-container transition-colors">
                                All
                              </button>
                              <button onClick={() => setSelectedFeatures([])}
                                className="text-xs px-3 py-1 rounded border border-outline-variant/40 bg-surface-container-lowest text-on-surface-variant hover:bg-surface-container transition-colors">
                                None
                              </button>
                            </div>
                          </div>
                          <div className="space-y-1.5 max-h-64 overflow-y-auto">
                            {dataset.columns
                              .filter(c => c.name !== trainConfig.target_column)
                              .map(col => {
                                const allCols = dataset.columns.filter(c => c.name !== trainConfig.target_column).map(c => c.name)
                                const isSelected = selectedFeatures.length === 0 || selectedFeatures.includes(col.name)
                                const isNumeric = col.dtype?.includes('float') || col.dtype?.includes('int')
                                return (
                                  <div key={col.name} onClick={() => setSelectedFeatures(prev => {
                                    const cur = prev.length === 0 ? allCols : prev
                                    return cur.includes(col.name) ? cur.filter(f => f !== col.name) : [...cur, col.name]
                                  })} className={`flex items-center gap-3 p-3 rounded-xl cursor-pointer transition-all border ${
                                    isSelected
                                      ? 'border-primary/30 bg-primary-fixed/20'
                                      : 'border-outline-variant/30 bg-surface-container-lowest'
                                  }`}>
                                    <input type="checkbox" readOnly checked={isSelected} className="rounded" />
                                    <span className="text-xs font-mono flex-1 text-on-surface">{col.name}</span>
                                    <span className={`text-[10px] px-2 py-0.5 rounded font-label ${
                                      isNumeric ? 'bg-tertiary-fixed/30 text-on-tertiary-fixed-variant' : 'bg-secondary-fixed text-on-secondary-fixed-variant'
                                    }`}>{col.dtype}</span>
                                    {col.missing_pct > 0 && (
                                      <span className="text-[10px] text-error">{col.missing_pct}% missing</span>
                                    )}
                                  </div>
                                )
                              })}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  <div className="flex gap-3">
                    <button onClick={() => askAI('What model architecture and hyperparameters do you recommend for this dataset?')}
                      className="px-5 py-2 bg-surface-container-high text-primary font-bold text-sm rounded-md hover:bg-surface-container-highest transition-colors flex items-center gap-2">
                      <span className="material-symbols-outlined text-base">psychology</span>
                      ASK AI
                    </button>
                  </div>
                </div>
              )}

              {/* STAGE 4: TRAINING */}
              {stage === 4 && (
                <div className="fade-up space-y-6">
                  <div className="flex items-end justify-between">
                    <div>
                      <p className="font-label text-xs uppercase tracking-widest text-on-surface-variant mb-1">On-Device Computation</p>
                      <h2 className="font-headline text-3xl font-extrabold text-primary tracking-tighter">Training</h2>
                      <p className="text-sm text-on-surface-variant mt-1">
                        {training?.active
                          ? `Epoch ${training.epoch}/${training.total_epochs}`
                          : training?.status === 'complete' ? 'Complete' : 'Starting...'}
                      </p>
                    </div>
                    {training?.active && (
                      <button onClick={stopTraining}
                        className="px-5 py-2 border-2 border-error text-error font-bold text-sm rounded-md hover:bg-error-container transition-colors flex items-center gap-2">
                        <span className="material-symbols-outlined text-base">stop</span>
                        Stop
                      </button>
                    )}
                  </div>

                  {training && (
                    <div className="bg-surface-container-low p-6 rounded-xl">
                      <div className="flex justify-between items-center mb-3">
                        <span className="font-label text-xs uppercase tracking-widest text-on-surface-variant">Epoch Progress</span>
                        <span className="font-label text-xs font-bold text-primary">
                          {training.total_epochs > 0 ? `${Math.round((training.epoch / training.total_epochs) * 100)}%` : '0%'}
                        </span>
                      </div>
                      <div className="h-1.5 bg-surface-container rounded-full overflow-hidden">
                        <div className="h-full rounded-full transition-all duration-300"
                          style={{
                            width: `${training.total_epochs > 0 ? (training.epoch / training.total_epochs) * 100 : 0}%`,
                            background: training.active ? '#00193c' : training.status === 'complete' ? '#88d982' : '#ba1a1a',
                          }} />
                      </div>
                    </div>
                  )}

                  {training?.metrics && (
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <MetricBox label="Train Loss" value={training.metrics.train_loss?.toFixed(4)} mono />
                      <MetricBox label="Val Loss" value={training.metrics.val_loss?.toFixed(4)} mono />
                      <MetricBox label="Accuracy" value={training.metrics.accuracy ? `${(training.metrics.accuracy * 100).toFixed(1)}%` : '—'} mono />
                      <MetricBox label="F1 Score" value={training.metrics.f1?.toFixed(3)} mono />
                    </div>
                  )}

                  {trainHistory.length > 2 && (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                      <div className="bg-surface-container-low p-6 rounded-xl">
                        <MiniChart data={trainHistory.map(h => h.val_loss)} color="#00193c" height={90} label="Validation Loss" />
                      </div>
                      <div className="bg-surface-container-low p-6 rounded-xl">
                        <MiniChart data={trainHistory.map(h => h.accuracy)} color="#7b41b3" height={90} label="Accuracy" />
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* STAGE 5: EVALUATE */}
              {stage === 5 && training?.metrics && (
                <div className="fade-up space-y-6">
                  <div>
                    <p className="font-label text-xs uppercase tracking-widest text-on-surface-variant mb-1">Performance Assessment</p>
                    <h2 className="font-headline text-3xl font-extrabold text-primary tracking-tighter">Evaluation Results</h2>
                    <p className="text-sm text-on-surface-variant mt-1">Training complete. Review performance before deciding next steps.</p>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <MetricBox label="Final Accuracy" value={`${(training.metrics.accuracy * 100).toFixed(1)}%`} mono large />
                    <MetricBox label="F1 Score" value={training.metrics.f1?.toFixed(3)} mono large />
                    <MetricBox label="Train Loss" value={training.metrics.train_loss?.toFixed(4)} mono large />
                    <MetricBox label="Val Loss" value={training.metrics.val_loss?.toFixed(4)} mono large />
                  </div>

                  {trainHistory.length > 2 && (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                      <div className="bg-surface-container-low p-6 rounded-xl">
                        <MiniChart data={trainHistory.map(h => h.train_loss)} color="#00193c" height={100} label="Train Loss (full history)" />
                      </div>
                      <div className="bg-surface-container-low p-6 rounded-xl">
                        <MiniChart data={trainHistory.map(h => h.accuracy)} color="#88d982" height={100} label="Accuracy (full history)" />
                      </div>
                    </div>
                  )}
                </div>
              )}

            </div>
          </div>

          {/* ═══ RIGHT PANEL — AI CO-PILOT ═══ */}
          <aside className="w-80 flex-shrink-0 border-l border-slate-200/50 bg-primary text-on-primary flex flex-col overflow-hidden relative">
            {/* Decorative blur */}
            <div className="absolute -right-16 -top-16 w-48 h-48 bg-secondary rounded-full blur-[60px] opacity-20 pointer-events-none" />
            <div className="absolute -left-8 bottom-0 w-24 h-24 bg-secondary-fixed-dim rounded-full blur-[50px] opacity-10 pointer-events-none" />

            <div className="relative z-10 flex items-center gap-3 px-6 py-4 border-b border-on-primary-fixed-variant/30">
              <span className="material-symbols-outlined text-secondary-fixed-dim">psychology</span>
              <h3 className="font-headline font-bold text-base">AI Co-Pilot</h3>
              {aiLoading && (
                <span className="material-symbols-outlined text-secondary-fixed-dim text-sm ml-auto animate-spin">progress_activity</span>
              )}
              {!aiLoading && (
                <span className="font-label text-[10px] font-bold tracking-wider text-on-primary-container ml-auto">Qwen 2.5</span>
              )}
            </div>

            <div ref={aiChatRef} className="relative z-10 flex-1 overflow-y-auto p-4 flex flex-col gap-3">
              {aiMessages.length === 0 && (
                <div className="text-xs text-on-primary-container text-center px-4 py-10 leading-relaxed">
                  The AI co-pilot will analyze your data and provide recommendations as you progress through the pipeline.
                </div>
              )}
              {aiMessages.map((m, i) => (
                <div key={i} className={`px-3 py-2.5 text-xs leading-relaxed whitespace-pre-wrap break-words ${
                  m.role === 'user'
                    ? 'bg-white/10 rounded-xl rounded-br-sm self-end max-w-[88%] text-on-primary'
                    : m.role === 'system'
                    ? 'bg-white/5 rounded-lg self-center text-on-primary-container border border-white/10 max-w-full text-center'
                    : 'bg-white/5 rounded-xl rounded-tl-sm self-start max-w-[88%] text-on-primary-container border border-white/10'
                }`}>
                  {m.text}
                </div>
              ))}
            </div>

            <div className="relative z-10 p-4 border-t border-on-primary-fixed-variant/30">
              <div className="flex gap-2">
                <input
                  value={aiInput} onChange={e => setAiInput(e.target.value)}
                  onKeyDown={e => {
                    if (e.key === 'Enter' && aiInput.trim()) {
                      askAI(aiInput)
                      setAiInput('')
                    }
                  }}
                  placeholder="Ask the AI co-pilot..."
                  className="flex-1 px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-on-primary text-xs placeholder:text-on-primary-container focus:outline-none focus:border-secondary-fixed-dim"
                />
                <button
                  onClick={() => { if (aiInput.trim()) { askAI(aiInput); setAiInput('') } }}
                  disabled={aiLoading}
                  className="px-3 py-2 bg-secondary text-on-secondary rounded-lg font-bold text-xs hover:opacity-90 transition-opacity disabled:opacity-50"
                >
                  <span className="material-symbols-outlined text-base">send</span>
                </button>
              </div>
            </div>
          </aside>
        </div>
      </div>

      {/* FAB — Save */}
      <div className="fixed bottom-8 right-8 z-50">
        <button className="w-14 h-14 bg-secondary text-on-secondary rounded-full shadow-2xl flex items-center justify-center hover:scale-105 transition-transform active:opacity-80">
          <span className="material-symbols-outlined">save</span>
        </button>
      </div>
    </div>
  )
}


// ── Small reusable pieces ───────────────────────────────────

function MetricBox({ label, value, mono, large, warn }) {
  return (
    <div className="bg-surface-container-lowest p-5 rounded-xl border border-outline-variant/20">
      <p className="font-label text-[10px] text-on-surface-variant uppercase tracking-widest mb-2">{label}</p>
      <p className={`font-headline font-bold ${large ? 'text-3xl' : 'text-xl'} ${warn ? 'text-error' : 'text-primary'} ${mono ? 'font-mono' : ''}`}>
        {value ?? '—'}
      </p>
    </div>
  )
}

function ConfigField({ label, value, onChange, type = 'text', options, placeholder, step }) {
  return (
    <div>
      <p className="font-label text-[10px] text-on-surface-variant uppercase tracking-widest mb-2">{label}</p>
      {options ? (
        <select
          value={value}
          onChange={e => onChange(e.target.value)}
          className="w-full px-3 py-2.5 bg-surface-container-lowest border border-outline-variant/40 rounded-lg text-on-surface text-sm focus:outline-none focus:border-primary"
        >
          {options.map(o => <option key={o} value={o}>{o}</option>)}
        </select>
      ) : (
        <input
          type={type} value={value} step={step}
          onChange={e => onChange(e.target.value)}
          placeholder={placeholder}
          className={`w-full px-3 py-2.5 bg-surface-container-lowest border border-outline-variant/40 rounded-lg text-on-surface text-sm focus:outline-none focus:border-primary ${type === 'number' ? 'font-mono' : ''}`}
        />
      )}
    </div>
  )
}
