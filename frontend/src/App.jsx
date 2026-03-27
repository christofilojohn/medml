import { useState, useEffect, useRef, useCallback } from 'react'

// ═══════════════════════════════════════════════════════════════
// MedML Forge — On-Device Clinical AI Pipeline Dashboard
// ═══════════════════════════════════════════════════════════════
// Architecture:
//   React UI → Python ML Worker (scan/train/cleanup)
//            → Qwen LLM via llama.cpp (reasoning)
//   All local. Zero patient data leaves the machine.
// ═══════════════════════════════════════════════════════════════


// ── Suggested questions per stage (non-technical audience) ───
const STAGE_QUESTIONS = {
  scan:    [
    'Is my dataset large enough to train a model?',
    'Are there any red flags I should know about?',
    'What format does my data need to be in?',
  ],
  preview: [
    'Which column should be my prediction target?',
    'Is the data balanced enough to train on?',
    'Should I be worried about the missing values?',
  ],
  clean:   [
    'Should I run the cleanup before training?',
    'What happens if I skip the cleanup step?',
    'Will cleanup remove too many of my records?',
  ],
  config:  [
    'Which model type should I pick for my data?',
    'What do these settings actually do?',
    'What are the safest settings to start with?',
  ],
  train:   [
    'Is my training going well so far?',
    'What does the loss curve mean?',
    'Should I stop early or let it finish?',
  ],
  eval:    [
    'Are these results good enough to use clinically?',
    'What does the accuracy score actually mean?',
    'What should I do next with this model?',
  ],
}

const STAGES = [
  { id: 'scan',    icon: 'biotech',                  label: 'Scan',      desc: 'Point to your local data directory' },
  { id: 'preview', icon: 'visibility',               label: 'Preview',   desc: 'Inspect and understand your dataset' },
  { id: 'clean',   icon: 'cleaning_services',        label: 'Cleanup',   desc: 'Fix quality issues' },
  { id: 'config',  icon: 'settings_input_component', label: 'Configure', desc: 'Model & training settings' },
  { id: 'train',   icon: 'model_training',           label: 'Training',  desc: 'On-device model training' },
  { id: 'eval',    icon: 'analytics',                label: 'Evaluate',  desc: 'Performance assessment' },
]

// ── Markdown renderer for AI messages ────────────────────────
function MarkdownMessage({ text }) {
  const renderInline = (str) => {
    const regex = /(\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`)/g
    const parts = []
    let last = 0, m, k = 0
    while ((m = regex.exec(str)) !== null) {
      if (m.index > last) parts.push(str.slice(last, m.index))
      if (m[2]) parts.push(<strong key={k++} className="font-semibold text-on-primary">{m[2]}</strong>)
      else if (m[3]) parts.push(<em key={k++} className="italic">{m[3]}</em>)
      else if (m[4]) parts.push(<code key={k++} className="bg-white/20 px-1 rounded font-mono text-[10px]">{m[4]}</code>)
      last = regex.lastIndex
    }
    if (last < str.length) parts.push(str.slice(last))
    return parts.length === 0 ? str : parts
  }

  const lines = text.split('\n')
  const result = []
  let listItems = null
  let listOrdered = false

  const flushList = () => {
    if (!listItems) return
    const Tag = listOrdered ? 'ol' : 'ul'
    result.push(
      <Tag key={result.length} className={`${listOrdered ? 'list-decimal' : 'list-disc'} ml-4 space-y-0.5 my-1`}>
        {listItems.map((item, i) => <li key={i}>{renderInline(item)}</li>)}
      </Tag>
    )
    listItems = null
  }

  lines.forEach((line, i) => {
    if (/^#{1,3}\s/.test(line)) {
      flushList()
      result.push(<p key={i} className="font-semibold text-secondary-fixed-dim mt-2 mb-0.5">{line.replace(/^#{1,3}\s/, '')}</p>)
    } else if (/^[-*•]\s/.test(line)) {
      if (listOrdered || !listItems) { flushList(); listItems = []; listOrdered = false }
      listItems.push(line.slice(2).trim())
    } else if (/^\d+\.\s/.test(line)) {
      if (!listOrdered || !listItems) { flushList(); listItems = []; listOrdered = true }
      listItems.push(line.replace(/^\d+\.\s/, ''))
    } else if (/^---+$/.test(line.trim())) {
      flushList()
      result.push(<hr key={i} className="border-white/10 my-1.5" />)
    } else if (line.trim() === '') {
      flushList()
    } else {
      flushList()
      result.push(<p key={i} className="leading-relaxed">{renderInline(line)}</p>)
    }
  })
  flushList()
  return <div className="space-y-0.5 text-xs text-on-primary-container">{result}</div>
}

// ── Training history chart ───────────────────────────────────
// series = [{ label, color, data: number[] }]
function TrainingChart({ series, title }) {
  const allVals = series.flatMap(s => s.data).filter(v => v != null && !isNaN(v))
  if (allVals.length < 2) return null

  const maxLen = Math.max(...series.map(s => s.data.length))
  const rawMin = Math.min(...allVals)
  const rawMax = Math.max(...allVals)
  const pad = (rawMax - rawMin) * 0.08 || 0.05
  const yMin = Math.max(0, rawMin - pad)
  const yMax = rawMax + pad
  const yRange = yMax - yMin || 1

  const W = 560, H = 200
  const PL = 46, PR = 16, PT = 14, PB = 30
  const cw = W - PL - PR
  const ch = H - PT - PB

  const tx = i => PL + (i / Math.max(maxLen - 1, 1)) * cw
  const ty = v => PT + (1 - (v - yMin) / yRange) * ch

  const yTicks = 4
  const xTicks = Math.min(6, maxLen)

  return (
    <div className="bg-surface-container-low p-5 rounded-xl">
      <div className="flex items-center justify-between mb-3">
        <p className="font-headline font-bold text-primary text-sm">{title}</p>
        <div className="flex items-center gap-5">
          {series.map(s => (
            <div key={s.label} className="flex items-center gap-1.5">
              <div className="w-5 h-0.5 rounded-full" style={{ background: s.color }} />
              <span className="font-label text-[10px] uppercase tracking-wider text-on-surface-variant">{s.label}</span>
            </div>
          ))}
        </div>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', display: 'block' }}>
        <defs>
          {series.map(s => (
            <linearGradient key={s.label} id={`tg-${s.label.replace(/\W/g, '')}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={s.color} stopOpacity="0.2" />
              <stop offset="100%" stopColor={s.color} stopOpacity="0" />
            </linearGradient>
          ))}
        </defs>

        {/* Horizontal grid + Y labels */}
        {Array.from({ length: yTicks + 1 }, (_, i) => {
          const v = yMin + (i / yTicks) * yRange
          const y = ty(v)
          return (
            <g key={i}>
              <line x1={PL} y1={y} x2={W - PR} y2={y} stroke="#e1e3e4" strokeWidth="1" />
              <text x={PL - 5} y={y + 3.5} textAnchor="end" fontSize="9" fill="#747781">
                {yRange < 0.1 ? v.toFixed(4) : v.toFixed(3)}
              </text>
            </g>
          )
        })}

        {/* X axis labels */}
        {Array.from({ length: xTicks }, (_, i) => {
          const idx = Math.round((i / (xTicks - 1)) * (maxLen - 1))
          const x = tx(idx)
          return (
            <g key={i}>
              <line x1={x} y1={H - PB} x2={x} y2={H - PB + 4} stroke="#c4c6d1" strokeWidth="1" />
              <text x={x} y={H - PB + 13} textAnchor="middle" fontSize="9" fill="#747781">{idx + 1}</text>
            </g>
          )
        })}
        <text x={W / 2} y={H - 2} textAnchor="middle" fontSize="9" fill="#c4c6d1">epoch</text>

        {/* Axes */}
        <line x1={PL} y1={PT} x2={PL} y2={H - PB} stroke="#c4c6d1" strokeWidth="1" />
        <line x1={PL} y1={H - PB} x2={W - PR} y2={H - PB} stroke="#c4c6d1" strokeWidth="1" />

        {/* Series */}
        {series.map(s => {
          const pts = s.data
            .map((v, i) => (v != null && !isNaN(v) ? `${tx(i).toFixed(2)},${ty(v).toFixed(2)}` : null))
            .filter(Boolean)
          if (pts.length < 2) return null
          const last = pts[pts.length - 1].split(',')
          const area = `${PL},${H - PB} ${pts.join(' ')} ${last[0]},${H - PB}`
          const gradId = `tg-${s.label.replace(/\W/g, '')}`
          const lastV = s.data.filter(v => v != null && !isNaN(v)).at(-1)
          return (
            <g key={s.label}>
              <polygon points={area} fill={`url(#${gradId})`} />
              <polyline points={pts.join(' ')} fill="none" stroke={s.color}
                strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              <circle cx={last[0]} cy={last[1]} r="4" fill={s.color} />
              {/* Latest value label */}
              <text x={parseFloat(last[0]) - 4} y={parseFloat(last[1]) - 7}
                textAnchor="end" fontSize="9" fill={s.color} fontWeight="600">
                {lastV?.toFixed(4)}
              </text>
            </g>
          )
        })}
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

  // Continue training
  const [continueEpochs, setContinueEpochs] = useState(20)
  const [checkpoint, setCheckpoint] = useState(null) // { metrics } snapshot before continuing

  // Model gallery
  const [showGallery, setShowGallery] = useState(false)

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

  // ── Build full app context for AI ────────────────────────
  // Called fresh on every AI request so it always reflects the latest state.
  const buildContext = () => {
    const ctx = {
      current_stage: STAGES[stage]?.label,
      stages_completed: [...completed].map(i => STAGES[i]?.label),
    }

    // Dataset summary
    if (dataset?.summary) {
      ctx.dataset = {
        type: dataset.type,
        ...dataset.summary,
        // strip verbose fields
        columns: undefined,
        preview: undefined,
      }
      if (dataset.columns) {
        ctx.dataset.column_names = dataset.columns.map(c => c.name)
        ctx.dataset.columns_with_missing = dataset.columns
          .filter(c => c.missing_pct > 0)
          .map(c => `${c.name} (${c.missing_pct}% missing)`)
      }
    }

    // Cleanup
    if (cleanupResult) {
      ctx.cleanup = {
        actions_taken: cleanupResult.actions_taken,
        rows_before: cleanupResult.rows_before,
        rows_after: cleanupResult.rows_after,
        rows_removed: (cleanupResult.rows_before || 0) - (cleanupResult.rows_after || 0),
      }
    }

    // Feature selection
    ctx.feature_selection = { mode: featureMode }
    if (selectedFeatures.length > 0) {
      ctx.feature_selection.selected = selectedFeatures
      ctx.feature_selection.selected_count = selectedFeatures.length
    } else {
      ctx.feature_selection.selected = 'all features'
    }
    if (autoFeatures) {
      ctx.feature_selection.method = autoFeatures.method
      ctx.feature_selection.target_column = autoFeatures.target_column
      ctx.feature_selection.ai_recommended = autoFeatures.selected
      ctx.feature_selection.total_available = autoFeatures.all_features?.length
      // Top 10 by importance score
      const imp = autoFeatures.importances || {}
      ctx.feature_selection.top_importances = Object.entries(imp)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 10)
        .map(([feat, score]) => ({ feature: feat, importance: Number(score.toFixed(4)) }))
    }

    // Training configuration
    ctx.training_config = {
      model_type: trainConfig.model_type,
      target_column: trainConfig.target_column,
      epochs: trainConfig.epochs,
      learning_rate: trainConfig.lr,
      batch_size: trainConfig.batch_size,
    }

    // Live training state + results
    if (training) {
      ctx.training = {
        status: training.status,
        epoch: training.epoch,
        total_epochs: training.total_epochs || trainConfig.epochs,
        progress_pct: training.progress,
        latest_metrics: training.metrics
          ? {
              accuracy: training.metrics.accuracy?.toFixed(4),
              f1: training.metrics.f1?.toFixed(4),
              train_loss: training.metrics.train_loss?.toFixed(4),
              val_loss: training.metrics.val_loss?.toFixed(4),
            }
          : null,
      }

      if (trainHistory.length >= 2) {
        // Send last 5 epochs so AI can see the trend
        ctx.training.recent_history = trainHistory.slice(-5).map(h => ({
          train_loss: h.train_loss?.toFixed(4),
          val_loss: h.val_loss?.toFixed(4),
          accuracy: h.accuracy?.toFixed(4),
          f1: h.f1?.toFixed(4),
        }))

        // Derive a simple trend indicator
        const first = trainHistory[0]
        const last = trainHistory[trainHistory.length - 1]
        const valGap = (last.val_loss || 0) - (last.train_loss || 0)
        ctx.training.trend = {
          loss_improved: last.train_loss < first.train_loss,
          accuracy_improved: (last.accuracy || 0) > (first.accuracy || 0),
          possible_overfitting: valGap > 0.15,
          val_train_loss_gap: valGap.toFixed(4),
        }
      }
    }

    return ctx
  }

  // ── Ask AI (Qwen reasoning) ───────────────────────────────
  const askAI = async (prompt, extraContext = null) => {
    setAiMessages(prev => [...prev, { role: 'user', text: prompt }])
    setAiLoading(true)
    try {
      // Always build a fresh full-state context; caller can add extra fields
      const context = { ...buildContext(), ...(extraContext || {}) }
      const r = await fetch('/reason', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, context }),
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

  // Safe fetch helper — always returns parsed JSON or { error } even on HTML responses
  const fetchJSON = async (url, opts = {}) => {
    const r = await fetch(url, opts)
    const text = await r.text()
    try {
      return JSON.parse(text)
    } catch {
      return { error: `Server error ${r.status}: ${text.slice(0, 120)}` }
    }
  }

  const subscribeSSE = (onComplete) => {
    if (sseRef.current) sseRef.current.close()
    // Small delay so the training thread has time to flip active=true
    // before the SSE stream checks the state
    setTimeout(() => {
      const sse = new EventSource('/train/stream')
      sseRef.current = sse
      sse.onmessage = (e) => {
        const d = JSON.parse(e.data)
        setTraining(d)
        if (d.metrics) setTrainHistory(prev => [...prev, d.metrics].slice(-500))
        if (!d.active && d.status === 'complete') {
          sse.close()
          completeStage(4)
          setStage(5)
          onComplete(d)
        }
      }
      sse.onerror = () => sse.close()
    }, 300)
  }

  const continueTraining = async () => {
    setCheckpoint({ metrics: { ...training?.metrics } })
    try {
      const data = await fetchJSON('/train/continue', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ epochs: continueEpochs }),
      })
      if (data.error) {
        setAiMessages(prev => [...prev, { role: 'system', text: `Cannot continue: ${data.error}` }])
        setCheckpoint(null)
        return
      }
      setStage(4)
      subscribeSSE(d => askAI(
        `Continuation training finished. New accuracy=${d.metrics.accuracy?.toFixed(4)}, f1=${d.metrics.f1?.toFixed(4)}, val_loss=${d.metrics.val_loss?.toFixed(4)}. Did scores improve compared to before?`
      ))
    } catch (e) {
      setAiMessages(prev => [...prev, { role: 'system', text: `Continue error: ${e.message}` }])
      setCheckpoint(null)
    }
  }

  const revertTraining = async () => {
    try {
      const r = await fetch('/train/revert', { method: 'POST' })
      const data = await r.json()
      if (data.error) {
        setAiMessages(prev => [...prev, { role: 'system', text: `Revert error: ${data.error}` }])
        return
      }
      if (data.metrics) setTraining(prev => ({ ...prev, metrics: data.metrics, status: 'complete', active: false }))
      setCheckpoint(null)
      askAI('Reverted to previous checkpoint. The extra training did not improve scores enough to keep.')
    } catch (e) {
      setAiMessages(prev => [...prev, { role: 'system', text: `Revert error: ${e.message}` }])
    }
  }

  // ── Render ────────────────────────────────────────────────
  return (
    <div className="font-body text-on-surface min-h-screen bg-surface">

      {/* ═══ SIDEBAR ═══ */}
      <aside className="fixed left-0 top-0 h-screen flex flex-col py-6 bg-slate-50 border-r border-slate-200/50 w-64 z-50">
        <div className="px-6 mb-5">
          <h2 className="font-body text-xs font-black uppercase tracking-widest text-primary">PIPELINE</h2>
        </div>

        <nav className="flex-1 flex flex-col gap-0.5 overflow-y-auto">
          {STAGES.map((s, i) => {
            const active = i === stage
            const done = completed.has(i)
            const locked = i > stage + 1 && !done
            const NEEDS = [null, 'Scan first', 'Preview first', 'Preview first', 'Select target column', 'Run training first']
            return (
              <button key={s.id}
                onClick={() => !locked && setStage(i)}
                className={`flex items-start px-4 py-2.5 mx-2 rounded-md text-left transition-all duration-200 group ${
                  active
                    ? 'bg-primary text-white shadow-sm'
                    : locked
                    ? 'text-slate-300 cursor-default'
                    : done
                    ? 'text-slate-500 hover:translate-x-0.5 cursor-pointer'
                    : 'text-slate-600 hover:translate-x-0.5 cursor-pointer hover:bg-slate-100'
                }`}
              >
                <span className="material-symbols-outlined mr-3 text-xl flex-shrink-0 mt-0.5">
                  {done && !active ? 'check_circle' : locked ? 'lock' : s.icon}
                </span>
                <div className="min-w-0">
                  <span className="font-body text-sm font-medium uppercase tracking-widest block">{s.label}</span>
                  {locked && NEEDS[i] && (
                    <span className="text-[10px] text-slate-300 font-label normal-case tracking-normal leading-tight block mt-0.5">
                      Needs: {NEEDS[i]}
                    </span>
                  )}
                  {done && !active && (
                    <span className="text-[10px] text-tertiary-fixed-dim font-label normal-case tracking-normal leading-tight block mt-0.5">
                      Complete
                    </span>
                  )}
                </div>
              </button>
            )
          })}
        </nav>

        {/* ─── Gallery button ─── */}
        <div className="px-4 pb-2">
          <button onClick={() => setShowGallery(true)}
            className="w-full py-2 rounded-xl border border-slate-200 text-slate-500 text-xs font-label font-semibold flex items-center justify-center gap-2 hover:bg-slate-100 hover:text-primary transition-colors">
            <span className="material-symbols-outlined text-base">photo_library</span>
            Model Gallery
          </button>
        </div>

        {/* ─── Initialize Engine CTA ─── */}
        <div className="px-4 pt-4 border-t border-slate-200/50 space-y-2">
          {(() => {
            const isTabular = dataset?.type === 'tabular'
            const hasTarget = !isTabular || !!trainConfig.target_column
            const hasDataset = completed.has(0)
            const canTrain = hasDataset && hasTarget
            const hint = !hasDataset
              ? 'Scan your dataset first'
              : !hasTarget
              ? 'Set a target column in Configure'
              : null
            return (
              <>
                {hint && (
                  <p className="text-[10px] text-center font-label leading-tight px-2" style={{ color: '#b45309' }}>
                    <span className="material-symbols-outlined text-[12px] align-middle mr-0.5">info</span>
                    {hint}
                  </p>
                )}
                {!hint && (
                  <p className="text-[10px] text-center font-label leading-tight" style={{ color: '#005312' }}>
                    <span className="material-symbols-outlined text-[12px] align-middle mr-0.5">check_circle</span>
                    Ready to train
                  </p>
                )}
                <button
                  onClick={() => { completeStage(3); startTraining() }}
                  disabled={!canTrain}
                  className={`w-full py-3 rounded-xl font-headline font-bold text-sm flex items-center justify-center gap-2 transition-all ${
                    canTrain
                      ? 'bg-primary text-white shadow-lg hover:opacity-90 active:scale-95'
                      : 'bg-slate-200 text-slate-400 cursor-not-allowed'
                  }`}
                >
                  <span className="material-symbols-outlined text-base">play_arrow</span>
                  START TRAINING
                </button>
              </>
            )
          })()}
          <div className="pb-1">
            <div className="flex justify-between text-[10px] text-on-surface-variant mb-1 font-label">
              <span>Progress</span>
              <span>{completed.size}/{STAGES.length}</span>
            </div>
            <div className="h-1 bg-surface-container rounded-full overflow-hidden">
              <div className="h-full bg-primary rounded-full transition-all duration-500"
                style={{ width: `${(completed.size / STAGES.length) * 100}%` }} />
            </div>
          </div>
        </div>
      </aside>

      <div className="ml-64 h-screen flex flex-col overflow-hidden">

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
        <div className="flex flex-1 min-h-0">

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

                    {/* Guided next step */}
                    <div className="border-t border-outline-variant/15 pt-4 flex items-center justify-between">
                      <p className="text-xs text-on-surface-variant">
                        {scanResult ? '✓ Dataset scanned successfully' : 'Enter your dataset path above'}
                      </p>
                      {scanResult && (
                        <button onClick={goNext}
                          className="px-4 py-2 bg-primary text-on-primary rounded-md text-xs font-bold flex items-center gap-1.5 hover:opacity-90 transition-all">
                          Preview data
                          <span className="material-symbols-outlined text-sm">arrow_forward</span>
                        </button>
                      )}
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

                  {/* Inline AI ask chips */}
                  <AskChips questions={[
                    'Is my dataset large enough to get useful results?',
                    'Which column should be my prediction target?',
                    'Are there any red flags I should address?',
                  ]} onAsk={askAI} />

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

                  {/* Guided next step */}
                  <div className="border-t border-outline-variant/15 pt-4 flex items-center justify-between">
                    <p className="text-xs text-on-surface-variant">
                      Inspect your data, then continue to fix any quality issues.
                    </p>
                    <button onClick={goNext}
                      className="px-4 py-2 bg-primary text-on-primary rounded-md text-xs font-bold flex items-center gap-1.5 hover:opacity-90 transition-all">
                      Go to Cleanup
                      <span className="material-symbols-outlined text-sm">arrow_forward</span>
                    </button>
                  </div>
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

                    <AskChips questions={[
                      'Should I run the cleanup or will it remove too much data?',
                      'What does imputing missing values mean?',
                    ]} onAsk={askAI} />

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
                <div className="fade-up space-y-8">
                  <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
                    <div>
                      <p className="font-label text-xs uppercase tracking-widest text-on-surface-variant mb-1">Configuration Engine</p>
                      <h2 className="font-headline text-4xl font-extrabold text-primary tracking-tighter">Calibrate Parameters</h2>
                    </div>
                    <button onClick={() => setTrainConfig({ epochs: 50, lr: 0.001, batch_size: 32, model_type: 'auto', target_column: '' })}
                      className="px-5 py-2 bg-surface-container-high text-primary font-bold text-sm rounded-md hover:bg-surface-container-highest transition-colors">
                      RESET DEFAULTS
                    </button>
                  </div>

                  {/* 2×2 illustrated parameter cards */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <LRCard value={trainConfig.lr} onChange={v => setTrainConfig(c => ({ ...c, lr: v }))} />
                    <EpochsCard value={trainConfig.epochs} onChange={v => setTrainConfig(c => ({ ...c, epochs: v }))} />
                    <BatchCard value={trainConfig.batch_size} onChange={v => setTrainConfig(c => ({ ...c, batch_size: v }))} />
                    <ModelTypeCard
                      modelType={trainConfig.model_type}
                      targetColumn={trainConfig.target_column}
                      targetOptions={scanResult?.summary?.target_candidates}
                      onModelChange={v => setTrainConfig(c => ({ ...c, model_type: v }))}
                      onTargetChange={v => setTrainConfig(c => ({ ...c, target_column: v }))}
                    />
                  </div>

                  {/* Feature Selection */}
                  {dataset?.type === 'tabular' && dataset?.columns?.length > 0 && (
                    <div className="bg-surface-container-low p-8 rounded-xl">
                      <div className="flex items-center justify-between border-b border-outline-variant/20 pb-4 mb-6">
                        <div className="flex items-center gap-3">
                          <span className="material-symbols-outlined text-secondary">account_tree</span>
                          <h3 className="font-headline text-lg font-bold text-primary">
                            Feature Selection
                            {selectedFeatures.length > 0 && (
                              <span className="ml-2 text-sm text-secondary font-normal">{selectedFeatures.length} selected</span>
                            )}
                            {selectedFeatures.length === 0 && (
                              <span className="ml-2 text-sm text-on-surface-variant font-normal">all columns</span>
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
                              {m === 'auto' ? '⚡ Auto (AI)' : '✋ Manual'}
                            </button>
                          ))}
                        </div>
                      </div>

                      {featureMode === 'auto' ? (
                        <div className="space-y-5">
                          {/* Info banner */}
                          <div className="flex items-start gap-3 p-4 bg-secondary-fixed/30 rounded-xl border border-secondary/20">
                            <span className="material-symbols-outlined text-secondary text-base mt-0.5 shrink-0">psychology</span>
                            <div className="space-y-1">
                              <p className="text-xs font-bold text-primary">Statistical importance testing</p>
                              <p className="text-xs text-on-surface-variant leading-relaxed">
                                Uses <strong>PyImpetus (PIMP)</strong> — a permutation-based method — to identify features that genuinely improve predictions.
                                Falls back to <strong>Random Forest importance</strong> if PyImpetus isn't installed.
                                Only numeric columns are evaluated; the target column is automatically excluded.
                              </p>
                            </div>
                          </div>

                          {/* Requirements check */}
                          {!trainConfig.target_column && !scanResult?.summary?.target_candidates?.length && (
                            <div className="flex items-start gap-3 p-3 bg-error-container/30 rounded-xl border border-error/20">
                              <span className="material-symbols-outlined text-error text-sm mt-0.5 shrink-0">warning</span>
                              <p className="text-xs text-error leading-relaxed">
                                No target column set. Set one in the Architecture card above or the backend will auto-detect it during selection.
                              </p>
                            </div>
                          )}

                          <button
                            disabled={autoRunning}
                            onClick={async () => {
                              setAutoRunning(true)
                              setAutoFeatures(null)
                              try {
                                const res = await fetch('/feature-select', {
                                  method: 'POST',
                                  headers: { 'Content-Type': 'application/json' },
                                  body: JSON.stringify({ target_column: trainConfig.target_column || '' }),
                                })
                                if (!res.ok) {
                                  const err = await res.json().catch(() => ({}))
                                  setAiMessages(prev => [...prev, { role: 'system', text: `Feature selection error: ${err.error || res.statusText}` }])
                                  return
                                }
                                const data = await res.json()
                                if (data.error) {
                                  setAiMessages(prev => [...prev, { role: 'system', text: `Feature selection error: ${data.error}` }])
                                  return
                                }
                                setAutoFeatures(data)
                                // If nothing selected (e.g. all pass threshold), keep all
                                const sel = data.selected?.length > 0 ? data.selected : data.all_features
                                setSelectedFeatures(sel)
                                askAI(`Feature selection complete using ${data.method}. Target: "${data.target_column}". Selected ${sel.length} of ${data.all_features.length} features. Top features: ${sel.slice(0, 5).join(', ')}. What do you recommend next?`)
                              } catch(e) {
                                setAiMessages(prev => [...prev, { role: 'system', text: `Feature selection connection error: ${e.message}` }])
                              } finally {
                                setAutoRunning(false)
                              }
                            }}
                            className={`px-5 py-2.5 font-bold text-sm rounded-md flex items-center gap-2 transition-all ${
                              autoRunning
                                ? 'bg-surface-container text-on-surface-variant cursor-wait'
                                : 'bg-secondary text-on-secondary hover:opacity-90 shadow-sm'
                            }`}>
                            <span className="material-symbols-outlined text-base">
                              {autoRunning ? 'hourglass_empty' : 'play_arrow'}
                            </span>
                            {autoRunning ? 'Running analysis…' : 'Run Feature Selection'}
                          </button>

                          {autoRunning && (
                            <div className="flex items-center gap-3 p-4 bg-surface-container rounded-xl">
                              <span className="material-symbols-outlined text-secondary animate-spin text-base">progress_activity</span>
                              <div>
                                <p className="text-xs font-bold text-primary">Analysing feature importance…</p>
                                <p className="text-xs text-on-surface-variant">This may take 30–120 seconds for large datasets.</p>
                              </div>
                            </div>
                          )}

                          {autoFeatures && !autoRunning && (
                            <div className="space-y-3">
                              {/* Result header */}
                              <div className="flex flex-wrap items-center gap-3">
                                <div className="flex items-center gap-2 px-3 py-1.5 bg-tertiary-fixed/30 rounded-full">
                                  <span className="material-symbols-outlined text-on-tertiary-fixed-variant text-sm">check_circle</span>
                                  <span className="font-label text-xs font-bold text-on-tertiary-fixed-variant">
                                    {selectedFeatures.length}/{autoFeatures.all_features.length} selected
                                  </span>
                                </div>
                                <span className="text-xs text-on-surface-variant">
                                  Method: <strong>{autoFeatures.method}</strong>
                                </span>
                                {autoFeatures.target_column && (
                                  <span className="text-xs text-on-surface-variant">
                                    Target: <strong className="font-mono">{autoFeatures.target_column}</strong>
                                  </span>
                                )}
                                <button onClick={() => {
                                  setSelectedFeatures(autoFeatures.all_features)
                                }} className="ml-auto text-xs text-secondary hover:underline">Select all</button>
                                <button onClick={() => {
                                  setSelectedFeatures(autoFeatures.selected?.length > 0 ? autoFeatures.selected : autoFeatures.all_features)
                                }} className="text-xs text-secondary hover:underline">Reset to AI picks</button>
                              </div>

                              {autoFeatures.selected?.length === 0 && (
                                <div className="flex items-start gap-2 p-3 bg-error-container/30 rounded-xl border border-error/20">
                                  <span className="material-symbols-outlined text-error text-sm shrink-0 mt-0.5">info</span>
                                  <p className="text-xs text-error">
                                    PyImpetus found no statistically significant features at p&lt;0.05. All features have been kept. Consider lowering the significance threshold or using Random Forest mode.
                                  </p>
                                </div>
                              )}

                              {/* Feature list */}
                              <div className="space-y-1.5 max-h-72 overflow-y-auto pr-1">
                                {autoFeatures.all_features.map(feat => {
                                  const imp = autoFeatures.importances?.[feat] || 0
                                  const maxImp = Math.max(...Object.values(autoFeatures.importances || { _: 1 }))
                                  const isSelected = selectedFeatures.includes(feat)
                                  const isAIPick = autoFeatures.selected?.includes(feat)
                                  return (
                                    <div key={feat} onClick={() => setSelectedFeatures(prev =>
                                      prev.includes(feat) ? prev.filter(f => f !== feat) : [...prev, feat]
                                    )} className={`flex items-center gap-3 p-3 rounded-xl cursor-pointer transition-all border ${
                                      isSelected
                                        ? 'border-on-tertiary-fixed-variant/50 bg-tertiary-fixed/15'
                                        : 'border-outline-variant/30 bg-surface-container-lowest hover:bg-surface-container'
                                    }`}>
                                      <input type="checkbox" readOnly checked={isSelected}
                                        className="rounded shrink-0 pointer-events-none" />
                                      <span className="text-xs font-mono flex-1 text-on-surface min-w-0 truncate">{feat}</span>
                                      {isAIPick && (
                                        <span className="font-label text-[9px] bg-secondary/10 text-secondary px-1.5 py-0.5 rounded shrink-0">AI pick</span>
                                      )}
                                      <div className="w-16 h-1.5 bg-surface-container rounded-full overflow-hidden shrink-0">
                                        <div className="h-full bg-on-tertiary-fixed-variant rounded-full transition-all"
                                          style={{ width: `${maxImp > 0 ? (imp / maxImp) * 100 : 0}%` }} />
                                      </div>
                                      <span className="text-[10px] text-on-surface-variant font-mono w-11 text-right shrink-0">{imp.toFixed(3)}</span>
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

                  <AskChips questions={[
                    'Which model type should I pick for my data?',
                    'What are the safest settings to start with?',
                    'How many epochs do I actually need?',
                  ]} onAsk={askAI} />
                </div>
              )}

              {/* STAGE 4: TRAINING */}
              {stage === 4 && (
                <div className="fade-up space-y-6">

                  {/* Header */}
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <p className="font-label text-xs uppercase tracking-widest text-on-surface-variant mb-1">On-Device Computation</p>
                      <h2 className="font-headline text-4xl font-extrabold text-primary tracking-tighter">Training</h2>
                      <div className="flex items-center gap-3 mt-2 flex-wrap">
                        {training?.active && <>
                          <span className="flex items-center gap-1.5">
                            <span className="w-2 h-2 rounded-full bg-tertiary-fixed-dim animate-pulse" style={{ boxShadow: '0 0 6px #88d982' }} />
                            <span className="font-label text-xs font-bold tracking-wider text-on-tertiary-fixed-variant">LIVE</span>
                          </span>
                          <span className="text-sm text-on-surface-variant">
                            Epoch {training.epoch} / {training.total_epochs}
                          </span>
                          {training.metrics?.train_loss != null && (
                            <span className="text-sm text-on-surface-variant font-mono">
                              loss {training.metrics.train_loss.toFixed(4)}
                            </span>
                          )}
                        </>}
                        {!training && <p className="text-sm text-on-surface-variant">Initializing…</p>}
                        {training && !training.active && training.status === 'complete' && (
                          <span className="flex items-center gap-1.5 text-on-tertiary-fixed-variant font-bold text-sm">
                            <span className="material-symbols-outlined text-base">check_circle</span>
                            Complete
                          </span>
                        )}
                      </div>
                    </div>
                    {training?.active && (
                      <button onClick={stopTraining}
                        className="shrink-0 px-5 py-2 border-2 border-error text-error font-bold text-sm rounded-md hover:bg-error-container/30 transition-colors flex items-center gap-2">
                        <span className="material-symbols-outlined text-base">stop</span>
                        Stop
                      </button>
                    )}
                  </div>

                  {/* Progress bar */}
                  {training ? (
                    <div className="bg-surface-container-low p-5 rounded-xl space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="font-label text-[10px] uppercase tracking-widest text-on-surface-variant">Progress</span>
                        <span className="font-mono text-xs font-bold text-primary">
                          {training.epoch ?? 0} / {training.total_epochs ?? '?'} epochs
                          {training.total_epochs > 0 && ` · ${Math.round((training.epoch / training.total_epochs) * 100)}%`}
                        </span>
                      </div>
                      <div className="h-2 bg-surface-container rounded-full overflow-hidden">
                        <div className="h-full rounded-full transition-all duration-500"
                          style={{
                            width: `${training.total_epochs > 0 ? (training.epoch / training.total_epochs) * 100 : 0}%`,
                            background: training.active
                              ? 'linear-gradient(90deg, #00193c, #24467c)'
                              : training.status === 'complete' ? '#88d982' : '#ba1a1a',
                          }} />
                      </div>
                    </div>
                  ) : (
                    <div className="bg-surface-container-low p-8 rounded-xl flex items-center gap-4 text-on-surface-variant">
                      <span className="material-symbols-outlined animate-spin text-2xl text-primary">progress_activity</span>
                      <div>
                        <p className="font-bold text-sm text-primary">Waiting for first epoch…</p>
                        <p className="text-xs mt-0.5">The training loop is starting up. Metrics will appear shortly.</p>
                      </div>
                    </div>
                  )}

                  {/* Live metric cards */}
                  {training?.metrics && (
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <MetricBox label="Train Loss" value={training.metrics.train_loss?.toFixed(4)} mono />
                      <MetricBox label="Val Loss"   value={training.metrics.val_loss?.toFixed(4)}   mono />
                      <MetricBox label="Accuracy"   value={training.metrics.accuracy != null ? `${(training.metrics.accuracy * 100).toFixed(1)}%` : '—'} mono />
                      <MetricBox label="F1 Score"   value={training.metrics.f1?.toFixed(3)} mono />
                    </div>
                  )}

                  {/* Charts */}
                  {trainHistory.length >= 2 ? (
                    <div className="space-y-4">
                      <TrainingChart
                        title="Loss History"
                        series={[
                          { label: 'Train Loss', color: '#00193c', data: trainHistory.map(h => h.train_loss) },
                          { label: 'Val Loss',   color: '#7b41b3', data: trainHistory.map(h => h.val_loss) },
                        ]}
                      />
                      <TrainingChart
                        title="Accuracy & F1"
                        series={[
                          { label: 'Accuracy', color: '#005312', data: trainHistory.map(h => h.accuracy) },
                          { label: 'F1 Score', color: '#24467c', data: trainHistory.map(h => h.f1) },
                        ]}
                      />
                    </div>
                  ) : training?.metrics && (
                    <div className="bg-surface-container-low p-6 rounded-xl flex items-center gap-3 text-on-surface-variant">
                      <span className="material-symbols-outlined text-outline">bar_chart</span>
                      <p className="text-sm">Charts will appear after 2 epochs of data.</p>
                    </div>
                  )}
                </div>
              )}

              {/* STAGE 5: EVALUATE */}
              {stage === 5 && training?.metrics && (
                <div className="fade-up space-y-6">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <p className="font-label text-xs uppercase tracking-widest text-on-surface-variant mb-1">Performance Assessment</p>
                      <h2 className="font-headline text-3xl font-extrabold text-primary tracking-tighter">Evaluation Results</h2>
                      <p className="text-sm text-on-surface-variant mt-1">Training complete. Review performance before deciding next steps.</p>
                    </div>
                    <button onClick={() => setShowGallery(true)}
                      className="shrink-0 flex items-center gap-2 px-4 py-2 rounded-xl border border-outline-variant/40 text-sm font-label font-semibold text-on-surface-variant hover:bg-surface-container hover:text-primary transition-colors">
                      <span className="material-symbols-outlined text-base">photo_library</span>
                      Gallery
                    </button>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <MetricBox label="Final Accuracy" value={`${(training.metrics.accuracy * 100).toFixed(1)}%`} mono large />
                    <MetricBox label="F1 Score" value={training.metrics.f1?.toFixed(3)} mono large />
                    <MetricBox label="Train Loss" value={training.metrics.train_loss?.toFixed(4)} mono large />
                    <MetricBox label="Val Loss" value={training.metrics.val_loss?.toFixed(4)} mono large />
                  </div>

                  {trainHistory.length > 2 && (
                    <div className="space-y-4">
                      <TrainingChart
                        title="Loss History"
                        series={[
                          { label: 'Train Loss', color: '#00193c', data: trainHistory.map(h => h.train_loss) },
                          { label: 'Val Loss', color: '#7b41b3', data: trainHistory.map(h => h.val_loss).filter(v => v != null) },
                        ]}
                      />
                      <TrainingChart
                        title="Accuracy & F1"
                        series={[
                          { label: 'Accuracy', color: '#005312', data: trainHistory.map(h => h.accuracy) },
                          { label: 'F1 Score', color: '#88d982', data: trainHistory.map(h => h.f1).filter(v => v != null) },
                        ]}
                      />
                    </div>
                  )}

                  {/* ── Refine model card ── */}
                  <div className="bg-surface-container-low p-6 rounded-xl space-y-4">
                    <div className="flex items-center gap-2">
                      <span className="material-symbols-outlined text-primary text-xl">model_training</span>
                      <div>
                        <h3 className="font-headline font-bold text-primary text-sm">Refine Your Model</h3>
                        <p className="text-xs text-on-surface-variant">Train for more epochs to try improving scores</p>
                      </div>
                    </div>

                    {/* Score delta — shown after a continuation run */}
                    {checkpoint && (
                      <div className="bg-surface rounded-lg border border-outline-variant/20 p-4 space-y-2">
                        <p className="font-label text-[10px] uppercase tracking-widest text-on-surface-variant mb-1">Score change after continuation</p>
                        {[
                          { label: 'Accuracy', before: checkpoint.metrics.accuracy, after: training.metrics.accuracy, higherBetter: true, fmt: v => `${(v * 100).toFixed(1)}%` },
                          { label: 'F1 Score', before: checkpoint.metrics.f1,       after: training.metrics.f1,       higherBetter: true, fmt: v => v?.toFixed(4) },
                          { label: 'Val Loss', before: checkpoint.metrics.val_loss,  after: training.metrics.val_loss, higherBetter: false, fmt: v => v?.toFixed(4) },
                        ].map(({ label, before, after, higherBetter, fmt }) => {
                          if (before == null || after == null) return null
                          const improved = higherBetter ? after > before : after < before
                          const delta = after - before
                          return (
                            <div key={label} className="flex items-center justify-between text-xs">
                              <span className="text-on-surface-variant">{label}</span>
                              <span className="font-mono">
                                <span className="text-on-surface-variant">{fmt(before)} → </span>
                                <span className={`font-bold ${improved ? 'text-on-tertiary-fixed-variant' : 'text-error'}`}>
                                  {fmt(after)}
                                  <span className="font-normal ml-1 opacity-70">({delta >= 0 ? '+' : ''}{delta.toFixed(4)})</span>
                                </span>
                              </span>
                            </div>
                          )
                        })}
                      </div>
                    )}

                    {/* Epoch picker + action buttons */}
                    <div className="flex items-center gap-3 flex-wrap">
                      <span className="text-xs text-on-surface-variant font-label">Additional epochs:</span>
                      <div className="flex items-center gap-1">
                        <button onClick={() => setContinueEpochs(e => Math.max(5, e - 5))}
                          className="w-7 h-7 rounded-md bg-surface-container text-primary font-bold hover:bg-surface-container-high transition-colors flex items-center justify-center text-base">−</button>
                        <span className="font-mono text-sm font-bold text-primary w-10 text-center">{continueEpochs}</span>
                        <button onClick={() => setContinueEpochs(e => Math.min(500, e + 5))}
                          className="w-7 h-7 rounded-md bg-surface-container text-primary font-bold hover:bg-surface-container-high transition-colors flex items-center justify-center text-base">+</button>
                      </div>
                      <div className="flex gap-2 ml-auto">
                        {checkpoint && (
                          <button onClick={revertTraining}
                            className="px-4 py-2 rounded-md border-2 border-error text-error text-xs font-bold hover:bg-error-container/30 transition-colors flex items-center gap-1.5">
                            <span className="material-symbols-outlined text-sm">undo</span>
                            Revert
                          </button>
                        )}
                        <button onClick={continueTraining}
                          className="px-5 py-2 bg-primary text-on-primary text-xs font-bold rounded-md hover:opacity-90 transition-all flex items-center gap-1.5 shadow-md">
                          <span className="material-symbols-outlined text-sm">play_arrow</span>
                          Train {continueEpochs} more
                        </button>
                      </div>
                    </div>

                    {checkpoint && (
                      <p className="text-[11px] text-on-surface-variant flex items-center gap-1">
                        <span className="material-symbols-outlined text-[13px]">history</span>
                        Checkpoint saved — revert if scores got worse
                      </p>
                    )}
                  </div>

                  <AskChips questions={[
                    'Are these results good enough to use with real patients?',
                    'What does the accuracy score mean in practice?',
                    'What should I do next with this model?',
                  ]} onAsk={askAI} />

                  <div className="flex gap-3 pt-2">
                    <a
                      href="/model/download"
                      download
                      className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-primary text-on-primary text-sm font-semibold hover:opacity-90 transition-opacity"
                    >
                      <span className="material-symbols-outlined text-base">download</span>
                      Download Model (.pt)
                    </a>
                    <a
                      href="/model/download_meta"
                      download
                      className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-surface-container text-on-surface text-sm font-semibold border border-outline hover:bg-surface-container-high transition-colors"
                    >
                      <span className="material-symbols-outlined text-base">data_object</span>
                      Download Metadata (.json)
                    </a>
                  </div>
                </div>
              )}

            </div>
          </div>

          {/* ═══ RIGHT PANEL — AI CO-PILOT ═══ */}
          <aside className="w-80 flex-shrink-0 border-l border-slate-200/50 bg-primary text-on-primary flex flex-col overflow-hidden relative">
            {/* Decorative blur */}
            <div className="absolute -right-16 -top-16 w-48 h-48 bg-secondary rounded-full blur-[60px] opacity-20 pointer-events-none" />
            <div className="absolute -left-8 bottom-0 w-24 h-24 bg-secondary-fixed-dim rounded-full blur-[50px] opacity-10 pointer-events-none" />

            <div className="relative z-10 flex items-center gap-3 px-5 py-4 border-b border-on-primary-fixed-variant/30">
              <span className="material-symbols-outlined text-secondary-fixed-dim text-xl">psychology</span>
              <div>
                <h3 className="font-headline font-bold text-sm leading-tight">AI Co-Pilot</h3>
                <p className="font-label text-[10px] text-on-primary-container leading-tight">Plain-language guidance</p>
              </div>
              {aiLoading
                ? <div className="ml-auto w-4 h-4 border-2 border-secondary-fixed-dim border-t-transparent rounded-full animate-spin" />
                : <span className="font-label text-[10px] font-bold tracking-wider text-on-primary-container ml-auto opacity-60">Qwen 2.5</span>
              }
            </div>

            <div ref={aiChatRef} className="relative z-10 flex-1 overflow-y-auto p-4 flex flex-col gap-2.5">

              {/* Empty state: show suggested questions for current stage */}
              {aiMessages.length === 0 && (
                <div className="space-y-3 pt-2">
                  <p className="font-label text-[10px] uppercase tracking-widest text-on-primary-container/60 text-center">Ask me anything</p>
                  {(STAGE_QUESTIONS[STAGES[stage].id] || []).map((q, i) => (
                    <button key={i} onClick={() => askAI(q)} disabled={aiLoading}
                      className="w-full text-left px-3 py-2.5 rounded-xl bg-white/8 border border-white/12 text-xs text-on-primary-container hover:bg-white/16 hover:text-on-primary transition-colors leading-snug disabled:opacity-40">
                      <span className="material-symbols-outlined text-[13px] mr-1.5 align-text-bottom opacity-60">help_outline</span>
                      {q}
                    </button>
                  ))}
                </div>
              )}

              {/* Messages */}
              {aiMessages.map((m, i) => (
                <div key={i} className={`px-3 py-2.5 rounded-xl break-words ${
                  m.role === 'user'
                    ? 'bg-white/12 rounded-br-sm self-end max-w-[90%] text-xs text-on-primary'
                    : m.role === 'system'
                    ? 'bg-white/5 rounded-lg self-center text-[11px] text-on-primary-container border border-white/10 max-w-full text-center'
                    : 'bg-white/6 rounded-tl-sm self-start max-w-[95%] border border-white/10'
                }`}>
                  {m.role === 'ai'
                    ? <MarkdownMessage text={m.text} />
                    : <span className="text-xs leading-relaxed">{m.text}</span>
                  }
                </div>
              ))}

              {/* Follow-up suggestions after AI responds */}
              {!aiLoading && aiMessages.length > 0 && aiMessages[aiMessages.length - 1].role === 'ai' && (
                <div className="space-y-1.5 pt-1 border-t border-white/8">
                  <p className="font-label text-[9px] uppercase tracking-widest text-on-primary-container/50 pt-1">Follow-up</p>
                  {(STAGE_QUESTIONS[STAGES[stage].id] || []).slice(0, 2).map((q, i) => (
                    <button key={i} onClick={() => askAI(q)} disabled={aiLoading}
                      className="w-full text-left px-2.5 py-2 rounded-lg bg-white/6 border border-white/10 text-[11px] text-on-primary-container hover:bg-white/14 hover:text-on-primary transition-colors leading-snug">
                      {q}
                    </button>
                  ))}
                </div>
              )}
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
                  placeholder="Type your question..."
                  className="flex-1 px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-on-primary text-xs placeholder:text-on-primary-container/60 focus:outline-none focus:border-secondary-fixed-dim"
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

      {/* Model Gallery modal */}
      {showGallery && (
        <ModelGallery
          onClose={() => setShowGallery(false)}
          onLoad={(data) => {
            setTraining({ active: false, status: 'complete', metrics: data.metrics, epoch: data.meta?.config?.epochs, total_epochs: data.meta?.config?.epochs })
            setTrainHistory([])
            completeStage(4)
            setStage(5)
          }}
        />
      )}
    </div>
  )
}


// ── Small reusable pieces ───────────────────────────────────

// ── Model type display config ────────────────────────────────
const MODEL_TYPE_CONFIG = {
  mlp:      { icon: 'memory',       bg: '#00193c', label: 'MLP Neural Net' },
  cnn:      { icon: 'blur_on',      bg: '#7b41b3', label: 'CNN' },
  resnet:   { icon: 'hub',          bg: '#005312', label: 'ResNet' },
  logistic: { icon: 'linear_scale', bg: '#b45309', label: 'Logistic Regression' },
  auto:     { icon: 'auto_awesome', bg: '#1e40af', label: 'Auto-Selected' },
}

function formatSavedAt(ts) {
  if (!ts) return '—'
  // ts = "20240101_120000"
  try {
    const [date, time] = ts.split('_')
    return `${date.slice(6, 8)}/${date.slice(4, 6)}/${date.slice(0, 4)}  ${time.slice(0, 2)}:${time.slice(2, 4)}`
  } catch { return ts }
}

function ModelCard({ model, onLoad, onDelete, deleting, onRename }) {
  const [editing, setEditing] = useState(false)
  const [nick, setNick] = useState(model.nickname || '')
  const typeKey = (model.config?.model_type || 'auto').toLowerCase()
  const tc = MODEL_TYPE_CONFIG[typeKey] || MODEL_TYPE_CONFIG.auto
  const datasetName = model.dataset_name || model.config?.data_path?.split('/').pop() || 'Dataset'
  const metrics = model.final_metrics || {}
  const featureCount = model.feature_columns?.length ?? model.n_features

  const handleRenameSubmit = () => {
    onRename(nick.trim())
    setEditing(false)
  }

  return (
    <div className="bg-surface-container-lowest rounded-2xl border border-outline-variant/20 overflow-hidden flex flex-col shadow-sm hover:shadow-lg transition-shadow">
      {/* Colored header */}
      <div className="px-5 py-4 flex items-start gap-3" style={{ background: tc.bg }}>
        <span className="material-symbols-outlined text-white text-4xl mt-0.5">{tc.icon}</span>
        <div className="min-w-0 flex-1">
          <p className="font-label text-[10px] text-white/60 uppercase tracking-widest">{tc.label}</p>
          {editing ? (
            <div className="flex gap-1 mt-0.5">
              <input autoFocus value={nick} onChange={e => setNick(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter') handleRenameSubmit(); if (e.key === 'Escape') setEditing(false) }}
                className="flex-1 bg-white/20 text-white placeholder:text-white/50 text-sm font-bold rounded px-2 py-0.5 outline-none border border-white/40 min-w-0"
                placeholder={datasetName} />
              <button onClick={handleRenameSubmit} className="text-white/80 hover:text-white">
                <span className="material-symbols-outlined text-base">check</span>
              </button>
            </div>
          ) : (
            <button onClick={() => setEditing(true)} className="text-left group w-full">
              <p className="font-headline font-bold text-white text-lg leading-tight truncate group-hover:opacity-80">
                {model.nickname || datasetName}
              </p>
              <p className="text-white/50 text-[10px]">{model.nickname ? datasetName : 'tap to rename'}</p>
            </button>
          )}
        </div>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-2 divide-x divide-outline-variant/20 border-b border-outline-variant/15">
        <div className="py-3 text-center">
          <p className="font-label text-[10px] text-on-surface-variant uppercase tracking-wide">Accuracy</p>
          <p className="font-headline font-bold text-2xl text-primary">
            {metrics.accuracy != null ? `${(metrics.accuracy * 100).toFixed(1)}%` : '—'}
          </p>
        </div>
        <div className="py-3 text-center">
          <p className="font-label text-[10px] text-on-surface-variant uppercase tracking-wide">F1 Score</p>
          <p className="font-headline font-bold text-2xl text-primary">{metrics.f1?.toFixed(3) ?? '—'}</p>
        </div>
      </div>

      {/* Config chips */}
      <div className="px-4 py-2.5 flex flex-wrap gap-1.5 border-b border-outline-variant/15">
        {model.config?.epochs && (
          <span className="px-2 py-0.5 bg-surface-container rounded-full text-[10px] text-on-surface-variant font-label">{model.config.epochs} epochs</span>
        )}
        {model.config?.lr && (
          <span className="px-2 py-0.5 bg-surface-container rounded-full text-[10px] text-on-surface-variant font-label">LR {model.config.lr}</span>
        )}
        {featureCount != null && (
          <span className="px-2 py-0.5 bg-surface-container rounded-full text-[10px] text-on-surface-variant font-label">{featureCount} features</span>
        )}
        {model.config?.target_column && (
          <span className="px-2 py-0.5 bg-primary/10 text-primary rounded-full text-[10px] font-label font-medium">→ {model.config.target_column}</span>
        )}
        <span className="px-2 py-0.5 bg-surface-container rounded-full text-[10px] text-on-surface-variant font-label ml-auto">{formatSavedAt(model.saved_at)}</span>
      </div>

      {/* Actions */}
      <div className="px-4 py-3 flex items-center gap-2">
        <button onClick={onLoad}
          className="flex-1 py-2 bg-primary text-on-primary rounded-lg text-xs font-bold flex items-center justify-center gap-1.5 hover:opacity-90 transition-opacity">
          <span className="material-symbols-outlined text-sm">upload_file</span>
          Load
        </button>
        <a href={`/model/gallery/${model.id}/download`}
          className="p-2 rounded-lg bg-surface-container text-on-surface-variant hover:bg-surface-container-high transition-colors"
          title="Download .pt">
          <span className="material-symbols-outlined text-base">download</span>
        </a>
        <button onClick={onDelete} disabled={deleting}
          className="p-2 rounded-lg text-error hover:bg-error-container/40 transition-colors disabled:opacity-40">
          <span className="material-symbols-outlined text-base">{deleting ? 'hourglass_empty' : 'delete'}</span>
        </button>
      </div>
    </div>
  )
}

function ModelGallery({ onClose, onLoad }) {
  const [models, setModels] = useState([])
  const [loading, setLoading] = useState(true)
  const [deletingId, setDeletingId] = useState(null)

  const refresh = () => {
    setLoading(true)
    fetch('/model/gallery')
      .then(r => r.json())
      .then(d => { setModels(d.models || []); setLoading(false) })
      .catch(() => setLoading(false))
  }

  useEffect(() => { refresh() }, [])

  const handleDelete = async (id) => {
    setDeletingId(id)
    await fetch(`/model/delete/${id}`, { method: 'DELETE' }).catch(() => {})
    setModels(m => m.filter(model => model.id !== id))
    setDeletingId(null)
  }

  const handleLoad = async (model) => {
    const r = await fetch(`/model/load/${model.id}`, { method: 'POST' })
    const data = await r.json()
    if (!data.error) { onLoad(data); onClose() }
  }

  const handleRename = async (id, nickname) => {
    await fetch(`/model/rename/${id}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ nickname }),
    }).catch(() => {})
    setModels(m => m.map(model => model.id === id ? { ...model, nickname } : model))
  }

  return (
    <div className="fixed inset-0 z-[200] flex items-start justify-center bg-black/60 backdrop-blur-sm pt-8 pb-4 px-4"
      onClick={e => e.target === e.currentTarget && onClose()}>
      <div className="w-full max-w-5xl max-h-full bg-surface rounded-2xl shadow-2xl flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-8 py-5 border-b border-outline-variant/20 flex-shrink-0">
          <div>
            <h2 className="font-headline text-2xl font-extrabold text-primary tracking-tighter flex items-center gap-3">
              <span className="material-symbols-outlined text-secondary">photo_library</span>
              Model Gallery
            </h2>
            <p className="text-sm text-on-surface-variant mt-0.5">
              {models.length} model{models.length !== 1 ? 's' : ''} saved locally · tap a name to rename
            </p>
          </div>
          <button onClick={onClose}
            className="w-9 h-9 flex items-center justify-center rounded-full hover:bg-surface-container transition-colors">
            <span className="material-symbols-outlined text-on-surface-variant">close</span>
          </button>
        </div>

        {/* Grid */}
        <div className="flex-1 overflow-y-auto p-6">
          {loading ? (
            <div className="flex items-center justify-center h-48">
              <div className="w-6 h-6 border-2 border-primary border-t-transparent rounded-full animate-spin" />
            </div>
          ) : models.length === 0 ? (
            <div className="text-center py-20">
              <span className="material-symbols-outlined text-6xl text-outline block mb-3">hub</span>
              <p className="text-on-surface-variant text-sm font-label">No models saved yet.</p>
              <p className="text-on-surface-variant text-xs mt-1">Complete a training run to save your first model.</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
              {models.map(model => (
                <ModelCard
                  key={model.id}
                  model={model}
                  onLoad={() => handleLoad(model)}
                  onDelete={() => handleDelete(model.id)}
                  deleting={deletingId === model.id}
                  onRename={(nick) => handleRename(model.id, nick)}
                />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function AskChips({ questions, onAsk }) {
  return (
    <div className="flex flex-wrap gap-2 py-1">
      <span className="flex items-center gap-1 text-[11px] text-on-surface-variant font-label">
        <span className="material-symbols-outlined text-sm text-secondary">psychology</span>
        Ask AI:
      </span>
      {questions.map((q, i) => (
        <button key={i} onClick={() => onAsk(q)}
          className="px-3 py-1.5 rounded-full bg-surface-container border border-secondary/20 text-xs text-secondary font-medium hover:bg-secondary hover:text-on-secondary transition-colors">
          {q}
        </button>
      ))}
    </div>
  )
}

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

// ── Illustrated parameter cards for the Configure stage ─────

function LRCard({ value, onChange }) {
  const min = 0.0001, max = 0.1
  const pct = ((value - min) / (max - min)) * 100
  const zone = value < 0.0008 ? 'cautious' : value > 0.015 ? 'aggressive' : 'optimal'
  const z = {
    cautious:   { label: 'Cautious — very stable', color: '#7b41b3', bg: '#f0dbff40', icon: 'moving',       desc: 'Tiny steps per update. Stable but slow — may need more epochs to converge.' },
    optimal:    { label: 'Optimal range',           color: '#005312', bg: '#a3f69c30', icon: 'check_circle', desc: 'Balanced convergence speed and stability. Works well for most clinical datasets.' },
    aggressive: { label: 'Aggressive — unstable risk', color: '#ba1a1a', bg: '#ffdad640', icon: 'warning',  desc: 'Large steps risk overshooting the loss minimum. Monitor training loss closely.' },
  }[zone]

  // Gradient descent path SVG
  const W = 280, H = 88
  const stepSize = 0.06 + (value / max) * 0.74
  let pos = 0.18
  const steps = []
  for (let i = 0; i < 6; i++) {
    steps.push(pos)
    const grad = 2 * (pos - 0.5)
    pos = Math.min(0.97, Math.max(0.03, pos - stepSize * grad))
  }
  const toSVG = p => ({ x: p * W, y: H - 10 - Math.pow(p - 0.5, 2) * (H - 24) })
  const pts = steps.map(toSVG)

  // Parabola path
  const parabola = Array.from({ length: 50 }, (_, i) => {
    const p = i / 49
    const { x, y } = toSVG(p)
    return `${x},${y}`
  }).join(' ')

  return (
    <div className="bg-surface-container-low p-6 rounded-xl space-y-4 border border-outline-variant/10 hover:border-outline-variant/30 transition-colors">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 mb-0.5">
            <span className="material-symbols-outlined text-primary text-base">trending_down</span>
            <h3 className="font-headline font-bold text-primary">Learning Rate</h3>
            <span className="font-label text-[9px] bg-surface-container text-on-surface-variant px-2 py-0.5 rounded tracking-wider">LR</span>
          </div>
          <p className="text-xs text-on-surface-variant">Step size per gradient descent update.</p>
        </div>
        <div className="text-right shrink-0">
          <p className="font-mono font-black text-2xl text-primary leading-none">{value.toFixed(4)}</p>
          <p className="font-label text-[10px] uppercase tracking-wider mt-0.5" style={{ color: z.color }}>{zone}</p>
        </div>
      </div>

      {/* Illustration */}
      <div className="bg-surface rounded-xl overflow-hidden border border-outline-variant/20">
        <svg width="100%" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet">
          {/* Zone shading */}
          <rect x={0} y={0} width={W * 0.18} height={H} fill="#f0dbff" opacity="0.45" />
          <rect x={W * 0.18} y={0} width={W * 0.62} height={H} fill="#a3f69c" opacity="0.18" />
          <rect x={W * 0.80} y={0} width={W * 0.20} height={H} fill="#ffdad6" opacity="0.45" />
          {/* Zone labels */}
          <text x="5" y="11" fontSize="7" fill="#622599" opacity="0.75" fontWeight="600">slow</text>
          <text x={W * 0.42} y="11" fontSize="7" fill="#005312" opacity="0.75" fontWeight="600">optimal</text>
          <text x={W * 0.82} y="11" fontSize="7" fill="#ba1a1a" opacity="0.75" fontWeight="600">fast</text>
          {/* Minimum dashed line */}
          <line x1={W / 2} y1={5} x2={W / 2} y2={H} stroke="#43474f" strokeWidth="1" strokeDasharray="3,3" opacity="0.3" />
          <text x={W / 2 + 3} y="20" fontSize="7" fill="#43474f" opacity="0.5">min</text>
          {/* Loss curve */}
          <polyline points={parabola} fill="none" stroke="#c4c6d1" strokeWidth="2" strokeLinecap="round" />
          {/* Descent path lines */}
          {pts.slice(0, -1).map((p, i) => (
            <line key={i} x1={p.x} y1={p.y} x2={pts[i + 1].x} y2={pts[i + 1].y}
              stroke={z.color} strokeWidth="1.5" opacity={0.9 - i * 0.12} />
          ))}
          {/* Descent dots */}
          {pts.map((p, i) => (
            <circle key={i} cx={p.x} cy={p.y} r={i === 0 ? 5 : 3.5}
              fill={i === 0 ? z.color : '#fff'} stroke={z.color} strokeWidth="1.5"
              opacity={1 - i * 0.12} />
          ))}
        </svg>
      </div>

      {/* Slider */}
      <div className="space-y-1.5">
        <input type="range" min={min} max={max} step={0.0001} value={value}
          onChange={e => onChange(parseFloat(e.target.value))}
          className="w-full"
          style={{ background: `linear-gradient(to right, ${z.color} 0%, ${z.color} ${pct}%, #e1e3e4 ${pct}%, #e1e3e4 100%)` }}
        />
        <div className="flex justify-between text-[10px] text-on-surface-variant font-mono">
          <span>1e-4</span><span className="text-on-surface-variant/60">|</span><span>0.01</span><span className="text-on-surface-variant/60">|</span><span>0.1</span>
        </div>
      </div>

      {/* Insight */}
      <div className="p-3 rounded-xl flex items-start gap-2.5 border" style={{ background: z.bg, borderColor: z.color + '40' }}>
        <span className="material-symbols-outlined text-sm mt-0.5 shrink-0" style={{ color: z.color }}>{z.icon}</span>
        <div>
          <p className="text-xs font-bold leading-none mb-1" style={{ color: z.color }}>{z.label}</p>
          <p className="text-xs text-on-surface-variant leading-relaxed">{z.desc}</p>
        </div>
      </div>
    </div>
  )
}

function EpochsCard({ value, onChange }) {
  const min = 10, max = 1000
  const pct = ((value - min) / (max - min)) * 100
  const zone = value < 80 ? 'underfit' : value > 600 ? 'overfit' : 'optimal'
  const z = {
    underfit: { label: 'Too few — underfitting', color: '#7b41b3', bg: '#f0dbff40', icon: 'show_chart',   desc: 'Not enough iterations. The model hasn\'t learned meaningful clinical patterns yet.' },
    optimal:  { label: 'Healthy range',          color: '#005312', bg: '#a3f69c30', icon: 'check_circle', desc: 'Good balance of learning depth vs. generalization. Suitable for most tasks.' },
    overfit:  { label: 'Overfitting risk',        color: '#ba1a1a', bg: '#ffdad640', icon: 'warning',      desc: 'Too many passes. The model may memorize training patients and fail on new ones.' },
  }[zone]

  // Pre-compute learning curves
  const W = 280, H = 88, N = 60
  const trainPts = [], valPts = []
  for (let i = 0; i < N; i++) {
    const t = i / (N - 1)
    const x = t * W
    const trainL = 0.82 * Math.exp(-3.2 * t) + 0.06 + 0.015 * Math.sin(t * 18) * Math.exp(-2 * t)
    const overfit = Math.max(0, t - 0.58) * 0.38
    const valL = 0.78 * Math.exp(-2.6 * t) + 0.11 + overfit + 0.012 * Math.sin(t * 14) * Math.exp(-t)
    const sy = v => H - 10 - Math.max(0, Math.min(1, v)) * (H - 22)
    trainPts.push(`${x},${sy(trainL)}`)
    valPts.push(`${x},${sy(valL)}`)
  }
  const markerX = ((value - min) / (max - min)) * W

  return (
    <div className="bg-surface-container-low p-6 rounded-xl space-y-4 border border-outline-variant/10 hover:border-outline-variant/30 transition-colors">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 mb-0.5">
            <span className="material-symbols-outlined text-primary text-base">repeat</span>
            <h3 className="font-headline font-bold text-primary">Epochs</h3>
            <span className="font-label text-[9px] bg-surface-container text-on-surface-variant px-2 py-0.5 rounded tracking-wider">PASSES</span>
          </div>
          <p className="text-xs text-on-surface-variant">Full dataset passes during training.</p>
        </div>
        <div className="text-right shrink-0">
          <p className="font-mono font-black text-2xl text-primary leading-none">{value}</p>
          <p className="font-label text-[10px] uppercase tracking-wider mt-0.5" style={{ color: z.color }}>{zone}</p>
        </div>
      </div>

      {/* Learning curves SVG */}
      <div className="bg-surface rounded-xl overflow-hidden border border-outline-variant/20">
        <svg width="100%" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet">
          {/* Zone backgrounds */}
          <rect x={0} y={0} width={W * 0.14} height={H} fill="#f0dbff" opacity="0.45" />
          <rect x={W * 0.14} y={0} width={W * 0.56} height={H} fill="#a3f69c" opacity="0.18" />
          <rect x={W * 0.70} y={0} width={W * 0.30} height={H} fill="#ffdad6" opacity="0.45" />
          <text x="4" y="11" fontSize="7" fill="#622599" opacity="0.75" fontWeight="600">under</text>
          <text x={W * 0.36} y="11" fontSize="7" fill="#005312" opacity="0.75" fontWeight="600">optimal</text>
          <text x={W * 0.73} y="11" fontSize="7" fill="#ba1a1a" opacity="0.75" fontWeight="600">overfit</text>
          {/* Val loss */}
          <polyline points={valPts.join(' ')} fill="none" stroke="#7b41b3" strokeWidth="1.5" strokeLinecap="round" opacity="0.65" />
          {/* Train loss */}
          <polyline points={trainPts.join(' ')} fill="none" stroke="#00193c" strokeWidth="2" strokeLinecap="round" />
          {/* Current epoch marker */}
          <line x1={markerX} y1={0} x2={markerX} y2={H} stroke={z.color} strokeWidth="2" strokeDasharray="4,3" />
          <circle cx={markerX} cy={H / 2} r="4.5" fill={z.color} />
          {/* Legend */}
          <rect x={W - 58} y={H - 22} width="8" height="3" rx="1.5" fill="#00193c" />
          <text x={W - 47} y={H - 16} fontSize="7" fill="#43474f">train loss</text>
          <rect x={W - 58} y={H - 11} width="8" height="3" rx="1.5" fill="#7b41b3" opacity="0.65" />
          <text x={W - 47} y={H - 5} fontSize="7" fill="#43474f">val loss</text>
        </svg>
      </div>

      {/* Slider */}
      <div className="space-y-1.5">
        <input type="range" min={min} max={max} step={10} value={value}
          onChange={e => onChange(parseInt(e.target.value))}
          className="w-full"
          style={{ background: `linear-gradient(to right, ${z.color} 0%, ${z.color} ${pct}%, #e1e3e4 ${pct}%, #e1e3e4 100%)` }}
        />
        <div className="flex justify-between text-[10px] text-on-surface-variant font-mono">
          <span>10</span><span className="text-on-surface-variant/60">|</span><span>500</span><span className="text-on-surface-variant/60">|</span><span>1000</span>
        </div>
      </div>

      <div className="p-3 rounded-xl flex items-start gap-2.5 border" style={{ background: z.bg, borderColor: z.color + '40' }}>
        <span className="material-symbols-outlined text-sm mt-0.5 shrink-0" style={{ color: z.color }}>{z.icon}</span>
        <div>
          <p className="text-xs font-bold leading-none mb-1" style={{ color: z.color }}>{z.label}</p>
          <p className="text-xs text-on-surface-variant leading-relaxed">{z.desc}</p>
        </div>
      </div>
    </div>
  )
}

function BatchCard({ value, onChange }) {
  const min = 16, max = 512
  const pct = ((value - min) / (max - min)) * 100
  const zone = value <= 32 ? 'noisy' : value >= 256 ? 'large' : 'balanced'
  const z = {
    noisy:    { label: 'Noisy gradients',   color: '#7b41b3', bg: '#f0dbff40', icon: 'graphic_eq',  desc: 'Small batches inject noise that can help escape local minima but slow per-step progress.' },
    balanced: { label: 'Well balanced',     color: '#005312', bg: '#a3f69c30', icon: 'check_circle', desc: 'Good tradeoff between gradient quality and memory efficiency.' },
    large:    { label: 'Smooth — less noise', color: '#24467c', bg: '#d7e2ff40', icon: 'timeline',   desc: 'Accurate gradients but may converge to sharper, less generalizable solutions.' },
  }[zone]

  // Dot grid: 8×8 = 64 total, highlight proportion matching batch size
  const COLS = 16, ROWS = 4, TOTAL = COLS * ROWS
  const highlighted = Math.round((value / max) * TOTAL)
  const W = 280, H = 72
  const cx = (i) => ((i % COLS) + 0.5) * (W / COLS)
  const cy = (i) => (Math.floor(i / COLS) + 0.5) * (H / ROWS)

  // Noise waveform: amplitude inversely proportional to batch size
  const noiseAmp = 1 - (value - min) / (max - min)  // 0..1 (1=max noise)
  const wavePoints = Array.from({ length: 40 }, (_, i) => {
    const t = i / 39
    const x = t * W
    const noise = (Math.sin(t * 28 + 1) * 0.4 + Math.sin(t * 11) * 0.6) * noiseAmp
    const y = H / 2 + noise * (H * 0.36)
    return `${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')

  return (
    <div className="bg-surface-container-low p-6 rounded-xl space-y-4 border border-outline-variant/10 hover:border-outline-variant/30 transition-colors">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 mb-0.5">
            <span className="material-symbols-outlined text-primary text-base">dataset</span>
            <h3 className="font-headline font-bold text-primary">Batch Size</h3>
            <span className="font-label text-[9px] bg-surface-container text-on-surface-variant px-2 py-0.5 rounded tracking-wider">BATCH</span>
          </div>
          <p className="text-xs text-on-surface-variant">Records per gradient update step.</p>
        </div>
        <div className="text-right shrink-0">
          <p className="font-mono font-black text-2xl text-primary leading-none">{value}</p>
          <p className="font-label text-[10px] uppercase tracking-wider mt-0.5" style={{ color: z.color }}>per step</p>
        </div>
      </div>

      {/* Dot grid + waveform */}
      <div className="bg-surface rounded-xl overflow-hidden border border-outline-variant/20">
        <svg width="100%" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet">
          {/* Dots */}
          {Array.from({ length: TOTAL }, (_, i) => (
            <circle key={i} cx={cx(i)} cy={cy(i)} r="3.5"
              fill={i < highlighted ? z.color : '#e1e3e4'}
              opacity={i < highlighted ? 0.85 : 0.5}
            />
          ))}
          {/* Gradient noise waveform overlay */}
          <polyline points={wavePoints} fill="none" stroke={z.color} strokeWidth="1.5"
            opacity="0.35" strokeLinecap="round" />
          <text x="6" y={H - 5} fontSize="7.5" fill={z.color} fontWeight="600" opacity="0.8">
            {highlighted}/{TOTAL} shown · {value} records/step
          </text>
        </svg>
      </div>

      {/* Slider */}
      <div className="space-y-1.5">
        <input type="range" min={min} max={max} step={16} value={value}
          onChange={e => onChange(parseInt(e.target.value))}
          className="w-full"
          style={{ background: `linear-gradient(to right, ${z.color} 0%, ${z.color} ${pct}%, #e1e3e4 ${pct}%, #e1e3e4 100%)` }}
        />
        <div className="flex justify-between text-[10px] text-on-surface-variant font-mono">
          <span>16</span><span className="text-on-surface-variant/60">|</span><span>256</span><span className="text-on-surface-variant/60">|</span><span>512</span>
        </div>
      </div>

      <div className="p-3 rounded-xl flex items-start gap-2.5 border" style={{ background: z.bg, borderColor: z.color + '40' }}>
        <span className="material-symbols-outlined text-sm mt-0.5 shrink-0" style={{ color: z.color }}>{z.icon}</span>
        <div>
          <p className="text-xs font-bold leading-none mb-1" style={{ color: z.color }}>{z.label}</p>
          <p className="text-xs text-on-surface-variant leading-relaxed">{z.desc}</p>
        </div>
      </div>
    </div>
  )
}

function ModelTypeCard({ modelType, targetColumn, targetOptions, onModelChange, onTargetChange }) {
  const models = [
    { id: 'auto',     icon: 'auto_awesome',  label: 'Auto',     desc: 'AI picks the best fit' },
    { id: 'mlp',      icon: 'account_tree',  label: 'MLP',      desc: 'Tabular data' },
    { id: 'cnn',      icon: 'image_search',  label: 'CNN',      desc: 'Image data' },
    { id: 'resnet',   icon: 'layers',        label: 'ResNet',   desc: 'Deep images' },
    { id: 'logistic', icon: 'show_chart',    label: 'Logistic', desc: 'Linear baseline' },
  ]
  return (
    <div className="bg-surface-container-low p-6 rounded-xl space-y-4 border border-outline-variant/10 hover:border-outline-variant/30 transition-colors">
      <div>
        <div className="flex items-center gap-2 mb-0.5">
          <span className="material-symbols-outlined text-primary text-base">hub</span>
          <h3 className="font-headline font-bold text-primary">Architecture</h3>
        </div>
        <p className="text-xs text-on-surface-variant">Neural network model for training.</p>
      </div>

      <div className="grid grid-cols-3 gap-2">
        {models.map(m => (
          <button key={m.id} onClick={() => onModelChange(m.id)}
            className={`p-3 rounded-xl text-left transition-all border ${
              modelType === m.id
                ? 'bg-primary border-primary shadow-md'
                : 'bg-surface-container-lowest border-outline-variant/25 hover:border-outline-variant hover:bg-surface-container'
            }`}
          >
            <span className={`material-symbols-outlined text-xl block mb-1.5 ${modelType === m.id ? 'text-on-primary' : 'text-primary'}`}>
              {m.icon}
            </span>
            <p className={`font-bold text-xs ${modelType === m.id ? 'text-on-primary' : 'text-primary'}`}>{m.label}</p>
            <p className={`text-[10px] mt-0.5 leading-tight ${modelType === m.id ? 'text-on-primary-container' : 'text-on-surface-variant'}`}>{m.desc}</p>
          </button>
        ))}
      </div>

      <div className="pt-2 border-t border-outline-variant/20 space-y-2">
        <p className="font-label text-[10px] text-on-surface-variant uppercase tracking-widest">Target Column</p>
        {targetOptions?.length > 0 ? (
          <select value={targetColumn} onChange={e => onTargetChange(e.target.value)}
            className="w-full px-3 py-2.5 bg-surface-container-lowest border border-outline-variant/40 rounded-lg text-on-surface text-sm focus:outline-none focus:border-primary">
            <option value="">auto-detect</option>
            {targetOptions.map(o => <option key={o} value={o}>{o}</option>)}
          </select>
        ) : (
          <input type="text" value={targetColumn} onChange={e => onTargetChange(e.target.value)}
            placeholder="Column name or leave blank to auto-detect"
            className="w-full px-3 py-2.5 bg-surface-container-lowest border border-outline-variant/40 rounded-lg text-on-surface text-sm focus:outline-none focus:border-primary" />
        )}
        <p className="text-[10px] text-on-surface-variant">
          The column your model will learn to predict. If left blank, the backend auto-detects it.
        </p>
      </div>
    </div>
  )
}
