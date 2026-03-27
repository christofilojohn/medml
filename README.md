# 🧬 MedML Forge

**Privacy-First Clinical AI Pipeline — Train models on-device without sending patient data anywhere.**

MedML Forge is a complete MLOps tool for hospitals and bio-clinics that trivializes the journey from raw data to trained model to federated learning — all running locally.

## Architecture

```
┌────────────────────────────────────────────────────┐
│                  React Dashboard                    │
│   (Pipeline stages, data preview, live metrics)     │
│                    :5173                            │
└──────────┬───────────────────┬─────────────────────┘
           │                   │
    ┌──────▼──────┐    ┌──────▼──────┐
    │  ML Worker   │    │  Qwen 2.5   │
    │  (Python)    │    │  (llama.cpp) │
    │   :8081      │    │   :8080      │
    │              │    │              │
    │ • Data scan  │    │ • Reasoning  │
    │ • Training   │    │ • Advice     │
    │ • Cleanup    │    │ • Model rec  │
    │ • Preview    │    │              │
    └──────────────┘    └──────────────┘
           │
    ┌──────▼──────┐
    │  Local Data  │
    │  (never      │
    │   leaves)    │
    └──────────────┘
```

## Pipeline Stages

1. **Data Scan** — Point to a local directory. A Python script scans metadata (file counts, column types, resolutions, class distributions). No raw data is sent anywhere.

2. **Data Preview** — Tabular data: interactive table with column types, distributions, missing values. Images: thumbnail grid with class balance visualization.

3. **Cleanup** — Automated removal of duplicates, imputation of missing values, outlier capping, corrupted file removal.

4. **Configure** — Model selection (auto, MLP, CNN, ResNet, logistic) and hyperparameter tuning. The on-device Qwen LLM recommends settings based on your data.

5. **Training** — Real PyTorch training loop with live SSE-streamed metrics. Epoch-by-epoch loss curves, accuracy, F1 — all rendered in the dashboard in real-time.

6. **Evaluate** — Full training history visualization, final metrics summary.

7. **Federate** — Matchmaking with federated learning networks (Flower, NVIDIA FLARE). Only model gradients are shared, never raw data.

## Quick Start

### Native (macOS / Linux)

```bash
chmod +x start.sh
./start.sh
```

The launcher will:
- Auto-detect GPU (Apple Silicon / NVIDIA / AMD / CPU)
- Download Qwen 2.5 3B model (~2GB, one-time)
- Install Python ML dependencies
- Start all three services
- Open the dashboard at http://localhost:5173

### Docker

```bash
# NVIDIA GPU
docker build --build-arg BUILD_TYPE=cuda -t medml-forge .
docker run --gpus all -p 3000:3000 -v ~/.cache/medml-models:/models -v /path/to/data:/data medml-forge

# CPU only
docker build --build-arg BUILD_TYPE=cpu -t medml-forge .
docker run -p 3000:3000 -v ~/.cache/medml-models:/models -v /path/to/data:/data medml-forge
```

## Requirements

- **llama.cpp** (llama-server binary)
- **Node.js** 18+
- **Python** 3.10+ with: torch, flask, pandas, scikit-learn, pillow

## Privacy Guarantees

- ✅ All data processing runs locally
- ✅ LLM inference runs on your hardware
- ✅ Scanner sends only metadata (column names, types, counts)
- ✅ Training happens entirely on-device
- ✅ Federation only shares encrypted model gradients
- ❌ No cloud APIs, no telemetry, no data uploads

## Configuration

Copy `.env.example` to `.env` and customize. Key options:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_NGL` | 999 | GPU layers (999 = all) |
| `LLM_THREADS` | 4 | CPU threads |
| `LLM_CONTEXT` | 4096 | Context window |
| `UI_PORT` | 5173 | Dashboard port |
| `ML_WORKER_PORT` | 8081 | ML Worker port |
