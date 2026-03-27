#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════╗
# ║          MedML Forge — Universal Launcher                    ║
# ║   Privacy-First Clinical AI Pipeline                         ║
# ║   Auto-detects: Apple Silicon | NVIDIA | AMD | CPU           ║
# ║                                                              ║
# ║   Components:                                                ║
# ║     1. Qwen 2.5 3B (on-device reasoning via llama.cpp)       ║
# ║     2. Python ML Worker (training, scanning, augmentation)   ║
# ║     3. React Dashboard (pipeline co-pilot UI)                ║
# ╚══════════════════════════════════════════════════════════════╝

set -euo pipefail
set -m

# ── Colors ──────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log()  { echo -e "${CYAN}[medml]${NC} $*"; }
ok()   { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[✗]${NC} $*"; exit 1; }
banner() {
  echo -e "${BOLD}${BLUE}"
  echo "  ╔═════════════════════════════════════════════╗"
  echo "  ║      🧬  MedML Forge                        ║"
  echo "  ║      Privacy-First Clinical AI Pipeline      ║"
  echo "  ║      Qwen 2.5 · On-Device Training           ║"
  echo "  ╚═════════════════════════════════════════════╝"
  echo -e "${NC}"
}

# ── Config ───────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "$SCRIPT_DIR/.env" ]]; then
  set -a; source "$SCRIPT_DIR/.env"; set +a
fi

MODEL_DIR="${MEDML_MODEL_DIR:-$HOME/.cache/medml-models}"
MODEL_FILE="Qwen2.5-3B-Instruct-Q4_K_M.gguf"
MODEL_REPO="bartowski/Qwen2.5-3B-Instruct-GGUF"
# Minimum expected model size in bytes (~1.9GB for Q4_K_M 3B)
MODEL_MIN_SIZE=1500000000
LLM_PORT="${LLM_PORT:-8080}"
ML_WORKER_PORT="${ML_WORKER_PORT:-8081}"
UI_PORT="${UI_PORT:-5173}"
CONTEXT="${LLM_CONTEXT:-4096}"
THREADS="${LLM_THREADS:-4}"

# ── Cleanup on exit ──────────────────────────────────────────
declare -a PIDS=()
cleanup() {
  echo ""
  log "Shutting down all services..."
  if [[ ${#PIDS[@]} -gt 0 ]]; then
    for pid in "${PIDS[@]}"; do
      kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
    done
  fi
  wait 2>/dev/null || true
  ok "Goodbye!"
}
trap cleanup EXIT INT TERM

# ── GPU Detection (same as Adaptive Dashboard) ──────────────
detect_gpu() {
  GPU_TYPE="cpu"
  NGL=0

  if [[ "$(uname)" == "Darwin" ]]; then
    ARCH=$(uname -m)
    if [[ "$ARCH" == "arm64" ]]; then
      GPU_TYPE="apple"
      NGL=999
      CHIP=$(system_profiler SPHardwareDataType 2>/dev/null | grep "Chip" | awk -F': ' '{print $2}' | xargs)
      ok "Apple Silicon detected: ${CHIP:-M-series}"
      return
    fi
  fi

  if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "")
    if [[ -n "$GPU_NAME" ]]; then
      GPU_TYPE="nvidia"
      NGL=999
      ok "NVIDIA GPU detected: $GPU_NAME"
      return
    fi
  fi

  if command -v rocm-smi &>/dev/null || [[ -d /opt/rocm ]]; then
    GPU_TYPE="amd"
    NGL=999
    ok "AMD GPU detected"
    return
  fi

  warn "No GPU detected — running on CPU (training will be slower)"
  NGL=0
  THREADS="${LLM_THREADS:-8}"
}

# ── Dependency checks ────────────────────────────────────────
check_deps() {
  log "Checking dependencies..."

  # llama-server
  if ! command -v llama-server &>/dev/null; then
    echo ""
    warn "llama-server not found."
    echo ""
    if [[ "$(uname)" == "Darwin" ]]; then
      echo -e "  ${BOLD}brew install llama.cpp${NC}"
    else
      echo -e "  ${BOLD}Build from source:${NC}"
      echo "    git clone https://github.com/ggerganov/llama.cpp"
      echo "    cd llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build -j8"
      echo "    sudo cp build/bin/llama-server /usr/local/bin/"
    fi
    err "Please install llama-server and re-run."
  fi
  ok "llama-server: $(which llama-server)"

  # Node.js
  if ! command -v node &>/dev/null; then
    err "Node.js not found. Install v18+: https://nodejs.org"
  fi
  ok "Node.js: $(node --version)"

  # Python 3
  if ! command -v python3 &>/dev/null; then
    err "Python 3 not found. Install Python 3.10+."
  fi
  ok "Python: $(python3 --version)"

  # Python deps
  log "Checking Python ML dependencies..."
  python3 -c "import torch, flask, pandas, sklearn, PIL" 2>/dev/null || {
    warn "Installing Python dependencies..."
    pip3 install torch torchvision flask flask-cors pandas scikit-learn pillow pyarrow openpyxl \
      --break-system-packages --quiet 2>/dev/null || \
    pip3 install torch torchvision flask flask-cors pandas scikit-learn pillow pyarrow openpyxl --quiet
    ok "Python deps installed"
  }
  ok "Python ML stack ready"
}

# ── Model download ───────────────────────────────────────────
ensure_model() {
  mkdir -p "$MODEL_DIR"
  MODEL_PATH="$MODEL_DIR/$MODEL_FILE"

  # Check if file exists AND is large enough (catches partial/stub downloads)
  if [[ -f "$MODEL_PATH" ]]; then
    FILE_SIZE=$(stat -f%z "$MODEL_PATH" 2>/dev/null || stat -c%s "$MODEL_PATH" 2>/dev/null || echo 0)
    if (( FILE_SIZE < MODEL_MIN_SIZE )); then
      warn "Model file exists but is only $(du -sh "$MODEL_PATH" | awk '{print $1}') — likely corrupt or incomplete."
      warn "Removing and re-downloading..."
      rm -f "$MODEL_PATH"
    fi
  fi

  if [[ ! -f "$MODEL_PATH" ]]; then
    log "Downloading Qwen 2.5 3B Instruct Q4_K_M (~2GB)..."
    echo -e "${YELLOW}  One-time download. Location: $MODEL_DIR${NC}"
    echo ""

    HF_URL="https://huggingface.co/${MODEL_REPO}/resolve/main/$MODEL_FILE"

    # Prefer huggingface-cli if available and functional
    DOWNLOADED=false
    if command -v huggingface-cli &>/dev/null && huggingface-cli download --help &>/dev/null 2>&1; then
      log "Trying huggingface-cli..."
      if huggingface-cli download "$MODEL_REPO" "$MODEL_FILE" \
          --local-dir "$MODEL_DIR" --local-dir-use-symlinks False 2>/dev/null; then
        # huggingface-cli might put file in a subdirectory — find it
        if [[ ! -f "$MODEL_PATH" ]]; then
          FOUND=$(find "$MODEL_DIR" -name "$MODEL_FILE" -type f 2>/dev/null | head -1)
          if [[ -n "$FOUND" && "$FOUND" != "$MODEL_PATH" ]]; then
            mv "$FOUND" "$MODEL_PATH"
          fi
        fi
        [[ -f "$MODEL_PATH" ]] && DOWNLOADED=true
      fi
      if ! $DOWNLOADED; then
        warn "huggingface-cli download did not produce expected file — falling back to curl"
      fi
    fi

    # Fallback: curl with redirect following
    if ! $DOWNLOADED; then
      if command -v curl &>/dev/null; then
        log "Downloading via curl..."
        curl -L --progress-bar --fail -o "$MODEL_PATH" "$HF_URL" || {
          rm -f "$MODEL_PATH"
          err "curl download failed. URL: $HF_URL"
        }
      elif command -v wget &>/dev/null; then
        wget --show-progress -O "$MODEL_PATH" "$HF_URL" || {
          rm -f "$MODEL_PATH"
          err "wget download failed. URL: $HF_URL"
        }
      else
        err "No download tool found. Install curl or: pip install huggingface-hub"
      fi
    fi

    # Final validation
    if [[ ! -f "$MODEL_PATH" ]]; then
      err "Model file not found after download. Try manually:\n  curl -L -o '$MODEL_PATH' '$HF_URL'"
    fi

    FILE_SIZE=$(stat -f%z "$MODEL_PATH" 2>/dev/null || stat -c%s "$MODEL_PATH" 2>/dev/null || echo 0)
    if (( FILE_SIZE < MODEL_MIN_SIZE )); then
      ACTUAL_SIZE=$(du -sh "$MODEL_PATH" | awk '{print $1}')
      rm -f "$MODEL_PATH"
      err "Downloaded file is too small ($ACTUAL_SIZE) — expected ~2GB.\n  The file was likely an HTML error page, not the actual model.\n  Try manually:\n    curl -L -o '$MODEL_PATH' '$HF_URL'"
    fi

    ok "Model downloaded: $MODEL_FILE ($(du -sh "$MODEL_PATH" | awk '{print $1}'))"
  else
    ok "Model ready: $MODEL_FILE ($(du -sh "$MODEL_PATH" | awk '{print $1}'))"
  fi
}

# ── Start LLM Server (Qwen for reasoning) ───────────────────
start_llm() {
  log "Starting Qwen reasoning server on port $LLM_PORT..."
  log "  GPU: $GPU_TYPE | NGL: $NGL | Threads: $THREADS | Context: $CONTEXT"

  FA_FLAG=""
  if llama-server --help 2>&1 | grep -q "\-fa\|flash.attn"; then
    FA_FLAG="-fa"
  fi

  llama-server \
    -m "$MODEL_DIR/$MODEL_FILE" \
    -c "$CONTEXT" \
    -ngl "$NGL" \
    -t "$THREADS" \
    --port "$LLM_PORT" \
    --host 127.0.0.1 \
    $FA_FLAG \
    --log-disable \
    > "$SCRIPT_DIR/.llm.log" 2>&1 &

  LLM_PID=$!
  PIDS+=($LLM_PID)

  echo -n "  Waiting for LLM"
  for i in $(seq 1 120); do
    sleep 0.75
    if curl -sf "http://localhost:$LLM_PORT/health" &>/dev/null; then
      echo ""
      ok "LLM server ready (pid: $LLM_PID)"
      return
    fi
    echo -n "."
    if ! kill -0 "$LLM_PID" 2>/dev/null; then
      echo ""
      err "llama-server crashed. Check .llm.log:\n$(tail -20 "$SCRIPT_DIR/.llm.log")"
    fi
  done
  echo ""
  err "LLM server timeout. Check .llm.log"
}

# ── Start Python ML Worker ───────────────────────────────────
start_ml_worker() {
  log "Starting ML Worker on port $ML_WORKER_PORT..."

  ML_WORKER_PORT=$ML_WORKER_PORT \
  LLM_PORT=$LLM_PORT \
  GPU_TYPE=$GPU_TYPE \
  python3 "$SCRIPT_DIR/ml-worker/server.py" \
    > "$SCRIPT_DIR/.ml-worker.log" 2>&1 &

  ML_PID=$!
  PIDS+=($ML_PID)

  echo -n "  Waiting for ML Worker"
  for i in $(seq 1 30); do
    sleep 0.5
    if curl -sf "http://127.0.0.1:$ML_WORKER_PORT/health" &>/dev/null; then
      echo ""
      ok "ML Worker ready (pid: $ML_PID)"
      return
    fi
    echo -n "."
    if ! kill -0 "$ML_PID" 2>/dev/null; then
      echo ""
      err "ML Worker crashed. Check .ml-worker.log:\n$(tail -20 "$SCRIPT_DIR/.ml-worker.log")"
    fi
  done
  echo ""
  warn "ML Worker slow to start — check .ml-worker.log"
}

# ── Start Frontend ───────────────────────────────────────────
start_frontend() {
  FRONTEND_DIR="$SCRIPT_DIR/frontend"
  log "Starting frontend..."

  (
    cd "$FRONTEND_DIR"
    if [[ ! -d "node_modules" ]] || [[ "package.json" -nt "node_modules/.package-lock.json" ]]; then
      log "Installing npm dependencies..."
      npm install --silent
      ok "Dependencies installed"
    fi
    npm run dev > "$SCRIPT_DIR/.ui.log" 2>&1
  ) &
  UI_PID=$!
  PIDS+=($UI_PID)

  echo -n "  Waiting for UI"
  for i in $(seq 1 60); do
    sleep 0.5
    if curl -sf "http://localhost:$UI_PORT" &>/dev/null; then
      echo ""
      ok "UI ready (pid: $UI_PID)"
      return
    fi
    echo -n "."
    if ! kill -0 "$UI_PID" 2>/dev/null; then
      echo ""
      err "Frontend crashed. Check .ui.log:\n$(tail -20 "$SCRIPT_DIR/.ui.log")"
    fi
  done
  echo ""
  warn "UI slow to start — check .ui.log"
}

# ── Open browser ─────────────────────────────────────────────
open_browser() {
  sleep 0.5
  URL="http://localhost:$UI_PORT"
  if [[ "$(uname)" == "Darwin" ]]; then
    open "$URL" 2>/dev/null || true
  elif command -v xdg-open &>/dev/null; then
    xdg-open "$URL" 2>/dev/null || true
  fi
}

# ── Main ─────────────────────────────────────────────────────
main() {
  banner
  detect_gpu
  [[ -n "${LLM_NGL:-}" ]] && NGL="$LLM_NGL"
  check_deps
  ensure_model
  echo ""
  start_llm
  start_ml_worker
  start_frontend
  echo ""
  echo -e "${BOLD}${GREEN}  ✨ MedML Forge running!${NC}"
  echo -e "  ${CYAN}Dashboard:${NC}  http://localhost:$UI_PORT"
  echo -e "  ${CYAN}LLM:${NC}       http://localhost:$LLM_PORT  (Qwen 2.5 — reasoning)"
  echo -e "  ${CYAN}ML Worker:${NC} http://127.0.0.1:$ML_WORKER_PORT  (training/scanning)"
  echo -e "  ${CYAN}GPU:${NC}       $GPU_TYPE (ngl=$NGL)"
  echo ""
  echo -e "  Press ${BOLD}Ctrl+C${NC} to stop all services"
  echo ""

  open_browser &
  wait
}

main "$@"
