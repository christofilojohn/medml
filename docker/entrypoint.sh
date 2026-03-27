#!/bin/bash
set -e

echo "╔═════════════════════════════════════════════╗"
echo "║      🧬  MedML Forge                        ║"
echo "║      Privacy-First Clinical AI Pipeline      ║"
echo "╚═════════════════════════════════════════════╝"

# Start ML Worker
echo "[medml] Starting ML Worker..."
ML_WORKER_PORT=${ML_WORKER_PORT:-8081} \
LLM_PORT=8080 \
GPU_TYPE=${BUILD_TYPE:-cpu} \
python3 /app/ml-worker/server.py &

# Start LLM
echo "[medml] Starting Qwen LLM..."
llama-server \
  -m "${MODEL_PATH}" \
  -c "${LLM_CONTEXT:-4096}" \
  -ngl "${LLM_NGL:-999}" \
  -t "${LLM_THREADS:-4}" \
  --port 8080 \
  --host 127.0.0.1 \
  --log-disable &

# Start nginx
echo "[medml] Starting nginx..."
exec nginx -g 'daemon off;'
