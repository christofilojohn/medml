# ╔══════════════════════════════════════════════════════════╗
# ║  MedML Forge — Multi-stage Docker Image                 ║
# ║  Supports: NVIDIA CUDA · AMD ROCm · CPU                 ║
# ║  (Apple Silicon uses start.sh, not Docker)              ║
# ╚══════════════════════════════════════════════════════════╝

# ── Stage 1: Build llama.cpp ─────────────────────────────────
ARG BUILD_TYPE=cuda

FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS builder-cuda
FROM rocm/dev-ubuntu-22.04:5.7 AS builder-rocm
FROM ubuntu:22.04 AS builder-cpu

FROM builder-${BUILD_TYPE} AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git cmake build-essential ninja-build \
    libcurl4-openssl-dev wget curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN git clone --depth 1 --branch b4887 https://github.com/ggerganov/llama.cpp .

ARG BUILD_TYPE=cuda
RUN if [ "$BUILD_TYPE" = "cuda" ]; then \
    cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -GNinja; \
    elif [ "$BUILD_TYPE" = "rocm" ]; then \
    cmake -B build -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release -GNinja; \
    else \
    cmake -B build -DCMAKE_BUILD_TYPE=Release -GNinja; \
    fi \
    && cmake --build build --target llama-server

# ── Stage 2: Build frontend ─────────────────────────────────
FROM node:20-alpine AS frontend-builder
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --silent
COPY frontend/ ./
RUN npm run build

# ── Stage 3: Runtime ────────────────────────────────────────
FROM ubuntu:22.04 AS runtime

ARG BUILD_TYPE=cuda
ENV BUILD_TYPE=${BUILD_TYPE}
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    ca-certificates gnupg curl wget \
    python3 python3-pip \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs nginx supervisor \
    && rm -rf /var/lib/apt/lists/*

# Python ML deps
RUN pip3 install --no-cache-dir \
    torch torchvision flask flask-cors \
    pandas scikit-learn pillow pyarrow openpyxl

# CUDA runtime
RUN if [ "$BUILD_TYPE" = "cuda" ]; then \
    apt-get update && apt-get install -y curl gnupg && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | gpg --dearmor -o /usr/share/keyrings/cuda-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" > /etc/apt/sources.list.d/cuda.list && \
    apt-get update && apt-get install -y libcublas-12-2 && \
    rm -rf /var/lib/apt/lists/* ; \
    fi

# Copy binaries
COPY --from=builder /build/build/bin/llama-server /usr/local/bin/llama-server
RUN chmod +x /usr/local/bin/llama-server

# Copy frontend build
COPY --from=frontend-builder /app/dist /var/www/medml

# Copy ML worker
COPY ml-worker/ /app/ml-worker/

# Nginx config
RUN cat > /etc/nginx/sites-enabled/default << 'EOF'
server {
  listen 3000;
  root /var/www/medml;
  index index.html;
  client_max_body_size 10M;

  location /v1/ {
    proxy_pass http://127.0.0.1:8080;
    proxy_set_header Host $host;
    proxy_read_timeout 300s;
    proxy_buffering off;
  }

  location /scan { proxy_pass http://127.0.0.1:8081; proxy_read_timeout 120s; }
  location /train { proxy_pass http://127.0.0.1:8081; proxy_read_timeout 600s; proxy_buffering off; }
  location /dataset { proxy_pass http://127.0.0.1:8081; }
  location /cleanup { proxy_pass http://127.0.0.1:8081; }
  location /reason { proxy_pass http://127.0.0.1:8081; proxy_read_timeout 120s; }
  location /health { proxy_pass http://127.0.0.1:8081; }

  location / {
    try_files $uri $uri/ /index.html;
  }
}
EOF

COPY docker/supervisord.conf /etc/supervisor/supervisord.conf
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 3000

VOLUME ["/models", "/data"]
ENV MODEL_PATH=/models/Qwen2.5-3B-Instruct-Q4_K_M.gguf
ENV LLM_CONTEXT=4096
ENV LLM_THREADS=4
ENV LLM_NGL=999
ENV ML_WORKER_PORT=8081

RUN useradd --system --no-create-home --shell /sbin/nologin medml \
    && sed -i 's/^user www-data;/user medml;/' /etc/nginx/nginx.conf \
    && mkdir -p /var/lib/nginx/body /var/lib/nginx/proxy \
    && chown -R medml:medml /var/www/medml /var/log /var/lib/nginx /app \
    && touch /run/nginx.pid && chown medml:medml /run/nginx.pid
USER medml

ENTRYPOINT ["/entrypoint.sh"]
