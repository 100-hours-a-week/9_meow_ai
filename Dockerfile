# syntax=docker/dockerfile:1

# 빌드 스테이지 - CUDA 12.1.1 (PyTorch 2.5.1 호환)
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

# Python 3.10 설치 (프로젝트 버전 3.10.15와 일치)
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-dev python3.10-venv \
    build-essential git curl \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# pip 업그레이드
RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /app
COPY requirements.txt .

# PyTorch 먼저 설치 (CUDA 12.1 호환 버전)
RUN pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 나머지 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 프로젝트 파일 복사
COPY ai_server ai_server
COPY scripts scripts

# 런타임 스테이지 - CUDA 12.1.1 런타임
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Python 3.10 런타임 설치
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-distutils \
    supervisor curl \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# pip 설치 (런타임용)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

WORKDIR /app

# 빌드 결과물 복사
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin  
COPY --from=builder /app/ai_server ai_server
COPY --from=builder /app/scripts scripts

# GPU 캐시 정리 스크립트 실행 권한 부여
RUN chmod +x /app/scripts/gpu_cache_cleanup.sh

# 환경변수 (일관성 있게 수정)
ENV PYTHONPATH=/app \
    VLLM_MODEL_PATH="haebo/meow-clovax-v2" \
    VLLM_HOST="0.0.0.0" \
    VLLM_PORT="8001" \
    VLLM_SERVED_MODEL_NAME="meow-clovax-v2" \
    VLLM_GPU_MEMORY_UTILIZATION="0.4" \
    VLLM_MAX_MODEL_LEN="512" \
    VLLM_MAX_NUM_BATCHED_TOKENS="512" \
    VLLM_MAX_NUM_SEQS="4" \
    CUDA_VISIBLE_DEVICES=0 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CUDA_MODULE_LOADING=LAZY \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64,garbage_collection_threshold:0.9 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000 8001

# 헬스체크
HEALTHCHECK --interval=60s --timeout=15s --start-period=120s --retries=2 \
    CMD curl -f http://localhost:8000/health || exit 1

# supervisord 설정 개선 (순차 실행 보장)
RUN mkdir -p /etc/supervisor/conf.d /var/log/supervisor && \
    { \
      echo '[supervisord]'; \
      echo 'nodaemon=true'; \
      echo 'logfile=/var/log/supervisor/supervisord.log'; \
      echo 'logfile_maxbytes=50MB'; \
      echo 'loglevel=info'; \
      echo '[program:cache-cleanup]'; \
      echo 'command=bash /app/scripts/gpu_cache_cleanup.sh'; \
      echo 'autostart=true'; \
      echo 'autorestart=false'; \
      echo 'startretries=3'; \
      echo 'exitcodes=0'; \
      echo 'priority=10'; \
      echo 'stdout_logfile=/var/log/supervisor/cache_cleanup.log'; \
      echo 'stderr_logfile=/var/log/supervisor/cache_cleanup_error.log'; \
      echo '[program:vllm]'; \
      echo 'command=bash scripts/direct_vllm_start.sh'; \
      echo 'autostart=true'; \
      echo 'autorestart=true'; \
      echo 'startretries=3'; \
      echo 'exitcodes=0'; \
      echo 'priority=50'; \
      echo 'stdout_logfile=/var/log/supervisor/vllm.log'; \
      echo 'stderr_logfile=/var/log/supervisor/vllm_error.log'; \
      echo '[program:fastapi]'; \
      echo 'command=python3 -u -m uvicorn ai_server.main:app --host 0.0.0.0 --port 8000 --workers 1'; \
      echo 'autostart=true'; \
      echo 'autorestart=true'; \
      echo 'startretries=5'; \
      echo 'priority=100'; \
      echo 'stdout_logfile=/var/log/supervisor/fastapi.log'; \
      echo 'stderr_logfile=/var/log/supervisor/fastapi_error.log'; \
    } > /etc/supervisor/conf.d/app.conf

CMD ["supervisord", "-c", "/etc/supervisor/conf.d/app.conf"]
