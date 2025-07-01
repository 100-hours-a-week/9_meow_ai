# syntax=docker/dockerfile:1

# 빌드 스테이지 - CUDA 12.1.1 (PyTorch 2.5.1 호환)
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

# Python 3.10 설치 (프로젝트 버전 3.10.15와 일치)
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-dev python3.10-venv \
    build-essential git curl \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# pip 업그레이드
RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /app
COPY requirements.txt .

# PyTorch 먼저 설치 (CUDA 12.1 호환 버전)
RUN pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 나머지 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

COPY ai_server ai_server

# 런타임 스테이지 - CUDA 12.1.1 런타임
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Python 3.10 런타임 설치
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-distutils \
    supervisor curl \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# pip 설치 (런타임용)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

WORKDIR /app

# 빌드 결과물 복사
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin  
COPY --from=builder /app/ai_server ai_server

# supervisord 설정
RUN mkdir -p /etc/supervisor/conf.d && \
    echo '[supervisord]' > /etc/supervisor/conf.d/app.conf && \
    echo 'nodaemon=true' >> /etc/supervisor/conf.d/app.conf && \
    echo 'silent=true' >> /etc/supervisor/conf.d/app.conf && \
    echo 'logfile=/dev/null' >> /etc/supervisor/conf.d/app.conf && \
    echo 'logfile_maxbytes=0' >> /etc/supervisor/conf.d/app.conf && \
    echo '' >> /etc/supervisor/conf.d/app.conf && \
    echo '[program:vllm]' >> /etc/supervisor/conf.d/app.conf && \
    echo 'command=python3 -m ai_server.external.vLLM.server.vllm_launcher --action start' >> /etc/supervisor/conf.d/app.conf && \
    echo 'autostart=true' >> /etc/supervisor/conf.d/app.conf && \
    echo 'autorestart=true' >> /etc/supervisor/conf.d/app.conf && \
    echo 'stdout_logfile=/dev/null' >> /etc/supervisor/conf.d/app.conf && \
    echo 'stderr_logfile=/dev/null' >> /etc/supervisor/conf.d/app.conf && \
    echo 'redirect_stderr=false' >> /etc/supervisor/conf.d/app.conf && \
    echo '' >> /etc/supervisor/conf.d/app.conf && \
    echo '[program:fastapi]' >> /etc/supervisor/conf.d/app.conf && \
    echo 'command=python3 -m uvicorn ai_server.main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level error' >> /etc/supervisor/conf.d/app.conf && \
    echo 'autostart=true' >> /etc/supervisor/conf.d/app.conf && \
    echo 'autorestart=true' >> /etc/supervisor/conf.d/app.conf && \
    echo 'stdout_logfile=/dev/null' >> /etc/supervisor/conf.d/app.conf && \
    echo 'stderr_logfile=/dev/null' >> /etc/supervisor/conf.d/app.conf && \
    echo 'redirect_stderr=false' >> /etc/supervisor/conf.d/app.conf && \
    echo 'depends_on=vllm' >> /etc/supervisor/conf.d/app.conf

# 환경변수
ENV PYTHONPATH=/app \
    VLLM_MODEL_PATH="haebo/Meow-HyperCLOVAX-1.5B_SFT-FFT_fp32_0629fe" \
    CUDA_VISIBLE_DEVICES=0 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CUDA_MODULE_LOADING=LAZY \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000 8001

# 헬스체크
HEALTHCHECK --interval=30s --timeout=15s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health && curl -f http://localhost:8001/v1/models

CMD ["supervisord", "-c", "/etc/supervisor/conf.d/app.conf"]
