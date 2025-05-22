# syntax=docker/dockerfile:1

# 1) Build 단계: Python 종속성 설치
FROM python:3.11-slim AS builder
WORKDIR /app

# 캐시 효율을 위한 requirements 분리
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY ai_server ai_server
COPY config.py key_manager.py ./

# 2) Runtime 단계: 컨테이너 경량화
FROM python:3.11-slim
WORKDIR /app

# 빌드 스테이지에서 설치된 패키지 복사
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /app /app

# FastAPI 기본 포트
EXPOSE 8000

# 컨테이너 시작 시 Uvicorn으로 앱 실행
CMD ["uvicorn", "ai_server.main:app", "--host", "0.0.0.0", "--port", "8000"]
