# syntax=docker/dockerfile:1

# 1) Build 단계: Python 패키지 설치
FROM python:3.11-slim AS builder
WORKDIR /app

# 의존성 설치
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 코드 전체 복사
COPY ai_server ai_server

# 2) Runtime 단계: 경량 컨테이너
FROM python:3.11-slim
WORKDIR /app

# 빌드 스테이지에서 설치된 패키지 가져오기
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
# 앱 코드 복사
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/ai_server ai_server

# FastAPI 기본 포트
EXPOSE 8000

# 컨테이너 시작 시 Uvicorn 실행
CMD ["python3", "-m", "uvicorn", "ai_server.main:app", "--host", "0.0.0.0", "--port", "8000"]
