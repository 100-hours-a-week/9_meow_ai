# AI Text Transformation Server

http://34.64.213.48:8000
SNS 포스팅/댓글/채팅을 고양이/강아지 말투로 변환하는 AI API 서버입니다.

## 🎯 기능

- 텍스트를 고양이/강아지 말투로 변환
- 다양한 감정 상태 지원 (일반, 행복, 호기심, 슬픔, 놀람, 화남)
- FastAPI 기반의 RESTful API 제공
- vLLM을 이용한 고성능 로컬 AI 추론

## 📦 설치 방법

1. **저장소 클론**
```bash
git clone https://github.com/100-hours-a-week/9_meow_ai.git
cd 9_meow_ai
```

2. **가상환경 생성 및 활성화**
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
```

3. **의존성 설치**
```bash
pip install -r requirements.txt
```

## 🚀 서버 실행 방법

### 1단계: vLLM 서버 시작 (터미널 1)
```bash
# 스크립트로 간단 실행 - 자동으로 모델 다운로드 및 LoRA 서빙
python scripts/start_vllm.py
```

**동작 과정:**
- 🔽 허깅페이스에서 베이스 모델 `naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B` 다운로드
- 🔽 허깅페이스에서 LoRA 어댑터 `haebo/Meow-HyperCLOVAX-1.5B_LoRA_fp16_0527` 다운로드  
- 🔧 베이스 모델 + LoRA 어댑터를 결합하여 vLLM 서버 시작
- 🚀 서버 실행: http://localhost:8001

### 2단계: FastAPI 서버 시작 (새 터미널 2)
```bash
cd 9_meow_ai
source .venv/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH
python -m uvicorn ai_server.main:app --host 0.0.0.0 --port 8000
```

### 3단계: 테스트
브라우저에서 http://localhost:8000/docs 접속하여 API 테스트

## 📖 API 사용 방법

**엔드포인트:** `POST /generate/post`

**요청 본문:**
```json
{
    "content": "변환할 원본 텍스트",
    "emotion": "일반",
    "post_type": "고양이"
}
```

**cURL 예제:**
```bash
curl -X POST "http://localhost:8000/generate/post" \
     -H "Content-Type: application/json" \
     -d '{
       "content": "오늘 날씨가 정말 좋네요!",
       "emotion": "행복",
       "post_type": "고양이"
     }'
```

## 🐛 문제 해결

```bash
# GPU 메모리 부족 시
python scripts/start_vllm.py --gpu-memory-utilization 0.6

# 포트가 이미 사용 중인 경우
python scripts/start_vllm.py --port 8002
```

# 9_meow_ai

AI 서버 프로젝트 - 고양이/강아지 말투 변환 서비스

## 🚀 주요 기능

- **고양이/강아지 말투 변환**: 일반 텍스트를 귀여운 동물 말투로 변환
- **다중 모델 지원**: LoRA와 풀 파인튜닝 모델 자동 감지
- **동적 모델 감지**: 실행 중인 vLLM 서버 모델을 자동으로 인식
- **한국어 최적화**: 200자 입력 → 400자 출력 최적화 설정

## 📋 지원 모델 (모델명 변경 가능)

| 모델 | 타입 | 용도 | GPU 메모리 |
|------|------|------|-----------|
| `haebo/Meow-HyperCLOVAX-1.5B_LoRA_fp16_0619i` | LoRA | 어댑터 기반 | 60% (~14GB) |
| `haebo/Meow-HyperCLOVAX-1.5B_FullFT_fp32_0619i` | 풀 파인튜닝 | 완전 학습 | 50% (~11.5GB) |

## 🛠️ 설치 및 설정

### 필요 조건
- Python 3.10+
- NVIDIA GPU (CUDA 12.4+)
- vLLM 라이브러리

### 환경 설정
```bash
# 가상환경 활성화
source /srv/shared/timmy/.venv/bin/activate

# 프로젝트 디렉토리로 이동
cd /srv/shared/timmy/9_meow_ai

# PYTHONPATH 설정
export PYTHONPATH=$PWD:$PYTHONPATH
```

## 🎯 간단 사용법

### 1단계: vLLM 서버 시작
```bash
# LoRA 모델로 시작
python scripts/model_manager.py switch haebo/Meow-HyperCLOVAX-1.5B_LoRA_fp16_0619i

# 또는 풀 파인튜닝 모델로 시작
python scripts/model_manager.py switch haebo/Meow-HyperCLOVAX-1.5B_FullFT_fp32_0619i
```

### 2단계: FastAPI 서버 시작 (새 터미널)
```bash
# 환경 설정
source /srv/shared/timmy/.venv/bin/activate
cd /srv/shared/timmy/9_meow_ai
export PYTHONPATH=$PWD:$PYTHONPATH

# 서버 시작 (자동으로 실행 중인 모델 감지)
python -m uvicorn ai_server.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3단계: API 테스트
```bash
# 고양이 말투 변환
curl -X POST "http://localhost:8000/generate/post" \
     -H "Content-Type: application/json" \
     -d '{
       "content": "오늘 날씨가 정말 좋네요!",
       "emotion": "happy",
       "post_type": "cat"
     }'

# 강아지 말투 변환
curl -X POST "http://localhost:8000/generate/post" \
     -H "Content-Type: application/json" \
     -d '{
       "content": "배가 고파요...",
       "emotion": "sad",
       "post_type": "dog"
     }'
```

## 📊 API 엔드포인트

### POST /generate/post
**요청:**
```json
{
  "content": "변환할 텍스트",
  "emotion": "happy|sad|normal|angry|curious",
  "post_type": "cat|dog"
}
```

**응답:**
```json
{
  "status_code": 200,
  "message": "Successfully transformed text",
  "data": "변환된 고양이/강아지 말투 텍스트"
}
```

## 🔧 고급 사용법

### 모델 관리
```bash
# 모델 목록 확인
python scripts/model_manager.py list

# 서버 상태 확인
python scripts/model_manager.py status

# 모델 전환 (서버 재시작)
python scripts/model_manager.py switch <모델_이름>

# 서버 중지/재시작
python scripts/model_manager.py stop
python scripts/model_manager.py restart
```

### 환경 변수 설정 (선택사항)
```bash
# 특정 모델 강제 지정
export VLLM_ACTIVE_MODEL="haebo/Meow-HyperCLOVAX-1.5B_FullFT_fp32_0619i"
```

## 🎉 핵심 특징

- **자동 감지**: FastAPI 서버가 실행 중인 vLLM 모델을 자동으로 감지
- **원스텝 실행**: 모델 전환과 서버 시작을 한 번에 처리
- **안정성 우선**: GPU 메모리 여유 확보로 OOM 방지
- **한국어 최적화**: 토큰 계산 및 동시성 한국어 기준 설정

## 🚨 문제 해결

### 일반적인 오류
1. **GPU 메모리 부족**: 다른 GPU 프로세스 종료 후 재시작
2. **모델 로딩 실패**: 허깅페이스 토큰 및 모델 접근 권한 확인
3. **API 404 오류**: FastAPI 서버 재시작으로 모델 설정 동기화

### 로그 확인
```bash
# 서버 상태 실시간 확인
python scripts/model_manager.py status

# 모델 감지 확인
# FastAPI 서버 로그에서 🎯, 🌍, 🔧 마크 확인
```

## 📝 참고사항

- **첫 실행**: 모델 다운로드로 5-10분 소요 (이후 30초-1분)
- **동시 사용자**: LoRA 8명, 풀FT 12명 기준 최적화
- **메모리 사용**: NVIDIA L4 (23GB) 기준 안정적 설정
- **API 주소**: http://localhost:8000/docs (Swagger UI)