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