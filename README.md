# AI Text Transformation Server

http://34.64.213.48:8000
SNS 포스팅/댓글/채팅을 고양이/강아지 말투로 변환하는 AI API 서버입니다.

## 🎯 기능

- 텍스트를 고양이/강아지 말투로 변환
- 다양한 감정 상태 지원 (normal, happy, curious, sad, grumpy, angry)
- FastAPI 기반의 RESTful API 제공
- vLLM을 이용한 고성능 로컬 AI 추론
- LoRA와 풀 파인튜닝 모델 자동 감지 및 전환

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

**지원되는 모델 목록 확인:**
```bash
python scripts/model_manager.py list
```

**모델 전환 및 서버 시작:**
```bash
# LoRA 모델로 시작 (권장)
python scripts/model_manager.py switch haebo/Meow-HyperCLOVAX-1.5B_LoRA_fp16_0619i

# 또는 풀 파인튜닝 모델로 시작
python scripts/model_manager.py switch haebo/Meow-HyperCLOVAX-1.5B_FullFT_fp32_0619i
```

**동작 과정:**
- 🔽 허깅페이스에서 모델 자동 다운로드 (첫 실행 시)
- 🔧 모델을 vLLM 서버로 서빙 시작
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

### POST /generate/post (포스트 변환)

**요청 본문:**
```json
{
    "content": "변환할 원본 텍스트",
    "emotion": "happy",
    "post_type": "cat"
}
```

**지원되는 값:**
- **emotion**: `normal`, `happy`, `curious`, `sad`, `grumpy`, `angry`
- **post_type**: `cat`, `dog`

**cURL 예제:**
```bash
# 고양이 말투 변환 (행복)
curl -X POST "http://localhost:8000/generate/post" \
     -H "Content-Type: application/json" \
     -d '{
       "content": "오늘 날씨가 정말 좋네요!",
       "emotion": "happy",
       "post_type": "cat"
     }'

# 강아지 말투 변환 (슬픔)
curl -X POST "http://localhost:8000/generate/post" \
     -H "Content-Type: application/json" \
     -d '{
       "content": "배가 고파요...",
       "emotion": "sad",
       "post_type": "dog"
     }'
```

### POST /generate/comment (댓글 변환)

**요청 본문:**
```json
{
    "content": "변환할 댓글 원본 텍스트",
    "post_type": "cat"
}
```

**cURL 예제:**
```bash
curl -X POST "http://localhost:8000/generate/comment" \
     -H "Content-Type: application/json" \
     -d '{
       "content": "정말 재미있네요!",
       "post_type": "cat"
     }'
```

## 🔧 모델 관리

### 기본 명령어
```bash
# 지원되는 모델 목록 확인
python scripts/model_manager.py list

# 서버 상태 확인
python scripts/model_manager.py status

# 서버 시작/중지/재시작
python scripts/model_manager.py start
python scripts/model_manager.py stop
python scripts/model_manager.py restart

# 모델 타입 감지
python scripts/model_manager.py detect [모델경로]
```

### 지원 모델

| 모델 | 타입 | 용도 | GPU 메모리 |
|------|------|------|-----------|
| `haebo/Meow-HyperCLOVAX-1.5B_LoRA_fp16_0619i` | LoRA | 어댑터 기반 | ~60% |
| `haebo/Meow-HyperCLOVAX-1.5B_FullFT_fp32_0619i` | 풀 파인튜닝 | 완전 학습 | ~50% |

## 🐛 문제 해결

```bash
# GPU 메모리 부족 시
python scripts/model_manager.py switch [모델명] --gpu-memory-utilization 0.6

# 포트가 이미 사용 중인 경우
python scripts/model_manager.py switch [모델명] --port 8002

# 서버 재시작
python scripts/model_manager.py restart
```

## 💡 개발 팁

- **첫 실행**: 모델 다운로드로 인해 시간이 오래 걸릴 수 있습니다
- **모델 저장**: 다운로드된 모델은 `./models/` 디렉토리에 캐시됩니다
- **GPU 메모리**: LoRA 모델이 메모리 효율적입니다
- **API 문서**: http://localhost:8000/docs 에서 실시간 테스트 가능