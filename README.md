# AI Text Transformation Server

http://34.64.213.48:8000
SNS 포스팅/댓글/채팅을 고양이/강아지 말투로 변환하는 AI API 서버입니다.

## 기능

- 텍스트를 고양이/강아지 말투로 변환
- 다양한 감정 상태 지원 (normal, happy, curious, sad, grumpy, angry)
- FastAPI 기반의 RESTful API 제공
- vLLM을 이용한 고성능 로컬 AI 추론

## 설치 방법

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

## 서버 실행 방법

### 1단계: vLLM 서버 시작 (터미널 1)

**현재 모델 정보 확인:**
```bash
python scripts/model_manager.py info
```

**서버 시작:**
```bash
python scripts/model_manager.py start
```

**동작 과정:**
- 허깅페이스에서 모델 자동 다운로드 (첫 실행 시)
- `haebo/Meow-HyperCLOVAX-1.5B_SFT-FFT_fp32_0629fe` 모델 로드
- 서버 실행: http://localhost:8001

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

## 🔧 서버 관리

### 기본 명령어
```bash
# 현재 모델 정보 확인
python scripts/model_manager.py info

# 서버 상태 확인
python scripts/model_manager.py status

# 서버 시작/중지/재시작
python scripts/model_manager.py start
python scripts/model_manager.py stop
python scripts/model_manager.py restart
```

## 📋 환경변수 설정

모델 경로를 변경하려면 환경변수를 설정할 수 있습니다:

```bash
export VLLM_MODEL_PATH="다른모델경로"
python scripts/model_manager.py start
```

## 🐳 Docker 실행

```bash
# Docker Compose로 실행
docker-compose up -d

# 또는 개별 빌드 및 실행
docker build -t meow-ai .
docker run -p 8000:8000 -p 8001:8001 --gpus all meow-ai
```

## 🧪 테스트

```bash
# 단위 테스트 실행
pytest tests/

# 특정 테스트 실행
pytest tests/unit_test.py::test_post_transformation_service
```

## 📝 모델 정보

- **모델**: haebo/Meow-HyperCLOVAX-1.5B_SFT-FFT_fp32_0629fe
- **타입**: 풀 파인튜닝 모델
- **용도**: 한국어 텍스트를 고양이/강아지 말투로 변환
- **기반**: HyperCLOVA-X 1.5B

## 🔧 성능 최적화

### GPU 메모리 설정
- GPU 메모리 사용률: 60%
- 최대 모델 길이: 1536 토큰
- 동시 처리 시퀀스: 12개

### 환경변수로 조정
```bash
export VLLM_GPU_MEMORY_UTILIZATION=0.6
export VLLM_MAX_MODEL_LEN=1536
export VLLM_MAX_NUM_SEQS=12
```
