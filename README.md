# AI Text Transformation Server

SNS 포스팅/댓글/채팅을 고양이/강아지 말투로 변환하는 AI API 서버입니다.

## 기능

- 텍스트를 고양이/강아지 말투로 변환
- 다양한 감정 상태 지원 (일반, 행복, 호기심, 슬픔, 놀람, 화남)
- FastAPI 기반의 RESTful API 제공
- API 키 풀링을 통한 처리량 최적화
- 캐싱을 통한 응답 시간 단축
- 배치 처리를 통한 효율적인 요청 처리

## 기술 스택

- Python 3.8+
- FastAPI
- LangChain
- Google Generative AI (Gemini)
- Pydantic
- asyncio

## 설치 방법

1. 저장소 클론
```bash
git clone https://github.com/100-hours-a-week/9_meow_ai.git
cd <working directory>
```

2. 가상환경 생성 및 활성화
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 또는
.venv\Scripts\activate  # Windows
```

3. 의존성 설치
```bash
pip install -r requirements.txt
```

4. 환경 변수 설정
`.env` 파일을 생성하고 다음 내용을 추가합니다:
```
GOOGLE_API_KEYS=["key1", "key2", "key3"]  # 여러 개의 API 키를 리스트로 설정
```

## 실행 방법

1. PYTHONPATH 설정
```bash
export PYTHONPATH=/path/to/haebo:$PYTHONPATH
```

2. 서버 실행
```bash
uvicorn ai_server.main:app --reload
```

3. API 문서 확인
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API 사용 방법

### 텍스트 변환 API

**엔드포인트:** `POST /generate/post`

**요청 본문:**
```json
{
    "content": "변환할 원본 텍스트",
    "emotion": "일반",  // "일반", "행복", "호기심", "슬픔", "놀람", "화남" 중 선택
    "post_type": "고양이"  // "고양이", "강아지" 중 선택
}
```

**응답:**
```json
{
    "fixed_content": "변환된 텍스트"
}
```

**에러 응답:**
```json
{
    "error": "에러 메시지",
    "detail": "상세 에러 정보"
}
```

**에러 코드:**
- 400: 잘못된 요청 (잘못된 감정 상태나 동물 타입)
- 429: 요청 제한 초과
- 500: 서버 내부 오류

## 프로젝트 구조

```
root/
├── ai_server/                    # 메인 서버 코드
│   ├── main.py                  # FastAPI 메인 애플리케이션
│   ├── config.py                # 환경 설정 및 구성
│   ├── schemas.py               # Pydantic 데이터 모델
│   ├── model.py                 # 텍스트 변환 서비스
│   └── prompt.py                # 프롬프트 생성 및 관리
├── core/                        # 핵심 기능 모듈
│   ├── key_pool.py             # API 키 풀링 및 관리
│   ├── cache.py                # LRU 캐시 구현
│   └── batch.py                # 배치 요청 처리
├── test/                       # 테스트 코드
│   └── key_pool_test.py        # API 키 풀 테스트
├── requirements.txt            # 의존성 목록
└── README.md                   # 프로젝트 문서
```

## 성능 최적화

### API 키 풀링 (key_pool.py)
- 여러 API 키를 효율적으로 관리하여 처리량 향상
- 분당 요청 제한 관리
- 키 사용량 기반 자동 순환
- 비동기 락을 통한 동시성 제어

### 캐싱 레이어 (cache.py)
- LRU 캐시로 반복 요청 처리 시간 단축
- TTL 기반 캐시 항목 만료
- 동시성 안전한 캐시 접근
- 메모리 사용량 제한

### 배치 처리 (batch.py)
- 동적 배치 크기 조정 (기본값: 10)
- 최대 대기 시간 설정 (기본값: 2초)
- 비동기 병렬 처리
- 개별 요청 추적 및 결과 전달

## 개발 가이드

### 코드 컨벤션
- Black 코드 포맷터 사용
- Pylint 린터 사용
- 타입 힌트 사용

### 테스트
```bash
python3 -m uvicorn ai_server.main:app --reload --port 8002
```

## 라이센스

MIT License
