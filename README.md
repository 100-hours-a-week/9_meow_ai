# AI Text Transformation Server

SNS 포스팅/댓글/채팅을 고양이/강아지 말투로 변환하는 AI API 서버입니다.

## 기능

- 텍스트를 고양이/강아지 말투로 변환
- 다양한 감정 상태 지원 (일반, 행복, 호기심, 슬픔, 놀람, 화남)
- FastAPI 기반의 RESTful API 제공
- API 키 풀링을 통한 처리량 최적화

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
cd 9_meow_ai
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
GOOGLE_API_KEYS=["key1"]   # 한 개의 API 키를 리스트로 설정
GOOGLE_API_KEYS_list=["key1", "key2", "key3"]   # 여러 개의 API 키를 리스트로 설정
```

## 실행 방법

1. PYTHONPATH 설정
```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```

2. 서버 실행
```bash
python3 -m uvicorn ai_server.main:app --host 0.0.0.0 --port 8000 --proxy-headers --reload
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
- 503: API 키 사용량 초과
- 500: 서버 내부 오류

## 프로젝트 구조

```
9_meow_ai/
├── ai_server/                    # 메인 서버 코드
│   ├── __init__.py             # 패키지 초기화
│   ├── main.py                 # FastAPI 메인 애플리케이션
│   ├── config.py               # 환경 설정 및 구성
│   ├── schemas.py              # Pydantic 데이터 모델
│   ├── model.py                # 텍스트 변환 서비스
│   ├── prompt.py               # 프롬프트 생성 및 관리
│   └── key_manager.py         # API 키 관리 및 풀링
├── requirements.txt            # 의존성 목록
└── README.md                   # 프로젝트 문서
```

## 성능 최적화

### API 키 관리 (key_manager.py)
- 여러 API 키를 효율적으로 관리하여 처리량 향상
- 분당 요청 제한 관리
- 키 사용량 기반 자동 순환
- 비동기 락을 통한 동시성 제어
- API 키 풀 초기화 및 관리
- 키 사용 가능 여부 확인 및 할당
- 키 사용 완료 후 반환

## 개발 가이드

### 코드 컨벤션
- Black 코드 포맷터 사용
- Pylint 린터 사용
- 타입 힌트 사용