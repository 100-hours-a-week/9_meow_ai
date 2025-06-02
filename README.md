# AI Text Transformation Server

SNS 포스팅/댓글/채팅을 고양이/강아지 말투로 변환하는 AI API 서버입니다.

## 기능

- 텍스트를 고양이/강아지 말투로 변환
- 다양한 감정 상태 지원 (일반, 행복, 호기심, 슬픔, 놀람, 화남)
- FastAPI 기반의 RESTful API 제공
- API 키 풀링을 통한 처리량 최적화

## 기술 스택

- Python 3.12.9
- FastAPI 0.109.2
- Uvicorn 0.27.1
- LangChain 0.1.9
- Google Generative AI 0.4.1
- Google AI Generative Language 0.4.0
- LangChain Google GenAI 0.0.11
- LangChain Core 0.1.53
- LangChain Community 0.0.38
- Pydantic 2.11.4
- Python-dotenv 1.0.1
- Jinja2 3.1.3
- SQLAlchemy 2.0.40
- Starlette 0.36.3

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
# API 키 설정 (공백 없이 정확한 형식으로 입력)
GOOGLE_API_KEYS=["key1","key2","key3"]   # 여러 키 사용 시
# 또는
GOOGLE_API_KEYS=["key1"]   # 단일 키 사용 시
```

주의: 환경 변수 값은 공백 없이 정확한 형식으로 입력해야 합니다.
- 올바른 형식: `GOOGLE_API_KEYS=["key1","key2","key3"]`
- 잘못된 형식: `GOOGLE_API_KEYS = ["key1", "key2", "key3"]`

## 실행 방법

1. PYTHONPATH 설정
```bash
# 현재 디렉토리를 Python 모듈 검색 경로에 추가
# 이는 ai_server 패키지를 Python이 찾을 수 있게 하기 위함입니다
export PYTHONPATH=$PWD:$PYTHONPATH
```

2. 서버 실행
```bash
python3 -m uvicorn ai_server.main:app --host 0.0.0.0 --port 8000 --proxy-headers
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
```str
변환된 텍스트
```

**에러 응답:**
```json
{
    "detail": "에러 메시지"
}
```

**에러 코드:**
- 400: 잘못된 요청 (잘못된 감정 상태나 동물 타입)
- 503: API 키 사용량 초과
- 500: 서버 내부 오류

## 프로젝트 구조

```
9_meow_ai/
├── .github/                    # GitHub 관련 설정
├── .venv/                      # Python 가상환경
├── ai_server/                  # 메인 서버 코드
│   ├── __init__.py            # 패키지 초기화 파일
│   ├── config.py              # 환경 설정 관리
│   ├── key_manager.py         # API 키 풀 관리
│   ├── main.py                # FastAPI 메인 애플리케이션
│   ├── comment/               # 댓글 변환 관련 모듈
│   │   ├── comment_model.py   # 댓글 변환 서비스
│   │   ├── comment_prompt.py  # 댓글 프롬프트 생성기
│   │   └── comment_schemas.py # 댓글 관련 스키마
│   └── post/                  # 포스트 변환 관련 모듈
│       ├── post_model.py      # 포스트 변환 서비스
│       ├── post_prompt.py     # 포스트 프롬프트 생성기
│       └── post_schemas.py    # 포스트 관련 스키마
├── .gitignore                 # Git 무시 파일 목록
├── README.md                  # 프로젝트 문서
└── requirements.txt           # Python 패키지 의존성
```

## 성능 최적화

### API 키 관리 (key_manager.py)
- 여러 API 키를 효율적으로 관리하여 처리량 향상
- 분당 요청 제한 관리

## 파일별 코드 설명

### ai_server/main.py
FastAPI 기반의 메인 애플리케이션 파일입니다. 주요 기능:
- FastAPI 앱 초기화 및 CORS 설정
- API 키 풀 초기화
- 루트 엔드포인트 (`/`) 구현
- 텍스트 변환 엔드포인트 (`/generate/post`) 구현
- 에러 처리 및 HTTP 예외 처리

### ai_server/schemas.py
Pydantic 모델을 정의하는 파일입니다. 주요 기능:
- `Emotion` Enum: 감정 상태 정의 (일반, 행복, 호기심, 슬픔, 놀람, 화남)
- `PostType` Enum: 동물 타입 정의 (고양이, 강아지)
- `PostRequest`: API 요청 데이터 모델
- `ErrorResponse`: API 에러 응답 모델

### ai_server/model.py
텍스트 변환 서비스의 핵심 로직을 구현한 파일입니다. 주요 기능:
- `TransformationService` 클래스: Google Gemini AI를 사용한 텍스트 변환
- LangChain과 Google Generative AI 통합
- 비동기 텍스트 변환 처리
- 프롬프트 생성 및 LLM 호출

### ai_server/key_manager.py
API 키 관리를 담당하는 파일입니다. 주요 기능:
- `APIKeyPool` 클래스: API 키 풀링 구현
- 비동기 락을 통한 동시성 제어
- 환경 변수에서 API 키 로드
- 키 순환 및 할당 관리

### ai_server/config.py
애플리케이션 설정을 관리하는 파일입니다. 주요 기능:
- `Settings` 클래스: 환경 변수 기반 설정 관리
- 캐시 설정, 배치 크기, 대기 시간 등 구성
- LRU 캐시를 통한 설정 인스턴스 관리

### ai_server/prompt.py
프롬프트 생성 및 관리를 담당하는 파일입니다. 주요 기능:
- `PromptGenerator` 클래스: 동물별, 감정별 프롬프트 템플릿 관리
- 고양이/강아지별 기본 프롬프트 정의
- 감정별 스타일 가이드 정의
- 프롬프트 템플릿 생성 및 포맷팅

### requirements.txt
프로젝트 의존성 패키지 목록을 관리하는 파일입니다.

### .env
환경 변수 설정 파일입니다. 주요 설정:
- `GOOGLE_API_KEYS`: API 키 리스트 (단일 키는 ["key1"], 여러 키는 ["key1", "key2", "key3"] 형태)
- 기타 환경 변수 설정

## CI/CD yaml 
```
name: Python application

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.12'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Run tests
      run: |
        pytest --cache-clear
```