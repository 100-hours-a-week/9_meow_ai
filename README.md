# 9_team_meow_ai
# AI Text Transformation Server

SNS 포스팅/댓글/채팅을 고양이 말투로 변환하는 AI API 서버입니다.

## 기능

- 텍스트를 고양이 말투로 변환
- 다양한 감정 상태 지원 (일반, 행복, 호기심, 슬픔, 까칠, 화남)
- FastAPI 기반의 RESTful API 제공

## 기술 스택

- Python 3.8+
- FastAPI
- LangChain
- Google Generative AI (Gemini)
- Pydantic

## 설치 방법

1. 저장소 클론
```bash
git clone https://github.com/your-username/ai-text-transformation.git
cd ai-text-transformation
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
GOOGLE_API_KEY=your_google_api_key
```

## 실행 방법

1. 서버 실행
```bash
uvicorn ai_server.main:app --reload
```

2. API 문서 확인
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API 사용 방법

### 텍스트 변환 API

**엔드포인트:** `POST /generate/post`

**요청 본문:**
```json
{
    "content": "변환할 원본 텍스트",
    "emotion": "일반",  // "일반", "행복", "호기심", "슬픔", "까칠", "화남" 중 선택
    "post_type": "고양이"
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

## 개발 가이드

### 프로젝트 구조
```
ai_server/
├── __init__.py
├── main.py           # FastAPI 서버 및 엔드포인트
├── config.py         # 환경 설정
├── schemas.py        # Pydantic 모델
├── templates.py      # 프롬프트 템플릿
└── transformation.py # 변환 서비스
```

### 코드 컨벤션
- Black 코드 포맷터 사용
- Pylint 린터 사용
- 타입 힌트 사용

## 라이센스

MIT License
