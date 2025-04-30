haebo/
├── .venv/                      # Python 가상환경
├── ai_server/                 # AI 서버 관련 로직 모듈
│   ├── __init__.py
│   ├── batch_processor.py     # 일괄 처리 관련 로직
│   ├── cache.py               # 캐싱 처리 로직
│   ├── config.py              # 설정 값 관리
│   ├── key_pool.py            # API 키 풀 관리 등 관련 기능
│   ├── key_pool_test.py       # API 키 풀 실행 테스트 코드 
│   ├── main.py                # FastAPI 진입점 또는 실행 로직
│   ├── schemas.py             # Pydantic 기반 요청/응답 데이터 구조 정의
│   ├── templates.py           # 프롬프트 또는 변환 템플릿 정의
│   └── transformation.py      # 핵심 변환 로직 (예: 말투 변환, 스타일 변화)
├── .gitignore
├── Feedback.txt               # 피드백 로그 또는 설명 파일로 추정
├── install.sh                 # 초기 환경설정 스크립트
├── README.md                  # 프로젝트 설명
└── requirements.txt           # Python 의존성 목록

## API 키 풀링 (key_pool.py)
> 목적: 
- 여러 API 키를 효율적으로 관리하여 처리량 향상

> 주요 기능:
- API 키 사용량 추적
- 요청 제한(rate limiting) 관리
- 키 순환(rotation) 처리

> 작동 방식:
- 각 키의 사용 상태와 요청 횟수 추적
- 분당 최대 요청 수 제한 적용
- 가장 적게 사용된 키 우선 할당
- 비동기 락을 통한 동시성 제어

## 캐싱 레이어 (cache.py)
> 목적
- 반복되는 요청에 대한 응답 시간 단축

> 주요 기능:
- LRU(Least Recently Used) 캐싱
- TTL(Time To Live) 기반 만료
- 동시성 안전한 캐시 접근

> 작동 방식:
- 요청 파라미터를 기반으로 고유 키 생성
- 캐시 용량 제한 관리
- 만료된 항목 자동 제거
- 비동기 락을 통한 스레드 안전성 보장

## 배치 처리 (batch_processor.py)
> 목적: 
- 여러 요청을 그룹화하여 효율적으로 처리

> 주요 기능:
- 동적 배치 크기 조정
- 최대 대기 시간 설정
- 병렬 요청 처리

> 작동 방식:
- 요청을 배치에 추가하고 Future 반환
- 배치 크기 또는 대기 시간 도달 시 처리
- 비동기 태스크로 병렬 처리
- 결과를 개별 Future에 전달