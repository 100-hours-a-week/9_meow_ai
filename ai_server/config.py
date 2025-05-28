from typing import List, Optional
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """애플리케이션 설정을 관리하는 클래스
    
    환경 변수와 .env 파일에서 설정을 자동으로 로드합니다.
    """
    # API 키 설정
    GOOGLE_API_KEYS: List[str]  # Google API 키 목록
    
    # 허깅페이스 모델 설정
    HUGGINGFACE_MODEL_PATH: str  # 허깅페이스 모델 경로
    HUGGINGFACE_TOKEN: Optional[str] = None  # 허깅페이스 API 토큰
    
    # 모델 생성 파라미터
    MODEL_MAX_LENGTH: int  # 최대 생성 길이 (기존 호환성 유지)
    MODEL_MAX_NEW_TOKENS: int = 256  # 최대 생성 토큰 수 (기본값 256)
    MODEL_TEMPERATURE: float  # 생성 온도
    MODEL_TOP_P: float  # 누적 확률 임계값
    
    # 모델 로딩 설정
    PRELOAD_MODELS: bool = True  # 서버 시작 시 모델 사전 로드 여부 (기본값: True)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings() -> Settings:
    """설정 인스턴스를 반환합니다. 캐시되어 재사용됩니다."""
    return Settings() 