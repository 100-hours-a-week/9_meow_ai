from typing import List
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """애플리케이션 설정을 관리하는 클래스
    
    환경 변수와 .env 파일에서 설정을 자동으로 로드합니다.
    """
    # API 키 설정
    GOOGLE_API_KEYS: List[str]  # Google API 키 목록
    
    # 배치 처리 설정
    MAX_BATCH_SIZE: int = 10  # 최대 배치 크기
    MAX_WAIT_TIME: float = 2.0  # 최대 대기 시간 (초)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings() -> Settings:
    """설정 인스턴스를 반환합니다. 캐시되어 재사용됩니다."""
    return Settings() 