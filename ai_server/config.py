from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    """애플리케이션 설정을 관리하는 클래스
    
    환경 변수와 .env 파일에서 설정을 자동으로 로드합니다.
    """
    # API 키 설정
    GOOGLE_API_KEYS: List[str]  # Google API 키 목록

    
    # 허깅페이스 인증 토큰
    HUGGINGFACE_TOKEN: Optional[str] = None  # 허깅페이스 API 토큰
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """설정 인스턴스를 반환합니다. 캐시되어 재사용됩니다."""
    return Settings() 
