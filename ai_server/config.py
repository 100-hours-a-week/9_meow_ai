from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import BaseModel, Field
from functools import lru_cache


class InferenceConfig(BaseModel):
    """추론 파라미터 중앙 설정 - 성능 최적화"""
    post_max_tokens: int = Field(default=300, description="포스트 생성 최대 토큰 수")
    post_temperature: float = Field(default=0.3, description="포스트 생성 온도 - 일관성 제어")
    post_top_p: float = Field(default=0.9, description="포스트 생성 top_p - 다양성 제어")
    post_top_k: int = Field(default=2, description="포스트 생성 top_k - 토큰 선택 범위")
    post_stop_tokens: List[str] = Field(default=["</s>", "<|endoftext|>", "\n\n"], description="포스트 생성 중지 토큰")
    
    # 댓글 생성용 파라미터
    comment_temperature: float = Field(default=0.5, description="댓글 생성 온도")


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


# 전역 추론 설정 인스턴스
inference_config = InferenceConfig()


@lru_cache()
def get_inference_config() -> InferenceConfig:
    """추론 설정 인스턴스 반환"""
    return inference_config


@lru_cache()
def get_settings() -> Settings:
    """설정 인스턴스를 반환합니다. 캐시되어 재사용됩니다."""
    return Settings() 
