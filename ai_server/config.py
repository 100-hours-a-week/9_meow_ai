from typing import List
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    GOOGLE_API_KEYS: List[str]
    MAX_BATCH_SIZE: int = 10
    MAX_WAIT_TIME: float = 2.0
    CACHE_CAPACITY: int = 1000
    CACHE_TTL_SECONDS: int = 3600
    MAX_REQUESTS_PER_MIN: int = 60

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings() -> Settings:
    return Settings() 