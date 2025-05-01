from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    GOOGLE_API_KEY: str
    
    class Config:
        env_file = ".env" 
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings() -> Settings:
    return Settings() 