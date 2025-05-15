from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

# 포스트 타입
class PostType(str, Enum):
    CAT = "cat"
    DOG = "dog"

class Emotion(str, Enum):
    NORMAL = "normal"
    HAPPY = "happy"
    CURIOUS = "curious"
    SAD = "sad"
    GRUMPY = "grumpy"
    ANGRY = "angry"

class PostRequest(BaseModel):
    content: str = Field(..., description="변환할 원본 텍스트")
    emotion: Emotion = Field(..., description="감정 상태")
    post_type: PostType = Field(..., description="동물 타입")

# 에러 응답 스키마
class ErrorResponse(BaseModel):
    error: str = Field(..., description="에러 메시지")
    detail: Optional[str] = Field(None, description="상세 에러 정보") 