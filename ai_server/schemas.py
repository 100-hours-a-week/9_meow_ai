from pydantic import BaseModel, Field
from typing import Optional, Literal
from enum import Enum

class Emotion(str, Enum):
    NORMAL = "일반"
    HAPPY = "행복"
    CURIOUS = "호기심"
    SAD = "슬픔"
    GRUMPY = "까칠"
    ANGRY = "화남"

class PostType(str, Enum):
    CAT = "고양이"
    DOG = "강아지"

class PostRequest(BaseModel):
    content: str = Field(..., description="변환할 원본 텍스트")
    emotion: Emotion = Field(..., description="감정 상태")
    post_type: PostType = Field(..., description="동물 타입")

class PostResponse(BaseModel):
    fixed_content: str = Field(..., description="변환된 텍스트")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="에러 메시지")
    detail: Optional[str] = Field(None, description="상세 에러 정보") 