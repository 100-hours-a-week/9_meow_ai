from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class Emotion(str, Enum):
    NORMAL = "normal"
    HAPPY = "happy"
    CURIOUS = "curious"
    SAD = "sad"
    GRUMPY = "grumpy"
    ANGRY = "angry"

class PostType(str, Enum):
    CAT = "cat"
    DOG = "dog"

class PostRequest(BaseModel):
    content: str = Field(..., description="변환할 원본 텍스트")
    emotion: Emotion = Field(..., description="감정 상태")
    post_type: PostType = Field(..., description="동물 타입")

class CommentRequest(BaseModel):
    text: str = Field(..., description="변환할 댓글 원본 텍스트")
    post_type: str = Field(..., description="동물 타입 (고양이 또는 강아지)")
    
class ErrorResponse(BaseModel):
    error: str = Field(..., description="에러 메시지")
    detail: Optional[str] = Field(None, description="상세 에러 정보") 