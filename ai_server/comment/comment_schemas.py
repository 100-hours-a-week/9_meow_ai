from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

# 댓글 타입 
class CommentType(str, Enum):
    CAT = "cat"
    DOG = "dog"

class CommentRequest(BaseModel):
    content: str = Field(..., description="변환할 댓글 원본 텍스트")
    post_type: CommentType = Field(..., description="동물 타입 (고양이 또는 강아지)")

# 에러 응답 스키마
class ErrorResponse(BaseModel):
    error: str = Field(..., description="에러 메시지")
    detail: Optional[str] = Field(None, description="상세 에러 정보") 