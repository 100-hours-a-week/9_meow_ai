from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional

# 댓글 타입 
class CommentType(str, Enum):
    CAT = "cat"
    DOG = "dog"

# 감정 상태 Enum 추가
class CommentEmotion(str, Enum):
    NORMAL = "normal"

class CommentRequest(BaseModel):
    content: str = Field(..., description="변환할 댓글 원본 텍스트")
    post_type: CommentType = Field(..., description="동물 타입 (고양이 또는 강아지)")
    emotion: CommentEmotion = Field(default=CommentEmotion.NORMAL, description="감정 상태 (기본값: normal)")

class CommentResponse(BaseModel):
    status_code: int = Field(..., description="응답 상태 코드")
    message: str = Field(..., description="응답 메시지")
    data: Optional[str] = Field(None, description="변환된 텍스트, 에러 발생 시 None")