from pydantic import BaseModel, Field
from enum import Enum

# 채팅 동물 타입
class ChatAnimalType(str, Enum):
    CAT = "cat"
    DOG = "dog"
    HAMSTER = "hamster"
    MONKEY = "monkey"
    RACCOON = "raccoon"

class ChatRequest(BaseModel):
    text: str = Field(..., description="변환할 원본 텍스트")
    post_type: ChatAnimalType = Field(..., description="동물 타입")

class ChatResponse(BaseModel):
    status_code: int = Field(..., description="응답 상태 코드")
    message: str = Field(..., description="변환된 메시지") 