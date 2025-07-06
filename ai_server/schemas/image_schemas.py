from pydantic import BaseModel, Field, HttpUrl
from typing import List, Literal
from enum import Enum

class AnimalType(str, Enum):
    """동물 타입 열거형"""
    CAT = "cat"
    DOG = "dog"

class ImageSearchRequest(BaseModel):
    """이미지 검색 요청 스키마"""
    image_url: HttpUrl = Field(
        ...,
        description="검색할 이미지의 URL",
        example="https://example.com/image.jpg"
    )
    animal_type: AnimalType = Field(
        ...,
        description="동물 종류 (cat 또는 dog)",
        example="cat"
    )
    n_results: int = Field(
        default=3,
        ge=1,
        le=10,
        description="반환할 유사 이미지 개수 (1-10)",
        example=3
    )

class ImageSearchResponse(BaseModel):
    """이미지 검색 응답 스키마"""
    status_code: int = Field(
        ...,
        description="HTTP 상태 코드",
        example=200
    )
    message: str = Field(
        ...,
        description="응답 메시지",
        example="이미지 검색 성공"
    )
    data: List[str] = Field(
        ...,
        description="유사한 이미지 URL 리스트",
        example=[
            "https://example.com/similar1.jpg",
            "https://example.com/similar2.jpg",
            "https://example.com/similar3.jpg"
        ]
    )

class ErrorResponse(BaseModel):
    """에러 응답 스키마"""
    status_code: int = Field(
        ...,
        description="HTTP 상태 코드",
        example=400
    )
    message: str = Field(
        ...,
        description="에러 메시지",
        example="잘못된 요청입니다"
    )
    data: None = Field(
        default=None,
        description="에러 시 데이터는 null"
    ) 