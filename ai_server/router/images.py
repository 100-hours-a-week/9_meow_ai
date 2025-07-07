from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from ai_server.schemas.image_schemas import (
    ImageSearchRequest, 
    ImageSearchResponse, 
    ErrorResponse
)
from ai_server.services.image_search import get_image_search_service, ImageSearchService
import logging
from typing import List

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post(
    "/search",
    response_model=ImageSearchResponse,
    responses={
        200: {"model": ImageSearchResponse, "description": "이미지 검색 성공"},
        400: {"model": ErrorResponse, "description": "잘못된 요청"},
        422: {"model": ErrorResponse, "description": "검증 오류"},
        500: {"model": ErrorResponse, "description": "서버 오류"}
    },
    summary="이미지 유사도 검색",
    description="업로드된 이미지와 유사한 이미지 3개를 검색합니다."
)
async def search_similar_images(
    request: ImageSearchRequest,
    image_service: ImageSearchService = Depends(get_image_search_service)
) -> ImageSearchResponse:
    """
    이미지 유사도 검색 API
    
    - **image_url**: 검색할 이미지의 URL
    - **animal_type**: 동물 종류 (cat 또는 dog)
    - **n_results**: 반환할 유사 이미지 개수 (1-10, 기본값: 3)
    
    Returns:
        ImageSearchResponse: 유사한 이미지 URL 리스트
    """
    try:
        # 이미지 검색 실행
        similar_images = image_service.search_similar_images(
            image_url=str(request.image_url),
            animal_type=request.animal_type.value,
            n_results=request.n_results
        )
        
        logger.info(f"Image search completed: {len(similar_images)} results")
        
        return ImageSearchResponse(
            status_code=200,
            message="이미지 검색 성공",
            data=similar_images
        )
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Image search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="이미지 검색 중 오류가 발생했습니다"
        ) 