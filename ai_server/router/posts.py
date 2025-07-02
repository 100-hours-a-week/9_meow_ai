from fastapi import APIRouter, HTTPException
from ai_server.schemas.post_schemas import PostRequest, PostResponse
from ai_server.model.post_model import PostTransformationService

router = APIRouter()

@router.post("", 
    response_model=PostResponse,
    responses={
        200: {"model": PostResponse, "description": "Successfully transformed text"},
        400: {"model": PostResponse, "description": "Empty Input"},
        422: {"model": PostResponse, "description": "wrong post_type or emotion"},
        500: {"model": PostResponse, "description": "internal_server_error"}
    }
)
async def generate_post(request: PostRequest):
    """포스트 텍스트를 고양이/강아지 말투로 변환합니다."""
    # 빈 입력 체크
    if not request.content.strip():
        raise HTTPException(status_code=400, detail="Empty Input")

    try:
        # 스키마에서 정의된 타입을 사용하여 포스트 변환 서비스 실행
        post_service = PostTransformationService()  # 기본 모델 사용
        transformed_content = await post_service.transform_post(
            content=request.content,
            emotion=request.emotion,
            post_type=request.post_type
        )
        
        # 성공 응답
        return PostResponse(
            status_code=200,
            message="Successfully transformed text",
            data=transformed_content
        )
        
    except ValueError as ve:
        # Enum 값이 잘못된 경우 등
        if "Enum" in str(ve) or "post_type" in str(ve) or "emotion" in str(ve):
            raise HTTPException(status_code=422, detail="wrong post_type or emotion")
        else:
            raise

    except Exception:
        # 기타 모든 에러는 500으로 처리
        raise HTTPException(status_code=500, detail="internal_server_error") 