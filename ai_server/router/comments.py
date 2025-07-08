from fastapi import APIRouter, HTTPException
from ai_server.schemas.converter_schemas import CommentRequest, CommentResponse
from ai_server.model.comment_model import CommentTransformationService

router = APIRouter()

@router.post("", 
    response_model=CommentResponse,
    responses={
        200: {"model": CommentResponse, "description": "Successfully transformed text"},
        400: {"model": CommentResponse, "description": "Empty Input"},
        422: {"model": CommentResponse, "description": "wrong post_type or emotion"},
        500: {"model": CommentResponse, "description": "internal_server_error"}
    }
)
async def generate_comment(request: CommentRequest):
    """댓글 텍스트를 고양이/강아지 말투로 변환합니다."""
    # 빈 입력 체크
    if not request.content.strip():
        raise HTTPException(status_code=400, detail="Empty Input")

    try:
        # CommentTransformationService를 사용하여 vLLM 추론 로직 적용
        comment_service = CommentTransformationService()
        transformed_content = await comment_service.transform_comment(
            content=request.content,
            emotion=request.emotion,
            post_type=request.post_type
        )
        
        # 성공 응답
        return CommentResponse(
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