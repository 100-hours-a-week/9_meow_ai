from fastapi import APIRouter, HTTPException
from ai_server.schemas.chat_schemas import ChatRequest, ChatResponse
from ai_server.model.chat_model import ChatTransformationService

router = APIRouter()

@router.post("", 
    response_model=ChatResponse,
    responses={
        200: {"model": ChatResponse, "description": "Successfully transformed text"},
        400: {"model": ChatResponse, "description": "Empty Input"},
        422: {"model": ChatResponse, "description": "wrong post_type"},
        500: {"model": ChatResponse, "description": "internal_server_error"}
    }
)
async def generate_chat(request: ChatRequest):
    """채팅 텍스트를 동물 말투로 변환합니다."""
    # 빈 입력 체크
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty Input")

    try:
        # ChatTransformationService를 사용하여 규칙 기반 변환 적용
        chat_service = ChatTransformationService()
        transformed_content = chat_service.transform_chat(
            text=request.text,
            post_type=request.post_type
        )
        
        # 성공 응답
        return ChatResponse(
            status_code=200,
            message=transformed_content
        )
        
    except ValueError as ve:
        # Enum 값이 잘못된 경우 등
        if "Enum" in str(ve) or "post_type" in str(ve):
            raise HTTPException(status_code=422, detail="wrong post_type")
        else:
            raise

    except Exception:
        # 기타 모든 에러는 500으로 처리
        raise HTTPException(status_code=500, detail="internal_server_error")
