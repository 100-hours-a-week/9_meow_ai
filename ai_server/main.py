from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from ai_server.key_manager import initialize_key_pool
from ai_server.post.post_model import PostTransformationService
from ai_server.post.post_schemas import PostRequest, ErrorResponse
from ai_server.comment.comment_model import CommentTransformationService
from ai_server.comment.comment_schemas import ErrorResponse, CommentRequest


# API 키 풀 초기화
key_pool = initialize_key_pool()

# FastAPI 앱 초기화
app = FastAPI(
    title="AI Text Transformation Server",
    description="SNS 포스팅/댓글/채팅을 고양이/강아지 말투로 변환하는 AI API 서버",
    version="0.1.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # 프론트엔드 주소
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"]
)

# 루트 엔드포인트
@app.get("/")
async def root():
    return {"message": "SNS 포스팅/댓글/채팅을 고양이/강아지 말투로 변환하는 AI API 서버"}

# 텍스트 변환 엔드포인트
@app.post("/generate/post", 
    response_class=PlainTextResponse,
    responses={
        200: {"description": "Successfully transformed text"},
        400: {"model": ErrorResponse},
        503: {"description": "Service temporarily unavailable"}
    }
)
async def generate_post(request: PostRequest) -> str:
    try:
        # 사용 가능한 API 키 획득
        api_key = await key_pool.get_available_key()
        if not api_key:
            raise HTTPException(
                status_code=503,
                detail="현재 모든 API 키가 사용 중입니다. 잠시 후 다시 시도해주세요."
            )

        # 스키마에서 정의된 타입을 사용하여 포스트 변환 서비스 실행
        post_service = PostTransformationService(api_key=api_key)
        transformed_content = await post_service.transform_post(
            content=request.content,
            emotion=request.emotion,
            post_type=request.post_type
        )
        return transformed_content

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 댓글 변환 엔드포인트
@app.post("/generate/comment", 
    response_class=PlainTextResponse,
    responses={
        200: {"description": "Successfully transformed text"},
        400: {"model": ErrorResponse},
        503: {"description": "Service temporarily unavailable"}
    }
)
async def generate_comment(request: CommentRequest) -> str:
    try:
        # 사용 가능한 API 키 풀에서 키 획득
        api_key = await key_pool.get_available_key()
        if not api_key:
            raise HTTPException(
                status_code=503,
                detail="현재 모든 API 키가 사용 중입니다. 잠시 후 다시 시도해주세요."
            )

        # 스키마에서 정의된 타입을 사용하여 댓글 변환 서비스 실행
        comment_service = CommentTransformationService(api_key=api_key)
        transformed_content = await comment_service.transform_comment(
            content=request.content,
            post_type=request.post_type
        )
        
        # 결과 반환(문자열)
        return transformed_content

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))