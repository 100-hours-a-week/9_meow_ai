from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ai_server.schemas import PostRequest, ErrorResponse
from ai_server.model import TransformationService
from ai_server.key_manager import initialize_key_pool
from fastapi.responses import PlainTextResponse

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
    allow_origins=["http://localhost:8080"],  # 백엔드 서버 주소
    allow_credentials=True,
    allow_methods=["POST", "GET"], # 필요한 HTTP 메서드만 허용
    allow_headers=["*"]  # 필요한 헤더만 허용
)

# 루트 엔드포인트
@app.get("/")
async def root():
    return {"message": "AI Text Transformation Server is running"}

# 텍스트 변환 엔드포인트
@app.post("/generate/post", response_class=PlainTextResponse, responses={400: {"model": ErrorResponse}})
async def generate_post(request: PostRequest) -> str:
    try:
        api_key = await key_pool.get_available_key()
        if not api_key:
            raise HTTPException(
                status_code=503,
                detail="현재 모든 API 키가 사용 중입니다. 잠시 후 다시 시도해주세요."
            )

        try:
            transformation_service = TransformationService(api_key=api_key)
            transformed_content = await transformation_service.transform_post(
                content=request.content,
                emotion=request.emotion,
                post_type=request.post_type
            )
            return transformed_content
        finally:
            await key_pool.release_key(api_key)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
