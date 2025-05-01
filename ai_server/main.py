from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas import PostRequest, PostResponse, ErrorResponse
from .transformation import TransformationService
from .config import get_settings

app = FastAPI(
    title="AI Text Transformation Server",
    description="SNS 포스팅/댓글/채팅을 고양이 말투 등으로 변환하는 AI API 서버",
    version="0.1.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 서비스 초기화
settings = get_settings()
transformation_service = TransformationService(api_key=settings.GOOGLE_API_KEY)

@app.get("/")
async def root():
    return {"message": "AI Text Transformation Server is running"}

@app.post("/generate/post", response_model=PostResponse, responses={400: {"model": ErrorResponse}})
async def generate_post(request: PostRequest) -> PostResponse:
    try:
        transformed_content = await transformation_service.transform_post(
            content=request.content,
            emotion=request.emotion,
            post_type=request.post_type
        )
        return PostResponse(fixed_content=transformed_content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
