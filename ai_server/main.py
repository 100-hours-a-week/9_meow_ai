from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from .schemas import PostRequest, ErrorResponse
from .transformation import TransformationService
from .config import get_settings

app = FastAPI(
    title="AI Text Transformation Server",
    description="SNS 포스팅/댓글/채팅을 고양이 말투 등으로 변환하는 AI API 서버",
    version="0.1.0"  #초기 개발 버전
)

# CORS 설정(모든 도메인에서 접근 허용(임시))
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 실제 도메인 목록으로 대체해야 함
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # 필요한 HTTP 메서드만 명시
    allow_headers=["*"],
)

# 서비스 초기화
settings = get_settings()
transformation_service = TransformationService(api_key=settings.GOOGLE_API_KEY)

@app.get("/")
async def root():
    return {"message": "AI Text Transformation Server is running"}

@app.post("/generate/post", response_class=PlainTextResponse, responses={400: {"model": ErrorResponse}})
async def generate_post(request: PostRequest) -> str:
    try:
        transformed_content = await transformation_service.transform_post(
            content=request.content,
            emotion=request.emotion,
            post_type=request.post_type
        )
        return transformed_content
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # 개발 환경에서는, 디버깅을 위해 에러 메시지를 반환
        # 프로덕션 환경에서는 자세한 에러 메시지를 로깅하고, 일반적인 메시지를 사용자에게 반환하는 것이 좋음
        raise HTTPException(status_code=500, detail=f"AI 서버 요청 실패: {str(e)}")
