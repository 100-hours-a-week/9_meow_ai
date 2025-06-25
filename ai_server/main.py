from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from ai_server.api.v1.api import api_router
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(
    title="AI Text Transformation Server",
    description="SNS 포스팅/댓글/채팅을 고양이/강아지 말투로 변환하는 AI API 서버",
    version="2.1.0",
    docs_url="/docs",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 백엔드 주소
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"]
)

# 검증 오류 처리기
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """검증 오류 처리"""
    error_msg = "잘못된 입력값"
    
    # 오류 상세 정보 추출
    for error in exc.errors():
        if error["type"] == "enum":
            field_name = error["loc"][-1]
            if field_name == "post_type":
                error_msg = "동물 타입은 'cat' 또는 'dog'여야 합니다"
            elif field_name == "emotion":
                error_msg = "감정 상태가 올바르지 않습니다"
    
    return JSONResponse(
        status_code=422,
        content={
            "status_code": 422,
            "message": error_msg,
            "data": None
        }
    )

# HTTP 예외 처리기
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """HTTP 예외 처리"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status_code": exc.status_code,
            "message": exc.detail,
            "data": None
        }
    )

# 일반 예외 처리기
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "status_code": 500,
            "message": "internal_server_error",
            "data": None
        }
    )

# API 라우터 포함
app.include_router(api_router)

# 루트 엔드포인트
@app.get("/")
async def root():
    """루트 엔드포인트 - 서버 상태 확인"""
    return {
        "message": "SNS 포스팅/댓글/채팅을 고양이/강아지 말투로 변환하는 AI API 서버",
        "version": "2.1.0",
        "status": "healthy"
    }

# 헬스체크 엔드포인트
@app.get("/health")
async def health_check():
    """헬스체크 엔드포인트"""
    return {"status": "healthy"}