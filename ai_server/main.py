from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from ai_server.router.api import api_router
import logging
import asyncio
import threading
import os

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 이미지 데이터베이스 구축 상태
image_db_status = {"status": "not_started", "message": "이미지 데이터베이스가 아직 구축되지 않았습니다"}

def build_image_database_background():
    """백그라운드에서 이미지 데이터베이스 구축"""
    try:
        image_db_status["status"] = "building"
        image_db_status["message"] = "이미지 데이터베이스 구축 중..."
        
        logger.info("백그라운드에서 이미지 데이터베이스 구축 시작...")
        
        # ChromaDB telemetry 비활성화
        os.environ["ANONYMIZED_TELEMETRY"] = "False"
        
        from ai_server.scripts.build_image_database import build_database
        build_database()
        
        image_db_status["status"] = "completed"
        image_db_status["message"] = "이미지 데이터베이스 구축 완료"
        logger.info("이미지 데이터베이스 구축 완료!")
        
    except Exception as e:
        image_db_status["status"] = "failed"
        image_db_status["message"] = f"이미지 데이터베이스 구축 실패: {str(e)}"
        logger.error(f"이미지 데이터베이스 구축 실패: {e}")

# FastAPI 앱 초기화
app = FastAPI(
    title="AI Text Transformation & Image Search Server",
    description="SNS 포스팅/댓글/채팅을 고양이/강아지 말투로 변환하고 유사 이미지를 검색하는 AI API 서버",
    version="3.0.0",
    docs_url="/docs",
)

@app.on_event("startup")
async def startup_event():
    """앱 시작 시 실행되는 이벤트"""
    logger.info("FastAPI 서버 시작 중...")
    
    # 이미지 데이터베이스를 백그라운드에서 구축
    thread = threading.Thread(target=build_image_database_background, daemon=True)
    thread.start()
    
    logger.info("FastAPI 서버 시작 완료")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://localhost:5173", 
        "http://127.0.0.1:5173",
        "https://127.0.0.1:5173",
        "http://localhost:8080",
        "https://localhost:8080",
        "http://127.0.0.1:8080",
        "https://127.0.0.1:8080",
        "http://www.meowng.com",
        "https://www.meowng.com",
	"http://meowng.com",
	"https://meowng.com",
        "https://ds36vr51hmfa7.cloudfront.net",
        "http://3.39.3.208",
        "http://3.39.3.208:8080",
        "http://3.39.3.208:5173",
        "http://172.20.5.64:5173",
        "http://testdev.meowng.com"
    ],
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
        "message": "AI Text Transformation & Image Search Server",
        "description": "SNS 포스팅/댓글/채팅을 고양이/강아지 말투로 변환하고 유사 이미지를 검색하는 AI API 서버",
        "version": "3.0.0",
        "status": "healthy",
        "features": [
            "텍스트 변환 (고양이/강아지 말투)",
            "이미지 유사도 검색 (CLIP 기반)",
            "벡터 데이터베이스 (ChromaDB)"
        ]
    }

# 헬스체크 엔드포인트
@app.get("/health")
async def health_check():
    """헬스체크 엔드포인트"""
    return {"status": "healthy"}

# 이미지 데이터베이스 상태 확인 엔드포인트
@app.get("/image-db-status")
async def get_image_db_status():
    """이미지 데이터베이스 구축 상태 확인"""
    return image_db_status
