from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from ai_server.key_manager import initialize_key_pool
from ai_server.post.post_model import PostTransformationService
from ai_server.post.post_schemas import PostRequest, PostResponse
from ai_server.comment.comment_model import CommentTransformationService
from ai_server.comment.comment_schemas import CommentRequest, CommentResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from ai_server.models.model_manager import get_model_manager
import logging
from contextlib import asynccontextmanager

# 로깅 설정
logger = logging.getLogger(__name__)

# API 키 풀 초기화
key_pool = initialize_key_pool()

# 애플리케이션 라이프스팬 설정
@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 시작 및 종료 시 실행되는 비동기 컨텍스트 매니저"""
    try:
        # 서버 시작 시 실행 (startup)
        logger.info("서버 시작: 모델 백그라운드 로드 시작")
        # 모델 매니저 인스턴스 가져오기
        model_manager = get_model_manager()
        # 백그라운드에서 모델 로드 시작 (비동기적 로드)
        model_manager.initialize_models()
        logger.info("서버 시작: 모델 백그라운드 로드 요청 완료")
        
        # 서버가 즉시 요청 처리 시작 (모델 로딩 완료 기다리지 않음)
        yield
        
        # 서버 종료 시 실행 (shutdown)
        logger.info("서버 종료: 모델 리소스 정리 시작...")
        model_manager = get_model_manager()
        model_manager.cleanup()
        logger.info("서버 종료: 모델 리소스 정리 완료")
        
    except Exception as e:
        logger.error(f"서버 라이프스팬 오류: {str(e)}")
        # 초기화 실패 시에도 서버는 계속 실행
        yield

# FastAPI 앱 초기화 (lifespan 파라미터 사용)
app = FastAPI(
    title="AI Text Transformation Server",
    description="SNS 포스팅/댓글/채팅을 고양이/강아지 말투로 변환하는 AI API 서버",
    version="0.1.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # 백엔드 주소
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"]
)

# 검증 오류 처리기
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Enum 값 검증 오류 메시지 생성
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

# 루트 엔드포인트
@app.get("/")
async def root():
    return {"message": "SNS 포스팅/댓글/채팅을 고양이/강아지 말투로 변환하는 AI API 서버"}

# 텍스트 변환 엔드포인트
@app.post("/generate/post", 
    response_model=PostResponse,
    responses={
        200: {"model": PostResponse, "description": "Successfully transformed text"},
        400: {"model": PostResponse, "description": "Empty Input"},
        422: {"model": PostResponse, "description": "wrong post_type or emotion"},
        500: {"model": PostResponse, "description": "internal_server_error"}
    }
)
async def generate_post(request: PostRequest):
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

# 댓글 변환 엔드포인트
@app.post("/generate/comment", 
    response_model=CommentResponse,
    responses={
        200: {"model": CommentResponse, "description": "Successfully transformed text"},
        400: {"model": CommentResponse, "description": "Empty Input"},
        422: {"model": CommentResponse, "description": "wrong post_type"},
        500: {"model": CommentResponse, "description": "internal_server_error"},
        503: {"model": CommentResponse, "description": "API key is not available"}
    }
)
async def generate_comment(request: CommentRequest):
    # 빈 입력 체크
    if not request.content.strip():
        raise HTTPException(status_code=400, detail="Empty Input")
        
    # 사용 가능한 API 키 풀에서 키 획득
    api_key = await key_pool.get_available_key()
    if not api_key:
        raise HTTPException(status_code=503, detail="API key is not available")

    try:
        # 스키마에서 정의된 타입을 사용하여 댓글 변환 서비스 실행
        comment_service = CommentTransformationService(api_key=api_key)
        transformed_content = await comment_service.transform_comment(
            content=request.content,
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
        if "Enum" in str(ve) or "post_type" in str(ve):
            raise HTTPException(status_code=422, detail="wrong post_type")
        else:
            raise

    except Exception:
        # 기타 모든 에러는 500으로 처리
        raise HTTPException(status_code=500, detail="internal_server_error")