from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from ai_server.post.post_schemas import PostRequest, PostResponse
from ai_server.rule_based_converter.cat import cat_converter
from ai_server.rule_based_converter.dog import dog_converter
from ai_server.rule_based_converter.converter_schemas import CommentRequest, CommentResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from ai_server.post.post_model import PostTransformationService
import logging

# 로깅 설정
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(
    title="AI Text Transformation Server",
    description="SNS 포스팅/댓글/채팅을 고양이/강아지 말투로 변환하는 AI API 서버",
    version="0.1.0"
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
        500: {"model": CommentResponse, "description": "internal_server_error"}
    }
)
async def generate_comment(request: CommentRequest):
    # 빈 입력 체크
    if not request.content.strip():
        raise HTTPException(status_code=400, detail="Empty Input")

    try:
        # post_type에 따라 적절한 변환기 선택하게
        if request.post_type == "cat":
            transformed_content = cat_converter(request.content)
        elif request.post_type == "dog":
            transformed_content = dog_converter(request.content)
        else:
            raise HTTPException(status_code=422, detail="wrong post_type")
        
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