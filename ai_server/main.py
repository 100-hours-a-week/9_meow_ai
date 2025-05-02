from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ai_server.schemas import PostRequest, ErrorResponse
from ai_server.model import TransformationService
from ai_server.config import get_settings
import asyncio
import ast
import os
from typing import List
from contextlib import asynccontextmanager
from fastapi.responses import PlainTextResponse

# API 키 풀 관리 클래스
class APIKeyPool:
    def __init__(self, api_keys: List[str], max_requests_per_min: int = 5):
        self.api_keys = api_keys
        self.max_requests_per_min = max_requests_per_min
        self.key_usage = {key: 0 for key in api_keys}
        self.lock = asyncio.Lock()

    async def get_available_key(self) -> str:
        async with self.lock:
            for key in self.api_keys:
                if self.key_usage[key] < self.max_requests_per_min:
                    self.key_usage[key] += 1
                    return key
        return None

    async def release_key(self, key: str):
        async with self.lock:
            if key in self.key_usage:
                self.key_usage[key] = max(0, self.key_usage[key] - 1)

    async def reset_counters(self):
        async with self.lock:
            for key in self.api_keys:
                self.key_usage[key] = 0

    def get_available_key_count(self) -> int:
        return sum(1 for key in self.api_keys if self.key_usage[key] < self.max_requests_per_min)

# 설정 및 키 풀 초기화
settings = get_settings()
from dotenv import load_dotenv

# .env 파일 불러오기
load_dotenv()

# 단일 키 또는 다중 키 확인
single_key = os.getenv("GOOGLE_API_KEYS")
multiple_keys = os.getenv("GOOGLE_API_KEYS_list")

if multiple_keys:
    try:
        api_keys = ast.literal_eval(multiple_keys)
        if not isinstance(api_keys, list):
            print("Warning: GOOGLE_API_KEYS_list must be a list")
            api_keys = []
        else:
            print(f"Running in multi-key mode with {len(api_keys)} keys")
    except (ValueError, SyntaxError) as e:
        print(f"Warning: Failed to parse GOOGLE_API_KEYS_list: {e}")
        api_keys = []
elif single_key:
    try:
        api_keys = ast.literal_eval(single_key)
        if not isinstance(api_keys, list):
            print("Warning: GOOGLE_API_KEYS must be a list")
            api_keys = []
        else:
            print(f"Running in single-key mode with {len(api_keys)} keys")
    except (ValueError, SyntaxError) as e:
        print(f"Warning: Failed to parse GOOGLE_API_KEYS: {e}")
        api_keys = []
else:
    print("Warning: No API keys found. Please set either GOOGLE_API_KEYS or GOOGLE_API_KEYS_list")
    api_keys = []

if not api_keys:
    print("Warning: No valid API keys found")

key_pool = APIKeyPool(api_keys=api_keys, max_requests_per_min=5)

# lifespan 이벤트 처리기
@asynccontextmanager
async def lifespan(app: FastAPI):
    async def reset_counters_periodically():
        while True:
            await asyncio.sleep(60)
            await key_pool.reset_counters()

    task = asyncio.create_task(reset_counters_periodically())
    yield  # 서버 실행 시작
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

# FastAPI 앱 초기화
app = FastAPI(
    title="AI Text Transformation Server (Multi-Key)",
    description="여러 API 키를 사용하여 SNS 포스팅/댓글/채팅을 고양이 말투 등으로 변환하는 AI API 서버",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 루트 엔드포인트
@app.get("/")
async def root():
    return {"message": "AI Text Transformation Server (Multi-Key) is running"}

# 고양이 말투 포스트 생성 엔드포인트
@app.post("/generate/post", response_class=PlainTextResponse, responses={400: {"model": ErrorResponse}})
async def generate_post(request: PostRequest) -> str:
    try:
        # 사용 가능한 API 키 가져오기
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
            # API 키 반환
            await key_pool.release_key(api_key)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
