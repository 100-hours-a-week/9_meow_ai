"""
vLLM 서버 클라이언트
포스트 문장 생성을 위한 간소화된 HTTP 통신
"""

import logging
from typing import Dict, List, Optional
import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class CompletionRequest(BaseModel):
    """포스트 생성 요청 모델 - 성능 최적화된 파라미터"""
    prompt: str
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    stop: Optional[List[str]] = None


class VLLMAsyncClient:
    """비동기 vLLM 클라이언트 - 포스트 생성 최적화"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None
    
    @property
    def client(self) -> httpx.AsyncClient:
        """비동기 HTTP 클라이언트 인스턴스"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),  # 포스트 생성용 단축
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
            )
        return self._client
    
    async def completion(self, request: CompletionRequest) -> Dict:
        """포스트 텍스트 생성 요청"""
        try:
            payload = {
                "model": "HyperCLOVAX-1.5B_LoRA_fp16",
                "prompt": request.prompt,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "stop": request.stop
            }
            
            response = await self.client.post(
                f"{self.base_url}/v1/completions",
                json=payload
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"포스트 생성 요청 실패: {e}")
            raise
    
    async def close(self):
        """클라이언트 종료"""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close() 