"""
vLLM 서버 클라이언트
FastAPI 서버에서 vLLM 서버(포트 8001)와 HTTP 통신
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union, AsyncIterator
import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class CompletionRequest(BaseModel):
    """완성 요청 모델"""
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.3
    top_p: float = 0.2
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None


class VLLMAsyncClient:
    """비동기 vLLM 클라이언트"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None
    
    @property
    def client(self) -> httpx.AsyncClient:
        """비동기 HTTP 클라이언트 인스턴스"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0),
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
            )
        return self._client
    
    async def health_check(self) -> bool:
        """서버 상태 확인"""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check 실패: {e}")
            return False
    
    async def list_models(self) -> Dict:
        """사용 가능한 모델 목록 조회"""
        try:
            response = await self.client.get(f"{self.base_url}/v1/models")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"모델 목록 조회 실패: {e}")
            raise
    
    async def completion(self, request: CompletionRequest) -> Dict:
        """텍스트 완성 요청"""
        try:
            payload = {
                "model": "HyperCLOVAX-1.5B_LoRA_fp16",
                "prompt": request.prompt,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "frequency_penalty": request.frequency_penalty,
                "presence_penalty": request.presence_penalty,
                "stop": request.stop
            }
            
            response = await self.client.post(
                f"{self.base_url}/v1/completions",
                json=payload
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"완성 요청 실패: {e}")
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