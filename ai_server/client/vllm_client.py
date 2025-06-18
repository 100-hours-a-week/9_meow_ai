"""
vLLM 서버 클라이언트
FastAPI 서버에서 vLLM 서버(포트 8001)와 HTTP 통신
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union, AsyncIterator
import httpx
import requests
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    """채팅 메시지 모델"""
    role: str
    content: str


class CompletionRequest(BaseModel):
    """완성 요청 모델"""
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    stream: bool = False


class ChatCompletionRequest(BaseModel):
    """채팅 완성 요청 모델"""
    messages: List[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    stream: bool = False


class VLLMClient:
    """동기 vLLM 클라이언트"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
    
    def health_check(self) -> bool:
        """서버 상태 확인"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check 실패: {e}")
            return False
    
    def list_models(self) -> Dict:
        """사용 가능한 모델 목록 조회"""
        try:
            response = self.session.get(f"{self.base_url}/v1/models", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"모델 목록 조회 실패: {e}")
            raise
    
    def completion(self, request: CompletionRequest) -> Dict:
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
                "stop": request.stop,
                "stream": request.stream
            }
            
            response = self.session.post(
                f"{self.base_url}/v1/completions",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"완성 요청 실패: {e}")
            raise
    
    def chat_completion(self, request: ChatCompletionRequest) -> Dict:
        """채팅 완성 요청"""
        try:
            payload = {
                "model": "HyperCLOVAX-1.5B_LoRA_fp16",
                "messages": [msg.dict() for msg in request.messages],
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "frequency_penalty": request.frequency_penalty,
                "presence_penalty": request.presence_penalty,
                "stop": request.stop,
                "stream": request.stream
            }
            
            response = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"채팅 완성 요청 실패: {e}")
            raise
    
    def close(self):
        """클라이언트 종료"""
        self.session.close()


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
                "stop": request.stop,
                "stream": request.stream
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
    
    async def chat_completion(self, request: ChatCompletionRequest) -> Dict:
        """채팅 완성 요청"""
        try:
            payload = {
                "model": "HyperCLOVAX-1.5B_LoRA_fp16",
                "messages": [msg.dict() for msg in request.messages],
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "frequency_penalty": request.frequency_penalty,
                "presence_penalty": request.presence_penalty,
                "stop": request.stop,
                "stream": request.stream
            }
            
            response = await self.client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"채팅 완성 요청 실패: {e}")
            raise
    
    async def stream_completion(self, request: CompletionRequest) -> AsyncIterator[Dict]:
        """스트리밍 완성 요청"""
        request.stream = True
        
        try:
            payload = {
                "model": "HyperCLOVAX-1.5B_LoRA_fp16",
                "prompt": request.prompt,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "frequency_penalty": request.frequency_penalty,
                "presence_penalty": request.presence_penalty,
                "stop": request.stop,
                "stream": True
            }
            
            async with self.client.stream(
                "POST",
                f"{self.base_url}/v1/completions",
                json=payload
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]  # "data: " 제거
                        if data.strip() == "[DONE]":
                            break
                        try:
                            import json
                            yield json.loads(data)
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            logger.error(f"스트리밍 완성 요청 실패: {e}")
            raise
    
    async def stream_chat_completion(self, request: ChatCompletionRequest) -> AsyncIterator[Dict]:
        """스트리밍 채팅 완성 요청"""
        request.stream = True
        
        try:
            payload = {
                "model": "HyperCLOVAX-1.5B_LoRA_fp16",
                "messages": [msg.dict() for msg in request.messages],
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "frequency_penalty": request.frequency_penalty,
                "presence_penalty": request.presence_penalty,
                "stop": request.stop,
                "stream": True
            }
            
            async with self.client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=payload
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]  # "data: " 제거
                        if data.strip() == "[DONE]":
                            break
                        try:
                            import json
                            yield json.loads(data)
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            logger.error(f"스트리밍 채팅 완성 요청 실패: {e}")
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