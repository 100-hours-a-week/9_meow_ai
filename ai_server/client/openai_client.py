"""
OpenAI 호환 클라이언트
vLLM의 OpenAI 호환 API를 사용하는 클라이언트
"""

import logging
from typing import Dict, List, Optional, AsyncIterator
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types import Completion

logger = logging.getLogger(__name__)


class OpenAICompatibleClient:
    """OpenAI 호환 동기 클라이언트"""
    
    def __init__(self, base_url: str = "http://localhost:8001/v1", api_key: str = "dummy"):
        """
        OpenAI 호환 클라이언트 초기화
        
        Args:
            base_url: vLLM 서버의 OpenAI 호환 API 엔드포인트
            api_key: API 키 (vLLM에서는 필요없지만 OpenAI 클라이언트 호환성을 위해 설정)
        """
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model_name = "haebo/Meow-HyperCLOVAX-1.5B_LoRA_fp16_0527"
    
    def list_models(self) -> Dict:
        """사용 가능한 모델 목록 조회"""
        try:
            models = self.client.models.list()
            return models.dict()
        except Exception as e:
            logger.error(f"모델 목록 조회 실패: {e}")
            raise
    
    def completion(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.2,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None
    ) -> Completion:
        """텍스트 완성 요청"""
        try:
            response = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop
            )
            return response
        except Exception as e:
            logger.error(f"완성 요청 실패: {e}")
            raise
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.2,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None
    ) -> ChatCompletion:
        """채팅 완성 요청"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop
            )
            return response
        except Exception as e:
            logger.error(f"채팅 완성 요청 실패: {e}")
            raise
    
    def stream_completion(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.2,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None
    ):
        """스트리밍 텍스트 완성 요청"""
        try:
            stream = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop,
                stream=True
            )
            
            for chunk in stream:
                yield chunk
                
        except Exception as e:
            logger.error(f"스트리밍 완성 요청 실패: {e}")
            raise
    
    def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.2,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None
    ):
        """스트리밍 채팅 완성 요청"""
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop,
                stream=True
            )
            
            for chunk in stream:
                yield chunk
                
        except Exception as e:
            logger.error(f"스트리밍 채팅 완성 요청 실패: {e}")
            raise


class AsyncOpenAICompatibleClient:
    """OpenAI 호환 비동기 클라이언트"""
    
    def __init__(self, base_url: str = "http://localhost:8001/v1", api_key: str = "dummy"):
        """
        OpenAI 호환 비동기 클라이언트 초기화
        
        Args:
            base_url: vLLM 서버의 OpenAI 호환 API 엔드포인트
            api_key: API 키 (vLLM에서는 필요없지만 OpenAI 클라이언트 호환성을 위해 설정)
        """
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model_name = "haebo/Meow-HyperCLOVAX-1.5B_LoRA_fp16_0527"
    
    async def list_models(self) -> Dict:
        """사용 가능한 모델 목록 조회"""
        try:
            models = await self.client.models.list()
            return models.dict()
        except Exception as e:
            logger.error(f"모델 목록 조회 실패: {e}")
            raise
    
    async def completion(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.2,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None
    ) -> Completion:
        """텍스트 완성 요청"""
        try:
            response = await self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop
            )
            return response
        except Exception as e:
            logger.error(f"완성 요청 실패: {e}")
            raise
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.2,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None
    ) -> ChatCompletion:
        """채팅 완성 요청"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop
            )
            return response
        except Exception as e:
            logger.error(f"채팅 완성 요청 실패: {e}")
            raise
    
    async def stream_completion(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.2,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None
    ) -> AsyncIterator:
        """스트리밍 텍스트 완성 요청"""
        try:
            stream = await self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop,
                stream=True
            )
            
            async for chunk in stream:
                yield chunk
                
        except Exception as e:
            logger.error(f"스트리밍 완성 요청 실패: {e}")
            raise
    
    async def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.2,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None
    ) -> AsyncIterator[ChatCompletionChunk]:
        """스트리밍 채팅 완성 요청"""
        try:
            stream = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop,
                stream=True
            )
            
            async for chunk in stream:
                yield chunk
                
        except Exception as e:
            logger.error(f"스트리밍 채팅 완성 요청 실패: {e}")
            raise
    
    async def close(self):
        """클라이언트 종료"""
        await self.client.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close() 