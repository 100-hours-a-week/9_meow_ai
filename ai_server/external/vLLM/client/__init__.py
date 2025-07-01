"""
vLLM 클라이언트 모듈 - 간소화 버전
"""

from .vllm_client import VLLMAsyncClient, CompletionRequest

__all__ = ["VLLMAsyncClient", "CompletionRequest"] 