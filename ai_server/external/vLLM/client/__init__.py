"""
vLLM 클라이언트 모듈
vLLM 서버와의 HTTP 통신을 위한 비동기 클라이언트
"""

__version__ = "1.0.0"
__author__ = "KTB Team Project"

from .vllm_client import VLLMAsyncClient, CompletionRequest

__all__ = ["VLLMAsyncClient", "CompletionRequest"] 