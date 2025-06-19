"""
vLLM 통합 모듈
서버와 클라이언트 기능을 모두 제공하는 통합 vLLM 패키지
"""

__version__ = "1.0.0"
__author__ = "KTB Team Project"

# 서버 모듈 임포트
from .server.vllm_config import VLLMConfig, VLLMServerArgs, get_vllm_config, update_vllm_config
from .server.vllm_launcher import VLLMLauncher

# 클라이언트 모듈 임포트
from .client.vllm_client import VLLMAsyncClient, CompletionRequest

__all__ = [
    # 서버 클래스
    "VLLMConfig",
    "VLLMServerArgs", 
    "VLLMLauncher",
    "get_vllm_config",
    "update_vllm_config",
    
    # 클라이언트 클래스
    "VLLMAsyncClient",
    "CompletionRequest"
] 