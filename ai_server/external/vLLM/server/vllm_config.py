"""
vLLM 서버 설정 관리
포스트 문장 생성을 위한 간소화된 설정
"""

import os
from pydantic_settings import BaseSettings
from pydantic import Field


class VLLMConfig(BaseSettings):
    """간소화된 vLLM 서버 설정"""
    
    # 서버 기본 설정
    host: str = Field(default="0.0.0.0", description="vLLM 서버 호스트")
    port: int = Field(default=8002, description="vLLM 서버 포트")
    
    # 모델 설정
    model_path: str = Field(
        default="haebo/meow-clovax-v3",
        description="사용할 모델 경로"
    )
    served_model_name: str = Field(
        default="meow-clovax-v3",
        description="서빙 모델명"
    )
    
    # 메모리 및 성능 설정 (이미지 검색 기능과 호환)
    gpu_memory_utilization: float = Field(default=0.4, description="GPU 메모리 사용률")
    max_model_len: int = Field(default=512, description="최대 모델 길이")
    max_num_batched_tokens: int = Field(default=512, description="배치 토큰 수")
    max_num_seqs: int = Field(default=4, description="동시 시퀀스 수")
    
    # 문제가 되는 boolean 설정들 제거
    # enable_prefix_caching, trust_remote_code 제거
    
    class Config:
        env_prefix = "VLLM_"


class VLLMServerArgs:
    """간소화된 vLLM 서버 실행 인자"""
    
    def __init__(self, config: VLLMConfig):
        self.config = config
    
    def get_server_args(self) -> list[str]:
        """풀파인튜닝 모델용 서버 인자 (안정적인 설정만)"""
        return [
            "--host", self.config.host,
            "--port", str(self.config.port),
            "--model", self.config.model_path,
            "--served-model-name", self.config.served_model_name,
            "--gpu-memory-utilization", str(self.config.gpu_memory_utilization),
            "--max-model-len", str(self.config.max_model_len),
            "--max-num-batched-tokens", str(self.config.max_num_batched_tokens),
            "--max-num-seqs", str(self.config.max_num_seqs),
        ]


# 전역 설정 인스턴스
vllm_config = VLLMConfig()


def get_vllm_config() -> VLLMConfig:
    """vLLM 설정 인스턴스 반환"""
    return vllm_config 