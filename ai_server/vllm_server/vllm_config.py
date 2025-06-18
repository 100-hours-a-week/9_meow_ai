"""
vLLM 서버 설정 관리
HyperCLOVAX-1.5B_LoRA_fp16 모델을 위한 설정
"""

import os
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class VLLMConfig(BaseSettings):
    """vLLM 서버 설정 클래스"""
    
    # 서버 기본 설정
    host: str = Field(default="0.0.0.0", description="vLLM 서버 호스트")
    port: int = Field(default=8001, description="vLLM 서버 포트")
    
    # 모델 설정
    model_name: str = Field(
        default="HyperCLOVAX-1.5B_LoRA_fp16",
        description="사용할 모델 이름"
    )
    model_path: str = Field(
        default="./models/HyperCLOVAX-1.5B_LoRA_fp16",
        description="모델 파일 경로"
    )
    
    # GPU 및 메모리 설정
    tensor_parallel_size: int = Field(default=1, description="텐서 병렬 처리 크기")
    gpu_memory_utilization: float = Field(default=0.8, description="GPU 메모리 사용률")
    max_model_len: int = Field(default=4096, description="최대 모델 길이")
    
    # 성능 최적화 설정
    enable_chunked_prefill: bool = Field(default=True, description="청크 프리필 활성화")
    max_num_batched_tokens: int = Field(default=8192, description="배치당 최대 토큰 수")
    max_num_seqs: int = Field(default=256, description="최대 시퀀스 수")
    
    # API 설정
    served_model_name: str = Field(
        default="HyperCLOVAX-1.5B_LoRA_fp16",
        description="API에서 제공할 모델 이름"
    )
    disable_log_stats: bool = Field(default=False, description="로그 통계 비활성화")
    
    # 토큰화 설정
    trust_remote_code: bool = Field(default=True, description="원격 코드 신뢰")
    
    # 환경 변수 설정
    class Config:
        env_prefix = "VLLM_"
        env_file = ".env"
        case_sensitive = False


class VLLMServerArgs(BaseModel):
    """vLLM 서버 실행 인자 생성"""
    
    def __init__(self, config: VLLMConfig):
        self.config = config
    
    def get_server_args(self) -> list[str]:
        """vLLM 서버 실행을 위한 명령행 인자 생성"""
        args = [
            "--host", self.config.host,
            "--port", str(self.config.port),
            "--model", self.config.model_path,
            "--served-model-name", self.config.served_model_name,
            "--tensor-parallel-size", str(self.config.tensor_parallel_size),
            "--gpu-memory-utilization", str(self.config.gpu_memory_utilization),
            "--max-model-len", str(self.config.max_model_len),
            "--max-num-batched-tokens", str(self.config.max_num_batched_tokens),
            "--max-num-seqs", str(self.config.max_num_seqs),
        ]
        
        if self.config.enable_chunked_prefill:
            args.append("--enable-chunked-prefill")
        
        if self.config.trust_remote_code:
            args.append("--trust-remote-code")
        
        if self.config.disable_log_stats:
            args.append("--disable-log-stats")
        
        return args


# 전역 설정 인스턴스
vllm_config = VLLMConfig()


def get_vllm_config() -> VLLMConfig:
    """vLLM 설정 인스턴스 반환"""
    return vllm_config


def update_vllm_config(**kwargs) -> VLLMConfig:
    """vLLM 설정 업데이트"""
    global vllm_config
    for key, value in kwargs.items():
        if hasattr(vllm_config, key):
            setattr(vllm_config, key, value)
    return vllm_config 