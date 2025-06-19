"""
vLLM 서버 설정 관리
포스트 문장 생성을 위한 간소화된 설정
"""

import os
from typing import Optional, List
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class VLLMConfig(BaseSettings):
    """vLLM 서버 설정 클래스 - 포스트 문장 생성 최적화"""
    
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
        description="base 모델 경로"
    )
    
    # LoRA 관련 설정
    enable_lora: bool = Field(default=True, description="LoRA 활성화")
    base_model_path: Optional[str] = Field(default=None, description="LoRA의 베이스 모델 경로")
    lora_modules: Optional[List[str]] = Field(default=None, description="LoRA 모듈 목록")
    
    # GPU 및 메모리 설정 (포스트 생성에 최적화)
    gpu_memory_utilization: float = Field(default=0.8, description="GPU 메모리 사용률")
    max_model_len: int = Field(default=1024, description="최대 모델 길이 - 포스트 생성용")
    
    # 포스트 생성 최적화 설정
    max_num_batched_tokens: int = Field(default=2048, description="배치당 최대 토큰 수 - 포스트용")
    max_num_seqs: int = Field(default=32, description="최대 시퀀스 수 - 포스트용")
    
    # API 설정
    served_model_name: str = Field(
        default="HyperCLOVAX-1.5B_LoRA_fp16",
        description="API에서 제공할 모델 이름"
    )
    
    # 토큰화 설정
    trust_remote_code: bool = Field(default=True, description="원격 코드 신뢰")
    
    # 환경 변수 설정
    class Config:
        env_prefix = "VLLM_"
        env_file = None
        case_sensitive = False
        extra = "ignore"


class VLLMServerArgs:
    """vLLM 서버 실행 인자 생성 - 포스트 생성 최적화"""
    
    def __init__(self, config: VLLMConfig):
        self.config = config
    
    def get_server_args(self) -> list[str]:
        """포스트 생성을 위한 vLLM 서버 실행 인자"""
        model_path = self.config.base_model_path if self.config.enable_lora else self.config.model_path
        
        args = [
            "--host", self.config.host,
            "--port", str(self.config.port),
            "--model", model_path,
            "--served-model-name", self.config.served_model_name,
            "--gpu-memory-utilization", str(self.config.gpu_memory_utilization),
            "--max-model-len", str(self.config.max_model_len),
            "--max-num-batched-tokens", str(self.config.max_num_batched_tokens),
            "--max-num-seqs", str(self.config.max_num_seqs),
            "--trust-remote-code",
        ]
        
        # LoRA 설정 (포스트 생성에 필수)
        if self.config.enable_lora:
            args.extend([
                "--enable-lora",
                "--max-loras", "1",  # 포스트 생성은 단일 LoRA만 사용
                "--max-lora-rank", "16",
            ])
            
            # LoRA 모듈 지정
            if self.config.lora_modules:
                for lora_module in self.config.lora_modules:
                    args.extend(["--lora-modules", lora_module])
        
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