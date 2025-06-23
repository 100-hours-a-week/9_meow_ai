"""
vLLM 서버 설정 관리
포스트 문장 생성을 위한 간소화된 설정
"""

import os
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import httpx
import logging

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """모델 타입 정의"""
    LORA = "lora"
    FULL_FINETUNED = "full_finetuned"


class ModelConfig(BaseModel):
    """개별 모델 설정"""
    model_type: ModelType
    model_path: str
    base_model_path: Optional[str] = None
    lora_modules: Optional[List[str]] = None
    gpu_memory_utilization: float = 0.8
    max_model_len: int = 1024
    max_num_batched_tokens: int = 2048
    max_num_seqs: int = 32
    served_model_name: Optional[str] = None


def get_default_models() -> Dict[str, ModelConfig]:
    """기본 지원 모델 설정 반환"""
    return {
        "haebo/Meow-HyperCLOVAX-1.5B_LoRA_fp16_0619i": ModelConfig(
            model_type=ModelType.LORA,
            model_path="haebo/Meow-HyperCLOVAX-1.5B_LoRA_fp16_0619i",
            base_model_path="naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B",
            lora_modules=["lora=haebo/Meow-HyperCLOVAX-1.5B_LoRA_fp16_0619i"],
            gpu_memory_utilization=0.6,   # 안정성 중심, 메모리 여유 확보
            max_model_len=1536,           # 한국어 600자 처리 
            max_num_batched_tokens=1536,  # 단일 요청 중심 배치
            max_num_seqs=8,               # 적은 동시 사용자 (8명)
            served_model_name="Meow-HyperCLOVAX-LoRA"
        ),
        "haebo/Meow-HyperCLOVAX-1.5B_FullFT_fp32_0619i": ModelConfig(
            model_type=ModelType.FULL_FINETUNED,
            model_path="haebo/Meow-HyperCLOVAX-1.5B_FullFT_fp32_0619i",
            base_model_path=None,
            lora_modules=None,
            gpu_memory_utilization=0.5,   # 낮은 메모리 
            max_model_len=1536,           # 한국어 600자 처리 
            max_num_batched_tokens=1536,  # 단일 요청 중심 배치
            max_num_seqs=12,              # 풀 파인튜닝 모델은 약간 더 많은 동시 처리
            served_model_name="Meow-HyperCLOVAX-FullFT"
        )
    }


def detect_active_model_from_server() -> Optional[str]:
    """실행 중인 vLLM 서버에서 활성 모델 감지"""
    try:
        response = httpx.get("http://localhost:8001/v1/models", timeout=5.0)
        if response.status_code == 200:
            models_data = response.json()
            if models_data.get("data"):
                # 첫 번째 모델의 root 정보에서 실제 모델 경로 추출
                model_info = models_data["data"][0]
                model_root = model_info.get("root", "")
                served_model_name = model_info.get("id", "")
                
                logger.info(f"🔍 서버에서 감지된 모델: {model_root} (서빙명: {served_model_name})")
                
                # 지원 모델 목록에서 매칭되는 모델 찾기
                default_models = get_default_models()
                for model_name, config in default_models.items():
                    if (model_root == config.model_path or 
                        model_root == config.base_model_path or
                        served_model_name == config.served_model_name):
                        logger.info(f"✅ 매칭된 모델: {model_name}")
                        return model_name
                
                logger.warning(f"⚠️ 지원되지 않는 모델: {model_root}")
                return None
    except Exception as e:
        logger.debug(f"서버 모델 감지 실패: {e}")
        return None
    
    return None


class VLLMConfig(BaseSettings):
    """vLLM 서버 설정 클래스 - 다중 모델 타입 지원"""
    
    # 서버 기본 설정
    host: str = Field(default="0.0.0.0", description="vLLM 서버 호스트")
    port: int = Field(default=8001, description="vLLM 서버 포트")
    
    # 지원 모델 설정 맵 (먼저 초기화)
    supported_models: Dict[str, ModelConfig] = Field(
        default_factory=get_default_models,
        description="지원하는 모델들의 설정"
    )
    
    # 현재 활성 모델 설정 - 동적으로 감지
    active_model: str = Field(
        default="",  # 빈 문자열로 초기화
        description="현재 사용할 모델 이름"
    )
    
    # 토큰화 설정
    trust_remote_code: bool = Field(default=True, description="원격 코드 신뢰")
    
    def __init__(self, **data):
        """초기화 시 active_model 동적 감지"""
        super().__init__(**data)
        
        # 1. 먼저 실행 중인 서버에서 모델 감지 시도
        if not self.active_model:
            detected_model = detect_active_model_from_server()
            if detected_model and detected_model in self.supported_models:
                self.active_model = detected_model
                logger.info(f"🎯 서버에서 자동 감지된 모델: {self.active_model}")
        
        # 2. 감지 실패 시 환경 변수 확인
        if not self.active_model:
            env_model = os.getenv("VLLM_ACTIVE_MODEL")
            if env_model and env_model in self.supported_models:
                self.active_model = env_model
                logger.info(f"🌍 환경 변수에서 설정된 모델: {self.active_model}")
        
        # 3. 여전히 없으면 기본값 사용
        if not self.active_model:
            available_models = list(self.supported_models.keys())
            if available_models:
                # 풀 파인튜닝 모델 우선, 없으면 첫 번째 모델
                fullft_models = [m for m in available_models if "FullFT" in m]
                self.active_model = fullft_models[0] if fullft_models else available_models[0]
                logger.info(f"🔧 기본 모델로 설정: {self.active_model}")
            else:
                raise ValueError("지원되는 모델이 없습니다.")
        
        # 최종 검증
        if self.active_model not in self.supported_models:
            available_models = list(self.supported_models.keys())
            raise ValueError(f"지원되지 않는 모델: {self.active_model}. 사용 가능한 모델: {available_models}")
    
    def get_current_model_config(self) -> ModelConfig:
        """현재 활성 모델의 설정 반환"""
        return self.supported_models[self.active_model]
    
    def is_lora_model(self) -> bool:
        """현재 모델이 LoRA 모델인지 확인"""
        return self.get_current_model_config().model_type == ModelType.LORA
    
    def is_full_finetuned_model(self) -> bool:
        """현재 모델이 풀 파인튜닝 모델인지 확인"""
        return self.get_current_model_config().model_type == ModelType.FULL_FINETUNED
    
    
    # 환경 변수 설정
    class Config:
        env_prefix = "VLLM_"
        env_file = None
        case_sensitive = False
        extra = "ignore"


class VLLMServerArgs:
    """vLLM 서버 실행 인자 생성 - 모델 타입별 최적화"""
    
    def __init__(self, config: VLLMConfig):
        self.config = config
        self.model_config = config.get_current_model_config()
    
    def get_server_args(self) -> list[str]:
        """모델 타입에 따른 최적화된 vLLM 서버 실행 인자"""
        if self.model_config.model_type == ModelType.LORA:
            return self._get_lora_args()
        elif self.model_config.model_type == ModelType.FULL_FINETUNED:
            return self._get_full_finetuned_args()
        else:
            raise ValueError(f"지원되지 않는 모델 타입: {self.model_config.model_type}")
    
    def _get_lora_args(self) -> list[str]:
        """LoRA 모델용 서버 인자"""
        args = [
            "--host", self.config.host,
            "--port", str(self.config.port),
            "--model", self.model_config.base_model_path,
            "--served-model-name", self.model_config.served_model_name,
            "--gpu-memory-utilization", str(self.model_config.gpu_memory_utilization),
            "--max-model-len", str(self.model_config.max_model_len),
            "--max-num-batched-tokens", str(self.model_config.max_num_batched_tokens),
            "--max-num-seqs", str(self.model_config.max_num_seqs),
            "--trust-remote-code",
            "--enable-lora",
            "--max-loras", "1",
            "--max-lora-rank", "16",
        ]
        
        # LoRA 모듈 추가
        if self.model_config.lora_modules:
            for lora_module in self.model_config.lora_modules:
                args.extend(["--lora-modules", lora_module])
        
        return args
    
    def _get_full_finetuned_args(self) -> list[str]:
        """풀 파인튜닝 모델용 서버 인자"""
        args = [
            "--host", self.config.host,
            "--port", str(self.config.port),
            "--model", self.model_config.model_path,
            "--served-model-name", self.model_config.served_model_name,
            "--gpu-memory-utilization", str(self.model_config.gpu_memory_utilization),
            "--max-model-len", str(self.model_config.max_model_len),
            "--max-num-batched-tokens", str(self.model_config.max_num_batched_tokens),
            "--max-num-seqs", str(self.model_config.max_num_seqs),
            "--trust-remote-code",
            # 풀 파인튜닝 모델 최적화 옵션
            "--enforce-eager",  # 메모리 최적화
            "--disable-custom-all-reduce",  # 안정성 향상
        ]
        
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


def switch_model(model_name: str) -> VLLMConfig:
    """활성 모델 전환"""
    global vllm_config
        if not vllm_config.validate_active_model(model_name):
        available_models = list(vllm_config.supported_models.keys())
        raise ValueError(f"지원되지 않는 모델: {model_name}. 사용 가능한 모델: {available_models}")
    
    vllm_config.active_model = model_name
    return vllm_config 