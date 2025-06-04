"""
허깅페이스 Meow-HyperCLOVAX 모델 구현 (T4 GPU 최적화 버전)
"""

import logging
import os
import torch
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import PeftModel
from ai_server.config import get_settings

# 로깅 설정
logger = logging.getLogger(__name__)

class PostModel:
    """
    파인튜닝된 Meow-HyperCLOVAX 모델을 로드하고 관리하는 클래스 (T4 GPU 전용)
    """
    # QLoRA 모델 경로 및 파라미터 직접 정의
    QLORA_MODEL_PATH = "haebo/Meow-HyperCLOVAX-1.5B_QLoRA_nf4_0527"
    BASE_MODEL_PATH = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"  # QLoRA의 베이스 모델
    MODEL_MAX_LENGTH = 512
    MODEL_MAX_NEW_TOKENS = 256
    MODEL_TEMPERATURE = 0.4
    MODEL_TOP_P = 0.9
    GPU_MEMORY_FRACTION = 0.9  # T4 GPU 메모리 사용 비율
    
    def __init__(self):
        """
        QLoRA 모델 초기화
        """
        # 모델 경로 설정
        self.qlora_model_path = self.QLORA_MODEL_PATH
        self.base_model_path = self.BASE_MODEL_PATH
        
        # 인증 토큰은 환경 변수에서 가져옴
        settings = get_settings()
        self.auth_token = settings.HUGGINGFACE_TOKEN
        
        # GPU 환경 최적화 설정
        self._setup_gpu()
        
        logger.info(f"QLoRA 모델 로드 시작: {self.qlora_model_path}")
        
        # QLoRA 모델 로드
        self._load_qlora_model()
        
        # 생성 설정 저장
        self.max_new_tokens = self.MODEL_MAX_NEW_TOKENS
        self.temperature = self.MODEL_TEMPERATURE
        self.top_p = self.MODEL_TOP_P
        
        logger.info(f"QLoRA 모델 로드 완료: {self.qlora_model_path}")
        self._initialized = True
    
    def _setup_gpu(self):
        """
        GPU 설정 초기화 및 최적화
        """
        # GPU 메모리 분할 크기 최적화
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        
        # CUDA 메모리 캐시 설정
        torch.cuda.empty_cache()
        
        # GPU 메모리 정보 로깅
        mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        mem_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU: {mem_name}, 메모리: 총 {mem_total:.2f}GB")
        logger.info(f"GPU 메모리 사용 제한: {int(self.GPU_MEMORY_FRACTION * 100)}%")
        logger.info("GPU 초기화 완료")
    
    def _load_qlora_model(self):
        """
        QLoRA 파인튜닝된 모델을 올바르게 로드
        """
        # 4bit 양자화 설정
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        # 베이스 모델 로드 (4bit 양자화)
        logger.info(f"베이스 모델 로드 중: {self.base_model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            token=self.auth_token,
            low_cpu_mem_usage=True,
            max_memory={0: f"{int(self.GPU_MEMORY_FRACTION * 100)}%"}
        )
        
        # QLoRA 어댑터 로드
        logger.info(f"QLoRA 어댑터 로드 중: {self.qlora_model_path}")
        self.model = PeftModel.from_pretrained(
            base_model,
            self.qlora_model_path,
            token=self.auth_token
        )
        
        # 토크나이저 로드 (QLoRA 모델에서)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.qlora_model_path,
            token=self.auth_token,
            model_max_length=self.MODEL_MAX_LENGTH
        )
        
        # 패딩 토큰 설정 (필요한 경우)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 파이프라인 설정
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device="cuda:0"
        )
    
    async def generate(
        self, 
        prompt: str, 
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        프롬프트를 기반으로 텍스트 생성
        """
        # 생성 설정
        generation_config = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "temperature": temperature or self.temperature,
            "top_p": top_p or self.top_p,
            "do_sample": True,
            "return_full_text": False,
            "pad_token_id": self.tokenizer.eos_token_id,
            **kwargs
        }
        
        # 텍스트 생성 및 결과 반환
        result = self.generator(prompt, **generation_config)
        return result[0]["generated_text"].strip()
    
    def unload(self):
        """모델 메모리에서 언로드"""
        if hasattr(self, "model"):
            del self.model
            del self.tokenizer
            del self.generator
            self._initialized = False
            torch.cuda.empty_cache()
            logger.info("QLoRA 모델 언로드 완료") 