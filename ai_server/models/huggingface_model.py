"""
허깅페이스 Meow-HyperCLOVAX 풀파인튜닝 모델 구현 (T4 GPU 최적화 버전)
"""

import logging
import os
import torch
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from ai_server.config import get_settings

# 로깅 설정
logger = logging.getLogger(__name__)

class PostModel:
    """
    풀파인튜닝된 Meow-HyperCLOVAX 모델을 로드하고 관리하는 클래스 (T4 GPU 전용)
    """
    # 풀파인튜닝 모델 경로 및 파라미터 정의
    FINE_TUNED_MODEL_PATH = "haebo/Meow-HyperCLOVAX-1.5B_FullFT_fp32_0527"
    MODEL_MAX_LENGTH = 2048 
    MODEL_MAX_NEW_TOKENS = 512  
    MODEL_TEMPERATURE = 0.4
    MODEL_TOP_P = 0.9
    
    def __init__(self):
        """
        풀파인튜닝 모델 초기화
        """
        # accelerate 관련 환경 변수 설정 (None 오류 방지)
        os.environ.setdefault("ACCELERATE_USE_CPU", "false")
        os.environ.setdefault("ACCELERATE_USE_DEEPSPEED", "false")
        os.environ.setdefault("ACCELERATE_USE_FSDP", "false")
        
        # 모델 경로 설정
        self.fine_tuned_model_path = self.FINE_TUNED_MODEL_PATH
        
        # 인증 토큰은 환경 변수에서 가져옴
        settings = get_settings()
        self.auth_token = settings.HUGGINGFACE_TOKEN
        
        # GPU 환경 최적화 설정
        self._setup_gpu()
        
        logger.info(f"파인튜닝 모델 로드 시작: {self.fine_tuned_model_path}")
        
        # 풀파인튜닝 모델 로드
        self._load_fine_tuned_model()
        
        # 생성 설정 저장
        self.max_new_tokens = self.MODEL_MAX_NEW_TOKENS
        self.temperature = self.MODEL_TEMPERATURE
        self.top_p = self.MODEL_TOP_P
        
        logger.info(f"파인튜닝 모델 로드 완료: {self.fine_tuned_model_path}")
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
        logger.info("GPU 초기화 완료")
    
    def _load_fine_tuned_model(self):
        """
        파인튜닝된 모델을 직접 로드
        """
        # 풀파인튜닝된 모델 직접 로드 (32비트)
        logger.info(f"파인튜닝 모델 로드 중: {self.fine_tuned_model_path}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.fine_tuned_model_path,
            device_map="auto",  # 자동 디바이스 매핑
            torch_dtype=torch.float32,  # 32비트 정밀도
            token=self.auth_token,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.fine_tuned_model_path,
            token=self.auth_token,
            model_max_length=self.MODEL_MAX_LENGTH,
            trust_remote_code=True
        )
        
        # 패딩 토큰 설정 (필요한 경우)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 모델을 평가 모드로 설정
        self.model.eval()
        
        logger.info("풀파인튜닝 모델과 토크나이저 로드 완료")
    
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
        
        Args:
            prompt: 입력 프롬프트
            max_new_tokens: 최대 생성 토큰 수
            temperature: 생성 온도 (창의성 조절)
            top_p: nucleus sampling 파라미터
            **kwargs: 추가 생성 파라미터
            
        Returns:
            생성된 텍스트
        """
        # 생성 파라미터 설정
        max_new_tokens = max_new_tokens or self.max_new_tokens
        temperature = temperature or self.temperature
        top_p = top_p or self.top_p
        
        # 입력 토큰화
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        
        # 어텐션 마스크 생성
        attention_mask = torch.ones_like(inputs)
        
        # 텍스트 생성
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,  # 반복 방지
                **kwargs
            )
        
        # 새로 생성된 토큰만 디코딩 (입력 제외)
        generated_tokens = outputs[0][inputs.shape[1]:]
        result = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return result.strip()
    
    def get_model_info(self) -> dict:
        """
        모델 정보 반환
        
        Returns:
            모델 정보 딕셔너리
        """
        return {
            "model_path": self.fine_tuned_model_path,
            "max_length": self.MODEL_MAX_LENGTH,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "device": str(self.model.device) if hasattr(self, 'model') else "not_loaded",
            "initialized": getattr(self, '_initialized', False)
        }
    
    def unload(self):
        """모델 메모리에서 언로드"""
        if hasattr(self, "model"):
            del self.model
            del self.tokenizer
            self._initialized = False
            torch.cuda.empty_cache()
            logger.info("풀파인튜닝 모델 언로드 완료") 