"""
허깅페이스 Meow-HyperCLOVAX 모델 구현
"""

import logging
import os
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from ai_server.config import get_settings

# 로깅 설정
logger = logging.getLogger(__name__)

class PostModel:
    """
    파인튜닝된 Meow-HyperCLOVAX 모델을 로드하고 관리하는 클래스
    """
    # 클래스 변수로 settings를 한 번만 로드
    settings = get_settings()
    
    def __init__(self):
        """
        모델 초기화 - GPU 환경에 최적화
        """
        # 설정 파일에서 값 가져오기
        self.model_path = self.settings.HUGGINGFACE_MODEL_PATH
        self.auth_token = self.settings.HUGGINGFACE_TOKEN
        
        # GPU 환경 최적화 설정
        self._setup_gpu()
        
        logger.info(f"모델 로드 시작: {self.model_path}")
        
        try:
            # 토크나이저 및 모델 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                token=self.auth_token,
                model_max_length=self.settings.MODEL_MAX_LENGTH
            )
            
            self._load_model()
            
            # 파이프라인 설정
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device="cuda:0"
            )
            
            # 생성 설정 저장
            self.max_new_tokens = self.settings.MODEL_MAX_NEW_TOKENS
            self.temperature = self.settings.MODEL_TEMPERATURE
            self.top_p = self.settings.MODEL_TOP_P
            
            logger.info(f"모델 로드 완료: {self.model_path}")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"모델 로드 실패: {str(e)}")
            raise RuntimeError(f"모델 로드 실패: {str(e)}")
    
    def _setup_gpu(self):
        """
        GPU 설정 초기화 및 최적화
        """
        # GPU 메모리 분할 크기 최적화
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        
        # CUDA 메모리 캐시 설정
        torch.cuda.empty_cache()
        
        # GPU 메모리 정보 로깅
        try:
            mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU 메모리: 총 {mem_total:.2f}GB")
        except Exception as e:
            logger.warning(f"GPU 메모리 정보 조회 실패: {str(e)}")
        
        logger.info("GPU 초기화 완료")
    
    def _load_model(self):
        """
        GPU에 최적화된 모델 로드
        """
        # 최적화 설정
        model_kwargs = {
            "token": self.auth_token,
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True
        }
        
        # 메모리 제한 설정
        gpu_memory = self.settings.GPU_MEMORY_FRACTION
        if gpu_memory and gpu_memory < 1.0:
            model_kwargs["max_memory"] = {0: f"{int(gpu_memory * 100)}%"}
        
        # 그래디언트 계산 비활성화 (추론 모드)
        with torch.no_grad():
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
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
        if not hasattr(self, "_initialized") or not self._initialized:
            raise RuntimeError("모델이 초기화되지 않았습니다.")
        
        try:
            # 생성 설정
            generation_config = {
                "max_new_tokens": max_new_tokens or self.max_new_tokens,
                "temperature": temperature or self.temperature,
                "top_p": top_p or self.top_p,
                "do_sample": True,
                "return_full_text": False,
                **kwargs
            }
            
            # 텍스트 생성 및 결과 반환
            result = self.generator(prompt, **generation_config)
            return result[0]["generated_text"].strip()
            
        except Exception as e:
            logger.error(f"텍스트 생성 실패: {str(e)}")
            raise RuntimeError(f"텍스트 생성 실패: {str(e)}")
    
    def unload(self):
        """모델 메모리에서 언로드"""
        if hasattr(self, "model") and self.model is not None:
            del self.model
            del self.tokenizer
            del self.generator
            self._initialized = False
            torch.cuda.empty_cache()
            logger.info("모델 언로드 완료") 