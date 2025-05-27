"""
허깅페이스 Meow-HyperCLOVAX 모델 구현
"""


import logging
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
    def __init__(
        self, 
        model_path: str = None, 
        device: str = None,
        use_auth_token: bool = False,
        max_new_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        **kwargs
    ):
        """
        모델 초기화
        
        Args:
            model_path: 모델 경로 또는 이름 (기본값: 설정에서 가져옴)
            device: 모델 실행 디바이스 (기본값: 자동 감지)
            use_auth_token: 허깅페이스 토큰 사용 여부
            max_new_tokens: 최대 생성 토큰 수 (기본값: 설정에서 가져옴)
            temperature: 생성 온도 (기본값: 설정에서 가져옴)
            top_p: 누적 확률 임계값 (기본값: 설정에서 가져옴)
            **kwargs: 추가 파라미터
        """
        # 설정 로드
        settings = get_settings()
            
        # 설정에서 값 로드
        self.model_path = model_path or settings.HUGGINGFACE_MODEL_PATH
        self.auth_token = settings.HUGGINGFACE_TOKEN if use_auth_token else None
        
        # 디바이스 설정 (자동 감지)
        if device is not None:
            # 사용자가 명시적으로 지정한 경우
            self.device = device
        elif torch.cuda.is_available():
            # GPU 사용 가능
            self.device = "cuda"
            logger.info("CUDA GPU를 사용합니다.")
        else:
            # GPU 사용 불가능
            self.device = "cpu"
            logger.warning("CPU를 사용.")
            
        logger.info(f"모델 로드 시작: {self.model_path}, 디바이스: {self.device}")
        
        try:
            # 토크나이저 로드 - max_length 설정
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                token=self.auth_token,
                model_max_length=settings.MODEL_MAX_LENGTH
            )
            
            # 모델 로드
            if self.device == "cuda":
                # GPU용 설정
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    token=self.auth_token,
                    torch_dtype=torch.float32
                ).cuda()
            else:
                # CPU용 설정
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    token=self.auth_token,
                    torch_dtype=torch.float16
                )
            
            # 생성 파이프라인 설정
            device_id = 0 if self.device == "cuda" else -1
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device_id
            )
            
            # 생성 설정 - max_length 대신 max_new_tokens 사용
            self.max_new_tokens = max_new_tokens or settings.MODEL_MAX_NEW_TOKENS
            self.temperature = temperature or settings.MODEL_TEMPERATURE
            self.top_p = top_p or settings.MODEL_TOP_P
            
            logger.info(f"모델 로드 완료: {self.model_path}")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"모델 로드 실패: {str(e)}")
            raise RuntimeError(f"모델 로드 실패: {str(e)}")
    
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
            prompt: 생성을 위한 프롬프트
            max_new_tokens: 최대 생성 토큰 수 (기본값: 초기화 값 사용)
            temperature: 생성 온도 (기본값: 초기화 값 사용)
            top_p: 누적 확률 임계값 (기본값: 초기화 값 사용)
            **kwargs: 추가 생성 파라미터
            
        Returns:
            생성된 텍스트
        """
        if not hasattr(self, "_initialized") or not self._initialized:
            raise RuntimeError("모델이 초기화되지 않았습니다.")
        
        try:
            # 생성 파라미터 설정 - max_length 대신 max_new_tokens 사용
            generation_config = {
                "max_new_tokens": max_new_tokens or self.max_new_tokens,
                "temperature": temperature or self.temperature,
                "top_p": top_p or self.top_p,
                "do_sample": True,
                "return_full_text": False,  # 입력 프롬프트 제외하고 결과만 반환
                **kwargs
            }
            
            # 텍스트 생성
            result = self.generator(
                prompt, 
                **generation_config
            )
            
            # 결과 반환 (첫 번째 생성 결과 사용)
            generated_text = result[0]["generated_text"].strip()
            return generated_text
            
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
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("모델 언로드 완료") 