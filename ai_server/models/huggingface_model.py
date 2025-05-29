"""
허깅페이스 Meow-HyperCLOVAX 모델 구현
"""

import logging
import os
from typing import Optional, Union, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from ai_server.config import get_settings

# 로깅 설정
logger = logging.getLogger(__name__)

class PostModel:
    """
    파인튜닝된 Meow-HyperCLOVAX 모델을 로드하고 관리하는 클래스
    """
    # 클래스 변수로 settings를 한 번만 로드 (모든 인스턴스가 공유)
    settings = get_settings()
    
    def __init__(self):
        """
        모델 초기화 - 설정 파일의 값만 사용
        """
        # 설정 파일에서 값 가져오기
        self.model_path = self.settings.HUGGINGFACE_MODEL_PATH
        self.auth_token = self.settings.HUGGINGFACE_TOKEN
        
        # GPU 메모리 설정 가져오기
        self.gpu_memory = self.settings.GPU_MEMORY_FRACTION
        
        # 디바이스 설정 및 CUDA 초기화 (GCP 환경 최적화)
        self._setup_device()
        
        logger.info(f"모델 로드 시작: {self.model_path}, 디바이스: {self.device}")
        
        try:
            # 토크나이저 로드 - max_length 설정
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                token=self.auth_token,
                model_max_length=self.settings.MODEL_MAX_LENGTH
            )
            
            # 모델 로드 (GCP GPU 환경에 최적화)
            self._load_model()
            
            # 생성 파이프라인 설정
            self._setup_pipeline()
            
            # 생성 설정 저장 (설정 파일 값 사용)
            self.max_new_tokens = self.settings.MODEL_MAX_NEW_TOKENS
            self.temperature = self.settings.MODEL_TEMPERATURE
            self.top_p = self.settings.MODEL_TOP_P
            
            logger.info(f"모델 로드 완료: {self.model_path}")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"모델 로드 실패: {str(e)}")
            raise RuntimeError(f"모델 로드 실패: {str(e)}")
    
    def _setup_device(self):
        """
        GPU 사용 가능 여부 확인 및 CUDA 설정 초기화
        GCP 환경에 최적화된 설정 적용
        """
        # 환경 변수 설정
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        
        # CUDA 가용성 체크
        if torch.cuda.is_available():
            try:
                # GPU 정보 로깅
                gpu_count = torch.cuda.device_count()
                gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
                logger.info(f"감지된 GPU: {gpu_count}개, 모델: {gpu_names}")
                
                # GPU 메모리 정보 로깅 (가능한 경우)
                try:
                    for i in range(gpu_count):
                        mem_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB 단위
                        mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                        mem_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                        logger.info(f"GPU {i} 메모리: 총 {mem_total:.2f}GB, 예약됨 {mem_reserved:.2f}GB, 할당됨 {mem_allocated:.2f}GB")
                except Exception as e:
                    logger.warning(f"GPU 메모리 정보 조회 실패: {str(e)}")
                
                # CUDA 초기화 테스트
                test_tensor = torch.tensor([1.0]).cuda()
                del test_tensor
                torch.cuda.empty_cache()
                
                # GPU 사용 가능
                self.device = "cuda"
                logger.info("CUDA GPU를 사용합니다.")
            except Exception as e:
                # CUDA 초기화 실패 시 CPU로 폴백
                logger.warning(f"CUDA 초기화 실패 (CPU로 전환): {str(e)}")
                self.device = "cpu"
        else:
            # GPU 사용 불가능
            self.device = "cpu"
            logger.warning("GPU를 찾을 수 없습니다. CPU를 사용합니다.")
    
    def _load_model(self):
        """
        모델 로드 - 디바이스 및 메모리 설정에 맞게 최적화
        """
        model_kwargs = {
            "token": self.auth_token,
            "device_map": "auto",  # 자동 장치 할당
        }
        
        if self.device == "cuda":
            # GPU 설정
            torch_dtype = torch.float16  # GCP 환경에서는 float16이 성능 최적화에 유리
            
            # 메모리 제한이 있는 경우 설정 추가
            if self.gpu_memory and self.gpu_memory < 1.0:
                model_kwargs["max_memory"] = {0: f"{int(self.gpu_memory * 100)}%"}
                logger.info(f"GPU 메모리 사용량 제한: {int(self.gpu_memory * 100)}%")
            
            model_kwargs["torch_dtype"] = torch_dtype
            logger.info(f"GPU 모델 로드 설정: {model_kwargs}")
            
            # 그래디언트 계산 비활성화 (추론 모드)
            with torch.no_grad():
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    **model_kwargs
                )
        else:
            # CPU 설정
            model_kwargs["torch_dtype"] = torch.float32  # CPU에서는 float32가 안정적
            logger.info(f"CPU 모델 로드 설정: {model_kwargs}")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )
    
    def _setup_pipeline(self):
        """
        생성 파이프라인 설정
        """
        if self.device == "cuda":
            # GPU 파이프라인 설정
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device="cuda:0"  # 첫 번째 GPU 사용
            )
        else:
            # CPU 파이프라인 설정
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1  # CPU 사용
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
        
        Args:
            prompt: 생성을 위한 프롬프트
            max_new_tokens: 최대 생성 토큰 수 (기본값: 설정 값 사용)
            temperature: 생성 온도 (기본값: 설정 값 사용)
            top_p: 누적 확률 임계값 (기본값: 설정 값 사용)
            **kwargs: 추가 생성 파라미터
            
        Returns:
            생성된 텍스트
        """
        if not hasattr(self, "_initialized") or not self._initialized:
            raise RuntimeError("모델이 초기화되지 않았습니다.")
        
        try:
            # 생성 파라미터 설정 
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
                logger.info("GPU 메모리 정리 완료")
            logger.info("모델 언로드 완료") 