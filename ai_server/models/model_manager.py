"""
모델 초기화 및 관리를 위한 모듈
"""

import os
import logging
import threading
import torch
from typing import Dict, Any

from ai_server.models.huggingface_model import PostModel

# 로깅 설정
logger = logging.getLogger(__name__)

class ModelManager:
    """
    모델 관리 클래스 - 모델 초기화 및 리소스 관리
    """
    _instance = None
    _lock = threading.Lock()
    
    # 모델 로딩 설정 직접 정의
    PRELOAD_MODELS = False  # GPU 메모리 절약을 위해 비활성화
    
    def __new__(cls):
        """싱글톤 패턴 구현"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """초기화 (스레드 안전)"""
        with self._lock:
            if self._initialized:
                return
            
            self._initialized = True
            self._models: Dict[str, Any] = {}
            
            # GPU 설정 및 초기화
            self._setup_gpu()
            
            # 모델 로딩 상태 추적
            self._model_ready: Dict[str, bool] = {"post": False}
            
            logger.info("ModelManager 초기화 완료")
            
            # 사전 로드 설정이 켜져 있으면 바로 모델 로드
            if self.PRELOAD_MODELS:
                self._load_models()
    
    def _setup_gpu(self):
        """GPU 설정 및 초기화"""
        # CUDA 초기화 전 설정
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        
        # GPU 정보 로깅
        gpu_count = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        
        logger.info(f"사용 가능한 GPU: {gpu_count}개")
        for i, name in enumerate(gpu_names):
            logger.info(f"GPU {i}: {name}")
            
        # CUDA 초기화
        torch.cuda.empty_cache()
        logger.info("CUDA 초기화 완료")
            
    def _load_models(self):
        """모든 모델 직접 로드"""
        try:
            logger.info("모델 로드 시작")
            
            # GPU 메모리 정리
            torch.cuda.empty_cache()
            
            # Post 모델 로드
            if "post" not in self._models:
                model = PostModel()
                self._models["post"] = model
                self._model_ready["post"] = True
                logger.info("Post 모델 로드 완료")
                
        except Exception as e:
            logger.error(f"모델 로드 실패: {str(e)}")
    
    def initialize_models(self, preload: bool = None):
        """
        모델 초기화
        
        Args:
            preload: 서버 시작 시 모델 미리 로드 여부 (기본값: 설정에서 가져옴)
        """
        # 설정에서 preload 값을 가져옴
        should_preload = preload if preload is not None else self.PRELOAD_MODELS
        
        if should_preload and not self._model_ready.get("post", False):
            self._load_models()
    
    async def get_post_model_async(self) -> PostModel:
        """
        Post 모델 인스턴스 비동기 반환
        
        Returns:
            PostModel 인스턴스
        """
        with self._lock:
            # 모델이 로드되어 있지 않으면 로드
            if not self._model_ready.get("post", False):
                self._load_models()
                
            # 모델 반환
            if "post" in self._models and self._model_ready["post"]:
                return self._models["post"]
            else:
                # 로딩 실패한 경우
                logger.error("Post 모델 로드 실패")
                raise RuntimeError("Post 모델 로드 실패")
    
    def is_model_ready(self, model_name: str) -> bool:
        """
        모델 로드 상태 확인
        
        Args:
            model_name: 확인할 모델 이름
            
        Returns:
            모델 로드 완료 여부
        """
        return self._model_ready.get(model_name, False)
    
    def cleanup(self):
        """모델 리소스 정리 (메모리 해제)"""
        logger.info("모델 리소스 정리 시작")
        
        with self._lock:
            # 모든 모델 인스턴스 언로드
            for model_name, model in list(self._models.items()):
                if hasattr(model, "unload"):
                    try:
                        model.unload()
                        logger.info(f"{model_name} 모델 언로드 완료")
                    except Exception as e:
                        logger.error(f"{model_name} 모델 언로드 실패: {str(e)}")
            
            # 모델 리스트 초기화
            self._models.clear()
            self._model_ready = {"post": False}
        
        # GPU 메모리 정리
        torch.cuda.empty_cache()
        logger.info("GPU 메모리 정리 완료")
            
        logger.info("모델 리소스 정리 완료")

def get_model_manager() -> ModelManager:
    """
    ModelManager 인스턴스 반환 (싱글톤)
    
    Returns:
        ModelManager 인스턴스
    """
    return ModelManager()