"""
모델 초기화 및 관리를 위한 모듈
"""

import os
import logging
import threading
from typing import Optional, Dict, Any

from ai_server.models.huggingface_model import PostModel
from ai_server.config import get_settings

# 로깅 설정
logger = logging.getLogger(__name__)

class ModelManager:
    """
    모델 관리 클래스 - 모델 초기화 및 리소스 관리
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """싱글톤 패턴 구현 (스레드 안전)"""
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
                
            # 설정 로드
            self.settings = get_settings()
            
            self._initialized = True
            self._models: Dict[str, Any] = {}
            logger.info("ModelManager 초기화 완료")
    
    def initialize_models(self, preload: bool = None):
        """
        모델 초기화
        
        Args:
            preload: 서버 시작 시 모델 미리 로드 여부 (기본값: 설정에서 가져옴)
        """
        with self._lock:
            try:
                # 설정에서 preload 값을 가져옴
                should_preload = preload if preload is not None else self.settings.PRELOAD_MODELS
                
                if should_preload:
                    # Post 모델 미리 로드
                    if "post" not in self._models:
                        self._models["post"] = PostModel()
                        logger.info("Post 모델 사전 로드 완료")
            except Exception as e:
                logger.error(f"모델 초기화 실패: {str(e)}")
                raise RuntimeError(f"모델 초기화 실패: {str(e)}")
    
    def get_post_model(self) -> PostModel:
        """
        Post 모델 인스턴스 반환
        
        Returns:
            PostModel 인스턴스
        """
        with self._lock:
            if "post" not in self._models:
                logger.info("Post 모델 로드 시작")
                try:
                    self._models["post"] = PostModel()
                    logger.info("Post 모델 로드 완료")
                except Exception as e:
                    logger.error(f"Post 모델 로드 실패: {str(e)}")
                    raise RuntimeError(f"Post 모델 로드 실패: {str(e)}")
            return self._models["post"]
    
    def cleanup(self):
        """
        모든 모델 정리 및 리소스 해제
        """
        with self._lock:
            for model_name, model in self._models.items():
                try:
                    if isinstance(model, PostModel):
                        model.unload()
                        logger.info(f"{model_name} 모델 정리 완료")
                except Exception as e:
                    logger.error(f"{model_name} 모델 정리 실패: {str(e)}")
            
            self._models = {}

# 싱글톤 인스턴스 가져오기 헬퍼 함수
def get_model_manager() -> ModelManager:
    """
    ModelManager 싱글톤 인스턴스 반환
    
    Returns:
        ModelManager 인스턴스
    """
    return ModelManager() 