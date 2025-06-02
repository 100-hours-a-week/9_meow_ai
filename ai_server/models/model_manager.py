"""
모델 초기화 및 관리를 위한 모듈
"""

import os
import logging
import threading
import asyncio
import torch
from typing import Dict, Any, Optional

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
            self._loading_threads: Dict[str, threading.Thread] = {}
            self._loading_events: Dict[str, threading.Event] = {}
            
            # 모델 로딩 상태 추적
            self._model_ready: Dict[str, bool] = {"post": False}
            
            # GPU 설정 및 초기화
            self._setup_gpu()
            
            logger.info("ModelManager 초기화 완료")
    
    def _setup_gpu(self):
        """GPU 설정 및 초기화"""
        try:
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
                
            # GPU 메모리 사용 제한 설정
            if self.settings.GPU_MEMORY_FRACTION < 1.0:
                logger.info(f"GPU 메모리 사용 제한: {int(self.settings.GPU_MEMORY_FRACTION * 100)}%")
                
        except Exception as e:
            logger.error(f"GPU 설정 초기화 중 오류 발생: {str(e)}")
    
    def initialize_models(self, preload: bool = None):
        """
        모델 초기화 - 백그라운드 스레드에서 모델 로드
        
        Args:
            preload: 서버 시작 시 모델 미리 로드 여부 (기본값: 설정에서 가져옴)
        """
        # 설정에서 preload 값을 가져옴
        should_preload = preload if preload is not None else self.settings.PRELOAD_MODELS
        
        if should_preload:
            # Post 모델 백그라운드에서 로드
            if "post" not in self._models and "post" not in self._loading_threads:
                logger.info("Post 모델 백그라운드 로드 시작")
                
                # 로딩 완료 이벤트 생성
                loading_event = threading.Event()
                self._loading_events["post"] = loading_event
                
                # 백그라운드 스레드 생성 및 시작
                thread = threading.Thread(
                    target=self._load_model_in_background,
                    args=("post",),
                    daemon=True
                )
                self._loading_threads["post"] = thread
                thread.start()
    
    def _load_model_in_background(self, model_name: str):
        """
        백그라운드 스레드에서 모델 로드
        
        Args:
            model_name: 로드할 모델 이름
        """
        try:
            logger.info(f"{model_name} 모델 백그라운드 로드 시작")
            
            # GPU 메모리 정리
            torch.cuda.empty_cache()
            
            # 모델 인스턴스 생성
            if model_name == "post":
                model = PostModel()
                
                # 로드 완료 후 모델 저장
                with self._lock:
                    self._models[model_name] = model
                    self._model_ready[model_name] = True
                
                # 로딩 완료 이벤트 설정
                if model_name in self._loading_events:
                    self._loading_events[model_name].set()
                
                logger.info(f"{model_name} 모델 백그라운드 로드 완료")
        except Exception as e:
            logger.error(f"{model_name} 모델 백그라운드 로드 실패: {str(e)}")
            # 로딩 실패 시에도 이벤트 설정
            if model_name in self._loading_events:
                self._loading_events[model_name].set()
    
    def get_post_model(self) -> PostModel:
        """
        Post 모델 인스턴스 반환 - 모델이 로드 중이면 완료될 때까지 대기
        
        Returns:
            PostModel 인스턴스
        """
        with self._lock:
            # 이미 로드된 모델이 있으면 즉시 반환
            if "post" in self._models and self._model_ready["post"]:
                return self._models["post"]
            
            # 모델이 백그라운드에서 로드 중이면
            if "post" in self._loading_threads and self._loading_threads["post"].is_alive():
                logger.info("Post 모델이 로드 중입니다. 완료될 때까지 대기합니다.")
            else:
                # 모델 로드가 시작되지 않았으면 새로 시작
                logger.info("Post 모델 로드 시작")
                # 로딩 완료 이벤트 생성
                loading_event = threading.Event()
                self._loading_events["post"] = loading_event
                
                # 백그라운드 스레드 생성 및 시작
                thread = threading.Thread(
                    target=self._load_model_in_background,
                    args=("post",),
                    daemon=True
                )
                self._loading_threads["post"] = thread
                thread.start()
        
        # 로딩 완료될 때까지 대기
        if "post" in self._loading_events:
            self._loading_events["post"].wait()
        
        # 로드 완료 후 모델 반환
        with self._lock:
            if "post" in self._models and self._model_ready["post"]:
                return self._models["post"]
            else:
                # 로딩 실패한 경우
                logger.error("Post 모델 로드 실패")
                raise RuntimeError("Post 모델 로드 실패")
    
    async def get_post_model_async(self) -> PostModel:
        """
        Post 모델 인스턴스 비동기 반환 - 모델이 로드 중이면 완료될 때까지 비동기 대기
        
        Returns:
            PostModel 인스턴스
        """
        with self._lock:
            # 이미 로드된 모델이 있으면 즉시 반환
            if "post" in self._models and self._model_ready["post"]:
                return self._models["post"]
            
            # 모델이 백그라운드에서 로드 중이면
            if "post" in self._loading_threads and self._loading_threads["post"].is_alive():
                logger.info("Post 모델이 로드 중입니다. 완료될 때까지 비동기 대기합니다.")
            else:
                # 모델 로드가 시작되지 않았으면 새로 시작
                logger.info("Post 모델 로드 시작")
                # 로딩 완료 이벤트 생성
                loading_event = threading.Event()
                self._loading_events["post"] = loading_event
                
                # 백그라운드 스레드 생성 및 시작
                thread = threading.Thread(
                    target=self._load_model_in_background,
                    args=("post",),
                    daemon=True
                )
                self._loading_threads["post"] = thread
                thread.start()
        
        # 비동기 대기 함수
        async def wait_for_event():
            while not self._loading_events["post"].is_set():
                await asyncio.sleep(0.1)
        
        # 로딩 완료될 때까지 비동기 대기
        if "post" in self._loading_events:
            await wait_for_event()
        
        # 로드 완료 후 모델 반환
        with self._lock:
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
        with self._lock:
            return model_name in self._model_ready and self._model_ready[model_name]
    
    def cleanup(self):
        """모델 리소스 정리 (메모리 해제)"""
        try:
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
        except Exception as e:
            logger.error(f"모델 리소스 정리 중 오류 발생: {str(e)}")

def get_model_manager() -> ModelManager:
    """
    ModelManager 인스턴스 반환 (싱글톤)
    
    Returns:
        ModelManager 인스턴스
    """
    return ModelManager()