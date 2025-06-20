"""
vLLM 서버 런처
다중 모델 타입 지원 (LoRA, 풀 파인튜닝)
"""

import os
import sys
import subprocess
import signal
import time
import logging
from typing import Optional
from pathlib import Path
import requests
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError

from .vllm_config import VLLMConfig, VLLMServerArgs, get_vllm_config, ModelType, switch_model


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelDetector:
    """모델 타입 자동 감지 클래스"""
    
    @staticmethod
    def detect_model_type(model_path: str) -> ModelType:
        """모델 경로로부터 모델 타입 감지"""
        try:
            # 허깅페이스 모델인 경우
            if "/" in model_path and not model_path.startswith(("./", "/")):
                return ModelDetector._detect_hf_model_type(model_path)
            
            # 로컬 모델인 경우
            return ModelDetector._detect_local_model_type(model_path)
            
        except Exception as e:
            logger.warning(f"모델 타입 감지 실패: {e}")
            raise ValueError(f"지원되지 않는 모델 타입: {model_path}")
    
    @staticmethod
    def _detect_hf_model_type(model_path: str) -> ModelType:
        """허깅페이스 모델 타입 감지"""
        try:
            api = HfApi()
            
            # 모델 정보 조회
            model_info = api.model_info(model_path)
            
            # 태그 기반 감지
            if model_info.tags:
                tags = [tag.lower() for tag in model_info.tags]
                if "lora" in tags or "peft" in tags:
                    return ModelType.LORA
                elif "fine-tuned" in tags or "finetuned" in tags:
                    return ModelType.FULL_FINETUNED
            
            # 파일 구조 기반 감지
            try:
                files = api.list_repo_files(model_path)
                
                # LoRA 관련 파일 확인
                lora_files = [
                    "adapter_config.json", 
                    "adapter_model.bin", 
                    "adapter_model.safetensors",
                    "peft_config.json"
                ]
                
                if any(f in files for f in lora_files):
                    return ModelType.LORA
                
                # 풀 모델 파일 확인
                full_model_files = [
                    "pytorch_model.bin",
                    "model.safetensors", 
                    "pytorch_model-00001-of-*.bin"
                ]
                
                if any(any(f.startswith(pattern.split('*')[0]) for f in files) 
                       for pattern in full_model_files):
                    return ModelType.FULL_FINETUNED
                    
            except Exception as e:
                logger.debug(f"파일 목록 조회 실패: {e}")
            
            # 모델명 기반 추론
            model_name_lower = model_path.lower()
            if "lora" in model_name_lower or "peft" in model_name_lower:
                return ModelType.LORA
            elif "ft" in model_name_lower or "finetun" in model_name_lower:
                return ModelType.FULL_FINETUNED
                
            raise ValueError(f"지원되지 않는 모델 타입: {model_path}")
            
        except RepositoryNotFoundError:
            logger.error(f"모델을 찾을 수 없습니다: {model_path}")
            raise ValueError(f"모델을 찾을 수 없습니다: {model_path}")
        except Exception as e:
            logger.error(f"허깅페이스 모델 감지 실패: {e}")
            raise ValueError(f"모델 감지 실패: {model_path}")
    
    @staticmethod
    def _detect_local_model_type(model_path: str) -> ModelType:
        """로컬 모델 타입 감지"""
        path = Path(model_path)
        
        if not path.exists():
            logger.warning(f"로컬 경로가 존재하지 않습니다: {model_path}")
            raise ValueError(f"로컬 경로가 존재하지 않습니다: {model_path}")
        
        # LoRA 관련 파일 확인
        lora_files = [
            "adapter_config.json",
            "adapter_model.bin", 
            "adapter_model.safetensors",
            "peft_config.json"
        ]
        
        if any((path / f).exists() for f in lora_files):
            return ModelType.LORA
        
        # 풀 모델 파일 확인
        full_model_files = [
            "pytorch_model.bin",
            "model.safetensors"
        ]
        
        if any((path / f).exists() for f in full_model_files):
            return ModelType.FULL_FINETUNED
        
        raise ValueError(f"지원되지 않는 모델 타입: {model_path}")


class VLLMLauncher:
    """vLLM 서버 런처 클래스 - 다중 모델 타입 지원"""
    
    def __init__(self, config: Optional[VLLMConfig] = None):
        self.config = config or get_vllm_config()
        self.process: Optional[subprocess.Popen] = None
        self.server_args = VLLMServerArgs(self.config)
        self.detector = ModelDetector()
        
    def validate_model_setup(self) -> bool:
        """현재 모델 설정 유효성 검사"""
        model_config = self.config.get_current_model_config()
        
        logger.info(f"모델 타입: {model_config.model_type}")
        logger.info(f"모델 경로: {model_config.model_path}")
        
        if model_config.model_type == ModelType.LORA:
            return self._validate_lora_model(model_config)
        elif model_config.model_type == ModelType.FULL_FINETUNED:
            return self._validate_full_finetuned_model(model_config)
        else:
            logger.error(f"지원되지 않는 모델 타입: {model_config.model_type}")
            return False
    
    def _validate_lora_model(self, model_config) -> bool:
        """LoRA 모델 검증"""
        # 베이스 모델 경로 확인
        if not model_config.base_model_path:
            logger.error("LoRA 모델에는 base_model_path가 필요합니다.")
            return False
        
        # 베이스 모델 존재 확인
        if not self._check_model_exists(model_config.base_model_path):
            logger.error(f"베이스 모델을 찾을 수 없습니다: {model_config.base_model_path}")
            return False
        
        # LoRA 어댑터 확인
        if not self._check_model_exists(model_config.model_path):
            logger.error(f"LoRA 어댑터를 찾을 수 없습니다: {model_config.model_path}")
            return False
        
        logger.info("✅ LoRA 모델 설정이 유효합니다.")
        return True
    
    def _validate_full_finetuned_model(self, model_config) -> bool:
        """풀 파인튜닝 모델 검증"""
        if not self._check_model_exists(model_config.model_path):
            logger.error(f"풀 파인튜닝 모델을 찾을 수 없습니다: {model_config.model_path}")
            return False
        
        logger.info("✅ 풀 파인튜닝 모델 설정이 유효합니다.")
        return True
    
    def _check_model_exists(self, model_path: str) -> bool:
        """모델 존재 여부 확인"""
        # 허깅페이스 모델 이름 형식 체크
        if "/" in model_path and not model_path.startswith(("./", "/")):
            try:
                api = HfApi()
                api.model_info(model_path)
                logger.info(f"허깅페이스 모델 확인: {model_path}")
                return True
            except RepositoryNotFoundError:
                return False
            except Exception as e:
                logger.warning(f"모델 확인 중 오류: {e}")
                return False
        
        # 로컬 경로 체크
        model_path_obj = Path(model_path)
        if model_path_obj.exists():
            logger.info(f"로컬 모델 확인: {model_path}")
            return True
        
        return False
    
    def switch_model(self, model_name: str) -> bool:
        """모델 전환"""
        try:
            # 현재 서버가 실행 중이면 중지
            if self.process and self.process.poll() is None:
                logger.info("기존 서버를 중지합니다...")
                self.stop_server()
                time.sleep(2)
            
            # 모델 전환
            switch_model(model_name)
            self.config = get_vllm_config()
            self.server_args = VLLMServerArgs(self.config)
            
            logger.info(f"모델이 '{model_name}'으로 전환되었습니다.")
            return True
            
        except Exception as e:
            logger.error(f"모델 전환 실패: {e}")
            return False
    
    def auto_detect_and_configure_model(self, model_path: str) -> bool:
        """모델 자동 감지 및 설정"""
        try:
            detected_type = self.detector.detect_model_type(model_path)
            logger.info(f"감지된 모델 타입: {detected_type}")
            
            # TODO: 감지된 타입에 따라 설정 자동 업데이트
            # 현재는 수동으로 설정된 모델만 지원
            
            return True
            
        except Exception as e:
            logger.error(f"모델 자동 감지 실패: {e}")
            return False
    
    def start_server(self) -> bool:
        """vLLM 서버 시작 - 모델 타입별 최적화"""
        if self.process and self.process.poll() is None:
            logger.warning("vLLM 서버가 이미 실행 중입니다.")
            return True
        
        if not self.validate_model_setup():
            logger.error("모델 설정 검증에 실패했습니다.")
            return False
        
        # 서버 실행 명령 구성
        cmd = ["python", "-m", "vllm.entrypoints.openai.api_server"] + self.server_args.get_server_args()
        
        model_config = self.config.get_current_model_config()
        logger.info(f"🚀 {model_config.model_type.value} 모델 서버를 시작합니다...")
        logger.info(f"모델: {self.config.active_model}")
        logger.info(f"실행 명령: {' '.join(cmd)}")
        
        try:
            # 환경 변수 설정
            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": "0",
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",  # 안정성 향상
            })
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                preexec_fn=os.setsid
            )
            
            logger.info(f"vLLM 서버가 시작되었습니다. PID: {self.process.pid}")
            return self._wait_for_server_ready()
            
        except Exception as e:
            logger.error(f"vLLM 서버 시작 실패: {e}")
            return False
    
    def _wait_for_server_ready(self, timeout: int = 300) -> bool:
        """서버 준비 상태 대기 - 모델 타입별 조정된 타임아웃"""
        url = f"http://{self.config.host}:{self.config.port}/v1/models"
        start_time = time.time()
        
        model_config = self.config.get_current_model_config()
        
        # 모델 타입별 타임아웃 조정
        if model_config.model_type == ModelType.LORA:
            timeout = 240  # LoRA는 베이스 모델 + 어댑터 로딩 시간
        elif model_config.model_type == ModelType.FULL_FINETUNED:
            timeout = 360  # 풀 파인튜닝 모델은 더 긴 로딩 시간
        
        logger.info(f"서버 준비 상태를 확인합니다... (최대 {timeout}초)")
        
        while time.time() - start_time < timeout:
            if self.process and self.process.poll() is not None:
                # 프로세스 오류 로그 출력
                stdout, stderr = self.process.communicate()
                logger.error("vLLM 서버 프로세스가 종료되었습니다.")
                if stderr:
                    logger.error(f"오류 로그: {stderr.decode()}")
                return False
            
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    logger.info(f"🎉 {model_config.model_type.value} 모델 서버가 준비되었습니다!")
                    logger.info(f"서빙 모델명: {model_config.served_model_name}")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(3)
        
        logger.error(f"서버 준비 대기 시간 초과 ({timeout}초)")
        return False
    
    def stop_server(self) -> bool:
        """vLLM 서버 중지"""
        if not self.process:
            logger.info("실행 중인 vLLM 서버가 없습니다.")
            return True
        
        if self.process.poll() is not None:
            logger.info("vLLM 서버가 이미 중지되었습니다.")
            return True
        
        logger.info("vLLM 서버를 중지합니다...")
        
        try:
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            
            try:
                self.process.wait(timeout=30)
                logger.info("vLLM 서버가 정상적으로 중지되었습니다.")
                return True
            except subprocess.TimeoutExpired:
                logger.warning("정상 종료 시간 초과. 강제 종료합니다.")
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                self.process.wait()
                return True
                
        except Exception as e:
            logger.error(f"서버 중지 중 오류 발생: {e}")
            return False
        finally:
            self.process = None
    
    def restart_server(self) -> bool:
        """vLLM 서버 재시작"""
        logger.info("vLLM 서버를 재시작합니다...")
        self.stop_server()
        time.sleep(5)
        return self.start_server()
    
    def get_server_status(self) -> dict:
        """서버 상태 정보 반환"""
        model_config = self.config.get_current_model_config()
        
        status = {
            "running": self.process is not None and self.process.poll() is None,
            "pid": self.process.pid if self.process else None,
            "active_model": self.config.active_model,
            "model_type": model_config.model_type.value,
            "served_model_name": model_config.served_model_name,
            "host": self.config.host,
            "port": self.config.port
        }
        
        return status


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="다중 모델 타입 지원 vLLM 서버 런처")
    parser.add_argument("--action", choices=["start", "stop", "restart", "switch", "status"], 
                       default="start", help="실행할 액션")
    parser.add_argument("--model", help="사용할 모델 이름 (switch 액션용)")
    parser.add_argument("--port", type=int, help="서버 포트")
    
    args = parser.parse_args()
    
    launcher = VLLMLauncher()
    
    if args.action == "start":
        success = launcher.start_server()
        sys.exit(0 if success else 1)
    elif args.action == "stop":
        success = launcher.stop_server()
        sys.exit(0 if success else 1)
    elif args.action == "restart":
        success = launcher.restart_server()
        sys.exit(0 if success else 1)
    elif args.action == "switch":
        if not args.model:
            logger.error("--model 인자가 필요합니다.")
            sys.exit(1)
        success = launcher.switch_model(args.model)
        if success:
            success = launcher.start_server()
        sys.exit(0 if success else 1)
    elif args.action == "status":
        status = launcher.get_server_status()
        print("=== vLLM 서버 상태 ===")
        for key, value in status.items():
            print(f"{key}: {value}")
        sys.exit(0)


if __name__ == "__main__":
    main() 