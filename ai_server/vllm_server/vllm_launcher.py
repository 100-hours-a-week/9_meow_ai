"""
vLLM 서버 런처
HyperCLOVAX-1.5B_LoRA_fp16 모델을 위한 vLLM 서버 시작 및 관리
"""

import os
import sys
import subprocess
import signal
import time
import logging
from typing import Optional, List
from pathlib import Path

from .vllm_config import VLLMConfig, VLLMServerArgs, get_vllm_config


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VLLMLauncher:
    """vLLM 서버 런처 클래스"""
    
    def __init__(self, config: Optional[VLLMConfig] = None):
        self.config = config or get_vllm_config()
        self.process: Optional[subprocess.Popen] = None
        self.server_args = VLLMServerArgs(self.config)
        
    def validate_model_path(self) -> bool:
        """모델 경로 유효성 검사"""
        model_path_str = self.config.model_path
        
        # 허깅페이스 모델 이름 형식 체크 (org/model-name)
        if "/" in model_path_str and not model_path_str.startswith("./") and not model_path_str.startswith("/"):
            logger.info(f"허깅페이스 모델 이름 감지: {model_path_str}")
            logger.info("vLLM이 자동으로 모델을 다운로드합니다.")
            return True
        
        # 로컬 경로 체크
        model_path = Path(model_path_str)
        if not model_path.exists():
            logger.error(f"모델 경로를 찾을 수 없습니다: {model_path}")
            return False
        
        # 모델 파일들 확인
        required_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
        missing_files = []
        
        for file_name in required_files:
            if not (model_path / file_name).exists():
                missing_files.append(file_name)
        
        if missing_files:
            logger.warning(f"일부 모델 파일이 누락되었습니다: {missing_files}")
            logger.info("vLLM이 자동으로 처리할 수 있는 경우가 있습니다.")
        
        return True
    
    def check_gpu_availability(self) -> bool:
        """GPU 사용 가능성 확인"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                logger.info(f"사용 가능한 GPU 수: {gpu_count}")
                
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    logger.info(f"GPU {i}: {gpu_name}, 메모리: {memory_total:.1f}GB")
                
                return gpu_count >= self.config.tensor_parallel_size
            else:
                logger.warning("CUDA를 사용할 수 없습니다. CPU 모드로 실행됩니다.")
                return False
        except ImportError:
            logger.error("PyTorch가 설치되지 않았습니다.")
            return False
    
    def start_server(self) -> bool:
        """vLLM 서버 시작"""
        if self.process and self.process.poll() is None:
            logger.warning("vLLM 서버가 이미 실행 중입니다.")
            return True
        
        # 사전 검사
        if not self.validate_model_path():
            return False
        
        self.check_gpu_availability()
        
        # 서버 실행 명령 구성
        cmd = ["python", "-m", "vllm.entrypoints.openai.api_server"] + self.server_args.get_server_args()
        
        logger.info("vLLM 서버를 시작합니다...")
        logger.info(f"실행 명령: {' '.join(cmd)}")
        
        try:
            # 환경 변수 설정
            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": "0",  # 첫 번째 GPU 사용
                "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",  # Flash Attention 사용
            })
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                preexec_fn=os.setsid  # 프로세스 그룹 생성
            )
            
            logger.info(f"vLLM 서버가 시작되었습니다. PID: {self.process.pid}")
            
            # 서버 시작 대기
            return self._wait_for_server_ready()
            
        except Exception as e:
            logger.error(f"vLLM 서버 시작 실패: {e}")
            return False
    
    def _wait_for_server_ready(self, timeout: int = 300) -> bool:
        """서버 준비 상태 대기"""
        import requests
        
        url = f"http://{self.config.host}:{self.config.port}/v1/models"
        start_time = time.time()
        
        logger.info("서버 준비 상태를 확인합니다...")
        
        while time.time() - start_time < timeout:
            if self.process and self.process.poll() is not None:
                logger.error("vLLM 서버 프로세스가 종료되었습니다.")
                return False
            
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    logger.info("vLLM 서버가 준비되었습니다!")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(5)
        
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
            # 프로세스 그룹 전체에 SIGTERM 전송
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            
            # 정상 종료 대기
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
        time.sleep(5)  # 포트 해제 대기
        return self.start_server()
    
    def get_server_status(self) -> dict:
        """서버 상태 정보 반환"""
        status = {
            "running": False,
            "pid": None,
            "config": self.config.dict(),
        }
        
        if self.process:
            status["running"] = self.process.poll() is None
            status["pid"] = self.process.pid
        
        return status


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="vLLM 서버 런처")
    parser.add_argument("--action", choices=["start", "stop", "restart", "status"], 
                       default="start", help="실행할 액션")
    parser.add_argument("--model-path", help="모델 경로")
    parser.add_argument("--port", type=int, help="서버 포트")
    
    args = parser.parse_args()
    
    # 설정 업데이트
    config_kwargs = {}
    if args.model_path:
        config_kwargs["model_path"] = args.model_path
    if args.port:
        config_kwargs["port"] = args.port
    
    if config_kwargs:
        from .vllm_config import update_vllm_config
        update_vllm_config(**config_kwargs)
    
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
    elif args.action == "status":
        status = launcher.get_server_status()
        print(f"서버 실행 상태: {status}")


if __name__ == "__main__":
    main() 