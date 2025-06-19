"""
vLLM 서버 런처
포스트 문장 생성을 위한 간소화된 vLLM 서버 시작 및 관리
"""

import os
import sys
import subprocess
import signal
import time
import logging
from typing import Optional
from pathlib import Path

from .vllm_config import VLLMConfig, VLLMServerArgs, get_vllm_config


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VLLMLauncher:
    """vLLM 서버 런처 클래스 - 포스트 생성 최적화"""
    
    def __init__(self, config: Optional[VLLMConfig] = None):
        self.config = config or get_vllm_config()
        self.process: Optional[subprocess.Popen] = None
        self.server_args = VLLMServerArgs(self.config)
        
    def validate_model_path(self) -> bool:
        """모델 경로 유효성 검사"""
        model_path_str = self.config.model_path
        
        # 허깅페이스 모델 이름 형식 체크
        if "/" in model_path_str and not model_path_str.startswith("./") and not model_path_str.startswith("/"):
            logger.info(f"허깅페이스 모델 감지: {model_path_str}")
            return True
        
        # 로컬 경로 체크
        model_path = Path(model_path_str)
        if not model_path.exists():
            logger.error(f"모델 경로를 찾을 수 없습니다: {model_path}")
            return False
        
        return True
    
    def start_server(self) -> bool:
        """vLLM 서버 시작 - 포스트 생성 최적화"""
        if self.process and self.process.poll() is None:
            logger.warning("vLLM 서버가 이미 실행 중입니다.")
            return True
        
        if not self.validate_model_path():
            return False
        
        # 서버 실행 명령 구성
        cmd = ["python", "-m", "vllm.entrypoints.openai.api_server"] + self.server_args.get_server_args()
        
        logger.info("포스트 생성용 vLLM 서버를 시작합니다...")
        logger.info(f"실행 명령: {' '.join(cmd)}")
        
        try:
            # 환경 변수 설정
            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": "0",
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
    
    def _wait_for_server_ready(self, timeout: int = 180) -> bool:
        """서버 준비 상태 대기 - 포스트 생성용으로 단축"""
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
                    logger.info("포스트 생성용 vLLM 서버가 준비되었습니다!")
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


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="포스트 생성용 vLLM 서버 런처")
    parser.add_argument("--action", choices=["start", "stop", "restart"], 
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


if __name__ == "__main__":
    main() 