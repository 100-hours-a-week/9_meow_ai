"""
vLLM 서버 런처 - 간소화 버전
"""

import subprocess
import time
import logging
import psutil
from typing import Optional
import requests

from ai_server.external.vLLM.server.vllm_config import VLLMConfig, VLLMServerArgs, get_vllm_config

logger = logging.getLogger(__name__)


class VLLMLauncher:
    """간소화된 vLLM 서버 런처"""
    
    def __init__(self, config: Optional[VLLMConfig] = None):
        self.config = config or get_vllm_config()
        self.process: Optional[subprocess.Popen] = None
        self.server_args = VLLMServerArgs(self.config)
    
    def is_running(self) -> bool:
        """서버 실행 상태 확인"""
        if self.process is None:
            return False
        
        # 프로세스 상태 확인
        if self.process.poll() is not None:
            return False
        
        # 서버 응답 확인
        return self._check_server_health()
    
    def _check_server_health(self) -> bool:
        """서버 상태 확인"""
        try:
            response = requests.get(f"http://{self.config.host}:{self.config.port}/v1/models", timeout=3)
            return response.status_code == 200
        except:
            return False
    
    def start_server(self) -> bool:
        """서버 시작"""
        if self.is_running():
            return True
        
        try:
            args = ["python", "-m", "vllm.entrypoints.openai.api_server"] + self.server_args.get_server_args()
            
            self.process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # 서버 준비 대기
            if self._wait_for_server_ready():
                logger.info("vLLM 서버가 시작되었습니다")
                return True
            else:
                logger.error("vLLM 서버 시작 실패")
                self.stop_server()
                return False
                
        except Exception as e:
            logger.error(f"vLLM 서버 시작 중 오류: {e}")
            return False
    
    def _wait_for_server_ready(self, timeout: int = 300) -> bool:
        """서버 준비 대기"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self._check_server_health():
                return True
            time.sleep(2)
        
        return False
    
    def stop_server(self) -> bool:
        """서버 중지"""
        if self.process is None:
            return True
        
        try:
            self.process.terminate()
            
            # 정상 종료 대기
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            
            self.process = None
            logger.info("vLLM 서버가 중지되었습니다")
            return True
            
        except Exception as e:
            logger.error(f"vLLM 서버 중지 중 오류: {e}")
            return False
    
    def restart_server(self) -> bool:
        """서버 재시작"""
        self.stop_server()
        time.sleep(2)
        return self.start_server()


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="vLLM 서버 런처")
    parser.add_argument("--action", choices=["start", "stop", "restart"], required=True)
    
    args = parser.parse_args()
    launcher = VLLMLauncher()
    
    if args.action == "start":
        success = launcher.start_server()
    elif args.action == "stop":
        success = launcher.stop_server()
    elif args.action == "restart":
        success = launcher.restart_server()
    
    exit(0 if success else 1)


if __name__ == "__main__":
    main() 