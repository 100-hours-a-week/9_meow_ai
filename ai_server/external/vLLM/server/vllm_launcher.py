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
import psutil
from typing import Optional, List
from pathlib import Path
import requests
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError

from ai_server.external.vLLM.server.vllm_config import VLLMConfig, VLLMServerArgs, get_vllm_config, ModelType


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
        
        # 시작 시 기존 프로세스 자동 연결 시도
        self._try_reconnect_existing_process()
    
    def _try_reconnect_existing_process(self) -> bool:
        """기존 실행 중인 vLLM 프로세스 자동 연결"""
        try:
            vllm_pids = self._find_vllm_processes()
            if vllm_pids:
                # 가장 최근 프로세스에 연결
                target_pid = max(vllm_pids)
                if self._connect_to_existing_process(target_pid):
                    logger.info(f"🔗 기존 vLLM 프로세스에 연결됨: PID {target_pid}")
                    return True
        except Exception as e:
            logger.debug(f"기존 프로세스 연결 실패: {e}")
        return False
    
    def _find_vllm_processes(self) -> List[int]:
        """실행 중인 vLLM 서버 프로세스 찾기"""
        vllm_pids = []
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info['cmdline']
                    if (cmdline and 
                        any('vllm.entrypoints.openai.api_server' in str(cmd) for cmd in cmdline)):
                        # 포트도 확인
                        if f"--port {self.config.port}" in ' '.join(cmdline) or str(self.config.port) in cmdline:
                            vllm_pids.append(proc.info['pid'])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            logger.debug(f"프로세스 검색 중 오류: {e}")
        
        return vllm_pids
    
    def _connect_to_existing_process(self, pid: int) -> bool:
        """기존 프로세스에 연결"""
        try:
            # PID로 프로세스 객체 생성 (추적 목적)
            proc = psutil.Process(pid)
            if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
                # subprocess.Popen 객체는 만들 수 없지만, PID는 저장
                self.process = type('MockProcess', (), {
                    'pid': pid,
                    'poll': lambda *args, **kwargs: None if proc.is_running() else 0,
                    'wait': lambda *args, **kwargs: proc.wait(timeout=kwargs.get('timeout')),
                    'terminate': lambda *args, **kwargs: proc.terminate(),
                    'kill': lambda *args, **kwargs: proc.kill()
                })()
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, Exception) as e:
            logger.debug(f"프로세스 연결 실패 (PID: {pid}): {e}")
        return False
    
    def is_running(self) -> bool:
        """vLLM 서버 실행 상태 확인 (다중 방법)"""
        # 방법 1: 프로세스 객체 기반 확인
        if self.process:
            try:
                # Mock 프로세스인 경우 poll() 메서드 사용
                if hasattr(self.process, 'poll') and callable(self.process.poll):
                    return self.process.poll() is None
                # 실제 subprocess.Popen인 경우
                elif hasattr(self.process, 'pid'):
                    return psutil.pid_exists(self.process.pid)
            except Exception as e:
                logger.debug(f"프로세스 객체 확인 실패: {e}")
        
        # 방법 2: PID 기반 프로세스 존재 확인
        if self._check_vllm_process_exists():
            # 기존 프로세스 재연결 시도
            self._try_reconnect_existing_process()
            return True
        
        # 방법 3: 포트 기반 HTTP 헬스체크
        return self._check_server_health()
    
    def _check_vllm_process_exists(self) -> bool:
        """PID 기반 vLLM 프로세스 존재 확인"""
        vllm_pids = self._find_vllm_processes()
        return len(vllm_pids) > 0
    
    def _check_server_health(self) -> bool:
        """HTTP 요청으로 서버 상태 확인"""
        try:
            url = f"http://{self.config.host}:{self.config.port}/health"
            response = requests.get(url, timeout=3)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            pass
        
        # /health 엔드포인트가 없는 경우 /v1/models 시도
        try:
            url = f"http://{self.config.host}:{self.config.port}/v1/models"
            response = requests.get(url, timeout=3)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def get_process_info(self) -> Optional[dict]:
        """실행 중인 vLLM 프로세스 상세 정보"""
        try:
            vllm_pids = self._find_vllm_processes()
            if not vllm_pids:
                return None
            
            # 가장 최근 프로세스 정보 반환
            target_pid = max(vllm_pids)
            proc = psutil.Process(target_pid)
            
            return {
                "pid": target_pid,
                "status": proc.status(),
                "cpu_percent": proc.cpu_percent(),
                "memory_percent": proc.memory_percent(),
                "create_time": proc.create_time(),
                "cmdline": proc.cmdline(),
                "connections": [
                    {"laddr": conn.laddr, "status": conn.status}
                    for conn in proc.connections()
                    if conn.laddr.port == self.config.port
                ]
            }
        except Exception as e:
            logger.debug(f"프로세스 정보 조회 실패: {e}")
            return None
    
    def start_server(self) -> bool:
        """vLLM 서버 시작 - 모델 타입별 최적화"""
        if self.process and self.process.poll() is None:
            logger.warning("vLLM 서버가 이미 실행 중입니다.")
            return True
        
        # 서버 실행 명령 구성
        cmd = ["python", "-m", "vllm.entrypoints.openai.api_server"] + self.server_args.get_server_args()
        
        model_config = self.config.get_current_model_config()
        logger.info(f"{model_config.model_type.value} 모델 서버를 시작합니다...")
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
                    logger.info(f"{model_config.model_type.value} 모델 서버가 준비되었습니다!")
                    logger.info(f"서빙 모델명: {model_config.served_model_name}")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(3)
        
        logger.error(f"서버 준비 대기 시간 초과 ({timeout}초)")
        return False
    
    def stop_server(self) -> bool:
        """vLLM 서버 중지 - 개선된 프로세스 처리"""
        # 프로세스 객체가 없는 경우, 실행 중인 vLLM 프로세스 찾아서 중지
        if not self.process:
            vllm_pids = self._find_vllm_processes()
            if not vllm_pids:
                logger.info("실행 중인 vLLM 서버가 없습니다.")
                return True
            
            # 모든 vLLM 프로세스 중지
            for pid in vllm_pids:
                try:
                    proc = psutil.Process(pid)
                    logger.info(f"발견된 vLLM 프로세스 중지 중: PID {pid}")
                    proc.terminate()
                    proc.wait(timeout=30)
                    logger.info(f"vLLM 프로세스가 정상적으로 중지됨: PID {pid}")
                except psutil.TimeoutExpired:
                    logger.warning(f"정상 종료 시간 초과. 강제 종료: PID {pid}")
                    proc.kill()
                except Exception as e:
                    logger.error(f"프로세스 중지 중 오류 (PID {pid}): {e}")
            return True
        
        # 프로세스 상태 확인
        try:
            if hasattr(self.process, 'poll') and callable(self.process.poll):
                if self.process.poll() is not None:
                    logger.info("vLLM 서버가 이미 중지되었습니다.")
                    self.process = None
                    return True
        except Exception:
            pass
        
        logger.info("vLLM 서버를 중지합니다...")
        
        try:
            # Mock 프로세스인 경우
            if hasattr(self.process, 'terminate') and not hasattr(self.process, 'communicate'):
                try:
                    self.process.terminate()
                    if hasattr(self.process, 'wait'):
                        self.process.wait(timeout=30)
                    logger.info("vLLM 서버가 정상적으로 중지되었습니다.")
                    return True
                except Exception as e:
                    logger.warning(f"정상 종료 실패, 강제 종료 시도: {e}")
                    if hasattr(self.process, 'kill'):
                        self.process.kill()
                    return True
            
            # 실제 subprocess.Popen인 경우
            else:
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
        """개선된 서버 상태 정보 반환"""
        model_config = self.config.get_current_model_config()
        
        # 기본 상태 정보
        is_running = self.is_running()
        process_info = self.get_process_info() if is_running else None
        
        status = {
            "running": is_running,
            "pid": process_info["pid"] if process_info else (self.process.pid if self.process else None),
            "active_model": self.config.active_model,
            "model_type": model_config.model_type.value,
            "served_model_name": model_config.served_model_name,
            "host": self.config.host,
            "port": self.config.port,
            
            # 추가 상태 정보
            "health_check": {
                "process_exists": self._check_vllm_process_exists(),
                "http_accessible": self._check_server_health(),
                "process_tracked": self.process is not None
            }
        }
        
        # 프로세스 상세 정보 추가
        if process_info:
            status.update({
                "process_status": process_info["status"],
                "cpu_percent": process_info["cpu_percent"],
                "memory_percent": process_info["memory_percent"],
                "uptime": time.time() - process_info["create_time"]
            })
        
        return status


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="다중 모델 타입 지원 vLLM 서버 런처")
    parser.add_argument("--action", choices=["start", "stop", "restart", "status"], 
                       default="start", help="실행할 액션")
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
    elif args.action == "status":
        status = launcher.get_server_status()
        print("=== vLLM 서버 상태 ===")
        for key, value in status.items():
            print(f"{key}: {value}")
        sys.exit(0)


if __name__ == "__main__":
    main() 